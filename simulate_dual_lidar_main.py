# 5_main.py
"""主程序,整合所有模块"""

from src.data.lidar_simulator import LidarSimulator
from src.processing.slam_processor import SLAMProcessor
from src.processing.object_detector import LidarObjectDetector
from src.processing.safety_analyzer import SafetyAnalyzer
from src.data.dual_lidar_simulator import DualLidarSimulator
import numpy as np
import time
import logging
from typing import Dict, Optional
import json
import ouster.sdk.client as client
from ouster.sdk import open_source
import open3d as o3d
import csv
import os
from datetime import datetime
from src.utils.lidar_processing_utils import transform_points_fast

class RealTimeLidarSystem:
    def __init__(self, config: Dict):
        """初始化实时处理系统"""
        self.config = config
        self.logger = self._setup_logger()

        # 创建output目录
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 生成CSV文件名（使用时间戳避免覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.output_dir, f"lidar_results_{timestamp}.csv")

        # 初始化CSV文件头
        self._initialize_csv()

        # 初始化各个模块
        # 初始化双LiDAR模拟器
        self.simulator = DualLidarSimulator(
            config['pcap_path1'],
            config['meta_path1'],
            config['pcap_path2'],
            config['meta_path2'],
            config['frame_rate']
        )

        self.slam = SLAMProcessor(config.get('slam_config', {}))
        try:
            data_source = open_source(config.get('pcap_path1'), sensor_idx=-1)
            self.metadata = data_source.metadata
            self.slam.initialize(self.metadata)
        except Exception as e:
            self.logger.error(f"加载元数据文件失败: {e}")
            raise

        self.detector = LidarObjectDetector(config.get('detector_config', {}))
        self.analyzer = SafetyAnalyzer(config.get('safety_config', {}))

        self.frame_count = 0
        self.processing_times = []
        self.total_triggers = 0
        self.trigger_frames = []
        self.transformation_matrix = []
        # 加载变换矩阵
        self.target_transform_matrix = np.load(
            '/home/yanan/Downloads/projects/test_code/2025_bus_dual_simulate_real_detect/data/self_ground_target_1778.npy')
        self.target_transform_inv = np.linalg.inv(self.target_transform_matrix)

    def _setup_logger(self):
        """配置日志"""
        logger = logging.getLogger("RealTimeLidarSystem")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _initialize_csv(self):
        """初始化CSV文件并写入表头"""
        headers = [
            'frame_id',
            'process_time_ms',
            # 安全分析指标
            'trigger_required',
            'trigger_reason',
            'lateral_distance',
            'longitudinal_distance',
            'safety_distance',
            'lateral_longitudinal_projected_position',
            'lateral_longitudinal_projected_velocity',
            'is_oncoming',
            # 检测对象相关
            'object_type',
            'object_pos',
            'object_vel',
            'object_dim',
            'object_yaw',
            # 自车信息
            'ego_pos',
            'ego_vel',

        ]
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _write_result_to_csv(self, result: Dict):
        """将处理结果写入CSV文件"""
        if result is None:
            return


        # 对每个检测到的对象和对应的安全分析结果写入一行
        for safety_result in result['safety_results']:
            object_info = safety_result.object_info
            risk_metrics = safety_result.risk_metrics

            row_data = [
                # 基本信息
                result['frame_id'],
                result['process_time'] * 1000,

                # 安全分析结果
                safety_result.trigger_required,
                safety_result.trigger_reason if safety_result.trigger_required else "None",
                risk_metrics['lateral_distance'],
                risk_metrics['longitudinal_distance'],
                risk_metrics['safety_distance'],
                object_info['ego_frame_position'],
                object_info['ego_frame_velocity'],
                object_info['is_oncoming'],



                # 对象信息
                object_info['type'],
                object_info['position'],
                object_info['velocity'],
                object_info['dimensions'],
                object_info['yaw'],
                # 自车信息
                result['detection_result'].ego_position,
                result['detection_result'].ego_velocity,

            ]

            # 写入CSV文件
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

    def process_frame(self, frame) -> Optional[Dict]:
        """处理单帧数据"""
        try:
            start_time = time.time()

            frame_start = time.time()
            slam_start = time.time()
            # 1. SLAM处理
            slam_result = self.slam.process_scan(frame.scan)
            # print('slam_result',slam_result)
            slam_time = time.time() - slam_start
            print(f"SLAM processing time: {slam_time * 1000:.1f}ms")
            if not slam_result:
                self.logger.warning("SLAM处理失败")
                return None
            # 2. 点云转换
            transform_start = time.time()
            # # 将点云转换到世界坐标系

            pcd = o3d.geometry.PointCloud()  # type: ignore
            # points_world = transform_points_fast(frame.points, slam_result.pose_matrix)
            final_transform = np.dot(self.target_transform_matrix,
                                     np.dot(slam_result.pose_matrix,
                                            self.target_transform_inv))
            points_world = transform_points_fast(frame.points, final_transform)
            # print(points_world.shape)

            pcd.points = o3d.utility.Vector3dVector(points_world)
            transform_time = time.time() - transform_start
            print(f"Point cloud transform time: {transform_time * 1000:.1f}ms")
            # 3. 目标检测
            detection_start = time.time()
            detection_result = self.detector.process_frame(
                frame.frame_id,
                pcd,
                # slam_result.pose_matrix,
                final_transform,
                slam_result.timestamp
            )
            detection_time = time.time() - detection_start
            print(f"Object detection time: {detection_time * 1000:.1f}ms")
            # 检查检测结果 - 如果没有检测到物体，返回一个安全的状态
            if detection_result is None:
                # 返回一个表示安全状态的结果
                return {
                    'frame_id': self.frame_count,
                    'slam_result': slam_result,
                    'detection_result': None,
                    'safety_results': [],  # 空列表表示没有风险
                    'process_time': time.time() - start_time
                }

            # 4. 安全分析
            safety_start = time.time()
            safety_results = self.analyzer.process_frame(
                detection_result.objects,
                {
                    'frame_id': self.frame_count,
                    'position': detection_result.ego_position,
                    'velocity': detection_result.ego_velocity
                },
                pcd  # 传入点云数据
            )

            safety_time = time.time() - safety_start
            print(f"Safety analysis time: {safety_time * 1000:.1f}ms")
            # 记录处理时间
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            total_time = time.time() - frame_start
            print(f"Total frame processing time: {total_time * 1000:.1f}ms\n")

            # 返回结果
            return {
                'frame_id': self.frame_count,
                'slam_result': slam_result,
                'detection_result': detection_result,
                'safety_results': safety_results,
                'process_time': process_time
            }

        except Exception as e:
            self.logger.error(f"处理错误: {e}")
            return None

    def run(self):
        """运行系统"""
        try:
            self.simulator.start()
            self.logger.info("系统启动")

            while True:
                # 获取数据
                get_data_start_time = time.time()
                frame = self.simulator.get_frame()
                if frame is None:
                    continue
                # print('get the frame data')
                get_data_end_time = time.time()
                get_data_time = get_data_end_time - get_data_start_time
                print(f'get_data_time : {get_data_time *1000:.1f}ms')
                # 处理数据
                result = self.process_frame(frame)
                if result:
                    self.frame_count += 1
                    # print('result',result,'frame_count',self.frame_count)
                    # 将结果写入CSV文件
                    self._write_result_to_csv(result)

                    # 检查安全触发
                    triggers = [r for r in result['safety_results'] if r.trigger_required]
                    if triggers:
                        self.logger.warning(
                            f"帧 {self.frame_count} 检测到安全风险! "
                            f"触发原因: {[t.trigger_reason for t in triggers]}"
                        )

                    # 输出处理状态
                    if self.frame_count % 10 == 0:  # 每10帧显示一次状态
                        avg_time = np.mean(self.processing_times[-10:])
                        self.logger.info(
                            f"已处理 {self.frame_count} 帧, "
                            f"平均处理时间: {avg_time * 1000:.1f}ms"
                        )

        except KeyboardInterrupt:
            self.logger.info("正在停止系统...")
        finally:
            self.simulator.stop()
            avg_time = np.mean(self.processing_times)
            # 在结束时获取并输出统计信息
            stats = self.get_final_statistics()
            self.total_triggers = stats['total_triggers']
            self.trigger_frames = stats['trigger_frames']
            self.logger.info(
                f"系统已停止\n"
                f"总帧数: {self.frame_count}\n"
                f"总触发次数: {self.total_triggers}\n"
                f"触发帧列表: {self.trigger_frames}\n"
                f"\n触发类型统计: {stats['trigger_stats']}"
                f"平均处理时间: {avg_time*1000:.1f}ms\n"
                f"最大处理时间: {max(self.processing_times)*1000:.1f}ms\n"
                f"最小处理时间: {min(self.processing_times)*1000:.1f}ms"
            )

    def get_final_statistics(self) -> Dict:
        """获取最终统计信息"""
        # 从安全分析器获取统计信息
        trigger_history = self.analyzer.get_trigger_history()
        trigger_frames = [result.frame_id for result in trigger_history]

        return {
            'total_frames': self.frame_count,
            'total_triggers': self.analyzer.total_triggers,
            'trigger_frames': trigger_frames,
            'trigger_stats': self.analyzer.trigger_stats,
            'avg_process_time': np.mean(self.processing_times) * 1000,
            'max_process_time': max(self.processing_times) * 1000,
            'min_process_time': min(self.processing_times) * 1000
        }
def main():
    """主函数"""
    # 系统配置
    config = {
        # 双LiDAR模拟器配置
        'pcap_path1': "/home/yanan/Downloads/data/raw_data/2025_bus/Left/20250124_1250_OS-1-128_122211001778.pcap",
        'meta_path1': "/home/yanan/Downloads/data/raw_data/2025_bus/Left/20250124_1250_OS-1-128_122211001778.json",
        'pcap_path2': "/home/yanan/Downloads/data/raw_data/2025_bus/Right/20250124_1250_OS-1-128_122211001621.pcap",
        'meta_path2': "/home/yanan/Downloads/data/raw_data/2025_bus/Right/20250124_1250_OS-1-128_122211001621.json",
        'frame_rate': 10.0,  # 10Hz

        # SLAM配置
        'slam_config': {
            'max_range': 75.0,
            'min_range': 1.0,
            'voxel_size': 1.0
        },

        # 检测器配置
        'detector_config': {
            'ground_removal': {
                'z_threshold': 0.28,
                'show_plane': False,
                'distance_threshold': 0.1,
                'ransac_n': 3,
                'num_iterations': 100000
            },
            'height_filter': {
                'min_height': -5,
                'max_height': 5
            },
            'coordinate_transform': {
                'yaw_angle_deg': 90,  #Rotate the point cloud counterclockwise by 90 degrees.
                'enable': True
            },
            'lidar': {
                'x_offset': 4,
                'y_offset': 1.3,
            },
            'roi': {
                'length': 20,
                'width': 7,
                'height': 4
            },
            'ego_roi': {
                 'length': 8,
                 'width': 2.8,
                 'height': 3.88
            },
            'clustering': {
                'voxel_size': 0.05,
                'eps': 1.25,
                'min_samples': 80,
                'min_points': 10
            },
            'visualization': {
                'enable': False,  # 总开关
                'show_ground_removal': True,  # 显示地面移除
                'show_height_filter': True,  # 显示高度过滤
                'show_roi': True,  # 显示ROI裁剪
                'show_clusters': True  # 显示聚类结果
            }
        },

        # 安全分析配置
        'safety_config': {
            'method': 'physics',
            'alpha': 0.1,
            'min_lateral_distance': 0.3,
            'min_longitudinal_distance': 1.5,
            'max_longitudinal_distance': 50.0,
            'block_lateral_threshold': 0.2,
            'reaction_time': 1.0,
            'deceleration': 7.84,
            'ego_half_width': 1
        }
    }

    # 创建并运行系统
    system = RealTimeLidarSystem(config)
    system.run()

if __name__ == "__main__":
    main()