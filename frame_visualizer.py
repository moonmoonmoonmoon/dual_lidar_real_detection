#!/usr/bin/env python3
# frame_visualizer.py
"""
针对特定帧的LiDAR处理过程可视化工具
可以指定某一帧，显示其处理过程中的各个步骤的可视化结果
"""
import argparse
import numpy as np
import open3d as o3d
import os
import time
from pathlib import Path
from ouster.sdk import open_source, client
from src.data.dual_lidar_simulator import DualLidarSimulator
from src.utils.visualization import Visualizer
from src.processing.object_detector import LidarObjectDetector
from src.processing.slam_processor import SLAMProcessor
from src.utils.lidar_processing_utils import transform_points_fast

class FrameVisualizer:
    def __init__(self, config):
        """初始化帧可视化工具"""
        self.config = config
        self.vis = Visualizer()

        # 初始化模拟器和处理器
        self.simulator = DualLidarSimulator(
            config['pcap_path1'],
            config['meta_path1'],
            config['pcap_path2'],
            config['meta_path2'],
            config['frame_rate']
        )

        # 初始化SLAM处理器
        self.slam = SLAMProcessor(config.get('slam_config', {}))
        try:
            data_source = open_source(config.get('pcap_path1'), sensor_idx=-1)
            self.metadata = data_source.metadata
            self.slam.initialize(self.metadata)
        except Exception as e:
            print(f"加载元数据文件失败: {e}")
            raise

        # 初始化目标检测器 - 修改配置以启用可视化
        detector_config = config.get('detector_config', {}).copy()
        # 启用所有可视化选项
        detector_config['visualization'] = {
            'enable': True,
            'show_ground_removal': True,
            'show_height_filter': True,
            'show_roi': True,
            'show_clusters': True
        }
        self.detector = LidarObjectDetector(detector_config)

        # 加载变换矩阵
        self.target_transform_matrix = np.load(
            '/home/yanan/Downloads/projects/test_code/2025_bus_dual_simulate_real_detect/data/self_ground_target_1778.npy')
        self.target_transform_inv = np.linalg.inv(self.target_transform_matrix)

    def process_specific_frame(self, target_frame_id):
        """处理指定帧ID的数据并可视化每个步骤"""
        print(f"正在查找帧 {target_frame_id}...")

        # 启动模拟器
        self.simulator.start()

        try:
            current_frame_id = 0
            while True:
                # 获取下一帧
                frame = self.simulator.get_frame()
                if frame is None:
                    continue

                current_frame_id += 1
                print(f"当前帧: {current_frame_id}", end='\r')

                # 找到目标帧
                if current_frame_id == target_frame_id:
                    print(f"\n找到目标帧 {target_frame_id}，开始处理...")
                    self._visualize_frame_processing(frame)
                    break

                # 如果已经超过目标帧，可能目标帧不存在
                if current_frame_id > target_frame_id:
                    print(f"\n错误: 目标帧 {target_frame_id} 不存在或已跳过")
                    break
        finally:
            # 停止模拟器
            self.simulator.stop()

    def _visualize_frame_processing(self, frame):
        """可视化一帧的所有处理步骤"""
        print("1. 处理SLAM...")
        slam_result = self.slam.process_scan(frame.scan)
        if not slam_result:
            print("SLAM处理失败")
            return

        print("2. 转换点云到世界坐标系...")
        final_transform = np.dot(self.target_transform_matrix,
                                 np.dot(slam_result.pose_matrix,
                                        self.target_transform_inv))
        points_world = transform_points_fast(frame.points, final_transform)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)

        # # 保存点云以便后续查看
        # os.makedirs("output/frame_visualizer", exist_ok=True)
        # raw_pcd_path = f"output/frame_visualizer/raw_pcd.pcd"
        # o3d.io.write_point_cloud(raw_pcd_path, pcd)
        # print(f"原始点云已保存到 {raw_pcd_path}")

        # 显示原始点云
        print("显示原始点云...")
        self.vis.visualize_point_cloud(pcd)

        # 使用目标检测器处理帧 - 将显示每个步骤的可视化结果
        print("3. 执行目标检测处理流程...")
        # 由于我们已启用目标检测器的可视化选项，
        # 下面的调用将在每个步骤停止并显示可视化结果
        detection_result = self.detector.process_frame(
            frame_id=frame.frame_id,
            pcd=pcd,
            final_transform=final_transform,
            timestamp=slam_result.timestamp
        )

        if detection_result:
            print(f"检测到 {len(detection_result.objects)} 个物体")
            for i, obj in enumerate(detection_result.objects):
                print(f"物体 {i + 1}: 类型={obj['label']}, 位置={obj['center']}, 速度={obj['velocity']}")
        else:
            print("目标检测失败或未检测到物体")


def main():
    parser = argparse.ArgumentParser(description='单帧可视化工具')
    parser.add_argument('frame_id', type=int, help='要可视化的帧ID')
    args = parser.parse_args()

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

        # 检测器配置 - 内部会修改visualization设置
        'detector_config': {
            'ground_removal': {
                'z_threshold': 0.3,
                'distance_threshold': 0.1,
                'ransac_n': 3,
                'num_iterations': 100000
            },
            'height_filter': {
                'min_height': 0.1,
                'max_height': 3
            },
            'coordinate_transform': {
                'yaw_angle_deg': 90,  # Rotate the point cloud counterclockwise by 90 degrees.
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
            }
        }
    }

    # 创建可视化工具并处理指定帧
    visualizer = FrameVisualizer(config)
    visualizer.process_specific_frame(args.frame_id)


if __name__ == "__main__":
    main()