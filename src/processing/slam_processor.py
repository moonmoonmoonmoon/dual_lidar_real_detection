# 2_slam_processor.py
# consider estimate pose matrix when data loss
"""实时SLAM处理模块,用于获取位姿估计"""

import numpy as np
from ouster.sdk.mapping.slam import KissBackend
import ouster.sdk.client as client
from dataclasses import dataclass
from typing import Optional, Tuple
from ouster.sdk import open_source

@dataclass
class SLAMResult:
    """SLAM处理结果"""
    pose_matrix: np.ndarray
    timestamp: np.uint64
    status: bool

class SLAMProcessor:
    def __init__(self, config: dict):
        """初始化SLAM处理器"""
        self.config = {
            'max_range': 75.0,
            'min_range': 1.0,
            'voxel_size': 1.0
        }
        if config:
            self.config.update(config)

        self.slam = None
        self.last_pose = np.eye(4)
        self.initialized = False

    def initialize(self, metadata):
        """使用sensor元数据初始化SLAM"""
        self.slam = KissBackend(
            metadata,
            max_range=self.config['max_range'],
            min_range=self.config['min_range'],
            voxel_size=self.config['voxel_size']
        )
        self.initialized = True
        print("SLAM初始化完成")
        # self.slam.update()

    def process_scan(self, scan) -> Optional[SLAMResult]:
        """处理单次扫描并返回位姿估计"""
        if not self.initialized:
            print('no initialize')
            return None

        try:
            # 检查点云数据质量
            data_is_incomplete = False
            if scan and scan[0]:
                # 获取点云数据
                range_data = scan[0].field(client.ChanField.RANGE)
                # 计算值为0的点的数量
                zero_points_count = np.sum(range_data == 0)
                # 计算总点数
                total_points = range_data.size
                # 计算零值点的占比
                zero_points_ratio = zero_points_count / total_points

                # 如果零值点占比超过阈值（例如70%），则认为数据不完整
                if zero_points_ratio > 0.87:
                    print(f"数据帧不完整，零值点占比: {zero_points_ratio:.2%}，使用位姿预测")
                    data_is_incomplete = True

            # 如果数据完整，正常更新SLAM
            if not data_is_incomplete:
                scans_w_poses = self.slam.update(scan)
                if not scans_w_poses:
                    return None

                # 获取有效列的位姿
                col = client.first_valid_column(scans_w_poses[0])
                current_pose = scans_w_poses[0].pose[col]
                scan_ts = scan[0].timestamp[col]

                # 记录当前位姿和时间戳用于未来预测
                if not hasattr(self, 'prev_poses'):
                    self.prev_poses = []
                    self.prev_timestamps = []

                self.prev_poses.append(current_pose)
                self.prev_timestamps.append(scan_ts)

                # 只保留最近几帧的历史
                if len(self.prev_poses) > 5:  # 保留5帧历史
                    self.prev_poses.pop(0)
                    self.prev_timestamps.pop(0)

                # 更新上一次位姿
                self.last_pose = current_pose
            else:
                # 数据不完整，使用线性插值预测位姿
                if hasattr(self, 'prev_poses') and len(self.prev_poses) >= 2:
                    # 获取当前时间戳
                    col = client.first_valid_column(scan[0])
                    scan_ts = scan[0].timestamp[col]

                    # 使用最近两帧计算运动趋势
                    last_pose = self.prev_poses[-1]
                    prev_pose = self.prev_poses[-2]
                    last_ts = self.prev_timestamps[-1]
                    prev_ts = self.prev_timestamps[-2]

                    # 计算位姿变化率
                    dt_history = last_ts - prev_ts
                    if dt_history <= 0:  # 避免除零错误
                        current_pose = last_pose.copy()
                    else:
                        # 计算从上一帧到当前帧的时间差
                        dt_current = scan_ts - last_ts

                        # 计算插值比例
                        ratio = dt_current / dt_history

                        # 计算相对位姿变化
                        relative_pose = np.dot(np.linalg.inv(prev_pose), last_pose)

                        # 对旋转部分进行插值（简化处理，直接使用矩阵乘法）
                        predicted_relative_pose = np.eye(4)
                        predicted_relative_pose[:3, :3] = relative_pose[:3, :3]  # 复制旋转部分

                        # 对平移部分进行线性插值
                        translation = relative_pose[:3, 3]
                        predicted_relative_pose[:3, 3] = translation * ratio

                        # 计算预测的位姿
                        current_pose = np.dot(last_pose, predicted_relative_pose)

                        print(f"使用线性插值预测位姿，插值比例: {ratio:.2f}")
                else:
                    # 历史帧不足，使用上一帧位姿
                    current_pose = self.last_pose
                    col = client.first_valid_column(scan[0])
                    scan_ts = scan[0].timestamp[col]
                    print("历史帧不足，使用上一帧位姿")

            return SLAMResult(
                pose_matrix=current_pose,
                timestamp=scan_ts,
                status=True
            )

        except Exception as e:
            print(f"SLAM处理错误: {e}")
            return None


    def get_relative_motion(self, pose1: np.ndarray, pose2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算两个位姿间的相对运动"""
        relative_pose = np.dot(np.linalg.inv(pose1), pose2)
        rotation = relative_pose[:3, :3]
        translation = relative_pose[:3, 3]
        return rotation, translation

# 测试代码
if __name__ == "__main__":
    # 示例配置
    config = {
        'max_range': 75.0,
        'min_range': 1.0,
        'voxel_size': 1.0
    }

    processor = SLAMProcessor(config)

    # 这里需要实际的metadata和scan数据来测试
    source_file_path = "/home/yanan/Downloads/projects/cluster/OneDrive_1_10-27-2024/pcap/OS-1-128_122344000701_1024x10_20240806_144732.pcap"
    data_source = open_source(source_file_path, sensor_idx=-1)
    metadata = data_source.metadata
    processor.initialize(metadata)
    for idx, scans in enumerate(data_source):

        result = processor.process_scan(scans)
    print("SLAM处理器就绪")
