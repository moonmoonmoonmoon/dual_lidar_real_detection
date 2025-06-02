from collections import deque
from pathlib import Path
import open3d as o3d
import numpy as np
import json
from typing import Dict, Optional

class PointCloudBuffer:
    """点云数据缓冲区管理类"""

    def __init__(self, pre_frames: int = 20, post_frames: int = 10):
        """
        初始化点云缓冲区

        Args:
            pre_frames: 触发前保存的帧数
            post_frames: 触发后保存的帧数
        """
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.buffer = deque(maxlen=pre_frames)  # 主缓冲区
        self.post_buffer = deque(maxlen=post_frames)  # 触发后的缓冲区
        self.is_triggered = False
        self.trigger_info = None
        self.save_root = Path("output/triggers")
        self.save_root.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame_id: int, points: np.ndarray,
                  scan: Optional[list] = None, transform: Optional[np.ndarray] = None) -> None:
        """
        添加新的点云帧到缓冲区

        Args:
            frame_id: 帧ID
            points: 点云数据
            scan: 原始扫描数据（可选）
            transform: 变换矩阵（可选）
        """
        frame_data = {
            'frame_id': frame_id,
            'points': points,
            'scan': scan,
            'transform': transform
        }

        if not self.is_triggered:
            # 正常模式：维护pre_frames大小的环形缓冲区
            self.buffer.append(frame_data)
        else:
            # 触发模式：收集post_frames帧
            self.post_buffer.append(frame_data)
            if len(self.post_buffer) >= self.post_frames:
                self._save_all_frames()
                self.reset()

    def trigger(self, frame_id: int, trigger_info: Dict) -> None:
        """
        触发保存操作

        Args:
            frame_id: 触发帧ID
            trigger_info: 触发相关信息
        """
        if not self.is_triggered:
            self.is_triggered = True
            self.trigger_info = {
                'frame_id': frame_id,
                'info': trigger_info
            }

    def _save_all_frames(self) -> None:
        """保存缓冲区中的所有点云帧"""
        if not self.trigger_info:
            return

        # 创建保存目录
        save_dir = self.save_root / f"trigger_{self.trigger_info['frame_id']}"
        save_dir.mkdir(exist_ok=True)

        # 保存触发信息
        with open(save_dir / "trigger_info.json", 'w') as f:
            json.dump(self.trigger_info['info'], f, indent=2)

        # 合并所有需要保存的帧
        # all_frames = list(self.buffer) + [self.trigger_info] + list(self.post_buffer)
        all_frames = list(self.buffer) + list(self.post_buffer)

        # 保存点云文件
        for idx, frame_data in enumerate(all_frames):
            relative_frame_id = idx - len(self.buffer) + 1 # 相对于触发帧的偏移

            # 创建点云对象并保存
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(frame_data['points'])

            # 使用相对帧号命名，便于后续分析
            pcd_path = save_dir / f"frame_{relative_frame_id:+03d}.pcd"
            o3d.io.write_point_cloud(str(pcd_path), pcd)

    def reset(self) -> None:
        """重置缓冲区状态"""
        self.is_triggered = False
        self.trigger_info = None
        self.post_buffer.clear()