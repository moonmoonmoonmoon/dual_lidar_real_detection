#!/usr/bin/env python3
"""
点云序列播放器 - 播放触发事件前后保存的点云序列
使用方法: python point_cloud_player.py [触发事件目录路径]
"""

import os
import sys
import time
import glob
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import argparse
import re

class PointCloudPlayer:
    def __init__(self, trigger_dir, loop=True, delay=0.1, colorize=True):
        """
        初始化点云播放器

        Args:
            trigger_dir: 触发事件目录路径
            loop: 是否循环播放
            delay: 帧间延迟时间(秒)
            colorize: 是否对点云进行着色
        """
        self.trigger_dir = Path(trigger_dir)
        self.loop = loop
        self.delay = delay
        self.colorize = colorize
        self.frame_files = []
        self.object_info = None
        self.current_frame_idx = 0

        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("点云序列播放器", width=1280, height=720)

        # 设置渲染参数
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 2.0

        # 加载触发数据
        self._load_data()

    def _load_data(self):
        """加载触发事件数据"""
        if not self.trigger_dir.exists():
            print(f"错误: 目录 {self.trigger_dir} 不存在")
            sys.exit(1)

        # 读取框信息
        box_info_path = self.trigger_dir / "trigger_info.json"
        if box_info_path.exists():
            with open(box_info_path, 'r') as f:
                self.object_info = json.load(f)
                print(f"加载触发信息: {self.object_info}")

        # 搜索所有PCD文件并按相对帧号排序
        pcd_files = sorted(
            self.trigger_dir.glob("frame_*.pcd"),
            key=lambda x: int(re.search(r'frame_([+-]\d+)\.pcd', x.name).group(1))
        )

        if not pcd_files:
            print(f"错误: 目录 {self.trigger_dir} 中没有找到PCD文件")
            sys.exit(1)

        self.frame_files = pcd_files
        print(f"找到 {len(self.frame_files)} 个点云帧")

    def _get_color_for_frame(self, frame_idx):
        """为不同类型的帧生成颜色"""
        rel_idx = int(re.search(r'frame_([+-]\d+)\.pcd', self.frame_files[frame_idx].name).group(1))

        if rel_idx < 0:
            # 触发前的帧 - 蓝色渐变
            intensity = 0.5 + 0.5 * (rel_idx / -20)  # -20 帧到 -1 帧
            return np.array([0.0, intensity, 1.0])
        elif rel_idx == 0:
            # 触发帧 - 红色
            return np.array([1.0, 0.0, 0.0])
        else:
            # 触发后的帧 - 绿色渐变
            intensity = 0.5 + 0.5 * (1 - rel_idx / 10)  # 1 帧到 10 帧
            return np.array([0.0, 1.0, intensity])

    def _create_bbox_for_object(self, obj_info):
        """根据目标信息创建边界框"""
        if not obj_info or 'object_info' not in obj_info:
            return None

        obj = obj_info['object_info']
        center = np.array(obj['position'])
        dimensions = np.array(obj['dimensions'])
        yaw = float(obj['yaw'])

        # 创建旋转矩阵
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 创建边界框
        bbox = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=R,
            extent=dimensions
        )
        bbox.color = np.array([1.0, 0.0, 0.0])  # 红色边界框

        return bbox

    def play(self):
        """播放点云序列"""
        pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(pcd)

        # 添加坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        self.vis.add_geometry(coord_frame)

        # 添加目标边界框（如果存在）
        bbox = self._create_bbox_for_object(self.object_info)
        if bbox:
            self.vis.add_geometry(bbox)

        view_control = self.vis.get_view_control()

        # 主循环
        running = True
        while running:
            # 读取当前帧
            frame_path = self.frame_files[self.current_frame_idx]
            rel_idx = int(re.search(r'frame_([+-]\d+)\.pcd', frame_path.name).group(1))

            # 加载点云
            current_pcd = o3d.io.read_point_cloud(str(frame_path))

            # 着色（可选）
            if self.colorize:
                color = self._get_color_for_frame(self.current_frame_idx)
                current_pcd.paint_uniform_color(color)

            # 更新点云
            pcd.points = current_pcd.points
            pcd.colors = current_pcd.colors
            self.vis.update_geometry(pcd)

            # 更新视图
            self.vis.poll_events()
            self.vis.update_renderer()

            # 显示帧信息
            frame_info = f"帧 {rel_idx:+03d}"
            if rel_idx == 0:
                frame_info += " (触发帧)"
            print(f"\r播放: {frame_info}", end="")

            # 等待
            time.sleep(self.delay)

            # 下一帧
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frame_files)

            # 检查是否循环
            if not self.loop and self.current_frame_idx == 0:
                running = False

            # 检查是否关闭窗口
            if not self.vis.poll_events():
                running = False

        self.vis.destroy_window()
        print("\n播放结束")


def find_latest_trigger_folder(base_dir="output/triggers"):
    """查找最新的触发文件夹"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"错误: 目录 {base_dir} 不存在")
        return None

    trigger_dirs = list(base_path.glob("trigger_*"))
    if not trigger_dirs:
        print(f"错误: 目录 {base_dir} 中没有找到触发文件夹")
        return None

    # 按修改时间排序
    latest_dir = max(trigger_dirs, key=lambda d: d.stat().st_mtime)
    return latest_dir


def main():
    parser = argparse.ArgumentParser(description="点云序列播放器")
    parser.add_argument("trigger_dir", nargs="?", type=str,
                        help="触发事件目录路径 (如果不提供，将使用最新的触发文件夹)")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--delay", type=float, default=0.1, help="帧间延迟时间(秒)")
    parser.add_argument("--no-color", action="store_true", help="禁用点云着色")

    args = parser.parse_args()

    # 如果未提供目录，查找最新的触发文件夹
    trigger_dir = args.trigger_dir
    if not trigger_dir:
        trigger_dir = find_latest_trigger_folder()
        if not trigger_dir:
            sys.exit(1)
        print(f"使用最新的触发文件夹: {trigger_dir}")

    # 创建并启动播放器
    player = PointCloudPlayer(
        trigger_dir=trigger_dir,
        loop=args.loop,
        delay=args.delay,
        colorize=not args.no_color
    )
    player.play()

if __name__ == "__main__":
    main()