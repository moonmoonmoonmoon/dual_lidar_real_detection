# 1_lidar_simulator.py
"""LiDAR数据流模拟器,用于模拟实时数据采集"""

import threading
import time
import queue
import logging
from pathlib import Path
from typing import Optional, Generator
import numpy as np
import open3d as o3d
from ouster.sdk import pcap, client, open_source
from dataclasses import dataclass
from ouster.sdk.client import LidarScan
from typing import List, Optional

@dataclass
class LidarFrame:
    """单帧LiDAR数据结构"""
    frame_id: int
    points: np.ndarray  # 点云数据
    # scan : LidarScan
    scan : List[Optional[client.LidarScan]]
    timestamp: float  # 时间戳


class LidarSimulator:
    def __init__(self, pcap_path: str, meta_path: str, frame_rate: float = 10.0):
        """初始化模拟器"""
        self.pcap_path = pcap_path
        self.meta_path = meta_path
        self.frame_interval = 1.0 / frame_rate
        self.frame_queue = queue.Queue(maxsize=4000)
        self.running = False
        self.thread = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """配置日志"""
        logger = logging.getLogger("LidarSimulator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


    def _read_pcap(self) -> Generator[LidarFrame, None, None]:
        """读取PCAP文件并生成点云帧"""
        try:
            pcap_source = pcap.PcapScanSource(self.pcap_path).single_source(0)
            point_metadata = pcap_source.metadata
            # # print('source',source,type(source))
            source = open_source(self.pcap_path, sensor_idx=-1)
            self.metadata = source.metadata
            xyzlut = client.XYZLut(point_metadata)
            # scans = iter(source)

            for idx, scan in enumerate(source):
                xyz = xyzlut(scan[0].field(client.ChanField.RANGE))
                points = xyz.reshape(-1, 3)
                valid_mask = ~np.any(np.isnan(points), axis=1)
                points = points[valid_mask]
                frame = LidarFrame(
                    frame_id=idx,
                    # points=pcd,
                    points=points,
                    scan = scan,
                    timestamp=time.time()
                )
                yield frame

        except Exception as e:
            self.logger.error(f"读取PCAP文件错误: {e}")
            raise

    def start(self):
        """启动模拟器"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("模拟器已启动")

    def stop(self):
        """停止模拟器"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.logger.info("模拟器已停止")

    def _simulation_loop(self):
        """模拟器主循环,按指定帧率产生数据"""
        try:
            frame_generator = self._read_pcap()
            next_frame_time = time.time()

            for frame in frame_generator:
                if not self.running:
                    break

                current_time = time.time()
                wait_time = next_frame_time - current_time

                if wait_time > 0:
                    time.sleep(wait_time)

                try:
                    self.frame_queue.put(frame, timeout=0.1)
                    next_frame_time = time.time() + self.frame_interval
                except queue.Full:
                    self.logger.warning("帧队列已满,跳过当前帧")

        except Exception as e:
            self.logger.error(f"模拟循环错误: {e}")
            self.running = False

    def get_frame(self, timeout: float = 1.0) -> Optional[LidarFrame]:
        """获取下一帧数据"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

def convert_to_o3d_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """将numpy点云数组转换为Open3D点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# 测试代码
if __name__ == "__main__":
    # 示例配置
    config = {
        'pcap_path': "/home/yanan/Downloads/projects/cluster/OneDrive_1_10-27-2024/pcap/OS-1-128_122344000701_1024x10_20240806_144732.pcap",
        'meta_path': "/home/yanan/Downloads/projects/cluster/OneDrive_1_10-27-2024/pcap/OS-1-128_122344000701_1024x10_20240806_144732.json",
        'frame_rate': 10.0  # 10Hz
    }

    # 测试模拟器
    simulator = LidarSimulator(**config)

    try:
        simulator.start()
        print("按Ctrl+C停止模拟器")

        while True:
            simulator_start_time = time.time()
            frame = simulator.get_frame()
            simulator_end_time = time.time()
            simulator_total_time = simulator_end_time - simulator_start_time
            print(f'simulator_total_time: {simulator_total_time * 1000:.1f}ms')
            if frame:
                print(f"处理帧 {frame.frame_id}, 点数: {len(frame.points)}")
                pcd = convert_to_o3d_pointcloud(frame.points)
                # 这里可以添加其他处理...
                print(f"处理帧 {frame.frame_id}")

    except KeyboardInterrupt:
        print("\n停止模拟...")
    finally:
        simulator.stop()