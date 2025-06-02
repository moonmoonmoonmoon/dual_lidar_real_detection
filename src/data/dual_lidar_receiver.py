# dual_lidar_receiver.py
"""Dual LiDAR data collection module"""
from contextlib import closing

import psutil
import threading
import time
import queue
import logging
from typing import List, Optional, Generator
import numpy as np
from ouster.sdk import client
from dataclasses import dataclass
from src.utils.lidar_processing_utils import transform_and_merge_clouds_fast


@dataclass
class LidarFrame:
    """single frame LiDAR data structure"""
    frame_id: int
    points: np.ndarray
    scan: List[Optional[client.LidarScan]]
    timestamp: float


class DualLidarReceiver:
    def __init__(self, lidar1_config: dict, lidar2_config: dict, frame_rate: float = 10.0):
        """初始化双LiDAR接收器"""
        self.lidar1_config = lidar1_config
        self.lidar2_config = lidar2_config
        # self.frame_interval = 1.0 / frame_rate

        # 帧队列
        self.frame_queue1 = queue.Queue(maxsize=4000)
        self.frame_queue2 = queue.Queue(maxsize=4000)
        self.combined_queue = queue.Queue(maxsize=4000)

        self.running = False
        self.threads = []
        self.logger = self._setup_logger()
        self.first_frame = True
        # self.sync_completed = False
        # self.time_diff_threshold = 0.01  # 10ms

    def _setup_logger(self):
        """configuration logs"""
        logger = logging.getLogger("DualLidarReceiver")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _read_lidar(self, hostname: str, lidar_port: int, frame_queue: queue.Queue):
    # def _read_lidar(self, hostname: str, lidar_port: int, imu_port:int, frame_queue: queue.Queue):
        """read data from LiDAR sensor"""
        try:
            # Add logging connection information
            self.logger.info(f"Connecting to LiDAR at {hostname}:{lidar_port}")
            # Establish sensor connection
            stream = client.Scans.stream(hostname, lidar_port, complete=False)
            # stream = client.Scans.stream(hostname, lidar_port, imu_port, complete=False)
            self.logger.info(f"Stream established with metadata: {stream.metadata}")
            metadata = stream.metadata
            xyzlut = client.XYZLut(metadata)

            frame_id = 0
            # next_frame_time = time.time()
            # print(self.running)

            while self.running:
                # print('1')
                for scan in stream:
                    # print('2')
                    if not self.running:
                        # print('3')
                        break

                    # obtain point cloud data
                    xyz = xyzlut(scan.field(client.ChanField.RANGE))
                    points = xyz.reshape(-1, 3)
                    valid_mask = ~np.any(np.isnan(points), axis=1)
                    points = points[valid_mask]
                    # print('xyz', points.shape,frame_id)
                    data_timestamp = scan.timestamp[0]
                    # print('time1', data_timestamp)

                    # create frame object
                    frame = LidarFrame(
                        frame_id=frame_id,
                        points=points,
                        scan=[scan],
                        # timestamp=time.time()
                        timestamp=float(data_timestamp)
                    )


                    try:
                        frame_queue.put(frame, timeout=0.1)
                        frame_id += 1
                    except queue.Full:
                        self.logger.warning("Frame queue is full, skipping current frame.")

        except Exception as e:
            self.logger.error(f"LiDAR read error: {e}")
            time.sleep(1)  # Retrying after 1 second
    


    def _combine_frames(self):
        """Merge point cloud frames from two LiDARs"""
        try:
            while self.running:
                frame1 = self.frame_queue1.get(timeout=10.0)
                frame2 = self.frame_queue2.get(timeout=10.0)

                if frame1 and frame2:
                    # Register and merge point clouds
                    combined_points = self._register_point_clouds(frame2.points, frame1.points)

                    combined_frame = LidarFrame(
                        frame_id=frame1.frame_id,
                        points=combined_points,
                        scan=frame1.scan,
                        timestamp=frame1.timestamp
                    )

                    self.combined_queue.put(combined_frame)

        except queue.Empty:
            self.logger.warning("Frame wait timeout")
        except Exception as e:
            self.logger.error(f"Frame merge error: {e}")
            self.running = False

    def _register_point_clouds(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Point cloud registration"""
        try:
            if self.first_frame:
                self.target_transform_matrix = np.load('data/ground_transform_target.npy')
                self.source_transform_matrix = np.load('data/ground_transform_source.npy')
                # self.registration_matrix = np.load('data/self_final_merge_matrix.npy')
                self.registration_matrix = np.load('data/final_transform_matrix.npy')
                self.first_frame = False

            combined_points = transform_and_merge_clouds_fast(
                source_points,
                target_points,
                self.source_transform_matrix,
                self.target_transform_matrix,
                self.registration_matrix
            )

            return combined_points

        except Exception as e:
            self.logger.error(f"Point cloud registration error: {e}")
            return np.vstack([source_points, target_points])


    def start(self):
        """Start data acquisition"""
        if self.running:
            return

        self.running = True

        # Create LiDAR reading thread
        thread1 = threading.Thread(
            target=self._read_lidar,
            args=(self.lidar1_config['hostname'],
                  self.lidar1_config['lidar_port'],
                  # self.lidar1_config['imu_port'],
                  self.frame_queue1)
        )
        thread2 = threading.Thread(
            target=self._read_lidar,
            args=(self.lidar2_config['hostname'],
                  self.lidar2_config['lidar_port'],
                  # self.lidar2_config['imu_port'],
                  self.frame_queue2)
        )

        # Create point cloud merging thread
        combine_thread = threading.Thread(target=self._combine_frames)

        # Start all threads
        thread1.daemon = True
        thread2.daemon = True
        combine_thread.daemon = True

        thread1.start()
        thread2.start()
        combine_thread.start()

        self.threads = [thread1, thread2, combine_thread]
        # self.threads = [thread1, thread2, combine_thread, monitor_thread]
        self.logger.info("Dual LiDAR receiver started")

    def stop(self):
        """Stop data acquisition"""
        self.running = False
        for thread in self.threads:
            thread.join()
        self.logger.info("Receiver stopped")

    def get_frame(self, timeout: float = 1.0) -> Optional[LidarFrame]:
        """Get the next merged frame"""
        try:
            return self.combined_queue.get(timeout=timeout)
        except queue.Empty:
            return None
