# dual_lidar_simulator.py
"""Dual LiDAR data stream simulator for real-time data acquisition and point cloud registration"""

import threading
import time
import queue
import logging
from pathlib import Path
from typing import Optional, Generator, Tuple, List
import numpy as np
import open3d as o3d
from ouster.sdk import pcap, client, open_source
from dataclasses import dataclass
from ouster.sdk.client import LidarScan
from src.utils.lidar_processing_utils import transform_and_merge_clouds_fast

@dataclass
class LidarFrame:
    """Single LiDAR frame data structure"""
    frame_id: int
    points: np.ndarray  # 点云数据
    scan: List[Optional[client.LidarScan]]
    timestamp: float  # 时间戳


class DualLidarSimulator:
    def __init__(self, pcap_path1: str, meta_path1: str,
                 pcap_path2: str, meta_path2: str, frame_rate: float = 10.0):
        """Initialize dual LiDAR simulator"""
        self.pcap_path1 = pcap_path1
        self.meta_path1 = meta_path1
        self.pcap_path2 = pcap_path2
        self.meta_path2 = meta_path2
        self.frame_interval = 1.0 / frame_rate
        # Create a frame queue for each LiDAR
        self.frame_queue1 = queue.Queue(maxsize=4000)
        self.frame_queue2 = queue.Queue(maxsize=4000)
        self.combined_queue = queue.Queue(maxsize=4000)
        self.running = False
        self.threads = []
        self.logger = self._setup_logger()
        self.transform_matrix1 = None  # Ground transformation matrix for the first LiDAR
        self.transform_matrix2 = None  # Ground transformation matrix for the second LiDAR
        self.registration_matrix = None  # Registration matrix between the two LiDARs
        self.first_frame = True  # Flag indicating whether it is the first frame

    def _setup_logger(self):
        """Configure logger"""
        logger = logging.getLogger("DualLidarSimulator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _read_pcap(self, pcap_path: str, meta_path: str) -> Generator[LidarFrame, None, None]:
        """Read PCAP file and generate point cloud frames"""
        try:
            pcap_source = pcap.PcapScanSource(pcap_path).single_source(0)
            point_metadata = pcap_source.metadata
            source = open_source(pcap_path, sensor_idx=-1)
            xyzlut = client.XYZLut(point_metadata)

            for idx, scan in enumerate(source):
                xyz = xyzlut(scan[0].field(client.ChanField.RANGE))
                points = xyz.reshape(-1, 3)
                valid_mask = ~np.any(np.isnan(points), axis=1)
                points = points[valid_mask]

                frame = LidarFrame(
                    frame_id=idx,
                    points=points,
                    scan=scan,
                    timestamp=time.time()
                )
                yield frame

        except Exception as e:
            self.logger.error(f"Error reading PCAP file: {e}")
            raise

    def _simulation_loop(self, pcap_path: str, meta_path: str, frame_queue: queue.Queue):
        """ Simulator loop for a single LiDAR"""
        try:
            frame_generator = self._read_pcap(pcap_path, meta_path)
            next_frame_time = time.time()

            for frame in frame_generator:
                if not self.running:
                    break

                current_time = time.time()
                wait_time = next_frame_time - current_time

                if wait_time > 0:
                    time.sleep(wait_time)

                try:
                    frame_queue.put(frame, timeout=0.1)
                    next_frame_time = time.time() + self.frame_interval
                except queue.Full:
                    self.logger.warning("Frame queue is full, skipping current frame")

        except Exception as e:
            self.logger.error(f"Simulation loop error: {e}")
            self.running = False

    def _combine_frames(self):
        """Merge point cloud frames from two LiDARs"""
        try:
            while self.running:
                # Retrieve frames from two queues
                frame1 = self.frame_queue1.get(timeout=10.0)
                frame2 = self.frame_queue2.get(timeout=10.0)

                if frame1 and frame2:
                    # Point cloud registration and merging
                    combined_points = self._register_point_clouds(frame2.points, frame1.points)

                    # Create merged frame
                    combined_frame = LidarFrame(
                        frame_id=frame1.frame_id,  # Use the frame ID from the first LiDAR
                        points=combined_points,
                        scan=frame1.scan,  # Temporarily use the scan data from the first LiDAR
                        timestamp=frame1.timestamp
                    )

                    # Put the merged frame into the combined queue
                    self.combined_queue.put(combined_frame)
                    # print('combined_queue_put', combined_frame.frame_id)

        except queue.Empty:
            self.logger.warning("Frame wait timeout")
        except Exception as e:
            self.logger.error(f"Frame merge error: {e}")
            self.running = False


    def _register_point_clouds(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Point cloud registration function"""
        try:
            if self.first_frame:
                # Initialize the transformation matrices (if needed)
                self.target_transform_matrix = np.load('/home/yanan/Downloads/projects/test_code/2025_bus_dual_simulate_real_detect/data/self_ground_target_1778.npy')  # target到地面的变换矩阵
                self.source_transform_matrix = np.load('/home/yanan/Downloads/projects/test_code/2025_bus_dual_simulate_real_detect/data/self_ground_source_1621.npy')  # source到地面的变换矩阵
                # self.registration_matrix = np.load('/home/yanan/Downloads/projects/test_code/2025_bus_dual_simulate_real_detect/data/final_transform_matrix.npy')  # merge矩阵
                # self.registration_matrix = np.load('/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/45_02/final_transform_matrix.npy')  # merge矩阵
                self.registration_matrix = np.load('/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_03/final_transform_matrix.npy')  # merge矩阵
                self.first_frame = False

            # Directly use the optimized function from lidar_processing_utils.py
            combined_points = transform_and_merge_clouds_fast(
                source_points,  # source point cloud
                target_points,  # target point cloud
                self.source_transform_matrix,  # Source-to-ground transformation matrix
                self.target_transform_matrix,  # target-to-ground transformation matrix
                self.registration_matrix  # merge matrix
            )

            return combined_points

        except Exception as e:
            print(f"Point cloud registration error: {e}")
            return np.vstack([source_points, target_points])  # 出错时直接拼接
    def compute_ground_transform(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """ Compute ground transformation matrix"""
        # Ground detection
        plane_model, _ = pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=100000
        )

        # Compute transformation matrix
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal_length = np.linalg.norm(normal)

        # 归一化法向量
        normal = normal / normal_length
        d = d / normal_length

        # 检查法向量方向
        if c < 0:
            normal = -normal
            d = -d

        # 计算旋转矩阵
        z_axis = np.array([0, 0, 1])
        if np.allclose(np.abs(np.dot(normal, z_axis)), 1.0):
            rotation_matrix = np.eye(3)
            if np.dot(normal, z_axis) < 0:
                rotation_matrix[2, 2] = -1
        else:
            rotation_axis = np.cross(normal, z_axis)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.clip(np.dot(normal, z_axis), -1.0, 1.0)
            rotation_angle = np.arccos(cos_angle)
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * rotation_angle)

        # 构建变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix

        # 计算平移向量
        point_on_plane = np.array([0, 0, -d / c])
        rotated_point = rotation_matrix @ point_on_plane
        transform_matrix[:3, 3] = -rotated_point

        return transform_matrix

    def start(self):
        """启动模拟器"""
        if self.running:
            return

        self.running = True

        # 创建两个LiDAR的模拟线程
        thread1 = threading.Thread(
            target=self._simulation_loop,
            args=(self.pcap_path1, self.meta_path1, self.frame_queue1)
        )
        thread2 = threading.Thread(
            target=self._simulation_loop,
            args=(self.pcap_path2, self.meta_path2, self.frame_queue2)
        )

        # 创建点云合并线程
        combine_thread = threading.Thread(target=self._combine_frames)

        # 启动所有线程
        thread1.daemon = True
        thread2.daemon = True
        combine_thread.daemon = True

        thread1.start()
        thread2.start()
        combine_thread.start()

        self.threads = [thread1, thread2, combine_thread]
        self.logger.info("双LiDAR模拟器已启动")

    def stop(self):
        """停止模拟器"""
        self.running = False
        for thread in self.threads:
            thread.join()
        self.logger.info("模拟器已停止")

    def get_frame(self, timeout: float = 1.0) -> Optional[LidarFrame]:
        """获取下一帧合并后的数据"""
        try:
            com_frame = self.combined_queue.get(timeout=timeout)
            print('combined_queue_get', com_frame.frame_id)  # 如果需要打印frame id
            return com_frame
        except queue.Empty:
            # print('None')
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
        'pcap_path1': "/home/yanan/Downloads/data/raw_data/2025_bus/Left/20250124_1250_OS-1-128_122211001778.pcap",
        'meta_path1': "/home/yanan/Downloads/data/raw_data/2025_bus/Left/20250124_1250_OS-1-128_122211001778.json",
        'pcap_path2': "/home/yanan/Downloads/data/raw_data/2025_bus/Right/20250124_1250_OS-1-128_122211001621.pcap",
        'meta_path2': "/home/yanan/Downloads/data/raw_data/2025_bus/Right/20250124_1250_OS-1-128_122211001621.json",
        'frame_rate': 10.0  # 10Hz
    }

    # 创建双LiDAR模拟器实例
    simulator = DualLidarSimulator(**config)

    # 创建输出目录用于保存测试结果
    output_dir = Path("output/test_dual_lidar_combined_points")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 启动模拟器
        simulator.start()
        print("模拟器已启动，按Ctrl+C停止")

        frame_count = 0
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 处理前100帧数据
        while frame_count < 2002:
            # 获取合并后的帧
            # simulator_start_time = time.time()
            frame = simulator.get_frame()
            # simulator_end_time = time.time()
            # simulator_total_time = simulator_end_time - simulator_start_time
            # print(f'simulator_total_time: {simulator_total_time * 1000:.1f}ms')
            if frame is None:
                continue

            # 创建Open3D点云对象用于可视化
            pcd = convert_to_o3d_pointcloud(frame.points)

            # 为不同LiDAR的点云设置不同颜色
            # 假设前半部分点是第一个LiDAR的，后半部分是第二个LiDAR的
            colors = np.zeros((len(frame.points), 3))
            mid_point = len(frame.points) // 2
            colors[:mid_point] = [1, 0, 0]  # 第一个LiDAR的点云显示为红色
            colors[mid_point:] = [0, 0, 1]  # 第二个LiDAR的点云显示为蓝色
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 保存每一帧的点云
            frame_path = output_dir / f"frame_{frame_count:06d}.pcd"
            o3d.io.write_point_cloud(str(frame_path), pcd)

            # 在控制台输出处理进度
            print(f"处理帧 {frame_count}, 点数: {len(frame.points)}")

            # 更新可视化
            if frame_count == 0:
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            # vis.update_renderer()
            # Set the rendering options
            opt = vis.get_render_option()
            opt.background_color = np.array([1, 1, 1])  # White background
            opt.point_size = 1.0  # Set point size

            frame_count += 1

            # 控制显示速度
            time.sleep(0.1)  # 限制为10Hz

    except KeyboardInterrupt:
        print("\n停止模拟...")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    finally:
        # 关闭可视化窗口
        vis.destroy_window()
        # 停止模拟器
        simulator.stop()
        print(f"\n测试完成，共处理 {frame_count} 帧")
        print(f"点云文件已保存至: {output_dir}")

        # 输出一些统计信息
        if frame_count > 0:
            # 读取最后一帧进行分析
            last_frame = o3d.io.read_point_cloud(str(output_dir / f"frame_{frame_count - 1:06d}.pcd"))
            print(f"\n最后一帧统计信息:")
            print(f"- 总点数: {len(last_frame.points)}")

            # 计算点云的边界框
            bbox = last_frame.get_axis_aligned_bounding_box()
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()

            print("\n点云范围:")
            print(f"- X: [{min_bound[0]:.2f}, {max_bound[0]:.2f}]")
            print(f"- Y: [{min_bound[1]:.2f}, {max_bound[1]:.2f}]")
            print(f"- Z: [{min_bound[2]:.2f}, {max_bound[2]:.2f}]")