# 3_object_detector.py
"""实时目标检测模块,用于检测和跟踪物体"""
import os
import time
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import List, Tuple, Dict
from src.utils.visualization import Visualizer
from src.utils.lidar_processing_utils import height_filter_fast, fast_euclidean_cluster, ground_points_filter_fast, compute_bbox
import numpy as np
from scipy.spatial.transform import Rotation as Ro

@dataclass
class DetectionResult:
    """检测结果数据结构"""
    frame_id: int
    objects: List[Dict]
    ego_position: List[float]
    ego_velocity: np.ndarray
    timestamp: float


class LidarObjectDetector:
    def __init__(self, config: dict = None):
        """初始化检测器"""
        self.config = {
            'ground_removal': {
                'z_threshold': 0.28,
                'show_plane': True,
                'distance_threshold': 0.1,
                'ransac_n': 3,
                'num_iterations': 100000
            },
            'height_filter': {
                'min_height': 0.1,
                'max_height': 3.0
            },
            'coordinate_transform': {
                'yaw_angle_deg': 90,
                'enable': True
            },
            'lidar': {
                'x_offset': 1.4,
                'y_offset': 0.88,
            },
            'roi': {
                'length': 20,
                'width': 6,
                'height': 4
            },
            'ego_roi': {
                'length': 3.4,
                'width': 1.86,
                'height': 2
            },
            'clustering': {
                'voxel_size': 0.05,
                'eps': 1.25,
                'min_samples': 80,
                'min_points': 10
            },
            'visualization': {
                'enable': True,  # 是否启用可视化
                'show_ground_removal': True,  # 显示地面移除
                'show_height_filter': False,  # 显示高度过滤
                'show_roi': False,  # 显示ROI裁剪
                'show_clusters': False  # 显示聚类结果
            }
        }
        if config:
            self.config.update(config)

        self.transform_matrix = None
        self.previous_objects = None
        self.ego_position = None
        self.previous_ego_position = None
        self.previous_timestamp = None
        self.lidar_x_offset = self.config['lidar']['x_offset']
        self.lidar_y_offset = self.config['lidar']['y_offset']
        self.vis = Visualizer() if self.config['visualization']['enable'] else None
        self.ground_plane_model = None  # 存储地面模型
        self.first_frame = True  # 标记是否是第一帧
        self.ground_plane_normal = None

    def process_frame(self, frame_id: int, pcd: o3d.geometry.PointCloud,
                      final_transform: np.ndarray, timestamp: np.uint64) -> DetectionResult:
        """处理单帧点云数据"""
        try:
            process_start = time.time()

            # 1. 坐标系对齐
            if self.config['coordinate_transform']['enable']:
                align_start = time.time()
                points = np.asarray(pcd.points)

                # 使用转换后的弧度值
                yaw_rad = self._get_yaw_angle_rad()
                aligned_points, full_rotation_matrix = self.align_coordinate_system(points,yaw_rad)
                # aligned_points, combined_transform = self.rotate_around_other_system_axis(points, final_transform,
                                                                                          # yaw_rad, 'z')
                aligned_pcd = o3d.geometry.PointCloud()
                aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
                align_time = time.time() - align_start
                print(f"Coordinate alignment time: {align_time * 1000:.1f}ms")
            else:
                aligned_pcd = pcd
                full_rotation_matrix = np.eye(4)

            # 2. 计算位置信息
            lidar_position, vehicle_center, right_front_position = self.calculate_positions(final_transform,
                                                                                            full_rotation_matrix)

            # 3. 地面检测和移除
            ground_start = time.time()
            non_ground_pcd, normal, d = self.remove_ground_vehicle_plane(aligned_pcd, lidar_position, vehicle_center,
                                                              right_front_position)
            # non_ground_pcd, ego_center_current = self.remove_ground_vehicle_plane(pcd, final_transform)
            ground_time = time.time() - ground_start
            print(f"Ground removal time: {ground_time * 1000:.1f}ms")

            # # 4. 高度过滤
            filter_start = time.time()
            filtered_pcd = self.height_filter(non_ground_pcd, normal, d)
            filter_time = time.time() - filter_start
            print(f"Height filter time: {filter_time * 1000:.1f}ms")

            # 5. ROI裁剪
            roi_start = time.time()
            roi_pcd = self.crop_roi_with_positions(filtered_pcd, final_transform, vehicle_center, full_rotation_matrix)
            roi_time = time.time() - roi_start
            print(f"ROI crop time: {roi_time * 1000:.1f}ms")

            # 6. 计算ego速度
            ego_velocity = self.calculate_ego_velocity(self.ego_position, 0.1)

            # 7. 聚类检测
            cluster_start = time.time()
            clusters, objects = self.cluster_points(roi_pcd, non_ground_pcd)
            cluster_time = time.time() - cluster_start
            print(f"Clustering time: {cluster_time * 1000:.1f}ms")

            # 8. 计算速度(如果有前一帧)
            velocity_start = time.time()
            objects = self.estimate_velocity(objects, 0.1)  # 假设帧间隔0.1s
            velocity_time = time.time() - velocity_start
            print(f"Velocity estimation time: {velocity_time * 1000:.1f}ms")

            detector_total = time.time() - process_start
            print(f"Total detection time: {detector_total * 1000:.1f}ms")

            # 保存当前检测结果
            self.previous_objects = objects

            return DetectionResult(
                frame_id=frame_id,  # 或使用实际帧ID
                objects=objects,
                ego_position=self.ego_position,
                ego_velocity=ego_velocity,
                timestamp=float(timestamp)  # 使用位姿矩阵中的时间戳
            )

        except Exception as e:
            print(f"检测错误: {e}")
            return None

    def _get_yaw_angle_rad(self) -> float:
        """将配置中的角度转换为弧度"""
        yaw_deg = self.config['coordinate_transform'].get('yaw_angle_deg', 90)
        return np.deg2rad(yaw_deg)


    def calculate_positions(self, final_matrix: np.ndarray, full_rotation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算激光雷达位置和自车位置

        Args:
            final_matrix: 变换矩阵

        Returns:
            lidar_position: 激光雷达位置
            vehicle_center: 自车中心位置
            rotation_matrix: 旋转矩阵
        """
        # 1. 定义车辆参考点的初始位置（相对于激光雷达）
        # 点1: 激光雷达位置
        lidar_pos_initial = np.array([0.0, 0.0, 0.0])
        lidar_pos_initial_transform = np.eye(4)
        lidar_pos_initial_transform[:3, 3] = lidar_pos_initial

        # 点2: 车辆中心点(ego center)
        ego_center_initial = np.array([-self.lidar_y_offset, self.lidar_x_offset, 0.0])
        ego_center_initial_transform = np.eye(4)
        ego_center_initial_transform[:3, 3] = ego_center_initial

        # 点3: 车辆右前方点(选择一个与激光雷达同高但在车辆不同位置的点)
        # 假设激光雷达在车辆左前方，车辆右前方的点可以设为
        right_front_initial = np.array([-2 * self.lidar_y_offset, 0, 0.0])
        right_front_initial_transform = np.eye(4)
        right_front_initial_transform[:3, 3] = right_front_initial

        # 2. 将三个点通过final_transform变换到旋转变换前当前帧
        lidar_pos = np.dot(full_rotation_matrix, np.dot(final_matrix, lidar_pos_initial_transform))[:3, 3]
        ego_center = np.dot(full_rotation_matrix, np.dot(final_matrix, ego_center_initial_transform))[:3, 3]
        right_front = np.dot(full_rotation_matrix, np.dot(final_matrix, right_front_initial_transform))[:3, 3]
        # print('v_p',vehicle_center)

        # 保存自车位置信息供其他方法使用
        self.ego_position = ego_center.tolist()

        return lidar_pos, ego_center, right_front


    def visualize_vehicle_plane(self, pcd: o3d.geometry.PointCloud,
                                lidar_pos: np.ndarray, ego_center: np.ndarray,
                                right_front: np.ndarray,
                                normal: np.ndarray, d: float,
                                grid_size: float = 20.0, grid_density: float = 1.0):
        """可视化车辆平面

        Args:
            pcd: 输入点云
            lidar_pos: 激光雷达位置
            ego_center: 车辆中心位置
            right_front: 车辆右前方位置
            normal: 平面法向量
            d: 平面方程中的d值
            grid_size: 平面网格大小
            grid_density: 平面网格密度
        """
        # 创建用于可视化的点云对象
        plane_vis = o3d.geometry.PointCloud()

        # 计算平面中心(使用三个点的中心作为平面中心)
        plane_center = (lidar_pos + ego_center + right_front) / 3.0

        # 创建平面网格点
        grid_points = []
        half_size = grid_size / 2.0
        step = grid_density
        for x in np.arange(-half_size, half_size, step):
            for y in np.arange(-half_size, half_size, step):
                # 给定x和y，计算平面上的z
                z = (-d - normal[0] * (plane_center[0] + x) - normal[1] * (plane_center[1] + y)) / normal[2]
                grid_points.append([plane_center[0] + x, plane_center[1] + y, z])

        # 转换为numpy数组
        plane_points = np.array(grid_points)

        # 设置平面点云
        plane_vis.points = o3d.utility.Vector3dVector(plane_points)

        # 设置平面点云颜色(使用红色)
        colors = np.zeros((len(plane_points), 3))
        colors[:, 0] = 1.0  # 红色
        plane_vis.colors = o3d.utility.Vector3dVector(colors)

        # 创建参考点可视化(三个点)
        ref_points = o3d.geometry.PointCloud()
        ref_points.points = o3d.utility.Vector3dVector(np.array([lidar_pos, ego_center, right_front]))
        # 设置参考点颜色(使用不同颜色区分三个点)
        ref_colors = np.array([
            [1.0, 0.0, 0.0],  # 红色 - 激光雷达
            [0.0, 1.0, 0.0],  # 绿色 - 车辆中心
            [0.0, 0.0, 1.0]  # 蓝色 - 车辆右前方
        ])
        ref_points.colors = o3d.utility.Vector3dVector(ref_colors)

        # 获取原始点云的点
        original_points = np.asarray(pcd.points)
        # 计算点到平面的有符号距离
        signed_distances = np.dot(original_points, normal) + d

        # 创建一个新的点云，用不同颜色显示上方和下方的点
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(original_points)

        # 计算点到平面的距离，并根据距离设置颜色
        distances = np.abs(signed_distances)
        max_dist = 5.0  # 最大距离阈值，超过这个值的使用同一颜色

        # 初始化颜色数组
        colors = np.zeros((len(original_points), 3))

        # 设置平面上方点的颜色(蓝色系)
        above_mask = signed_distances > 0
        # 颜色从浅蓝色到深蓝色，根据距离变化
        scale = np.minimum(distances[above_mask] / max_dist, 1.0)
        colors[above_mask, 2] = 0.5 + 0.5 * (1 - scale)  # 蓝色分量

        # 设置平面下方点的颜色(绿色系)
        below_mask = signed_distances <= 0
        # 颜色从浅绿色到深绿色，根据距离变化
        scale = np.minimum(distances[below_mask] / max_dist, 1.0)
        colors[below_mask, 1] = 0.5 + 0.5 * (1 - scale)  # 绿色分量

        colored_pcd.colors = o3d.utility.Vector3dVector(colors)

        # 创建可视化对象
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 添加点云和平面
        vis.add_geometry(colored_pcd)
        vis.add_geometry(plane_vis)
        vis.add_geometry(ref_points)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 1.0

        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # 设置初始视角
        ctr = vis.get_view_control()
        ctr.set_lookat(plane_center)

        # 运行可视化
        vis.run()
        vis.destroy_window()

    def remove_ground_vehicle_plane(self, pcd: o3d.geometry.PointCloud,
                                    lidar_position: np.ndarray, ego_center:np.ndarray, right_front_position: np.ndarray) -> Tuple[o3d.geometry.PointCloud, np.ndarray, float]:
        """基于车辆平面的地面移除

        使用三个车辆参考点定义一个平面，该平面与地面平行
        然后移除低于该平面特定距离的点作为地面点

        Args:
            pcd: 输入点云
            final_transform: 当前的变换矩阵

        Returns:
            non_ground_pcd: 移除地面后的点云
        """

        # 计算平面方程 ax + by + cz + d = 0
        # 平面法向量通过两个向量的叉乘计算
        v1 = ego_center - lidar_position
        v2 = right_front_position - lidar_position
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # 单位化法向量

        # 确保法向量指向上方（z分量为正）
        if normal[2] < 0:
            normal = -normal
        self.ground_plane_normal = normal

        # 计算平面方程的d系数
        d = -np.dot(normal, lidar_position)

        # 打印平面方程
        a, b, c = normal
        # print(f"车辆平面方程: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        if self.vis and self.config['ground_removal'].get('show_plane', False):
        # if (self.vis and self.config['ground_removal'].get('show_plane', False)) or self.first_frame:
            # 设置 first_frame 为 False 防止每帧都显示
            # if self.first_frame:
            #     self.first_frame = False

            # 调用新增的可视化方法
            self.visualize_vehicle_plane(
                pcd,
                lidar_position,
                ego_center,
                right_front_position,
                normal,
                d
            )
        # 4. 计算点到平面的距离
        points = np.asarray(pcd.points)
        signed_distances = np.dot(points, normal) + d

        # 5. 根据阈值过滤地面点
        ground_threshold = self.config['ground_removal'].get('z_threshold', 0.3)
        ground_mask = (signed_distances < 0) & (np.abs(signed_distances) > ground_threshold)
        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]

        # 创建地面和非地面点云对象
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_points)

        non_ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)

        # 可视化
        if self.vis and self.config['visualization']['show_ground_removal']:
            self.vis.visualize_ground(ground_pcd, non_ground_pcd)

        return non_ground_pcd, normal, d

    def height_filter(self, pcd: o3d.geometry.PointCloud, normal: np.ndarray, d: float) -> o3d.geometry.PointCloud:

        # 计算点到平面的距离
        points = np.asarray(pcd.points)
        signed_distances = np.dot(points, normal) + d

        # 根据阈值过滤
        min_height_threshold = self.config['height_filter'].get('min_height', 0.3)
        max_height_threshold = self.config['height_filter'].get('max_height', 0.7)
        effective_space_mask = (((signed_distances < 0) & (np.abs(signed_distances) < min_height_threshold)) |
                                ((signed_distances > 0) & (np.abs(signed_distances) < max_height_threshold)))

        filtered_points = points[effective_space_mask]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 可视化高度过滤结果
        if self.vis and self.config['visualization']['show_height_filter']:
            self.vis.visualize_point_cloud(filtered_pcd, color=[0.5, 0.5, 0.5])

        return filtered_pcd


    def compute_transform_matrix(self, plane_model):
        """计算变换矩阵"""
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


    def align_coordinate_system(self, points: np.ndarray, yaw_angle: float = np.pi / 2) -> Tuple[np.ndarray,np.ndarray]:
        """
        将点云绕Z轴旋转使其与车辆前进方向对齐

        Args:
            points: 输入点云数据，shape为(N, 3)
            yaw_angle: 绕Z轴旋转的角度，默认为π/2

        Returns:
            aligned_points: 旋转后的点云
        """
        # 构建绕Z轴旋转的变换矩阵
        rotation_matrix = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1]
        ])

        # 应用旋转变换
        aligned_points = np.dot(points, rotation_matrix.T)

        # 创建完整的旋转变换矩阵（4x4）
        full_rotation_matrix = np.eye(4)
        full_rotation_matrix[:3, :3] = rotation_matrix

        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)

        # if self.vis:
        #     self.vis.visualize_point_cloud(aligned_pcd)

        return aligned_points, full_rotation_matrix

    def rotate_around_other_system_axis(self, points, transform_matrix, angle, axis='z'):
        """
        直接在世界坐标系下，按其他坐标系的指定轴旋转点云

        参数:
            points: 世界坐标系下的点云数据, shape为(n, 3)
            transform_matrix: 从世界坐标系到其他坐标系的变换矩阵, shape为(4, 4)
            angle: 旋转角度, 单位为弧度
            axis: 旋转轴, 可以是'x', 'y'或'z'

        返回:
            rotated_points: 旋转后的点云, 仍在世界坐标系下
        """
        # 提取变换矩阵的旋转部分和平移部分
        rotation_part = transform_matrix[:3, :3]
        translation_part = transform_matrix[:3, 3]

        # 确定旋转轴在其他坐标系中的表示
        if axis == 'x':
            axis_in_other_system = np.array([1, 0, 0])
        elif axis == 'y':
            axis_in_other_system = np.array([0, 1, 0])
        elif axis == 'z':
            axis_in_other_system = np.array([0, 0, 1])
        else:
            raise ValueError("轴必须是 'x', 'y' 或 'z'")

        # 将旋转轴从其他坐标系变换到世界坐标系
        # 注意：只需要应用旋转部分，不需要平移
        axis_in_world_system = rotation_part @ axis_in_other_system

        # 标准化旋转轴
        axis_in_world_system = axis_in_world_system / np.linalg.norm(axis_in_world_system)

        # 获取其他坐标系原点在世界坐标系中的位置
        origin_in_world = translation_part

        # 创建一个从原点到其他坐标系原点的平移矩阵
        translate_to_other_origin = np.eye(4)
        translate_to_other_origin[:3, 3] = origin_in_world

        # 创建一个从其他坐标系原点回到原点的平移矩阵
        translate_back_to_origin = np.eye(4)
        translate_back_to_origin[:3, 3] = -origin_in_world

        # 使用Rodriguez公式创建绕任意轴的旋转矩阵
        # 或者使用scipy的Rotation
        rotation = Ro.from_rotvec(angle * axis_in_world_system)
        rotation_matrix = rotation.as_matrix()

        # 创建完整的旋转矩阵（在世界坐标系中）
        world_rotation_matrix = np.eye(4)
        world_rotation_matrix[:3, :3] = rotation_matrix

        # 组合变换：先平移到其他坐标系原点，然后旋转，最后平移回来
        combined_transform = translate_to_other_origin @ world_rotation_matrix @ translate_back_to_origin

        # 应用变换到点云
        # 将点云转为齐次坐标
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        # 应用变换
        rotated_homogeneous_points = (combined_transform @ homogeneous_points.T).T
        # 去掉齐次坐标的最后一列
        rotated_points = rotated_homogeneous_points[:, 0:3]

        return rotated_points, combined_transform

    def crop_roi_with_positions(self, pcd: o3d.geometry.PointCloud,
                                final_matrix: np.ndarray, vehicle_center: np.ndarray, full_rotation_matrix: np.ndarray) -> o3d.geometry.PointCloud:
        """
        使用预先计算的车辆中心位置进行ROI区域裁剪

        Args:
            pcd: 输入点云
            final_matrix: 变换矩阵
            vehicle_center: 预先计算的车辆中心位置

        Returns:
            filtered_pcd: 裁剪后的点云
        """
        box_size = np.array([
            self.config['roi']['length'],
            self.config['roi']['width'],
            self.config['roi']['height']
        ])

        combined_final_matrix = np.dot(full_rotation_matrix, np.dot(final_matrix, np.linalg.inv(full_rotation_matrix)))
        # 创建ROI边界框
        bbox = o3d.geometry.OrientedBoundingBox(
            center=vehicle_center,
            R=combined_final_matrix[:3, :3],
            extent=box_size
        )

        # 首先裁剪出ROI区域内的点云
        cropped_pcd = pcd.crop(bbox)

        # 创建ego车辆边界框
        ego_box_size = np.array([
            self.config['ego_roi']['length'],
            self.config['ego_roi']['width'],
            self.config['ego_roi']['height']
        ])
        ego_bbox = o3d.geometry.OrientedBoundingBox(
            center=vehicle_center,
            R=combined_final_matrix[:3, :3],
            extent=ego_box_size
        )

        # 获取ego区域内的点索引
        points = np.asarray(cropped_pcd.points)
        ego_indices = ego_bbox.get_point_indices_within_bounding_box(cropped_pcd.points)

        # 创建相反的掩码（保留ego box外的点）
        all_indices = np.arange(len(points))
        non_ego_indices = np.setdiff1d(all_indices, ego_indices)

        # 选择ego box外的点
        filtered_pcd = cropped_pcd.select_by_index(non_ego_indices)

        # 可视化ROI裁剪结果
        if self.vis and self.config['visualization']['show_roi']:
            bbox_param = [vehicle_center[0], vehicle_center[1], vehicle_center[2],
                          box_size[0], box_size[1], box_size[2],
                          combined_final_matrix[:3,:3]]
            ego_bbox_param = [vehicle_center[0], vehicle_center[1], vehicle_center[2],
                              ego_box_size[0], ego_box_size[1], ego_box_size[2],
                              combined_final_matrix[:3, :3]]

            self.vis.visualize_pcd_with_bbox_3d(pcd, [bbox_param, ego_bbox_param])

        return filtered_pcd


    def preprocess_points(self, pcd):
        """
        预处理点云
        :param pcd: 输入点云
        :return: 预处理后的点云
        """
        # 体素下采样
        pcd_down = pcd.voxel_down_sample(
            voxel_size=self.config['clustering']['voxel_size']
        )

        # 移除统计离群点
        cl, ind = pcd_down.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        # 估计法向量
        cl.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config['clustering']['voxel_size'] * 2,
                max_nn=30
            )
        )
        return cl

    def adjust_bbox_rotation(self, bbox, ground_normal):
        """
        调整边界框旋转矩阵使其与地面平行

        Args:
            bbox: 原始边界框 [x, y, z, w, l, h, yaw]
            ground_normal: 地面法向量 [a, b, c]

        Returns:
            调整后的边界框旋转矩阵
        """
        # 提取边界框参数
        center = bbox[:3]
        dimensions = bbox[3:6]
        yaw = bbox[6]

        # 原始的旋转矩阵(只考虑了yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

        # 确保地面法向量是单位向量
        ground_normal = ground_normal / np.linalg.norm(ground_normal)

        # 我们希望z轴与地面法向量对齐
        # 首先，确定x轴(保持与原来的x轴方向一致，但在地面上)
        x_axis = np.array([R_yaw[0, 0], R_yaw[1, 0], 0])

        # 如果x轴与法向量不垂直，我们需要调整它
        # 投影x轴到垂直于法向量的平面上
        x_axis = x_axis - np.dot(x_axis, ground_normal) * ground_normal

        # 归一化x轴
        x_axis = x_axis / np.linalg.norm(x_axis)

        # y轴为z轴(法向量)与x轴的叉积
        y_axis = np.cross(ground_normal, x_axis)

        # 构建新的旋转矩阵
        R_adjusted = np.column_stack((x_axis, y_axis, ground_normal))

        return R_adjusted

    def cluster_points(self, pcd: o3d.geometry.PointCloud, non_ground_pcd: o3d.geometry.PointCloud) -> Tuple[List, List]:
        """点云聚类"""
        # 检查点云是否为空或点数太少
        if len(pcd.points) < self.config['clustering']['min_points']:
            return [], []  # 返回空列表表示没有聚类结果
        try:
            # 预处理
            processed_pcd = self.preprocess_points(pcd)
            points = np.asarray(processed_pcd.points)
            # 再次检查预处理后的点数
            if len(points) < self.config['clustering']['min_points']:
                return [], []

            # # 使用优化后的聚类
            labels = fast_euclidean_cluster(
                points,
                self.config['clustering']['eps'],
                self.config['clustering']['min_samples']
            )

            # 处理聚类结果
            clusters = []
            objects = []
            unique_labels = np.unique(labels)

            # 存储所有边界框的参数，而不是直接渲染
            all_boxes = []

            for label in unique_labels:
                if label == -1:  # 跳过噪声点
                    continue

                # 获取当前聚类的点
                cluster_mask = labels == label
                cluster_points = points[cluster_mask]

                if len(cluster_points) < self.config['clustering']['min_points']:
                    continue

                #计算边界框
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                # # way1 PCA
                # bbox = cluster_pcd.get_oriented_bounding_box()
                #
                # all_boxes.append({
                #     'center': np.asarray(bbox.center),
                #     'dimensions': np.asarray(bbox.extent),
                #     'rotation_matrix': np.asarray(bbox.R),
                #     'pcd': cluster_pcd  # 保存点云以便后续可视化
                # })
                # # 提取边界框参数
                # center = np.asarray(bbox.center)
                # size = np.asarray(bbox.extent)
                # R = np.asarray(bbox.R)
                # yaw = np.arctan2(R[1, 0], R[0, 0])

                # way4
                # 使用compute_bbox函数计算边界框
                bbox = compute_bbox(cluster_points)

                # 解析边界框参数
                center = np.array([bbox[0], bbox[1], bbox[2]])
                size = np.array([bbox[3], bbox[4], bbox[5]])
                adjusted_R = self.adjust_bbox_rotation(bbox, self.ground_plane_normal)
                # 存储调整后的边界框
                all_boxes.append({
                    'center': center,
                    'dimensions': size,
                    'rotation_matrix': adjusted_R,
                    'pcd': cluster_pcd
                })

                # 从调整后的旋转矩阵中提取yaw角度
                yaw = np.arctan2(adjusted_R[1, 0], adjusted_R[0, 0])

                # 构建目标信息
                obj_info = {
                    'center': center,
                    'dimensions': size,
                    'yaw': yaw,
                    'points': cluster_points
                }

                # 分类物体
                if 2.1 < size[0] < 5 and 1.5 < size[1] < 2.5:  # 简单的车辆判断标准
                    obj_info['label'] = 'vehicle'
                else:
                    obj_info['label'] = 'blocks'

                clusters.append(cluster_points)
                objects.append(obj_info)

            # 最后统一渲染所有边界框
            if self.vis and self.config['visualization']['show_clusters'] and all_boxes:
                self.vis.visualize_all_clusters(non_ground_pcd, all_boxes)
            return clusters, objects

        except Exception as e:
            print(f"聚类错误: {e}")
            return [], []

    def estimate_velocity(self, current_objects: List[Dict], delta_t: float) -> List[Dict]:
        """估计物体速度"""
        # 检查当前物体列表是否为空
        if not current_objects:
            return []
        if self.previous_objects is None:
            # 第一帧,所有物体速度初始化为0
            for obj in current_objects:
                obj['velocity'] = np.zeros(3)
            return current_objects

        try:
            # 构建KD树用于匹配
            if len(self.previous_objects) == 0:
                # 如果前一帧没有物体，所有当前物体速度设为0
                for obj in current_objects:
                    obj['velocity'] = np.zeros(3)
                return current_objects

            # 构建KD树用于匹配
            prev_centers = np.array([obj['center'] for obj in self.previous_objects])
            tree = cKDTree(prev_centers)

            # 更新每个当前物体的速度
            for obj in current_objects:
                # 寻找最近的历史物体
                dist, idx = tree.query(obj['center'], k=1)

                if obj['label'] == 'vehicle':
                    if dist < 2.0:  # 匹配阈值
                        # 计算速度
                        displacement = obj['center'] - self.previous_objects[idx]['center']
                        obj['velocity'] = displacement / delta_t
                    else:
                        # 没有找到匹配,速度设为0
                        obj['velocity'] = np.zeros(3)
                else:
                    obj['velocity'] = np.zeros(3)

            return current_objects
        except Exception as e:
            print(f"速度估计错误: {e}")
            # 发生错误时，将所有物体速度设为0
            for obj in current_objects:
                obj['velocity'] = np.zeros(3)
            return current_objects

    def calculate_ego_velocity(self, current_position: List[float],
                               timestamp: float) -> np.ndarray:
        """计算ego速度

        Args:
            current_position: 当前位置 [x, y, z]
            timestamp: 当前时间戳

        Returns:
            velocity: [vx, vy, vz] 速度向量
        """
        if self.previous_ego_position is None or self.previous_timestamp is None:
            self.previous_ego_position = current_position
            self.previous_timestamp = timestamp
            return np.zeros(3)

        # 计算位置变化
        displacement = np.array(current_position) - np.array(self.previous_ego_position)

        # # 计算时间差 (确保时间戳单位统一)
        # dt = (timestamp - self.previous_timestamp)/10e-9
        # if dt == 0:
        #     return np.zeros(3)
        dt = 0.1

        # 计算速度
        velocity = displacement / dt

        # 更新上一帧的状态
        self.previous_ego_position = current_position
        self.previous_timestamp = timestamp

        return velocity

# 测试代码
if __name__ == "__main__":
    # 示例配置
    config = {
        'ground_removal': {
            'z_threshold': 0.2,
            'distance_threshold': 0.1,
            'ransac_n': 3,
            'num_iterations': 100000
        },
        'height_filter': {
            'min_height': 0.1,
            'max_height': 3.0
        },
        'coordinate_transform': {
            'yaw_angle_deg': 90,  # Rotate the point cloud counterclockwise by 90 degrees.
            'enable': True
        },
        'lidar':{
          'x_offset':1.4,
          'y_offset':0.88,
        },
        'roi': {
            'length': 20,
            'width': 6,
            'height': 4
        },
        'ego_roi':{
            'length': 3.6,
            'width': 2,
            'height': 2
        },
        'clustering': {
            'voxel_size': 0.05,
            'eps': 1.25,
            'min_samples': 80,
            'min_points': 10
        },
        'visualization': {
            'enable': True,  # 总开关
            'show_ground_removal': True,  # 显示地面移除
            'show_height_filter': True,  # 显示高度过滤
            'show_roi': True,  # 显示ROI裁剪
            'show_clusters': True  # 显示聚类结果
        }
    }

    detector = LidarObjectDetector(config)

    # 这里需要实际的点云数据和位姿矩阵来测试
    pcd_path = '../../output/test/scan_000001.pcd'
    pcd = o3d.io.read_point_cloud(pcd_path)

    pose_matrix = np.array([
        [9.99999502e-01, -7.82245691e-04, 6.19885549e-04, 4.70140755e-01],
        [7.81709297e-04, 9.99999320e-01, 8.65081326e-04, 1.21892632e-01],
        [-6.20561833e-04, -8.64596325e-04, 9.99999434e-01, 1.56693007e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    result = detector.process_frame(1,pcd, pose_matrix,1234567)
    print("目标检测器就绪")