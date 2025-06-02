# 3_object_detector.py
"""实时目标检测模块,用于检测和跟踪物体"""
import time
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import List, Tuple, Dict
from src.utils.visualization import Visualizer
from src.utils.lidar_processing_utils import height_filter_fast, fast_euclidean_cluster, ground_points_filter_fast

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
                'distance_threshold': 0.1,
                'ransac_n': 3,
                'num_iterations': 100000
            },
            'height_filter': {
                'min_height': 0.1,
                'max_height': 3.0
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

    def process_frame(self, frame_id: int, pcd: o3d.geometry.PointCloud,
                      final_transform: np.ndarray, timestamp: np.uint64) -> DetectionResult:
        """处理单帧点云数据"""
        try:
            process_start = time.time()

            # 1. 地面检测和移除
            ground_start = time.time()
            non_ground_pcd = self.remove_ground(pcd)
            ground_time = time.time() - ground_start
            print(f"Ground removal time: {ground_time * 1000:.1f}ms")

            # 2. 高度过滤
            filter_start = time.time()
            filtered_pcd = self.height_filter(non_ground_pcd)
            filter_time = time.time() - filter_start
            print(f"Height filter time: {filter_time * 1000:.1f}ms")

            # 3. 坐标系对齐（新增步骤）
            align_start = time.time()
            points = np.asarray(filtered_pcd.points)

            aligned_points = self.align_coordinate_system(points,np.pi/2)

            aligned_pcd = o3d.geometry.PointCloud()
            aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
            align_time = time.time() - align_start
            print(f"Coordinate alignment time: {align_time * 1000:.1f}ms")


            # 3. ROI裁剪
            roi_start = time.time()
            roi_pcd = self.crop_roi(aligned_pcd, final_transform, np.pi/2)
            roi_time = time.time() - roi_start
            print(f"ROI crop time: {roi_time * 1000:.1f}ms")

            # 4. 计算ego速度
            ego_velocity = self.calculate_ego_velocity(self.ego_position, 0.1)

            # 5. 聚类检测
            cluster_start = time.time()
            clusters, objects = self.cluster_points(roi_pcd)
            cluster_time = time.time() - cluster_start
            print(f"Clustering time: {cluster_time * 1000:.1f}ms")

            # 6. 计算速度(如果有前一帧)
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

    ## way1: fast remove ground through height value
    def remove_ground(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """基于高度阈值移除地面点

        Args:
            pcd: 输入点云

        Returns:
            non_ground_pcd: 移除地面后的点云
        """
        ground_removal_start = time.time()

        # 转换为numpy数组处理
        points = np.asarray(pcd.points)

        # 使用z阈值区分地面点和非地面点
        ground_mask = points[:, 2] <= self.config['ground_removal'].get('z_threshold', 0.4)
        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]

        # 创建地面和非地面点云对象
        non_ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)

        ground_removal_time = time.time() - ground_removal_start
        print(f'Ground removal time: {ground_removal_time * 1000:.1f}ms')

        # 如果启用可视化,显示地面移除结果
        if self.vis and self.config['visualization']['show_ground_removal']:
            ground_pcd = o3d.geometry.PointCloud()
            ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
            self.vis.visualize_ground(ground_pcd, non_ground_pcd)

        return non_ground_pcd

    ## way2: identify ground on time
    # def remove_ground(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    #     """地面检测和移除"""
    #     find_ground_start_time = time.time()
    #     # 只在第一帧进行地面检测
    #     if self.first_frame:
    #         # 使用Open3D的segment_plane进行地面检测
    #         plane_model, inliers = pcd.segment_plane(
    #             distance_threshold=self.config['ground_removal']['distance_threshold'],
    #             ransac_n=self.config['ground_removal']['ransac_n'],
    #             num_iterations=self.config['ground_removal']['num_iterations']
    #         )
    #         self.ground_plane_model = plane_model  # 保存地面模型
    #
    #         self.first_frame = False
    #         find_ground_time = time.time() - find_ground_start_time
    #         print(f'First frame ground detection time: {find_ground_time * 1000:.1f}ms')
    #
    #     # 使用保存的地面模型直接计算inliers
    #     ground_transform_start_time = time.time()
    #     points = np.asarray(pcd.points)
    #     a, b, c, d = self.ground_plane_model
    #     distances = np.abs(np.dot(points, [a, b, c]) + d)
    #     inliers = np.where(distances <= self.config['ground_removal']['distance_threshold'])[0]
    #
    #     # 分离地面点和非地面点
    #     non_ground_pcd = pcd.select_by_index(inliers, invert=True)
    #     ground_removel_time = time.time() - ground_transform_start_time
    #     print(f'Ground remove time: {ground_removel_time * 1000:.1f}ms')
    #
    #     if self.vis and self.config['visualization']['show_ground_removal']:
    #         ground_pcd = pcd.select_by_index(inliers)
    #         self.vis.visualize_ground(ground_pcd, non_ground_pcd)
    #     return non_ground_pcd

    ## way3: identify ground every time
    # def remove_ground(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    #     """地面检测和移除"""
    #     plane_model, inliers = pcd.segment_plane(
    #         distance_threshold=self.config['ground_removal']['distance_threshold'],
    #         ransac_n=self.config['ground_removal']['ransac_n'],
    #         num_iterations=self.config['ground_removal']['num_iterations']
    #     )
    #
    #     # 分离地面点和非地面点
    #     ground_pcd = pcd.select_by_index(inliers)
    #     non_ground_pcd = pcd.select_by_index(inliers, invert=True)
    #
    #     if self.vis and self.config['visualization']['show_ground_removal']:
    #         # ground_pcd = pcd.select_by_index(inliers)
    #         self.vis.visualize_ground(ground_pcd, non_ground_pcd)
    #     return non_ground_pcd


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



    def height_filter(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:

        points = np.asarray(pcd.points)
        filtered_points = height_filter_fast(
            points,
            self.config['height_filter']['min_height'],
            self.config['height_filter']['max_height']
        )

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 可视化高度过滤结果
        if self.vis and self.config['visualization']['show_height_filter']:
            self.vis.visualize_point_cloud(filtered_pcd, color=[0.5, 0.5, 0.5])

        return filtered_pcd

    def align_coordinate_system(self, points: np.ndarray, yaw_angle: float = np.pi / 2) -> np.ndarray:
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

        return aligned_points

    def crop_roi(self, pcd: o3d.geometry.PointCloud,
                 final_matrix: np.ndarray, yaw_angle:float) -> o3d.geometry.PointCloud:
        """ROI区域裁剪"""
        box_size = np.array([
            self.config['roi']['length'],
            self.config['roi']['width'],
            self.config['roi']['height']
        ])

        # 组合变换矩阵
        # combined_matrix = np.dot(self.transform_matrix, pose_matrix)
        # 构建绕Z轴旋转的变换矩阵
        rotation_matrix = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1]
        ])
        rotation_position = np.dot(rotation_matrix[:3,:3], final_matrix[:3,3])
        lidar_position = rotation_position
        # crop_center = lidar_position

        # 获取自车位置
        vehicle_center = lidar_position.copy()
        vehicle_center[0] -= self.lidar_x_offset
        vehicle_center[1] -= self.lidar_y_offset
        self.ego_position = vehicle_center.tolist()
        # print(self.ego_position)
        # print(pose_matrix)
        crop_center = vehicle_center
        # 创建ROI边界框
        bbox = o3d.geometry.OrientedBoundingBox(
            center=crop_center,
            R=final_matrix[:3, :3],
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
            R=final_matrix[:3, :3],
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
            # bbox_param = [crop_center[0], crop_center[1], crop_center[2],
            #               box_size[0], box_size[1], box_size[2],
            #               np.arctan2(combined_matrix[1, 0], combined_matrix[0, 0])]
            # ego_bbox_param = [vehicle_center[0], vehicle_center[1], vehicle_center[2],
            #                   ego_box_size[0], ego_box_size[1], ego_box_size[2],
            #                   np.arctan2(combined_matrix[1, 0], combined_matrix[0, 0])]
            bbox_param = [crop_center[0], crop_center[1], crop_center[2],
                          box_size[0], box_size[1], box_size[2],
                          np.arctan2(final_matrix[1, 0], final_matrix[0, 0])]
            ego_bbox_param = [vehicle_center[0], vehicle_center[1], vehicle_center[2],
                              ego_box_size[0], ego_box_size[1], ego_box_size[2],
                              np.arctan2(final_matrix[1, 0], final_matrix[0, 0])]
            self.vis.visualize_pcd_with_bbox_3d(filtered_pcd, [bbox_param, ego_bbox_param])

        return filtered_pcd
        # 3_object_detector.py (继续)
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


    def cluster_points(self, pcd: o3d.geometry.PointCloud) -> Tuple[List, List]:
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

            for label in unique_labels:
                if label == -1:  # 跳过噪声点
                    continue

                # 获取当前聚类的点
                cluster_mask = labels == label
                cluster_points = points[cluster_mask]

                if len(cluster_points) < self.config['clustering']['min_points']:
                    continue

                # 计算边界框
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                bbox = cluster_pcd.get_oriented_bounding_box()

                # 提取边界框参数
                center = np.asarray(bbox.center)
                size = np.asarray(bbox.extent)
                R = np.asarray(bbox.R)
                yaw = np.arctan2(R[1, 0], R[0, 0])

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

                # 可视化聚类结果
                if self.vis and self.config['visualization']['show_clusters']:
                    if objects:  # 如果有检测到物体，显示边界框
                        boxes = [[
                            float(obj['center'][0]), float(obj['center'][1]), float(obj['center'][2]),
                            float(obj['dimensions'][0]), float(obj['dimensions'][1]), float(obj['dimensions'][2]),
                            float(obj['yaw'])
                        ] for obj in objects]
                        self.vis.visualize_pcd_with_bbox_3d(processed_pcd, boxes)

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