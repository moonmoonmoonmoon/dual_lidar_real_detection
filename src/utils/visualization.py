# src/utils/visualization.py

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import patches

class Visualizer:
    def __init__(self):
        """初始化可视化器"""
        plt.ion()  # 开启交互模式
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)

    def create_pcd(points):
        """
        从点数组创建点云对象
        :param points: 点数组
        :return: 点云对象
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def plot_tracks(self, tracks, ego_pose=None):
        """
        可视化跟踪轨迹
        :param tracks: 跟踪对象列表
        :param ego_pose: 自车位置
        """
        plt.cla()

        # 绘制轨迹
        for track in tracks:
            positions = track.position_history
            if len(positions) < 2:
                continue

            positions = np.array(positions)
            # 根据目标类型选择颜色
            color = 'red' if track.is_dynamic else 'blue'

            # 绘制轨迹线
            plt.plot(positions[:, 0], positions[:, 1], '-', color=color, alpha=0.5)

            # 绘制当前位置
            current_pos = positions[-1]
            plt.plot(current_pos[0], current_pos[1], 'o', color=color)

            # 绘制边界框
            self._plot_box(current_pos[0], current_pos[1],
                           track.bbox[3], track.bbox[4],
                           track.bbox[6], color)

        # 绘制自车位置
        if ego_pose is not None:
            plt.plot(ego_pose[0], ego_pose[1], 'k^', markersize=10)

        plt.axis('equal')
        plt.grid(True)
        plt.pause(0.001)

    def plot_predictions(self, predictions, color='yellow'):
        """
        绘制预测轨迹
        :param predictions: 预测轨迹列表
        :param color: 轨迹颜色
        """
        positions = np.array([[p['center']['x'], p['center']['y']] for p in predictions])
        plt.plot(positions[:, 0], positions[:, 1], '--', color=color, alpha=0.5)

    def plot_collision_warning(self, text, position):
        """
        显示碰撞警告
        :param text: 警告文本
        :param position: 显示位置
        """
        plt.text(position[0], position[1], text,
                 color='red', fontsize=12, fontweight='bold')

    def _plot_box(self, x, y, width, length, yaw, color):
        """
        绘制旋转的边界框
        :param x, y: 中心位置
        :param width, length: 宽度和长度
        :param yaw: 航向角
        :param color: 颜色
        """
        corners = np.array([
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2]
        ])

        # 创建旋转矩阵
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])

        # 旋转角点并平移到中心位置
        rotated_corners = corners @ rot_matrix.T + np.array([x, y])

        # 绘制边界框
        plt.plot(rotated_corners[[0, 1, 2, 3, 0], 0],
                 rotated_corners[[0, 1, 2, 3, 0], 1],
                 '-', color=color, alpha=0.5)

    # @staticmethod
    def visualize_point_cloud(self, pcd, color=[0.5, 0.5, 0.5]):
        """
        可视化点云
        :param pcd: 点云对象
        :param color: 点云颜色
        """
        pcd.paint_uniform_color(color)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=960) # 增大窗口尺寸
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = 1.0

        # 添加坐标框架
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        vis.run()
        vis.destroy_window()

    def visualize_two_pcd(self,pcd1, pcd2):
        pcd1.paint_uniform_color([1, 0, 0])  # 红色
        pcd2.paint_uniform_color([0, 0, 1])  # 蓝色
        # Create a visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2)
        # Set the rendering options
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # White background
        opt.point_size = 1.0  # Set point size
        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def visualize_pcd_with_bbox(self,pcd, bboxes):
        """
        Visualize point cloud with bounding boxes.
        Args:
            pcd (o3d.geometry.PointCloud): Point cloud to visualize
            bboxes (list): List of bounding box parameters [x,y,z,w,l,h,yaw]
        """
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        # Add bounding boxes
        for bbox in bboxes:
            center = bbox[:3]
            extent = bbox[3:6]
            R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            obb.color = (1, 0, 0)  # Red color for bounding box
            vis.add_geometry(obb)
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # 白色背景
        opt.point_size = 1.0
        vis.run()
        vis.destroy_window()

    def visualize_pcd_with_bbox_3d(self, pcd, bboxes, z_min=-0.1, z_max=0.1, ground=False):
        """
        Visualize point cloud with 3D bounding boxes.
        :param pcd:
        :param bboxes:
        :param z_min:
        :param z_max:
        :return:
        """
        # Get points and colors
        points = np.asarray(pcd.points)
        colors = np.full_like(points, [0.5, 0.5, 0.5])  # Gray for point cloud
        # Filter ground points
        if ground:
            ground_points_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            colors[ground_points_mask] = [0, 0, 1]  # Blue for ground points
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Create a visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=960) # 增大窗口尺寸
        vis.add_geometry(pcd)
        # Add bounding boxes
        for bbox in bboxes:
            center = bbox[:3]
            extent = bbox[3:6]
            # R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
            R = bbox[6]
            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            obb.color = (1, 0, 0)  # Red color for bounding box
            vis.add_geometry(obb)
        # Set the rendering options
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # 白色背景
        opt.point_size = 1.0
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # # 设置视角
        # view_control = vis.get_view_control()
        # if len(bboxes) > 0:
        #     # 计算所有边界框的中心作为视角中心
        #     centers = np.array([bbox[:3] for bbox in bboxes])
        #     center = np.mean(centers, axis=0)
        #     view_control.set_lookat(center)
        #     view_control.set_front([0, 0, -1])
        #     view_control.set_up([0, 1, 0])
        #     view_control.set_zoom(0.8)  # 增大缩放比例

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def visualize_all_clusters(self, pcd, boxes):
        """统一可视化所有聚类结果"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=960) # 增大窗口尺寸

        points = np.asarray(pcd.points)
        colors = np.full_like(points, [0.5, 0.5, 0.5])  # Gray for point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)

        # 然后为每个点云创建并添加独立的边界框
        for box in boxes:
            obb = o3d.geometry.OrientedBoundingBox(
                center=box['center'],
                R=box['rotation_matrix'],
                extent=box['dimensions']
            )
            obb.color = [1, 0, 0]  # 红色边界框
            vis.add_geometry(obb)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = 1.0
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # 获取视图控制器并设置更合适的视角
        view_control = vis.get_view_control()

        # 计算所有边界框的中心作为视角中心
        centers = np.array([box['center'] for box in boxes])
        if len(centers) > 0:
            center = np.mean(centers, axis=0)
            # 设置相机位置，更近距离地观察目标
            view_control.set_lookat(center)
            view_control.set_front([0, 0, 1])  # 从上方向下看
            view_control.set_up([0, 1, 0])
            view_control.set_zoom(1.8)  # 增大缩放比例

        # 运行可视化
        vis.run()
        vis.destroy_window()


    def visualize_ground(self,ground_points, non_ground_points):
        """
        Visualize the ground plane with colored points.
        Args:
            ground_points (o3d.geometry.PointCloud): Ground points
            non_ground_points (o3d.geometry.PointCloud): Non-ground points
        """
        ground_points.paint_uniform_color([1, 0, 0])  # Red for ground
        non_ground_points.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for non-ground
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(ground_points)
        vis.add_geometry(non_ground_points)
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = 1.0
        vis.run()
        vis.destroy_window()

    def visualize_dynamic_points(self,dynamic_points, static_points):
        """
        Visualize dynamic and static points with different colors.
        Args:
            dynamic_points (o3d.geometry.PointCloud): Dynamic points
            static_points (o3d.geometry.PointCloud): Static points
        """
        dynamic_points.paint_uniform_color([1, 0, 0])  # Red for dynamic points
        static_points.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for static points
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(dynamic_points)
        vis.add_geometry(static_points)
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = 1.0
        vis.run()
        vis.destroy_window()

    def visualize_clusters(self,pcd, colors, labels):
        """
        可视化聚类结果
        :param pcd: 原始点云
        :param colors: 点云颜色数组
        :param labels: 聚类标签
        """
        # 创建可视化点云
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = pcd.points

        # 设置点云颜色
        # 将噪声点（label为-1）设置为灰色
        noise_indices = np.where(labels == -1)[0]
        colors[noise_indices] = [0.5, 0.5, 0.5]  # 灰色
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(vis_pcd)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # 白色背景
        opt.point_size = 1.0  # 设置点的大小

        # 运行可视化
        vis.run()
        vis.destroy_window()

    def visualize_tracking(self, tracked_objects, frame_id_mapping, frequency, km_per_h=True, arrow=False):
        """
        Visualize the tracking results.
        :param tracked_objects:
        :param frame_id_mapping:
        :param frequency:
        :param km_per_h:
        :param arrow:
        :return:
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        trajectories = {}
        if km_per_h:
            frequency = frequency * 3.6
        for frame_idx, (objects, frame_id) in enumerate(zip(tracked_objects, frame_id_mapping)):
            ax.clear()
            active_ids = []
            for obj in objects:
                bbox = obj['bbox']  # [x, y, w, l, yaw]
                print(f'Frame {frame_id}, Track ID: {obj["track_id"]}, yaw: {np.rad2deg(bbox[4]):.1f}')
                track_id = obj['track_id']
                speed = obj['speed']
                # print(speed)
                speed_magnitude = np.linalg.norm(speed) * frequency
                active_ids.append(track_id)
                # 使用bbox中的yaw创建旋转矩形
                rect = patches.Rectangle(
                    (bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2),
                    bbox[2], bbox[3],
                    angle=np.rad2deg(bbox[4]),
                    edgecolor='r',
                    facecolor='none',
                    alpha=1,  # 边框的透明度
                    linewidth=1.5,
                    rotation_point='center'
                )
                ax.add_patch(rect)
                if arrow:
                    arrow_length = 5.0  # 设置箭头长度
                    dx = arrow_length * np.cos(bbox[4])
                    dy = arrow_length * np.sin(bbox[4])
                    ax.arrow(bbox[0], bbox[1], dx, dy,
                             head_width=0.5, head_length=1.0, fc='r', ec='r')
                # 添加ID和速度标签
                ax.text(bbox[0] + bbox[2], bbox[1] + bbox[3], f'id:{track_id} v:{speed_magnitude:.1f}km/h',
                        color='black')
                if track_id not in trajectories:
                    trajectories[track_id] = []
                trajectories[track_id].append([bbox[0], bbox[1]])
                if track_id in trajectories:
                    trajectory = np.array(trajectories[track_id])
                    ax.plot(trajectory[:, 0], trajectory[:, 1], '-', color='r', linewidth=1, alpha=0.5)
            trajectories = {k: v for k, v in trajectories.items() if k in active_ids}
            ax.set_xlim(-55, 75)
            ax.set_ylim(-55, 75)
            ax.grid(True)
            ax.set_title(f'Frame ID: {frame_id}')  # 使用原始frame_id
            plt.pause(0.1)
        plt.show()


    def close(self):
        """关闭可视化窗口"""
        plt.close(self.fig)