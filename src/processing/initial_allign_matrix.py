import copy
import os
import sys
import numpy as np
import argparse
import pandas as pd
import open3d as o3d
from tqdm import tqdm


np.set_printoptions(suppress=True)

# Filter point cloud based on distance and remove zero rows and nans
def filter_point_cloud(point_cloud, max_distance):
    # remove zero rows
    point_cloud = point_cloud[~np.all(point_cloud[:, :3] == 0.0, axis=1)]

    # remove nans
    point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1), :]

    # remove points above 200 m distance (robosense) and above 120 m distance (ouster)
    distances = np.array(
        [np.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]) for point in point_cloud]
    )
    point_cloud = point_cloud[distances < max_distance, :]
    return point_cloud
def find_ground(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """计算地面变换矩阵"""
    # 地面检测
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.1,
        ransac_n=3,
        num_iterations=100000
    )

    # 计算变换矩阵
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
# Preprocess point cloud
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return pcd_down, pcd_fpfh

# Prepare point cloud for registration
def prepare_point_cloud(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

# Initial registration using RANSAC
def initial_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result

# Refine registration using ICP
def refine_registration(source, target, voxel_size, trans_init, use_point_to_plane):
    distance_threshold = voxel_size * 0.4
    if use_point_to_plane:
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
    else:
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
    return result


def check_and_correct_orientation(source, target):
    # 计算主方向
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # 使用PCA计算主方向
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)

    source_covariance = np.cov(source_points.T)
    target_covariance = np.cov(target_points.T)

    source_eigenvalues, source_eigenvectors = np.linalg.eigh(source_covariance)
    target_eigenvalues, target_eigenvectors = np.linalg.eigh(target_covariance)

    # 如果主方向相反，旋转180度
    if np.dot(source_eigenvectors[:, -1], target_eigenvectors[:, -1]) < 0:
        R = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        source.transform(T)

    return source, target

def register_point_clouds(
        point_cloud_source,
        point_cloud_target,
        initial_voxel_size,
        continuous_voxel_size,
        use_point_to_plane
):
    num_initial_registration_loops = 4

    # 使用numpy的inf替代sys.maxsize更符合数值计算
    inlier_rmse_best = np.inf
    fitness_best = 0
    loss_best = np.inf
    transformation_matrix = None

    # 提前转换点云格式，避免重复转换
    source = o3d.geometry.PointCloud()
    init_transform = np.eye(4)
    # 如果已知两个点云大致的相对位置，可以设置一个粗略的初始变换
    source.transform(init_transform)
    source.points = o3d.utility.Vector3dVector(point_cloud_source[:, 0:3])
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(point_cloud_target[:, 0:3])
    # 预计算降采样的点云，避免重复计算
    source_down = source.voxel_down_sample(continuous_voxel_size)
    target_down = target.voxel_down_sample(continuous_voxel_size)

    if use_point_to_plane:
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    # transformation_matrix = np.array([[0.8717529, -0.48938331, -0.02347018, 1.17635401],
    #                                  [0.48991357, 0.87014549, 0.053212, -0.67875347],
    #                                  [-0.00561859, -0.05788607, 0.99830738, -0.0290115 ],
    #                                  [0, 0, 0, 1]])
    for i in range(num_initial_registration_loops):
        source, target, source_init_down, target_init_down, source_fpfh, target_fpfh = prepare_point_cloud(
            source, target, initial_voxel_size
        )

        initial_result = initial_registration(
            source_init_down, target_init_down, source_fpfh, target_fpfh, initial_voxel_size
        )

        loss = initial_result.inlier_rmse + 1 / initial_result.fitness
        if loss < loss_best:
            transformation_matrix = initial_result.transformation
            inlier_rmse_best = initial_result.inlier_rmse
            fitness_best = initial_result.fitness
            loss_best = loss

    return transformation_matrix, inlier_rmse_best, fitness_best


def read_pcd(filepath, num_fields=3):
    """Read PCD file and return numpy array

    Args:
        filepath: Path to PCD file
        num_fields: Number of fields to read (3 for XYZ, 4 for XYZI)

    Returns:
        numpy.ndarray: Point cloud data
    """
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    if num_fields == 4 and hasattr(pcd, 'colors'):
        # Assuming intensity is stored in the red channel of colors
        intensities = np.asarray(pcd.colors)[:, 0]
        points = np.column_stack((points, intensities))
    return points

def create_pcd(points):
    """Create Open3D PointCloud object from numpy array

    Args:
        points: numpy array of shape (N, 3) containing XYZ coordinates

    Returns:
        open3d.geometry.PointCloud: Point cloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def visualize_pcd(source_points: np.ndarray,target_points:np.ndarray):
    """可视化配准结果

    Args:
        source_points: 原始源点云
        target_points: 原始目标点云
        combined_points: 配准后的组合点云
        transform: 源点云的总变换矩阵(包含地面对齐和配准变换)
        ground_transform2: 目标点云的地面变换矩阵
    """
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 设置不同颜色
    source_points.paint_uniform_color([1, 0, 0])  # 红色
    target_points.paint_uniform_color([0, 1, 0])  # 红色


    # 添加坐标系（便于观察）
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame)

    # 添加到可视化器
    vis.add_geometry(source_points)
    vis.add_geometry(target_points)
    # 更新视图
    vis.update_geometry(source_points)
    vis.update_geometry(target_points)

    # 设置视角
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.asarray([0, 0, 0])

    # 运行可视化
    vis.run()
    vis.destroy_window()

def main():
    # Define paths directly
    source_path = "../../data/1621/source_pcd_out_000000.pcd"  # 1621 lidar pcd path
    target_path = "../../data/1778/target_pcd_out_000000.pcd"  # 1778 lidar pcd path

    # Read source and target point clouds
    print("Reading source point cloud...")
    pcd_source0 = read_pcd(source_path)
    print("Reading target point cloud...")
    pcd_target0 = read_pcd(target_path)

    # Create Open3D point cloud objects
    print("Creating point cloud objects...")
    pcd_source = create_pcd(pcd_source0[:, :3])
    pcd_target = create_pcd(pcd_target0[:, :3])

    # Filter point clouds
    print("Filtering point clouds...")
    source_filtered = filter_point_cloud(np.asarray(pcd_source.points), max_distance=100)
    target_filtered = filter_point_cloud(np.asarray(pcd_target.points), max_distance=100)

    # ground_matrix_source = find_ground(pcd_source)
    # print('ground_matrix_source',ground_matrix_source)
    # np.save('../../data/ground_transform_source.npy',ground_matrix_source)

    # ground_matrix_target = find_ground(pcd_target)
    # print('ground_matrix_target',ground_matrix_target)
    # np.save('../../data/ground_transform_target.npy', ground_matrix_target)
    # ground_matrix_source = np.load("ground_transform_source.npy")
    # ground_matrix_target = np.load("ground_transform_target.npy")

    # 2025_bus
    ground_matrix_source = np.load("../../data/self_ground_source_1621.npy")
    ground_matrix_target = np.load("../../data/self_ground_target_1778.npy")

    pcd_source.transform(ground_matrix_source)

    pcd_target.transform(ground_matrix_target)
    visualize_pcd(pcd_source, pcd_target)
    pcd_source, pcd_target = check_and_correct_orientation(pcd_source, pcd_target)
    pcd_source_array = np.asarray(pcd_source.points)
    pcd_target_array = np.asarray(pcd_target.points)

    point_cloud_source = filter_point_cloud(pcd_source_array, max_distance=100)
    point_cloud_target = filter_point_cloud(pcd_target_array, max_distance=100)

    use_point_to_plane = True
    initial_voxel_size = 1
    continuous_voxel_size = 1

    print("Starting registration process...")
    # Perform registration
    transformation_matrix, inlier_rmse, fitness = register_point_clouds(
        point_cloud_source,
        point_cloud_target,
        initial_voxel_size,
        continuous_voxel_size,
        use_point_to_plane
    )

    print("\nRegistration Results:")
    print(f"RMSE: {inlier_rmse:.6f}")
    print(f"Fitness: {fitness:.6f}")
    print("\nTransformation Matrix:")
    print(transformation_matrix)

    # Save transformation matrix
    output_file = "initial_transformation_matrix.txt"
    np.savetxt(output_file, transformation_matrix)
    print(f"\nTransformation matrix saved to {output_file}")

    # Apply transformation to source point cloud
    pcd_source_transformed = copy.deepcopy(pcd_source)
    pcd_source_transformed.transform(transformation_matrix)

    # Visualize result
    print("\nVisualizing registration result...")
    visualize_pcd(pcd_source_transformed, pcd_target)

if __name__ == "__main__":
    main()