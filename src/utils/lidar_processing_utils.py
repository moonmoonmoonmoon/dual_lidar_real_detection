# lidar_processing_utils.py
from numba import jit, prange
import numpy as np


@jit(nopython=True)
def transform_points_fast(points, pose_matrix):
    """使用 Numba 加速点云坐标转换"""
    n_points = points.shape[0]

    points_world = np.empty((n_points, 3))

    for i in prange(n_points):
        x, y, z = points[i]
        points_world[i, 0] = pose_matrix[0, 0] * x + pose_matrix[0, 1] * y + pose_matrix[0, 2] * z + pose_matrix[0, 3]
        points_world[i, 1] = pose_matrix[1, 0] * x + pose_matrix[1, 1] * y + pose_matrix[1, 2] * z + pose_matrix[1, 3]
        points_world[i, 2] = pose_matrix[2, 0] * x + pose_matrix[2, 1] * y + pose_matrix[2, 2] * z + pose_matrix[2, 3]

    return points_world


@jit(nopython=True)
def combine_point_clouds_fast(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """使用numba加速点云合并"""
    n1, n2 = len(points1), len(points2)
    combined = np.empty((n1 + n2, 3))
    for i in prange(n1):
        combined[i] = points1[i]
    for i in prange(n2):
        combined[n1 + i] = points2[i]
    return combined


@jit(nopython=True)
def transform_and_merge_clouds_fast(source_points: np.ndarray,
                                    target_points: np.ndarray,
                                    transform_matrix_source: np.ndarray,
                                    transform_matrix_target: np.ndarray,
                                    merge_matrix: np.ndarray) -> np.ndarray:
    """
    使用numba加速的点云变换和合并函数
    points1: source点云
    points2: target点云
    transform_matrix1: target到地面的变换矩阵
    transform_matrix2: source到地面的变换矩阵
    merge_matrix: 地面上的对齐矩阵
    """
    # 1. 转换target点云到地面
    points2_ground = transform_points_fast(target_points, transform_matrix_target)


    # 2. 使用merge矩阵对齐source点云
    points1_aligned = transform_points_fast(source_points, merge_matrix)

    # 3. 合并点云
    combined_points = combine_point_clouds_fast(points1_aligned, points2_ground)

    return combined_points


@jit(nopython=True)
def height_filter_fast(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    """加速高度过滤"""
    mask = np.zeros(len(points), dtype=np.bool_)
    for i in prange(len(points)):
        if min_height <= points[i, 2] <= max_height:
            mask[i] = True
    return points[mask]


@jit(nopython=True)
def ground_points_filter_fast(points: np.ndarray, plane_model: np.ndarray, distance_threshold: float) -> np.ndarray:
    """使用numba加速的地面点分离
    Args:
        points: 点云数据 (N, 3)
        plane_model: 平面模型参数 [a, b, c, d]
        distance_threshold: 距离阈值
    Returns:
        mask: 非地面点的布尔掩码
    """
    a, b, c, d = plane_model
    n_points = len(points)
    mask = np.ones(n_points, dtype=np.bool_)  # True表示非地面点

    for i in prange(n_points):
        point = points[i]
        # 计算点到平面的距离
        distance = abs(point[0] * a + point[1] * b + point[2] * c + d)
        if distance <= distance_threshold:
            mask[i] = False  # 地面点

    return mask



@jit(nopython=True)
def fast_euclidean_cluster(points: np.ndarray,
                           eps: float,
                           min_samples: int) -> np.ndarray:
    """使用Numba加速的聚类核心计算"""
    n_points = len(points)
    labels = -np.ones(n_points, dtype=np.int32)
    current_label = 0

    # 第一步: 找到核心点
    core_points = np.zeros(n_points, dtype=np.bool_)
    for i in range(n_points):
        neighbors = 0
        for j in prange(n_points):
            if i != j:
                distance = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                if distance < eps:
                    neighbors += 1
                    if neighbors >= min_samples:
                        core_points[i] = True
                        break

    # 第二步: 扩展聚类
    for i in prange(n_points):
        if not core_points[i] or labels[i] >= 0:
            continue

        # 开始新的聚类
        labels[i] = current_label
        stack = [i]

        while stack:
            current = stack.pop()

            # 寻找邻居
            for j in prange(n_points):
                if labels[j] >= 0:
                    continue

                distance = np.sqrt(np.sum((points[current] - points[j]) ** 2))
                if distance < eps:
                    labels[j] = current_label
                    if core_points[j]:
                        stack.append(j)

        current_label += 1
    return labels

@jit(nopython=True)
def compute_transform_matrix_fast(plane_model: np.ndarray) -> np.ndarray:
    """计算变换矩阵"""
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_length = np.sqrt(np.sum(normal ** 2))

    normal = normal / normal_length
    d = d / normal_length

    # 确保法向量朝上
    if c < 0:
        normal = -normal
        d = -d

    # 构建旋转矩阵
    z_axis = np.array([0., 0., 1.])
    if abs(np.dot(normal, z_axis)) > 0.999:
        rotation_matrix = np.eye(3)
        if np.dot(normal, z_axis) < 0:
            rotation_matrix[2, 2] = -1
    else:
        rotation_axis = np.cross(normal, z_axis)
        rotation_axis = rotation_axis / np.sqrt(np.sum(rotation_axis ** 2))
        cos_angle = np.dot(normal, z_axis)
        sin_angle = np.sqrt(1 - cos_angle ** 2)

        # Rodrigues旋转公式
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)

    # 构建4x4变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    point_on_plane = np.array([0, 0, -d / c]) if abs(c) > 1e-10 else np.array([0, -d / b, 0]) if abs(
        b) > 1e-10 else np.array([-d / a, 0, 0])
    transform[:3, 3] = -(rotation_matrix @ point_on_plane)

    return transform

@jit(nopython=True)
def compute_distances_fast(points: np.ndarray) -> np.ndarray:
    """加速点之间距离计算"""
    n = len(points)
    distances = np.zeros((n, n))
    for i in prange(n):
        for j in prange(i + 1, n):
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


@jit(nopython=True)
def cluster_core_fast(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """加速DBSCAN核心点发现"""
    n_points = len(points)
    neighbor_counts = np.zeros(n_points, dtype=np.int32)

    for i in prange(n_points):
        for j in prange(n_points):
            if i != j:
                dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                if dist <= eps:
                    neighbor_counts[i] += 1

    core_points_mask = neighbor_counts >= min_samples
    return core_points_mask


@jit(nopython=True)
def optimized_clustering(points: np.ndarray, eps: float, min_samples: int):
    """优化的聚类实现
    Args:
        points: 输入点云 (N, 3)
        eps: 邻域半径
        min_samples: 最小样本数
    Returns:
        labels: 聚类标签 (-1表示噪声点)
    """
    # 1. 使用cluster_core_fast找到核心点
    core_points_mask = cluster_core_fast(points, eps, min_samples)

    # 2. 初始化标签
    n_points = len(points)
    labels = -np.ones(n_points, dtype=np.int32)  # 所有点初始化为-1(噪声点)
    current_label = 0

    # 3. 对每个核心点进行聚类扩展
    for i in prange(n_points):
        # 跳过非核心点或已标记的点
        if not core_points_mask[i] or labels[i] >= 0:
            continue

        # 标记当前核心点
        labels[i] = current_label

        # 寻找当前核心点的邻居
        for j in prange(n_points):
            if labels[j] >= 0:  # 跳过已标记的点
                continue

            # 计算距离
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            if dist <= eps:
                labels[j] = current_label

        current_label += 1

    return labels


@jit(nopython=True)
def estimate_plane_fast(points: np.ndarray, n_iterations: int,
                        distance_threshold: float) :
    """加速RANSAC平面估计"""
    best_plane = np.zeros(4)
    best_inliers_count = 0
    n_points = len(points)

    for _ in prange(n_iterations):
        # 随机选择3个点
        idx = np.random.randint(0, n_points, 3)
        p1, p2, p3 = points[idx]

        # 计算平面法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.sqrt(np.sum(normal ** 2))
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        plane = np.array([normal[0], normal[1], normal[2], d])

        # 计算点到平面的距离
        distances = np.abs(points @ normal + d)
        inliers = distances < distance_threshold
        inliers_count = np.sum(inliers)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_plane = plane

    return best_plane, best_inliers_count


@jit(nopython=True)
def estimate_plane_ransac(points: np.ndarray,
                          distance_threshold: float,
                          num_iterations: int):
    """使用RANSAC估计平面"""
    best_plane = np.zeros(4)
    best_inlier_mask = np.zeros(len(points), dtype=np.bool_)
    max_inliers = 0

    for _ in prange(num_iterations):
        # 随机选择3个点
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        # 计算平面方程 ax + by + cz + d = 0
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.sqrt(np.sum(normal ** 2))
        if norm == 0:
            continue
        normal = normal / norm
        # 使用手动展开的点积运算替代np.dot
        d = -(normal[0] * p1[0] + normal[1] * p1[1] + normal[2] * p1[2])

        # 计算所有点到平面的距离
        distances = np.empty(len(points))
        for i in prange(len(points)):
            distances[i] = abs(normal[0] * points[i, 0] + normal[1] * points[i, 1] +
                               normal[2] * points[i, 2] + d)
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_plane = np.array([normal[0], normal[1], normal[2], d])
            best_inlier_mask = inlier_mask

    return best_plane, best_inlier_mask


def get_improved_oriented_box(points, analyze_density=True):
    """
    改进的有向边界框算法，更好地处理不完整点云
    Args:
        points: 点云数据，np.ndarray，形状为(N, 3)
        analyze_density: 是否分析密度分布
    Returns:
        bbox_info: 包含边界框信息的字典
    """
    from sklearn.decomposition import PCA
    import numpy as np

    # 检查点云是否为空
    if len(points) < 10:
        return None

    # 1. 使用PCA计算主方向
    pca = PCA(n_components=3, random_state=42)  # 添加随机种子保证一致性
    pca.fit(points)

    # 主方向向量
    principal_axes = pca.components_

    # 计算在主方向上的投影
    projected_points = points @ principal_axes.T
    min_proj = np.min(projected_points, axis=0)
    max_proj = np.max(projected_points, axis=0)
    dimensions = max_proj - min_proj

    # 原始中心
    raw_center = (min_proj + max_proj) / 2
    adjusted_center = raw_center.copy()
    # print('a_c',adjusted_center)

    # 分析密度分布（如果需要）
    is_imbalanced = False
    if analyze_density:
        hist, _ = np.histogram(projected_points[:, 0], bins=10)
        normalized_hist = hist / np.sum(hist)

        left_half = np.sum(normalized_hist[:5])
        right_half = np.sum(normalized_hist[5:])
        imbalance = abs(left_half - right_half)
        is_imbalanced = imbalance > 0.3

        # 根据密度分布调整
        if is_imbalanced:
            # 左侧或右侧密集，对应调整
            if left_half > right_half:
                extension_factor = 0.5
                adjusted_center[0] += dimensions[0] * extension_factor / 2
                dimensions[0] *= (1 + extension_factor)
            else:
                extension_factor = 0.5
                adjusted_center[0] -= dimensions[0] * extension_factor / 2
                dimensions[0] *= (1 + extension_factor)
    # 转回原始坐标系
    center = (principal_axes.T @ adjusted_center)
    # center = (principal_axes.T @ adjusted_center) + np.mean(points, axis=0)
    # print('c',center)

    # 返回完整信息
    return {
        'center': center,
        'dimensions': dimensions,
        'rotation_matrix': principal_axes,
        'is_imbalanced': is_imbalanced
    }


@jit(nopython=True, cache=True)
def compute_bbox(points_coords):
    """计算具有最小xy平面投影面积的边界框，严格包络所有点"""
    if len(points_coords) == 0:
        # 处理空点集的情况
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 手动计算最小和最大坐标，不使用axis参数
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf

    for i in range(len(points_coords)):
        x, y, z = points_coords[i]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    min_coords = np.array([min_x, min_y, min_z])
    max_coords = np.array([max_x, max_y, max_z])
    centroid_coords = (min_coords + max_coords) / 2

    # 将点云移动到中心
    centered_points_coords = np.empty_like(points_coords)
    for i in range(len(points_coords)):
        centered_points_coords[i] = points_coords[i] - centroid_coords

    centered_points_2d_coords = centered_points_coords[:, :2]

    best_yaw = 0.0
    min_area = 1e38  # 用一个非常大的数代替infinity

    # 遍历所有可能的旋转角度
    for angle_deg in np.arange(0, 90, 0.5):  # 使用np.arange支持浮点数步长
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)

        # 将点云旋转
        rotated_points = np.empty((len(centered_points_2d_coords), 2))
        for i in range(len(centered_points_2d_coords)):
            x, y = centered_points_2d_coords[i]
            rotated_points[i, 0] = c * x + s * y
            rotated_points[i, 1] = -s * x + c * y

        # 手动计算旋转后点云的边界
        rot_min_x = np.inf
        rot_min_y = np.inf
        rot_max_x = -np.inf
        rot_max_y = -np.inf

        for i in range(len(rotated_points)):
            rot_min_x = min(rot_min_x, rotated_points[i, 0])
            rot_min_y = min(rot_min_y, rotated_points[i, 1])
            rot_max_x = max(rot_max_x, rotated_points[i, 0])
            rot_max_y = max(rot_max_y, rotated_points[i, 1])

        width = rot_max_x - rot_min_x
        length = rot_max_y - rot_min_y
        area = width * length

        if area < min_area:
            min_area = area
            best_yaw = angle

    # 使用最优角度对所有点应用旋转
    c, s = np.cos(best_yaw), np.sin(best_yaw)
    rotated_points_3d = np.empty_like(centered_points_coords)

    for i in range(len(centered_points_coords)):
        x, y, z = centered_points_coords[i]
        rotated_points_3d[i, 0] = c * x + s * y
        rotated_points_3d[i, 1] = -s * x + c * y
        rotated_points_3d[i, 2] = z

    # 手动计算旋转后的3D点云边界
    rot_min_x = np.inf
    rot_min_y = np.inf
    rot_min_z = np.inf
    rot_max_x = -np.inf
    rot_max_y = -np.inf
    rot_max_z = -np.inf

    for i in range(len(rotated_points_3d)):
        rot_min_x = min(rot_min_x, rotated_points_3d[i, 0])
        rot_min_y = min(rot_min_y, rotated_points_3d[i, 1])
        rot_min_z = min(rot_min_z, rotated_points_3d[i, 2])
        rot_max_x = max(rot_max_x, rotated_points_3d[i, 0])
        rot_max_y = max(rot_max_y, rotated_points_3d[i, 1])
        rot_max_z = max(rot_max_z, rotated_points_3d[i, 2])

    rot_min_coords = np.array([rot_min_x, rot_min_y, rot_min_z])
    rot_max_coords = np.array([rot_max_x, rot_max_y, rot_max_z])

    # 计算边界框尺寸
    size = rot_max_coords - rot_min_coords

    # 边界框中心（在旋转坐标系中）
    bbox_center_rotated = (rot_min_coords + rot_max_coords) / 2

    # 将边界框中心转回原始坐标系 - 使用逆旋转矩阵
    bbox_center = np.empty(3)
    # 逆旋转矩阵的应用
    bbox_center[0] = centroid_coords[0] + c * bbox_center_rotated[0] - s * bbox_center_rotated[1]
    bbox_center[1] = centroid_coords[1] + s * bbox_center_rotated[0] + c * bbox_center_rotated[1]
    bbox_center[2] = centroid_coords[2] + bbox_center_rotated[2]

    # 最终的边界框参数 [x,y,z,w,l,h,yaw]
    bbox = np.array([
        bbox_center[0],
        bbox_center[1],
        bbox_center[2],
        size[0],  # width
        size[1],  # length
        size[2],  # height
        best_yaw
    ])

    return bbox