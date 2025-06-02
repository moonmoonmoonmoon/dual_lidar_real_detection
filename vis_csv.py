import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def parse_vector(vector_str):
    """Parse vector string into x, y, z values"""
    # 移除方括号
    vector_str = vector_str.strip('[]')

    # 尝试按逗号分隔
    if ',' in vector_str:
        x, y, z = map(float, vector_str.split(','))
    # 如果没有逗号，按空格分隔
    else:
        x, y, z = map(float, vector_str.split())
    return x,y,z

def get_color_by_trigger(trigger_reason):
    """根据trigger_reason返回对应的颜色"""
    color_map = {
        'lateral': 'red',
        'lateral_with_width': 'orange',
        'both': 'purple',
        'lateral_oncoming': 'green',
        'longitudinal': 'brown'
    }
    return color_map.get(trigger_reason, 'gray')  # 默认灰色


def get_rect_coordinates(center_x, center_y, length, width, yaw):
    """
    计算考虑旋转后的矩形左下角坐标
    :param center_x: 中心点x坐标
    :param center_y: 中心点y坐标
    :param length: 矩形长度
    :param width: 矩形宽度
    :param yaw: 偏航角（弧度）
    :return: 矩形左下角坐标(x, y)
    """
    # 计算从中心到左下角的向量
    dx = -length / 2
    dy = -width / 2

    # 应用旋转
    rotated_dx = dx * np.cos(yaw) - dy * np.sin(yaw)
    rotated_dy = dx * np.sin(yaw) + dy * np.cos(yaw)

    # 计算最终坐标
    rect_x = center_x + rotated_dx
    rect_y = center_y + rotated_dy

    return rect_x, rect_y
def plot_frame(frame_data, ax=None):
    """Plot ego and object boxes for a single frame"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # 设置ego车辆参数
    ego_length, ego_width = 3.6, 2.0

    # 解析位置数据
    ego_x, ego_y,_ = parse_vector(frame_data['ego_pos'])
    obj_x, obj_y,_ = parse_vector(frame_data['object_pos'])

    # 解析速度来估计朝向（如果需要）
    ego_vx, ego_vy,_ = parse_vector(frame_data['ego_vel'])
    ego_yaw = np.arctan2(ego_vy, ego_vx)
    print(frame_data['object_dim'].strip('[]').split(' '))

    # 解析物体尺寸
    obj_dims = [float(val) for val in frame_data['object_dim'].strip('[]').split(' ') if val]
    obj_l, obj_w, obj_h = obj_dims[:3]  # 只取前三个值

    # obj_l, obj_w, obj_h = map(float, frame_data['object_dim'].strip('[]').split(' '))
    obj_yaw = float(frame_data['object_yaw'])

    # 获取trigger_reason对应的颜色
    obj_color = get_color_by_trigger(frame_data['trigger_reason'])

    # 计算正确的矩形起点坐标
    ego_rect_x, ego_rect_y = get_rect_coordinates(ego_x, ego_y, ego_length, ego_width, ego_yaw)
    obj_rect_x, obj_rect_y = get_rect_coordinates(obj_x, obj_y, obj_l, obj_w, obj_yaw)

    # 绘制ego车辆边界框
    ego_rect = Rectangle(
        (ego_rect_x, ego_rect_y),
        ego_length, ego_width,
        angle=np.degrees(ego_yaw),
        facecolor='blue',
        alpha=0.3,
        label='Ego Vehicle'
    )
    ax.add_patch(ego_rect)

    # 绘制物体边界框
    obj_rect = Rectangle(
        (obj_rect_x, obj_rect_y),
        obj_l, obj_w,
        angle=np.degrees(obj_yaw),
        facecolor=obj_color,
        alpha=0.3,
        label='Object'
    )
    ax.add_patch(obj_rect)
    # 添加中心点标记（用于验证）
    ax.plot(ego_x, ego_y, 'b+', markersize=10)  # ego中心
    ax.plot(obj_x, obj_y, 'r+', markersize=10)  # object中心

    # 绘制速度矢量
    ax.arrow(ego_x, ego_y, ego_vx, ego_vy,
             head_width=0.5, head_length=0.8, fc='blue', ec='blue', alpha=0.5)

    # 设置图形属性
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

    # 设置适当的显示范围
    margin = max(ego_length, ego_width, obj_l, obj_w) * 2
    ax.set_xlim(min(ego_x, obj_x) - margin, max(ego_x, obj_x) + margin)
    ax.set_ylim(min(ego_y, obj_y) - margin, max(ego_y, obj_y) + margin)

    # 添加帧信息
    ax.set_title(f"Frame: {frame_data['frame_id']}")

    return ax


def visualize_data(csv_path):
    """主函数：读取CSV并可视化数据"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 为每个触发帧创建可视化
    triggered_frames = df[df['trigger_required'] == True]

    for _, frame_data in triggered_frames.iterrows():
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_frame(frame_data, ax)
        plt.show()
# 使用示例
if __name__ == "__main__":
    # 替换为实际的CSV文件路径
    # csv_path = "./output/lidar_results_20241130_002836.csv"
    csv_path = "./output/lidar_results_20241216_233022.csv"
    visualize_data(csv_path)