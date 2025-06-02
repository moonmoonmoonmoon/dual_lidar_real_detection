import json
from pathlib import Path

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import deque
import open3d as o3d
from src.utils.point_cloud_buffer import PointCloudBuffer

@dataclass
class SafetyConfig:
    """安全配置参数"""
    min_lateral_distance: float = 0.3
    min_longitudinal_distance: float = 1.5
    max_longitudinal_distance: float = 50.0
    block_lateral_threshold: float = 0.2
    reaction_time: float = 1.0
    deceleration: float = 7.84
    method: str = 'physics'
    alpha: float = 0.1
    ego_half_width: float = 1.0


@dataclass
class SafetyResult:
    """安全分析结果"""
    frame_id: int
    trigger_required: bool
    trigger_reason: Optional[str]
    object_info: Dict
    risk_metrics: Dict
    point_cloud: Optional[o3d.geometry.PointCloud] = None  # 添加点云字段


class SafetyAnalyzer:
    def __init__(self, config: Dict = None):
        """初始化安全分析器"""
        self.config = SafetyConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.trigger_buffer = deque(maxlen=700)
        self.last_trigger_frame = 0
        self.total_triggers = 0
        self.trigger_stats = {
            'lateral': 0,
            'longitudinal': 0,
            'both': 0,
            'lateral_oncoming': 0,
            'lateral_with_width': 0
        }
        self.point_cloud_buffer = PointCloudBuffer(pre_frames=2, post_frames=1)
        # 创建保存目录
        self.save_dir = Path("output/triggers")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_trigger_data(self, result: SafetyResult):
        """保存触发时的数据"""
        if not result.point_cloud:
            return

        # 创建以id命名的目录
        trigger_dir = self.save_dir / f"frame_{result.frame_id}"
        trigger_dir.mkdir(exist_ok=True)

        # 保存点云
        pcd_path = trigger_dir / "point_cloud.pcd"
        o3d.io.write_point_cloud(str(pcd_path), result.point_cloud)

        # 只保存关键信息
        box_info = {
            "frame_id": result.frame_id,
            "trigger_reason": result.trigger_reason,
            "object": {
                "center": result.object_info['position'].tolist(),
                "dimensions": result.object_info['dimensions'].tolist(),
                "yaw": float(result.object_info['yaw'])
            }
        }
        # print('box_info',box_info)
        box_path = trigger_dir / "box_info.json"
        with open(box_path, 'w') as f:
            json.dump(box_info, f, indent=2)

    def calculate_safety_distance(self, ego_speed: float) -> float:
        """计算安全距离"""
        if self.config.method == 'physics':
            if ego_speed < 0 :
                ego_speed = abs(ego_speed)
                reaction_distance = ego_speed * self.config.reaction_time
                braking_distance = (ego_speed ** 2) / (2 * self.config.deceleration)
                safety_distance = reaction_distance + braking_distance
                # print(reaction_distance,braking_distance,safety_distance)
            else:
                safety_distance = self.config.min_longitudinal_distance
        else:
            if ego_speed < 0:
                ego_speed = abs(ego_speed)
                safety_distance = np.exp(self.config.alpha * ego_speed)
            else:
                safety_distance = self.config.min_longitudinal_distance

        return np.clip(safety_distance,
                       self.config.min_longitudinal_distance,
                       self.config.max_longitudinal_distance)

    def get_ego_direction(self, ego_vel: np.ndarray) -> np.ndarray:
        """获取ego的单位方向向量
        Args:
            ego_vel: ego速度向量 [vx, vy]
        Returns:
            direction: 单位方向向量 [dx, dy]
        """
        ego_speed = np.linalg.norm(ego_vel)
        if ego_speed < 0.001:  # 速度太小时假设朝向x轴正方向
            return np.array([1.0, 0.0])
        return ego_vel / ego_speed

    def project_to_ego_frame(self,
                             vector: np.ndarray,
                             ego_direction: np.ndarray) -> np.ndarray:
        """将向量投影到ego坐标系
        Args:
            vector: 要投影的向量 [x, y]
            ego_direction: ego的单位方向向量 [dx, dy]
        Returns:
            projected: [longitudinal, lateral] 投影后的纵向和横向分量
        """
        # 计算纵向投影（与ego方向同向的分量）
        longitudinal = np.dot(vector, ego_direction)

        # 计算横向投影
        # 1. 先计算纵向向量
        longitudinal_vector = longitudinal * ego_direction
        # 2. 用原向量减去纵向得到横向向量
        lateral_vector = vector - longitudinal_vector
        # 3. 计算横向长度（带符号，左正右负）
        lateral = np.linalg.norm(lateral_vector) * np.sign(np.cross(ego_direction, vector))

        return np.array([longitudinal, lateral])

    def is_oncoming_vehicle(self, obj_vel: np.ndarray, ego_vel: np.ndarray) -> bool:
        """判断是否是对向车辆"""
        # 获取两车运动方向
        obj_direction = self.get_ego_direction(obj_vel)
        ego_direction = self.get_ego_direction(ego_vel)

        # 计算方向向量夹角
        cos_angle = np.dot(obj_direction, ego_direction)
        # 夹角大于90度认为是对向车辆
        return cos_angle < 0
    def analyze_object(self, obj: Dict, ego_info: Dict) -> SafetyResult:
        """分析单个物体的安全风险"""
        # 获取ego的方向
        ego_vel = ego_info['velocity'][:2]
        ego_direction = self.get_ego_direction(ego_vel)

        # 计算相对位置投影
        relative_pos = obj['center'][:2] - ego_info['position'][:2]
        projected_pos = self.project_to_ego_frame(relative_pos, ego_direction)

        # 纵向和横向距离(取绝对值)
        longitudinal_distance = abs(projected_pos[0])
        lateral_distance = abs(projected_pos[1])

        obj_vel = obj['velocity'][:2]
        obj_half_width = obj['dimensions'][1]/2
        # 计算相对速度投影
        relative_vel = obj_vel - ego_vel
        projected_vel = self.project_to_ego_frame(relative_vel, ego_direction)
        # print('relative_speed',np.linalg.norm(ego_vel[:2] - obj_vel[:2]))
        # print('d_thre',self.calculate_safety_distance(-1.41),self.calculate_safety_distance(1.41))
        # 速度标量
        ego_speed = np.linalg.norm(ego_vel)
        obj_speed = np.linalg.norm(obj_vel)

        # 判断是否是对向车辆
        is_oncoming = self.is_oncoming_vehicle(obj_vel, ego_vel)
        print('effective_lateral_distance',lateral_distance - (self.config.ego_half_width + obj_half_width))
        # 计算安全距离阈值
        if obj['label'] == 'vehicle':
            if is_oncoming:
                # safety_distance = None
                # trigger = 0 < lateral_distance - (self.config.ego_half_width + obj_half_width) < self.config.min_lateral_distance
                # trigger_reason = "lateral_oncoming" if trigger else None
                safety_distance = self.config.min_longitudinal_distance
                lateral_trigger = 0 < lateral_distance - (
                            self.config.ego_half_width + obj_half_width) < self.config.min_lateral_distance
                longitudinal_trigger = longitudinal_distance < safety_distance
                trigger = lateral_trigger and longitudinal_trigger
                trigger_reason = "lateral_oncoming" if trigger else None
            else:
                # safety_distance = self.calculate_safety_distance(ego_speed)
                safety_distance = self.calculate_safety_distance(projected_vel[0])
                # print('safety_distance', safety_distance)
                lateral_trigger = 0 < lateral_distance - (self.config.ego_half_width + obj_half_width) < self.config.min_lateral_distance
                longitudinal_trigger = longitudinal_distance < safety_distance
                trigger = lateral_trigger and longitudinal_trigger

                if trigger:
                    trigger_reason = "both" if lateral_trigger and longitudinal_trigger \
                        else "lateral" if lateral_trigger else "longitudinal"
                else:
                    trigger_reason = None

        else:  # blocks
            trigger = 0 < lateral_distance - (self.config.ego_half_width + obj_half_width) < self.config.block_lateral_threshold
            trigger_reason = "lateral_with_width" if trigger else None
            safety_distance = None

        return SafetyResult(
            frame_id=ego_info['frame_id'],
            trigger_required=trigger,
            trigger_reason=trigger_reason,
            object_info={
                'type': obj['label'],
                'position': obj['center'],
                'velocity': obj['velocity'],
                'dimensions': obj['dimensions'],
                'yaw': obj['yaw'],
                'ego_frame_position': projected_pos,  # [longitudinal, lateral]
                'ego_frame_velocity': projected_vel,  # [v_longitudinal, v_lateral]
                'is_oncoming': is_oncoming
            },
            risk_metrics={
                'lateral_distance': lateral_distance,
                'longitudinal_distance': longitudinal_distance,
                'safety_distance': safety_distance,
                'relative_speed': obj_speed - ego_speed,
                'longitudinal_velocity': projected_vel[0],
                'lateral_velocity': projected_vel[1],
                # 添加原始投影值（带符号）
                'raw_longitudinal_distance': projected_pos[0],  # 正表示在前方
                'raw_lateral_distance': projected_pos[1]  # 正表示在左侧
            }
        )
    def process_frame(self, objects: List[Dict], ego_info: Dict, point_cloud: o3d.geometry.PointCloud = None) -> List[SafetyResult]:
        """处理一帧数据"""
        
        if point_cloud is not None:
            self.point_cloud_buffer.add_frame(
                frame_id=ego_info['frame_id'],
                points=np.asarray(point_cloud.points),
                transform=ego_info.get('transform_matrix')
            )
        results = []
        # trigger_frames = []
        # if not objects:
        #     return []  # 如果没有检测到物体，返回空列表

        for obj in objects:
            result = self.analyze_object(obj, ego_info)
            # 添加点云和时间戳信息
            result.point_cloud = point_cloud
            results.append(result)

            # 处理触发事件
            if result.trigger_required:
                frames_since_last = ego_info['frame_id'] - self.last_trigger_frame
                self.total_triggers += 1
                # trigger_frames.append(ego_info['frame_id'])
                if result.trigger_reason in self.trigger_stats:
                    self.trigger_stats[result.trigger_reason] += 1
                if frames_since_last >= 30:
                    trigger_info = {
                        'frame_id': ego_info['frame_id'],
                        'trigger_reason': result.trigger_reason,
                        'object_info': {
                            'type': obj['label'],
                            'position': obj['center'].tolist(),
                            'dimensions': obj['dimensions'].tolist(),
                            'yaw': float(obj['yaw'])
                        },
                        'risk_metrics': result.risk_metrics
                    }
                    self.point_cloud_buffer.trigger(ego_info['frame_id'], trigger_info)
                    self.trigger_buffer.append(result)
                    self.last_trigger_frame = ego_info['frame_id']
                    # 保存触发数据
                    # self.save_trigger_data(result)
        return results

    def get_trigger_history(self) -> List[SafetyResult]:
        """获取触发历史"""
        return list(self.trigger_buffer)


# 测试代码
if __name__ == "__main__":
    # 示例配置
    config = {
        'method': 'physics',
        'alpha': 0.1,
        'min_lateral_distance': 2.0,
        'min_longitudinal_distance': 1.0
    }

    analyzer = SafetyAnalyzer(config)

    # 示例数据：ego沿45度角行驶，object在其左前方
    test_object = {
        'label': 'vehicle',
        'center': np.array([10.0, 5.0, 0.0]),
        'velocity': np.array([4.0, 0.0, 0.0]),
        'dimensions': np.array([4.0, 2.0, 1.5])
    }

    test_ego = {
        'frame_id': 1,
        'position': np.array([0.0, 0.0, 0.0]),
        'velocity': np.array([3.0, 3.0, 0.0])  # 45度角行驶
    }

    # 测试分析
    result = analyzer.analyze_object(test_object, test_ego)
    print("\n测试场景：ego沿45度角行驶，object在其左前方")
    print(f"\n物体在ego坐标系下的位置: [纵向={result.object_info['ego_frame_position'][0]:.2f}, "
          f"横向={result.object_info['ego_frame_position'][1]:.2f}]")
    print(f"物体在ego坐标系下的速度: [纵向={result.object_info['ego_frame_velocity'][0]:.2f}, "
          f"横向={result.object_info['ego_frame_velocity'][1]:.2f}]")
    print(f"\n触发结果: {'需要触发' if result.trigger_required else '无需触发'}")
    print(f"是否对向车辆: {result.object_info['is_oncoming']}")
    if result.trigger_reason:
        print(f"触发原因: {result.trigger_reason}")
