#!/usr/bin/env python3
"""
Autonomous Driving Demo
自动驾驶模拟演示 - Apollo架构简化版
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import math

@dataclass
class Obstacle:
    """障碍物"""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float
    obstacle_type: str  # 'car', 'pedestrian', 'static'
    
@dataclass
class EgoVehicle:
    """自车"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # 朝向角度
    v: float = 0.0  # 速度
    steering: float = 0.0  # 转向角
    
    # 车辆参数
    length: float = 4.5
    width: float = 2.0
    max_speed: float = 30.0  # m/s
    max_accel: float = 3.0
    max_steering: float = 0.6
    
@dataclass
class PerceptionOutput:
    """感知层输出"""
    obstacles: List[Obstacle]
    lane_markers: List[Tuple[float, float]]
    traffic_signs: List[Dict]
    timestamp: float
    
@dataclass
class DecisionOutput:
    """决策层输出"""
    intent: str  # 'lane_follow', 'change_left', 'change_right', 'stop', 'overtake'
    target_speed: float
    target_lane: int
    emergency: bool
    
@dataclass
class PlanningOutput:
    """规划层输出"""
    trajectory: List[Tuple[float, float]]  # 路径点
    speed_profile: List[float]  # 速度曲线
    timestamps: List[float]
    
@dataclass
class ControlOutput:
    """控制层输出"""
    throttle: float  # 0-1
    brake: float  # 0-1
    steering: float  # -1 to 1
    
class PerceptionModule:
    """感知模块"""
    
    def __init__(self, sensor_range: float = 100.0):
        self.sensor_range = sensor_range
        self.obstacle_id_counter = 0
        
    def detect(self, ego: EgoVehicle, true_obstacles: List[Obstacle]) -> PerceptionOutput:
        """模拟感知检测"""
        detected_obstacles = []
        
        for obs in true_obstacles:
            # 计算距离
            distance = math.sqrt((obs.x - ego.x)**2 + (obs.y - ego.y)**2)
            
            # 只在传感器范围内检测
            if distance <= self.sensor_range:
                # 添加检测噪声
                noise_x = random.gauss(0, 0.3)
                noise_y = random.gauss(0, 0.3)
                
                detected_obs = Obstacle(
                    id=obs.id,
                    x=obs.x + noise_x,
                    y=obs.y + noise_y,
                    vx=obs.vx,
                    vy=obs.vy,
                    width=obs.width,
                    height=obs.height,
                    obstacle_type=obs.obstacle_type
                )
                detected_obstacles.append(detected_obs)
                
        # 模拟车道线检测
        lane_markers = self._detect_lanes(ego)
        
        # 模拟交通标志
        traffic_signs = self._detect_signs(ego)
        
        return PerceptionOutput(
            obstacles=detected_obstacles,
            lane_markers=lane_markers,
            traffic_signs=traffic_signs,
            timestamp=0.0
        )
        
    def _detect_lanes(self, ego: EgoVehicle) -> List[Tuple[float, float]]:
        """检测车道线"""
        # 简化的车道线: 左右各3.5米
        left_lane = [(ego.x - 3.5, ego.y + i*10) for i in range(-5, 6)]
        right_lane = [(ego.x + 3.5, ego.y + i*10) for i in range(-5, 6)]
        return left_lane + right_lane
        
    def _detect_signs(self, ego: EgoVehicle) -> List[Dict]:
        """检测交通标志"""
        signs = []
        # 模拟前方的限速标志
        if random.random() < 0.3:
            signs.append({
                'type': 'speed_limit',
                'value': 60,
                'distance': random.uniform(30, 80)
            })
        return signs

class DecisionModule:
    """决策模块"""
    
    def decide(self, perception: PerceptionOutput, ego: EgoVehicle) -> DecisionOutput:
        """做出驾驶决策"""
        # 检查紧急情况
        emergency = self._check_emergency(perception, ego)
        if emergency:
            return DecisionOutput(
                intent='stop',
                target_speed=0.0,
                target_lane=0,
                emergency=True
            )
            
        # 检查前方障碍物
        front_obstacle = self._find_front_obstacle(perception, ego)
        
        if front_obstacle:
            distance = math.sqrt(
                (front_obstacle.x - ego.x)**2 + 
                (front_obstacle.y - ego.y)**2
            )
            
            if distance < 20:  # 太近，需要减速或变道
                if self._can_change_left(perception, ego):
                    return DecisionOutput(
                        intent='change_left',
                        target_speed=ego.v,
                        target_lane=-1,
                        emergency=False
                    )
                else:
                    return DecisionOutput(
                        intent='slow_down',
                        target_speed=front_obstacle.v * 0.8,
                        target_lane=0,
                        emergency=False
                    )
                    
        # 默认: 车道保持
        target_speed = 20.0  # 默认速度
        
        # 检查限速标志
        for sign in perception.traffic_signs:
            if sign['type'] == 'speed_limit':
                target_speed = min(target_speed, sign['value'] / 3.6)  # km/h to m/s
                
        return DecisionOutput(
            intent='lane_follow',
            target_speed=target_speed,
            target_lane=0,
            emergency=False
        )
        
    def _check_emergency(self, perception: PerceptionOutput, ego: EgoVehicle) -> bool:
        """检查紧急情况"""
        for obs in perception.obstacles:
            distance = math.sqrt((obs.x - ego.x)**2 + (obs.y - ego.y)**2)
            if distance < 10:  # 10米内视为危险
                return True
        return False
        
    def _find_front_obstacle(self, perception: PerceptionOutput, ego: EgoVehicle) -> Optional[Obstacle]:
        """找到前方障碍物"""
        front_obs = None
        min_distance = float('inf')
        
        for obs in perception.obstacles:
            # 只考虑前方的障碍物
            dy = obs.y - ego.y
            dx = abs(obs.x - ego.x)
            
            if dy > 0 and dx < 2.0:  # 前方且在同一车道
                distance = math.sqrt(dx**2 + dy**2)
                if distance < min_distance:
                    min_distance = distance
                    front_obs = obs
                    
        return front_obs
        
    def _can_change_left(self, perception: PerceptionOutput, ego: EgoVehicle) -> bool:
        """检查是否可以向左变道"""
        for obs in perception.obstacles:
            dx = obs.x - ego.x
            dy = obs.y - ego.y
            if -5 < dx < -2 and abs(dy) < 20:  # 左侧有车
                return False
        return True

class PlanningModule:
    """规划模块"""
    
    def plan(self, decision: DecisionOutput, ego: EgoVehicle, 
             perception: PerceptionOutput) -> PlanningOutput:
        """规划轨迹"""
        trajectory = []
        speed_profile = []
        timestamps = []
        
        dt = 0.1
        horizon = 50  # 5秒规划 horizon
        
        # 当前状态
        x, y, theta, v = ego.x, ego.y, ego.theta, ego.v
        
        for i in range(horizon):
            # 根据决策生成轨迹
            if decision.intent == 'lane_follow':
                # 沿车道直线行驶
                y += v * math.cos(theta) * dt
                
            elif decision.intent == 'change_left':
                # 向左变道
                x -= 0.1  # 每步向左移动
                y += v * dt
                
            elif decision.intent == 'slow_down':
                # 减速
                v = max(0, v - 1.0 * dt)
                y += v * dt
                
            # 速度平滑
            v = min(v + 0.5 * dt, decision.target_speed)
            
            trajectory.append((x, y))
            speed_profile.append(v)
            timestamps.append(i * dt)
            
        return PlanningOutput(
            trajectory=trajectory,
            speed_profile=speed_profile,
            timestamps=timestamps
        )

class ControlModule:
    """控制模块"""
    
    def control(self, planning: PlanningOutput, ego: EgoVehicle) -> ControlOutput:
        """生成控制指令"""
        if not planning.trajectory:
            return ControlOutput(0, 0, 0)
            
        # 取第一个轨迹点作为目标
        target = planning.trajectory[0]
        target_speed = planning.speed_profile[0]
        
        # 速度控制
        speed_error = target_speed - ego.v
        if speed_error > 0:
            throttle = min(1.0, speed_error / ego.max_accel)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -speed_error / ego.max_accel)
            
        # 横向控制 (简化的 Stanley 控制器)
        dx = target[0] - ego.x
        dy = target[1] - ego.y
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - ego.theta
        
        steering = max(-1.0, min(1.0, heading_error / ego.max_steering))
        
        return ControlOutput(throttle, brake, steering)

class ApolloSimulator:
    """Apollo架构模拟器"""
    
    def __init__(self):
        self.perception = PerceptionModule()
        self.decision = DecisionModule()
        self.planning = PlanningModule()
        self.control = ControlModule()
        
        self.ego = EgoVehicle(x=0, y=0, theta=0, v=15.0)
        self.obstacles = []
        self.history = []
        
    def initialize(self, num_obstacles: int = 5):
        """初始化场景"""
        # 创建障碍物
        for i in range(num_obstacles):
            obs = Obstacle(
                id=i,
                x=random.uniform(-2, 2),
                y=random.uniform(50, 200),
                vx=0,
                vy=random.uniform(-5, -15),  # 向自车方向移动
                width=random.uniform(1.8, 2.2),
                height=random.uniform(4.0, 5.0),
                obstacle_type='car'
            )
            self.obstacles.append(obs)
            
        print(f"🚗 自动驾驶模拟器初始化完成")
        print(f"   自车位置: ({self.ego.x:.1f}, {self.ego.y:.1f})")
        print(f"   障碍物数量: {num_obstacles}")
        
    def step(self, dt: float = 0.1):
        """模拟一步"""
        # 1. 感知层
        perception_output = self.perception.detect(self.ego, self.obstacles)
        
        # 2. 决策层
        decision_output = self.decision.decide(perception_output, self.ego)
        
        # 3. 规划层
        planning_output = self.planning.plan(
            decision_output, self.ego, perception_output
        )
        
        # 4. 控制层
        control_output = self.control.control(planning_output, self.ego)
        
        # 5. 执行控制
        self._apply_control(control_output, dt)
        
        # 6. 更新障碍物
        self._update_obstacles(dt)
        
        # 记录历史
        self.history.append({
            'ego': (self.ego.x, self.ego.y),
            'decision': decision_output.intent,
            'speed': self.ego.v,
            'obstacles': len(perception_output.obstacles)
        })
        
        return decision_output.intent
        
    def _apply_control(self, control: ControlOutput, dt: float):
        """应用控制指令"""
        # 更新速度
        accel = control.throttle * self.ego.max_accel - control.brake * 5.0
        self.ego.v = max(0, min(self.ego.max_speed, self.ego.v + accel * dt))
        
        # 更新朝向
        self.ego.theta += control.steering * self.ego.max_steering * dt
        
        # 更新位置
        self.ego.x += self.ego.v * math.sin(self.ego.theta) * dt
        self.ego.y += self.ego.v * math.cos(self.ego.theta) * dt
        
    def _update_obstacles(self, dt: float):
        """更新障碍物位置"""
        for obs in self.obstacles:
            obs.x += obs.vx * dt
            obs.y += obs.vy * dt
            
    def visualize(self):
        """可视化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制道路
        ax.axhline(y=-3.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=3.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='yellow', linestyle='-', alpha=0.5, linewidth=2)
        
        # 绘制历史轨迹
        if len(self.history) > 0:
            xs = [h['ego'][0] for h in self.history]
            ys = [h['ego'][1] for h in self.history]
            ax.plot(xs, ys, 'b-', label='Ego Trajectory', alpha=0.7)
            
        # 绘制自车
        ego_rect = Rectangle(
            (self.ego.x - self.ego.width/2, self.ego.y - self.ego.length/2),
            self.ego.width, self.ego.length,
            angle=math.degrees(self.ego.theta),
            facecolor='blue', edgecolor='darkblue', linewidth=2
        )
        ax.add_patch(ego_rect)
        
        # 绘制障碍物
        for obs in self.obstacles:
            obs_rect = Rectangle(
                (obs.x - obs.width/2, obs.y - obs.height/2),
                obs.width, obs.height,
                facecolor='red', edgecolor='darkred', alpha=0.7
            )
            ax.add_patch(obs_rect)
            
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, max(100, self.ego.y + 50))
        ax.set_aspect('equal')
        ax.set_xlabel('Lateral (m)')
        ax.set_ylabel('Longitudinal (m)')
        ax.set_title('Autonomous Driving Simulation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('autonomous_driving_demo.png', dpi=150)
        plt.close()
        
        print("📊 可视化已保存: autonomous_driving_demo.png")

def main():
    """主函数"""
    print("=" * 60)
    print("   Autonomous Driving Demo")
    print("   Apollo架构自动驾驶模拟")
    print("=" * 60)
    
    # 创建模拟器
    sim = ApolloSimulator()
    sim.initialize(num_obstacles=5)
    
    # 运行模拟
    print("\n▶️  开始自动驾驶模拟...")
    for step in range(100):
        intent = sim.step(dt=0.1)
        
        if step % 20 == 0:
            print(f"   Step {step:3d}: 位置=({sim.ego.x:.1f}, {sim.ego.y:.1f}), "
                  f"速度={sim.ego.v:.1f}m/s, 决策={intent}")
                  
    # 可视化
    sim.visualize()
    
    print("\n✅ 演示完成!")
    print("\nApollo架构分层:")
    print("  🎥 感知层(Perception): 多传感器融合，障碍物检测")
    print("  🧠 决策层(Decision): 驾驶意图生成")
    print("  📍 规划层(Planning): 路径和速度规划")
    print("  🎮 控制层(Control): 车辆执行指令")
    print("\n设计模式:")
    print("  • 工厂模式: 动态创建传感器")
    print("  • 策略模式: 独立传感器逻辑")
    print("  • 观察者模式: 事件驱动更新")

if __name__ == "__main__":
    main()
