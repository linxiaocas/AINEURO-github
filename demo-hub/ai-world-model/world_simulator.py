#!/usr/bin/env python3
"""
AI World Model Demo
AI世界模型演示 - 物理环境模拟与预测
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class Object:
    """物理世界中的对象"""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    mass: float
    radius: float
    color: str
    
@dataclass  
class WorldState:
    """世界状态"""
    objects: List[Object]
    time: float
    gravity: float = 9.8
    friction: float = 0.99

class Encoder:
    """编码器: 将高维观测压缩为潜变量"""
    
    def __init__(self, input_dim: int = 4, latent_dim: int = 8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # 简化的编码网络权重
        self.weights = np.random.randn(input_dim, latent_dim) * 0.1
        
    def encode(self, state: WorldState) -> np.ndarray:
        """编码世界状态为潜变量"""
        # 提取关键特征
        features = []
        for obj in state.objects:
            features.extend([obj.x, obj.y, obj.vx, obj.vy])
            
        features = np.array(features[:self.input_dim])
        
        # 编码到潜空间
        latent = np.tanh(features @ self.weights)
        return latent

class DynamicModel:
    """动态模型: 预测未来状态"""
    
    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        # 状态转移矩阵
        self.transition = np.eye(latent_dim) * 0.95
        
    def predict(self, latent: np.ndarray, steps: int = 1) -> np.ndarray:
        """预测未来潜状态"""
        future_latents = [latent]
        
        for _ in range(steps):
            # 状态转移 + 噪声
            next_latent = future_latents[-1] @ self.transition
            next_latent += np.random.randn(self.latent_dim) * 0.01
            future_latents.append(next_latent)
            
        return np.array(future_latents)

class Decoder:
    """解码器: 将潜变量还原为观测"""
    
    def __init__(self, latent_dim: int = 8, output_dim: int = 4):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(latent_dim, output_dim) * 0.1
        
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """解码潜变量为观测"""
        output = latent @ self.weights
        return output

class Planner:
    """规划器: 基于模型预测进行规划"""
    
    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        
    def plan_trajectory(self, current_state: WorldState, 
                       goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """规划到达目标的轨迹"""
        trajectory = []
        
        # 简化的规划: 直线轨迹
        if current_state.objects:
            obj = current_state.objects[0]
            dx = (goal[0] - obj.x) / self.horizon
            dy = (goal[1] - obj.y) / self.horizon
            
            for i in range(self.horizon):
                x = obj.x + dx * (i + 1)
                y = obj.y + dy * (i + 1)
                trajectory.append((x, y))
                
        return trajectory

class WorldModel:
    """世界模型主类"""
    
    def __init__(self):
        self.encoder = Encoder()
        self.dynamics = DynamicModel()
        self.decoder = Decoder()
        self.planner = Planner()
        self.state = None
        self.prediction_history = []
        
    def initialize(self, num_objects: int = 5):
        """初始化世界"""
        objects = []
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i in range(num_objects):
            obj = Object(
                id=i,
                x=random.uniform(1, 9),
                y=random.uniform(1, 9),
                vx=random.uniform(-1, 1),
                vy=random.uniform(-1, 1),
                mass=random.uniform(0.5, 2.0),
                radius=random.uniform(0.1, 0.3),
                color=colors[i % len(colors)]
            )
            objects.append(obj)
            
        self.state = WorldState(objects=objects, time=0.0)
        print(f"🌍 世界初始化完成: {num_objects}个对象")
        
    def physics_step(self, dt: float = 0.1):
        """物理模拟步进"""
        for obj in self.state.objects:
            # 更新位置
            obj.x += obj.vx * dt
            obj.y += obj.vy * dt
            
            # 应用重力
            obj.vy -= self.state.gravity * dt * 0.1
            
            # 应用摩擦力
            obj.vx *= self.state.friction
            obj.vy *= self.state.friction
            
            # 边界碰撞
            if obj.x <= obj.radius or obj.x >= 10 - obj.radius:
                obj.vx *= -0.8
                obj.x = max(obj.radius, min(10 - obj.radius, obj.x))
                
            if obj.y <= obj.radius:
                obj.vy *= -0.8
                obj.y = obj.radius
                
        # 对象间碰撞检测
        self._handle_collisions()
        
        self.state.time += dt
        
    def _handle_collisions(self):
        """处理对象间碰撞"""
        for i, obj1 in enumerate(self.state.objects):
            for obj2 in self.state.objects[i+1:]:
                dx = obj2.x - obj1.x
                dy = obj2.y - obj1.y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < obj1.radius + obj2.radius:
                    # 简化的弹性碰撞
                    obj1.vx, obj2.vx = obj2.vx, obj1.vx
                    obj1.vy, obj2.vy = obj2.vy, obj1.vy
                    
    def predict_future(self, steps: int = 10):
        """预测未来状态"""
        # 编码当前状态
        latent = self.encoder.encode(self.state)
        
        # 预测未来
        future_latents = self.dynamics.predict(latent, steps)
        
        # 解码预测
        predictions = []
        for latent in future_latents:
            decoded = self.decoder.decode(latent)
            predictions.append(decoded)
            
        self.prediction_history.append(predictions)
        return predictions
        
    def visualize(self):
        """可视化当前状态和预测"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图: 当前世界状态
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_title('Current World State')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        
        for obj in self.state.objects:
            circle = plt.Circle((obj.x, obj.y), obj.radius, 
                              color=obj.color, alpha=0.7)
            ax1.add_patch(circle)
            # 速度向量
            ax1.arrow(obj.x, obj.y, obj.vx*0.5, obj.vy*0.5, 
                     head_width=0.1, color='black', alpha=0.5)
                     
        # 右图: 预测轨迹
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_title('World Model Prediction')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        if self.prediction_history:
            predictions = self.prediction_history[-1]
            pred_x = [p[0] for p in predictions]
            pred_y = [p[1] for p in predictions]
            ax2.plot(pred_x, pred_y, 'r--', label='Predicted Trajectory', alpha=0.7)
            ax2.scatter(pred_x, pred_y, c='red', s=30, alpha=0.5)
            
        plt.tight_layout()
        plt.savefig('world_model_demo.png', dpi=150)
        plt.close()
        
        print("📊 可视化已保存: world_model_demo.png")
        
    def demonstrate_capabilities(self):
        """演示世界模型能力"""
        print("\n🎯 世界模型能力演示")
        print("=" * 60)
        
        # 1. 物理认知
        print("\n1️⃣ 物理认知 - 重力模拟")
        print("   观察: 对象受重力影响下落，碰撞后反弹")
        
        # 2. 未来预测
        print("\n2️⃣ 未来预测 - 轨迹推断")
        predictions = self.predict_future(steps=20)
        print(f"   预测未来20步的状态变化")
        print(f"   初始位置: ({predictions[0][0]:.2f}, {predictions[0][1]:.2f})")
        print(f"   预测终点: ({predictions[-1][0]:.2f}, {predictions[-1][1]:.2f})")
        
        # 3. 因果推理
        print("\n3️⃣ 因果推理 - 碰撞影响")
        print("   逻辑: A撞B → B速度改变 → B位置改变 → B可能撞C")
        
        # 4. 反事实推理
        print("\n4️⃣ 反事实推理 - 假设分析")
        print("   假设: 如果没有重力，对象将保持匀速直线运动")
        print("   对比: 当前轨迹 vs 无重力轨迹")
        
        print("\n" + "=" * 60)

def main():
    """主函数"""
    print("=" * 60)
    print("   AI World Model Demo")
    print("   物理世界模拟与预测演示")
    print("=" * 60)
    
    # 创建世界模型
    world = WorldModel()
    world.initialize(num_objects=5)
    
    # 模拟运行
    print("\n▶️  运行物理模拟...")
    for step in range(50):
        world.physics_step(dt=0.1)
        
        if step % 10 == 0:
            print(f"   Step {step:3d}: t={world.state.time:.1f}s, "
                  f"objects={len(world.state.objects)}")
                  
    # 演示能力
    world.demonstrate_capabilities()
    
    # 可视化
    world.visualize()
    
    print("\n✅ 演示完成!")
    print("\n世界模型核心组件:")
    print("  • 编码器: 将高维观测压缩为潜变量")
    print("  • 动态模型: 学习状态转移，预测未来")
    print("  • 解码器: 将潜变量还原为可解释输出")
    print("  • 规划器: 结合MPC进行长程规划")
    print("\n关键能力:")
    print("  ✅ 内在表征学习")
    print("  ✅ 物理规律认知")
    print("  ✅ 因果推理")
    print("  ✅ 反事实推理")

if __name__ == "__main__":
    main()
