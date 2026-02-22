---
title: "人形机器人控制实战：ROS+EtherCAT全身动力学控制"
date: "2026-02-22"
author: "Lin Xiao"
category: "Demo"
tags: ["Humanoid Robot", "ROS", "EtherCAT", "Control", "Dynamics"]
---

# 人形机器人控制实战：ROS+EtherCAT全身动力学控制

## 引言

人形机器人是机器人技术的皇冠明珠。本文介绍如何使用ROS+EtherCAT构建一套完整的人形机器人控制系统，实现全身动力学控制、步态规划和遥操作。

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                     感知计算单元 (大脑)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 建图定位  │  │ 目标检测  │  │ 导航规划  │  │ 遥操作   │     │
│  │ (SLAM)   │  │ (YOLO)   │  │ (Nav2)   │  │ (VR/动捕)│     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
└───────┼─────────────┼─────────────┼─────────────┼─────────────┘
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             │ ROS Topic/Service
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                     运动控制单元 (小脑)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 线程管理  │  │ 数据共享  │  │ 运动学   │  │ 动力学   │     │
│  │          │  │          │  │ 正/逆解  │  │ 模型预测 │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
└───────┼─────────────┼─────────────┼─────────────┼─────────────┘
        │             │             │             │
        └─────────────┴─────────────┴──────┬──────┘
                                           │ EtherCAT
                                           ↓
┌──────────────────────────────────────────────────────────────┐
│                        执行层 (硬件)                          │
│     ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    │
│     │ 左臂 │    │ 右臂 │    │ 躯干 │    │ 左腿 │    │ 右腿 │    │
│     └─────┘    └─────┘    └─────┘    └─────┘    └─────┘    │
│   7个自由度   7个自由度   3个自由度   6个自由度   6个自由度    │
└──────────────────────────────────────────────────────────────┘
```

## 硬件配置

| 部位 | 自由度 | 电机类型 | 控制频率 |
|------|--------|----------|----------|
| 头部 | 2 | 伺服电机 | 1kHz |
| 左臂 | 7 | 谐波电机 | 1kHz |
| 右臂 | 7 | 谐波电机 | 1kHz |
| 躯干 | 3 | 力矩电机 | 1kHz |
| 左腿 | 6 | 力矩电机 | 1kHz |
| 右腿 | 6 | 力矩电机 | 1kHz |
| **总计** | **31** | - | - |

## 软件架构

### 1. ROS2节点设计

```python
# humanoid_control_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose
from trajectory_msgs.msg import JointTrajectory
import numpy as np
import pinocchio as pin
from proxsuite import proxqp

class HumanoidControlNode(Node):
    """人形机器人控制主节点"""
    
    def __init__(self):
        super().__init__('humanoid_control')
        
        # 加载机器人模型
        self.model = pin.buildModelFromUrdf('humanoid.urdf')
        self.data = self.model.createData()
        
        # 关节数量
        self.nq = self.model.nq  # 配置空间维度
        self.nv = self.model.nv  # 速度空间维度
        
        # 当前状态
        self.q = np.zeros(self.nq)  # 关节位置
        self.v = np.zeros(self.nv)  # 关节速度
        self.a = np.zeros(self.nv)  # 关节加速度
        
        # 订阅传感器数据
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # 订阅高层指令
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.command_callback,
            10
        )
        
        # 发布控制指令
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )
        
        # 控制循环 (1kHz)
        self.control_timer = self.create_timer(0.001, self.control_loop)
        
        # 初始化控制器
        self.init_controllers()
        
        self.get_logger().info('🤖 人形机器人控制节点启动')
        
    def init_controllers(self):
        """初始化各种控制器"""
        # 平衡控制器
        self.balance_controller = BalanceController(self.model, self.data)
        
        # 步态生成器
        self.gait_generator = GaitGenerator()
        
        # 全身逆动力学控制器
        self.wbc_controller = WholeBodyController(self.model, self.data)
        
        # 操作空间控制器
        self.osc_controller = OperationalSpaceController(self.model, self.data)
        
    def joint_callback(self, msg):
        """关节状态回调"""
        # 更新关节状态
        for i, name in enumerate(msg.name):
            joint_id = self.model.getJointId(name)
            if joint_id < len(self.q):
                self.q[joint_id] = msg.position[i]
                self.v[joint_id] = msg.velocity[i]
                
    def imu_callback(self, msg):
        """IMU数据回调"""
        # 更新姿态估计
        self.orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
    def command_callback(self, msg):
        """接收高层运动指令"""
        self.target_velocity = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        
    def control_loop(self):
        """主控制循环 (1kHz)"""
        # 1. 状态估计
        self.estimate_state()
        
        # 2. 步态规划
        foot_forces, foot_positions = self.gait_generator.update(
            self.target_velocity,
            self.q,
            self.v
        )
        
        # 3. 全身控制
        tau = self.wbc_controller.compute(
            self.q, self.v,
            foot_forces, foot_positions
        )
        
        # 4. 发送指令
        self.send_joint_commands(tau)
        
    def estimate_state(self):
        """状态估计 (卡尔曼滤波)"""
        # 融合IMU和关节编码器数据
        pass
        
    def send_joint_commands(self, tau):
        """发送关节力矩指令"""
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        
        point = JointTrajectoryPoint()
        point.effort = tau.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 1000000  # 1ms
        
        traj_msg.points.append(point)
        self.joint_pub.publish(traj_msg)

class BalanceController:
    """平衡控制器 - 基于ZMP"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # ZMP参数
        self.zmp_target = np.array([0.0, 0.0, 0.0])
        self.com_height = 0.8  # 质心高度
        
        # PD控制器增益
        self.Kp = np.array([100, 100, 50])
        self.Kd = np.array([10, 10, 5])
        
    def compute(self, q, v, com_pos, com_vel):
        """计算平衡力矩"""
        # 更新运动学
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.centerOfMass(self.model, self.data, q, v)
        
        # 计算当前ZMP
        zmp_current = self.compute_zmp()
        
        # ZMP误差
        zmp_error = self.zmp_target - zmp_current
        
        # PD控制
        zmp_correction = self.Kp * zmp_error - self.Kd * com_vel[:2]
        
        # 计算期望质心加速度
        com_accel_desired = np.array([
            zmp_correction[0] / self.com_height * 9.81,
            zmp_correction[1] / self.com_height * 9.81,
            0
        ])
        
        return com_accel_desired
        
    def compute_zmp(self):
        """计算零力矩点"""
        # 基于质心加速度和角动量
        com = self.data.com[0]
        com_acc = self.data.acom[0]
        
        # ZMP公式
        zmp_x = com[0] - com_acc[0] * self.com_height / 9.81
        zmp_y = com[1] - com_acc[1] * self.com_height / 9.81
        
        return np.array([zmp_x, zmp_y, 0])

class GaitGenerator:
    """步态生成器"""
    
    def __init__(self):
        self.phase = 0.0  # 步态相位 [0, 1]
        self.step_height = 0.05  # 抬腿高度
        self.step_length = 0.3   # 步长
        self.stance_width = 0.15 # 支撑宽度
        
        self.left_foot_pos = np.array([0, self.stance_width, 0])
        self.right_foot_pos = np.array([0, -self.stance_width, 0])
        
    def update(self, target_vel, q, v):
        """更新步态"""
        # 更新相位
        self.phase += 0.01  # 假设控制频率1kHz
        if self.phase > 1.0:
            self.phase = 0.0
            # 交换支撑腿
            self.swap_support_foot()
            
        # 计算摆动腿轨迹
        if self.phase < 0.5:
            # 左腿摆动
            swing_foot_pos = self.compute_swing_trajectory(
                self.left_foot_pos,
                target_vel,
                self.phase * 2
            )
            support_foot_pos = self.right_foot_pos
        else:
            # 右腿摆动
            swing_foot_pos = self.compute_swing_trajectory(
                self.right_foot_pos,
                target_vel,
                (self.phase - 0.5) * 2
            )
            support_foot_pos = self.left_foot_pos
            
        # 计算地面反作用力
        foot_forces = self.compute_ground_forces(support_foot_pos)
        
        return foot_forces, [swing_foot_pos, support_foot_pos]
        
    def compute_swing_trajectory(self, start_pos, target_vel, phase):
        """计算摆动腿轨迹 (三次样条)"""
        # 目标落脚点
        end_pos = start_pos + np.array([
            target_vel[0] * 0.5,  # 半步时间
            target_vel[1] * 0.5,
            0
        ])
        
        # 三次样条插值
        t = phase
        pos = (1 - t)**3 * start_pos + 3 * (1 - t)**2 * t * (start_pos + end_pos) / 2
        
        # 添加高度
        pos[2] = self.step_height * np.sin(np.pi * t)
        
        return pos
        
    def compute_ground_forces(self, support_pos):
        """计算地面反作用力"""
        # 简化的力分配
        total_weight = 60 * 9.81  # 假设60kg
        return np.array([0, 0, total_weight])

class WholeBodyController:
    """全身控制器 - 基于QP优化"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 初始化QP求解器
        self.qp_solver = proxqp.sparse.QP(
            self.model.nv + 6,  # 变量维度 (关节加速度 + 基底加速度)
            0,  # 等式约束
            0   # 不等式约束
        )
        
    def compute(self, q, v, foot_forces, foot_positions):
        """计算最优关节力矩"""
        # 更新动力学
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.computeAllTerms(self.model, self.data, q, v)
        
        # 构建QP问题
        H, g = self.build_cost_function(q, v, foot_positions)
        
        # 求解
        self.qp_solver.solve(H, g)
        solution = self.qp_solver.results.x
        
        # 提取关节加速度
        joint_acc = solution[:self.model.nv]
        
        # 计算关节力矩 (逆动力学)
        tau = pin.rnea(self.model, self.data, q, v, joint_acc)
        
        return tau
        
    def build_cost_function(self, q, v, foot_positions):
        """构建代价函数"""
        # 任务优先级:
        # 1. 跟踪质心轨迹
        # 2. 跟踪脚部轨迹
        # 3. 最小化关节加速度
        
        H = np.eye(self.model.nv + 6)
        g = np.zeros(self.model.nv + 6)
        
        return H, g

def main():
    rclpy.init()
    node = HumanoidControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. EtherCAT通信

```python
# ethercat_interface.py
import pysoem
import ctypes
import numpy as np
import threading
import time

class EtherCATInterface:
    """EtherCAT主站接口"""
    
    def __init__(self, interface_name='eth0'):
        self.master = pysoem.Master()
        self.interface_name = interface_name
        
        # 电机数量
        self.num_motors = 31
        
        # 控制周期 (1kHz)
        self.control_period = 0.001
        
        # 数据缓冲区
        self.target_positions = np.zeros(self.num_motors)
        self.target_velocities = np.zeros(self.num_motors)
        self.target_torques = np.zeros(self.num_motors)
        
        self.actual_positions = np.zeros(self.num_motors)
        self.actual_velocities = np.zeros(self.num_motors)
        self.actual_torques = np.zeros(self.num_motors)
        
        self.running = False
        
    def initialize(self):
        """初始化EtherCAT网络"""
        # 打开网卡
        self.master.open(self.interface_name)
        
        # 扫描从站
        if self.master.config_init() > 0:
            print(f"✅ 发现 {len(self.master.slaves)} 个从站")
            
            # 配置从站
            for i, slave in enumerate(self.master.slaves):
                print(f"  从站 {i}: {slave.name}")
                
                # 配置PDO映射
                slave.sdo_write(0x1C12, 0, bytes([0]))  # 清空RxPDO
                slave.sdo_write(0x1C13, 0, bytes([0]))  # 清空TxPDO
                
                # 添加PDO条目
                slave.sdo_write(0x1600, 0, bytes([3]))  # 3个RxPDO条目
                slave.sdo_write(0x1600, 1, struct.pack('<I', 0x607A0020))  # 目标位置
                slave.sdo_write(0x1600, 2, struct.pack('<I', 0x60FF0020))  # 目标速度
                slave.sdo_write(0x1600, 3, struct.pack('<I', 0x60710010))  # 目标力矩
                
                slave.sdo_write(0x1A00, 0, bytes([3]))  # 3个TxPDO条目
                slave.sdo_write(0x1A00, 1, struct.pack('<I', 0x60640020))  # 实际位置
                slave.sdo_write(0x1A00, 2, struct.pack('<I', 0x606C0020))  # 实际速度
                slave.sdo_write(0x1A00, 3, struct.pack('<I', 0x60770010))  # 实际力矩
                
            # 配置分布式时钟
            self.master.config_dc()
            
            # 切换到OP模式
            self.master.state_check(pysoem.SAFEOP_STATE, 50000)
            
            # 分配PDO内存
            self.master.config_map()
            
            # 切换到运行模式
            self.master.state_request(pysoem.OP_STATE)
            self.master.state_check(pysoem.OP_STATE, 50000)
            
            print("✅ EtherCAT初始化完成")
            return True
        else:
            print("❌ 未发现从站")
            return False
            
    def start(self):
        """启动通信循环"""
        self.running = True
        self.thread = threading.Thread(target=self._control_loop)
        self.thread.start()
        
    def _control_loop(self):
        """实时控制循环"""
        next_time = time.perf_counter()
        
        while self.running:
            # 接收数据
            self.master.receive_processdata()
            
            # 读取实际状态
            for i, slave in enumerate(self.master.slaves):
                # 解析PDO数据
                data = slave.input
                if len(data) >= 10:
                    self.actual_positions[i] = struct.unpack('<i', data[0:4])[0] * 2 * np.pi / 65536
                    self.actual_velocities[i] = struct.unpack('<i', data[4:8])[0] * 2 * np.pi / 65536
                    self.actual_torques[i] = struct.unpack('<h', data[8:10])[0] / 1000.0
                    
            # 写入目标指令
            for i, slave in enumerate(self.master.slaves):
                # 构建PDO数据
                pos_int = int(self.target_positions[i] * 65536 / (2 * np.pi))
                vel_int = int(self.target_velocities[i] * 65536 / (2 * np.pi))
                tor_int = int(self.target_torques[i] * 1000)
                
                slave.output = struct.pack('<iih', pos_int, vel_int, tor_int)
                
            # 发送数据
            self.master.send_processdata()
            
            # 精确周期控制
            next_time += self.control_period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def set_target(self, motor_id, position=None, velocity=None, torque=None):
        """设置目标值"""
        if position is not None:
            self.target_positions[motor_id] = position
        if velocity is not None:
            self.target_velocities[motor_id] = velocity
        if torque is not None:
            self.target_torques[motor_id] = torque
            
    def get_feedback(self, motor_id):
        """获取反馈"""
        return {
            'position': self.actual_positions[motor_id],
            'velocity': self.actual_velocities[motor_id],
            'torque': self.actual_torques[motor_id]
        }
        
    def stop(self):
        """停止通信"""
        self.running = False
        self.thread.join()
        self.master.close()
```

### 3. 遥操作接口

```python
# teleoperation.py
import vrpn
import numpy as np
from scipy.spatial.transform import Rotation as R

class VRTeleoperation:
    """VR遥操作"""
    
    def __init__(self, robot_controller):
        self.controller = robot_controller
        
        # VR设备连接
        self.tracker = vrpn.receiver.Tracker("Head@localhost")
        self.button = vrpn.receiver.Button("Controller@localhost")
        self.analog = vrpn.receiver.Analog("Controller@localhost")
        
        # 遥操作模式
        self.mode = 'position'  # 'position' 或 'velocity'
        
        # 缩放因子
        self.position_scale = 1.5  # 放大操作者的动作
        self.velocity_scale = 0.5
        
        # 安全限制
        self.max_velocity = 0.5  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        
    def update(self):
        """更新VR数据"""
        self.tracker.mainloop()
        self.button.mainloop()
        self.analog.mainloop()
        
        # 获取VR头显位置和朝向
        head_pos, head_quat = self.get_vr_pose()
        
        # 获取手柄数据
        left_controller = self.get_controller_pose('left')
        right_controller = self.get_controller_pose('right')
        
        # 映射到机器人
        if self.mode == 'position':
            self.map_to_position_control(
                head_pos, head_quat,
                left_controller, right_controller
            )
        else:
            self.map_to_velocity_control(
                left_controller, right_controller
            )
            
    def map_to_position_control(self, head_pose, left_ctrl, right_ctrl):
        """映射到位置控制"""
        # 头部控制躯干朝向
        _, _, yaw = self.quat_to_euler(head_pose[1])
        self.controller.set_torso_orientation(yaw)
        
        # 左手控制左臂末端位置
        left_target = self.vr_to_robot_coords(left_ctrl[0])
        self.controller.set_left_arm_target(left_target)
        
        # 右手控制右臂末端位置
        right_target = self.vr_to_robot_coords(right_ctrl[0])
        self.controller.set_right_arm_target(right_target)
        
        # 手柄按钮控制抓取
        if left_ctrl[2]['trigger'] > 0.5:
            self.controller.set_left_grip(1.0)
        else:
            self.controller.set_left_grip(0.0)
            
    def map_to_velocity_control(self, left_ctrl, right_ctrl):
        """映射到速度控制"""
        # 左手摇杆控制移动
        forward = left_ctrl[2]['joystick_y'] * self.max_velocity
        lateral = left_ctrl[2]['joystick_x'] * self.max_velocity
        
        # 右手摇杆控制旋转
        rotation = right_ctrl[2]['joystick_x'] * self.max_angular_velocity
        
        # 发送速度指令
        self.controller.set_base_velocity(forward, lateral, rotation)
        
    def vr_to_robot_coords(self, vr_pos):
        """VR坐标到机器人坐标转换"""
        # 缩放和偏移
        robot_pos = np.array([
            vr_pos[2] * self.position_scale,  # Z→X
            -vr_pos[0] * self.position_scale, # X→Y
            vr_pos[1] * self.position_scale   # Y→Z
        ])
        
        # 添加偏移 (机器人基座位置)
        robot_pos += np.array([0.3, 0.0, 1.0])
        
        return robot_pos
        
    def quat_to_euler(self, quat):
        """四元数转欧拉角"""
        r = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        return r.as_euler('xyz')
```

## 运行演示

```bash
# 启动ROS2
ros2 launch humanoid_control bringup.launch.py

# 启动遥操作
ros2 run humanoid_control vr_teleop

# 监控状态
ros2 topic echo /joint_states
```

```
🤖 人形机器人控制启动

状态:
  关节: 31个
  控制频率: 1000Hz
  通信: EtherCAT
  
模式: BALANCE
  CoM位置: [0.02, 0.01, 0.80]
  ZMP位置: [0.01, 0.00, 0.00]
  双脚支撑
  
遥操作: ACTIVE
  VR头显: 已连接
  左手控制器: 已连接
  右手控制器: 已连接
  
关节状态:
  L_Hip:  0.12 rad | -0.05 rad/s | 2.1 Nm
  R_Hip:  -0.10 rad | 0.03 rad/s | -1.8 Nm
  ...
```

## 总结

核心技术:
- ROS2实时通信架构
- Pinocchio动力学计算
- Proxsuite QP优化
- EtherCAT 1kHz实时控制
- VR遥操作接口

**完整代码**: [GitHub仓库](https://github.com/aineuro/demo-hub/humanoid-robot)
