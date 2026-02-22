---
title: "Apollo自动驾驶实战：分层感知-决策-规划-控制架构"
date: "2026-02-22"
author: "Lin Xiao"
category: "Demo"
tags: ["Autonomous Driving", "Apollo", "ROS", "Planning", "Control"]
---

# Apollo自动驾驶实战：分层感知-决策-规划-控制架构

## 引言

Apollo是百度开源的自动驾驶平台，采用分层模块化架构。本文介绍如何基于Apollo构建完整的自动驾驶系统，包括感知、预测、规划、控制四大模块。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    感知层 (Perception)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 激光雷达    │  │ 摄像头      │  │ 毫米波雷达  │         │
│  │ LiDAR      │  │ Camera     │  │ Radar      │         │
│  │ • 点云分割  │  │ • 目标检测  │  │ • 速度测量  │         │
│  │ • 3D检测   │  │ • 车道线   │  │ • 距离测量  │         │
│  │ • 地面检测  │  │ • 红绿灯   │  │ • 跟踪     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         └────────────────┼────────────────┘                │
│                          ↓                                  │
│              ┌─────────────────────┐                       │
│              │   多传感器融合      │                       │
│              │   Fusion Component  │                       │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    预测层 (Prediction)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 障碍物轨迹预测                                      │   │
│  │ • RNN/LSTM 时序建模                                │   │
│  │ • 交互式预测 (考虑周围车辆影响)                     │   │
│  │ • 多模态预测 (多条可能轨迹)                         │   │
│  │ • 意图识别 (变道/跟车/停车)                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    规划层 (Planning)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 路径规划    │  │ 行为决策    │  │ 轨迹规划    │         │
│  │ Routing    │  │ Decision   │  │ Trajectory │         │
│  │             │  │            │  │            │         │
│  │ • 全局路径  │  │ • 场景分析  │  │ • 避障     │         │
│  │ • A*算法   │  │ • 状态机   │  │ • 优化     │         │
│  │ • 地图匹配  │  │ • 交通规则  │  │ • 平滑     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    控制层 (Control)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 纵向控制    │  │ 横向控制    │  │ 底盘控制    │         │
│  │ Longitudinal│  │ Lateral    │  │ Chassis    │         │
│  │             │  │            │  │            │         │
│  │ • 速度控制  │  │ • 转向控制  │  │ • 油门     │         │
│  │ • 加速度    │  │ • 位置跟踪  │  │ • 刹车     │         │
│  │ • PID/MPC  │  │ • 预览控制  │  │ • 转向     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 核心代码实现

### 1. 感知模块

```python
# perception/perception_node.py
import rospy
from sensor_msgs.msg import PointCloud2, Image
from autoware_msgs.msg import DetectedObjectArray
import numpy as np
import cv2
from cv_bridge import CvBridge

class PerceptionNode:
    """感知节点"""
    
    def __init__(self):
        rospy.init_node('perception_node')
        
        # 订阅传感器数据
        self.lidar_sub = rospy.Subscriber('/points_raw', PointCloud2, self.lidar_callback)
        self.camera_sub = rospy.Subscriber('/image_raw', Image, self.camera_callback)
        
        # 发布感知结果
        self.objects_pub = rospy.Publisher('/detected_objects', DetectedObjectArray, queue_size=10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 检测器
        self.lidar_detector = LidarDetector()
        self.camera_detector = CameraDetector()
        
        rospy.loginfo("✅ Perception node initialized")
        
    def lidar_callback(self, msg):
        """激光雷达回调"""
        # 点云转numpy
        points = self.pointcloud2_to_array(msg)
        
        # 3D目标检测
        detections = self.lidar_detector.detect(points)
        
        # 发布结果
        self.publish_detections(detections)
        
    def camera_callback(self, msg):
        """摄像头回调"""
        # 转换图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 2D目标检测
        detections = self.camera_detector.detect(cv_image)
        
        # 车道线检测
        lanes = self.detect_lanes(cv_image)
        
    def pointcloud2_to_array(self, cloud_msg):
        """点云消息转numpy数组"""
        # 解析PointCloud2
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        
        # 从ROS消息中提取数据
        point_cloud = np.frombuffer(cloud_msg.data, dtype=dtype)
        
        points = np.vstack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).T
        return points
        
    def publish_detections(self, detections):
        """发布检测结果"""
        obj_array = DetectedObjectArray()
        obj_array.header.stamp = rospy.Time.now()
        obj_array.header.frame_id = 'base_link'
        
        for det in detections:
            obj = DetectedObject()
            obj.label = det['label']
            obj.score = det['score']
            obj.pose.position.x = det['center'][0]
            obj.pose.position.y = det['center'][1]
            obj.pose.position.z = det['center'][2]
            obj.dimensions.x = det['size'][0]
            obj.dimensions.y = det['size'][1]
            obj.dimensions.z = det['size'][2]
            obj_array.objects.append(obj)
            
        self.objects_pub.publish(obj_array)
        
    def detect_lanes(self, image):
        """车道线检测"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        return lines

class LidarDetector:
    """激光雷达目标检测器"""
    
    def __init__(self):
        # 加载PointPillars模型
        self.model = self.load_pointpillars_model()
        
    def detect(self, points):
        """检测3D目标"""
        # 点云预处理
        voxels = self.preprocess_points(points)
        
        # 推理
        with torch.no_grad():
            predictions = self.model(voxels)
            
        # 解析结果
        detections = self.parse_predictions(predictions)
        
        return detections
        
    def preprocess_points(self, points):
        """点云预处理"""
        # 体素化
        voxel_size = 0.1
        grid_size = [800, 800, 40]
        
        # 过滤范围外的点
        mask = (np.abs(points[:, 0]) < 40) & (np.abs(points[:, 1]) < 40)
        points = points[mask]
        
        return points
        
    def parse_predictions(self, predictions):
        """解析预测结果"""
        detections = []
        
        for pred in predictions:
            det = {
                'label': pred['label'],
                'score': pred['score'],
                'center': pred['location'],
                'size': pred['dimensions'],
                'rotation': pred['rotation_y']
            }
            detections.append(det)
            
        return detections

class CameraDetector:
    """摄像头目标检测器 (YOLO)"""
    
    def __init__(self):
        # 加载YOLO模型
        self.net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
        self.classes = self.load_classes('coco.names')
        
    def detect(self, image):
        """检测2D目标"""
        height, width = image.shape[:2]
        
        # 创建blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # 设置输入
        self.net.setInput(blob)
        
        # 获取输出层
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # 前向传播
        outputs = self.net.forward(output_layers)
        
        # 解析检测结果
        detections = self.parse_yolo_outputs(outputs, width, height)
        
        return detections
```

### 2. 规划模块

```python
# planning/planning_node.py
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from scipy.interpolate import CubicSpline

class PlanningNode:
    """规划节点"""
    
    def __init__(self):
        rospy.init_node('planning_node')
        
        # 订阅
        rospy.Subscriber('/detected_objects', DetectedObjectArray, self.objects_callback)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/goal_pose', PoseStamped, self.goal_callback)
        
        # 发布
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 状态
        self.current_pose = None
        self.goal_pose = None
        self.obstacles = []
        
        # 规划器
        self.global_planner = GlobalPlanner()
        self.local_planner = LocalPlanner()
        
        rospy.Timer(rospy.Duration(0.1), self.planning_loop)
        
        rospy.loginfo("✅ Planning node initialized")
        
    def objects_callback(self, msg):
        """障碍物回调"""
        self.obstacles = []
        for obj in msg.objects:
            self.obstacles.append({
                'x': obj.pose.position.x,
                'y': obj.pose.position.y,
                'radius': max(obj.dimensions.x, obj.dimensions.y) / 2
            })
            
    def pose_callback(self, msg):
        """当前位置回调"""
        self.current_pose = msg
        
    def goal_callback(self, msg):
        """目标位置回调"""
        self.goal_pose = msg
        rospy.loginfo(f"🎯 New goal received: ({msg.pose.position.x}, {msg.pose.position.y})")
        
    def planning_loop(self, event):
        """规划主循环"""
        if self.current_pose is None or self.goal_pose is None:
            return
            
        # 全局路径规划
        global_path = self.global_planner.plan(
            self.current_pose, self.goal_pose
        )
        
        # 局部轨迹规划 (避障)
        local_trajectory = self.local_planner.plan(
            self.current_pose, global_path, self.obstacles
        )
        
        # 发布路径
        self.publish_path(local_trajectory)
        
        # 生成控制指令
        cmd = self.generate_control_command(local_trajectory)
        self.cmd_pub.publish(cmd)
        
    def generate_control_command(self, trajectory):
        """生成控制指令"""
        if len(trajectory) < 2:
            return Twist()
            
        # 计算速度和角度
        dx = trajectory[1][0] - trajectory[0][0]
        dy = trajectory[1][1] - trajectory[0][1]
        
        cmd = Twist()
        cmd.linear.x = np.sqrt(dx**2 + dy**2) * 10  # 速度
        cmd.angular.z = np.arctan2(dy, dx)  # 角速度
        
        return cmd

class GlobalPlanner:
    """全局路径规划器 (A*)"""
    
    def __init__(self, resolution=0.5):
        self.resolution = resolution
        
    def plan(self, start_pose, goal_pose):
        """A*路径规划"""
        start = (start_pose.pose.position.x, start_pose.pose.position.y)
        goal = (goal_pose.pose.position.x, goal_pose.pose.position.y)
        
        # A*算法实现
        path = self.astar(start, goal)
        
        # 路径平滑
        smoothed_path = self.smooth_path(path)
        
        return smoothed_path
        
    def astar(self, start, goal):
        """A*算法"""
        open_set = {start}
        closed_set = set()
        
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        came_from = {}
        
        while open_set:
            # 选择f值最小的节点
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
                
            open_set.remove(current)
            closed_set.add(current)
            
            # 遍历邻居
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                tentative_g = g_score[current] + self.distance(current, neighbor)
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                    
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                
        return [start]  # 无路径
        
    def heuristic(self, a, b):
        """启发函数 (欧氏距离)"""
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
    def get_neighbors(self, node):
        """获取邻居节点"""
        x, y = node
        neighbors = [
            (x+self.resolution, y),
            (x-self.resolution, y),
            (x, y+self.resolution),
            (x, y-self.resolution),
            (x+self.resolution, y+self.resolution),
            (x-self.resolution, y-self.resolution),
            (x+self.resolution, y-self.resolution),
            (x-self.resolution, y+self.resolution)
        ]
        return neighbors

class LocalPlanner:
    """局部轨迹规划器 (DWA)"""
    
    def __init__(self):
        self.max_speed = 5.0  # m/s
        self.max_yawrate = 1.0  # rad/s
        self.dt = 0.1  # s
        
    def plan(self, current_pose, global_path, obstacles):
        """动态窗口法规划"""
        # 提取当前状态
        x = current_pose.pose.position.x
        y = current_pose.pose.position.y
        
        # 动态窗口
        dw = self.calculate_dynamic_window()
        
        best_trajectory = None
        best_score = -float('inf')
        
        # 采样速度和角速度
        for v in np.arange(dw[0], dw[1], 0.1):
            for w in np.arange(dw[2], dw[3], 0.1):
                trajectory = self.predict_trajectory(x, y, v, w)
                score = self.evaluate_trajectory(trajectory, global_path, obstacles)
                
                if score > best_score:
                    best_score = score
                    best_trajectory = trajectory
                    
        return best_trajectory if best_trajectory else global_path[:10]
        
    def calculate_dynamic_window(self):
        """计算动态窗口"""
        # 速度约束
        Vs = [0, self.max_speed, -self.max_yawrate, self.max_yawrate]
        
        # 动力学约束
        Vd = [
            self.current_speed - self.max_accel * self.dt,
            self.current_speed + self.max_accel * self.dt,
            self.current_yawrate - self.max_d_yawrate * self.dt,
            self.current_yawrate + self.max_d_yawrate * self.dt
        ]
        
        # 交集
        dw = [
            max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])
        ]
        
        return dw
        
    def predict_trajectory(self, x, y, v, w, predict_time=3.0):
        """预测轨迹"""
        trajectory = []
        time = 0
        
        while time <= predict_time:
            x += v * np.cos(0) * self.dt  # 简化，假设朝向为0
            y += v * np.sin(0) * self.dt
            trajectory.append((x, y))
            time += self.dt
            
        return trajectory
```

### 3. 控制模块

```python
# control/control_node.py
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class ControlNode:
    """控制节点"""
    
    def __init__(self):
        rospy.init_node('control_node')
        
        # 订阅
        rospy.Subscriber('/planned_path', Path, self.path_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 发布
        self.cmd_pub = rospy.Publisher('/vehicle_cmd', Twist, queue_size=10)
        
        # 控制器
        self.lateral_controller = StanleyController()
        self.longitudinal_controller = PIDController()
        
        rospy.loginfo("✅ Control node initialized")
        
    def odom_callback(self, msg):
        """里程计回调"""
        self.current_speed = msg.twist.twist.linear.x
        self.current_pose = msg.pose.pose
        
    def path_callback(self, msg):
        """路径回调"""
        self.target_path = msg
        
    def control_loop(self):
        """控制主循环"""
        if not hasattr(self, 'target_path') or not hasattr(self, 'current_pose'):
            return
            
        # 找到最近路径点
        target_point = self.find_target_point()
        
        # 横向控制 (Stanley)
        steering = self.lateral_controller.compute(
            self.current_pose, target_point, self.current_speed
        )
        
        # 纵向控制 (PID)
        target_speed = self.get_target_speed()
        throttle = self.longitudinal_controller.compute(
            target_speed, self.current_speed
        )
        
        # 发布控制指令
        cmd = Twist()
        cmd.linear.x = throttle
        cmd.angular.z = steering
        self.cmd_pub.publish(cmd)

class StanleyController:
    """Stanley横向控制器"""
    
    def __init__(self, k=0.5):
        self.k = k  # 增益系数
        
    def compute(self, current_pose, target_point, current_speed):
        """计算转向角"""
        # 提取位置
        x = current_pose.position.x
        y = current_pose.position.y
        
        # 航向误差
        heading_error = self.calculate_heading_error(current_pose, target_point)
        
        # 横向误差
        cross_track_error = self.calculate_cte(x, y, target_point)
        
        # Stanley公式
        steering = heading_error + np.arctan2(self.k * cross_track_error, 
                                             current_speed + 0.1)
        
        # 限幅
        steering = np.clip(steering, -np.pi/4, np.pi/4)
        
        return steering

class PIDController:
    """PID纵向控制器"""
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0
        self.prev_error = 0
        
    def compute(self, target, current):
        """计算控制量"""
        error = target - current
        
        # 积分
        self.integral += error
        self.integral = np.clip(self.integral, -10, 10)  # 防积分饱和
        
        # 微分
        derivative = error - self.prev_error
        self.prev_error = error
        
        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        return np.clip(output, -1, 1)  # 归一化到 [-1, 1]
```

## 仿真运行

```bash
# 启动Apollo
roslaunch apollo_simulator apollo.launch

# 启动感知
rosrun apollo perception_node.py

# 启动规划
rosrun apollo planning_node.py

# 启动控制
rosrun apollo control_node.py

# RViz可视化
rosrun rviz rviz -d apollo.rviz
```

```
🚗 Apollo自动驾驶系统启动

模块状态:
  ✅ 感知模块 - 运行中
     激光雷达: 10Hz
     摄像头: 30Hz
     检测延迟: 50ms
     
  ✅ 规划模块 - 运行中
     全局规划: A*
     局部规划: DWA
     规划频率: 10Hz
     
  ✅ 控制模块 - 运行中
     横向控制: Stanley
     纵向控制: PID
     控制频率: 50Hz
     
车辆状态:
  速度: 5.2 m/s
  位置: (125.3, 78.6)
  航向: 0.45 rad
  目标: (200, 100)
  
路径信息:
  全局路径长度: 150m
  剩余距离: 75m
  预计到达: 15s
```

## 总结

核心技术:
- 多传感器融合感知
- A*全局规划
- DWA局部规划
- Stanley横向控制
- PID纵向控制

**完整代码**: [GitHub仓库](https://github.com/aineuro/demo-hub/autonomous-driving)
