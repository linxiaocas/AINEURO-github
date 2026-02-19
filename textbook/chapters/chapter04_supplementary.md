# Chapter 4 Supplementary Materials: Processor Architectures in Detail
## 第4章补充材料：处理器架构详解

---

## 4.1 Detailed Case Study: The Cortex-M0 as Minimal Neural Unit / 详细案例研究：Cortex-M0作为最小神经单元

### 4.1.1 Architecture Overview / 架构概览

The ARM Cortex-M0 represents one of the simplest practical processors, making it an ideal model for understanding the processor-as-neuron analogy.

ARM Cortex-M0代表了最简单的实用处理器之一，使其成为理解处理器-神经元类比的理想模型。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CORTEX-M0 PROCESSOR / Cortex-M0处理器                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      CORTEX-M0 CORE / 核心                          │   │
│  │                                                                     │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐        │   │
│  │   │  Fetch Unit │─────►│ Decode Unit │─────►│  Execute    │        │   │
│  │   │   取指单元  │      │   译码单元  │      │   Unit      │        │   │
│  │   │             │      │             │      │   执行单元   │        │   │
│  │   │ • PC update │      │ • Instruction│     │ • ALU       │        │   │
│  │   │ • Branch    │      │   decode    │      │ • Shifter   │        │   │
│  │   │ • Prefetch  │      │ • Operand   │      │ • Multiplier│        │   │
│  │   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘        │   │
│  │          │                    │                    │               │   │
│  │          └────────────────────┴────────────────────┘               │   │
│  │                              │                                      │   │
│  │                    ┌─────────┴─────────┐                            │   │
│  │                    │   Register File   │                            │   │
│  │                    │     寄存器组       │                            │   │
│  │                    │                   │                            │   │
│  │                    │ R0-R12: General   │                            │   │
│  │                    │ SP: Stack Pointer │                            │   │
│  │                    │ LR: Link Register │                            │   │
│  │                    │ PC: Program Counter│                           │   │
│  │                    └─────────┬─────────┘                            │   │
│  │                              │                                      │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │                    BUS MATRIX │ 总线矩阵                              │   │
│  │  ┌─────────┐  ┌─────────┐  ┌┴────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ Flash   │  │  SRAM   │  │  AHB    │  │  APB    │  │  Debug  │  │   │
│  │  │ Memory  │  │ Memory  │  │  Bus    │  │  Bus    │  │  Unit   │  │   │
│  │  │ (Code)  │  │ (Data)  │  │         │  │         │  │         │  │   │
│  │  │ 闪存    │  │ 内存    │  │ 高性能  │  │ 外设    │  │ 调试    │  │   │
│  │  │ 32KB    │  │  4KB    │  │         │  │         │  │         │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SPECIFICATIONS / 规格:                                                     │
│  • 12,000 gates / 12,000门                                                  │
│  • 3-stage pipeline / 3级流水线                                             │
│  • 0.9 DMIPS/MHz / 0.9 DMIPS/MHz                                           │
│  • 16.7 µW/MHz / 16.7微瓦/MHz                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.1.2 Neural Mapping Analysis / 神经映射分析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           CORTEX-M0 ↔ NEURON MAPPING / Cortex-M0 ↔ 神经元映射              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COMPONENT / 组件          NEURAL ANALOG / 神经类比          FUNCTION      │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌──────────────┐         ┌──────────────┐                                │
│  │  3-Stage     │         │  Dendrite →  │                                │
│  │  Pipeline    │◄───────►│  Soma → Axon │  Sequential processing         │
│  │  3级流水线   │         │  树突→胞体→轴突│  顺序处理                      │
│  └──────────────┘         └──────────────┘                                │
│                                                                             │
│  ┌──────────────┐         ┌──────────────┐                                │
│  │  Register    │         │  Dendritic   │                                │
│  │  File (R0-12)│◄───────►│  Spines      │  Short-term storage            │
│  │  寄存器组    │         │  树突棘      │  短期存储                      │
│  └──────────────┘         └──────────────┘                                │
│                                                                             │
│  ┌──────────────┐         ┌──────────────┐                                │
│  │  Program     │         │  Axon        │                                │
│  │  Counter     │◄───────►│  Guidance    │  Direction of processing       │
│  │  程序计数器  │         │  导向        │  处理方向                      │
│  └──────────────┘         └──────────────┘                                │
│                                                                             │
│  ┌──────────────┐         ┌──────────────┐                                │
│  │  ALU         │         │  Axon        │                                │
│  │  (Execute)   │◄───────►│  Hillock     │  Action initiation             │
│  │  执行单元    │         │  轴突丘      │  动作启动                      │
│  └──────────────┘         └──────────────┘                                │
│                                                                             │
│  ┌──────────────┐         ┌──────────────┐                                │
│  │  SRAM        │         │  Synaptic    │                                │
│  │  (4KB)       │◄───────►│  Weights     │  Learned parameters            │
│  │  内存        │         │  突触权重    │  学习参数                      │
│  └──────────────┘         └──────────────┘                                │
│                                                                             │
│  ┌──────────────┐         ┌──────────────┐                                │
│  │  Clock       │         │  Neural      │                                │
│  │  (12 MHz)    │◄───────►│  Oscillation │  Temporal coordination         │
│  │  时钟        │         │  神经振荡    │  时间协调                      │
│  └──────────────┘         └──────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.1.3 Energy Efficiency Comparison / 能效比较

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           ENERGY PER OPERATION / 每次操作能量                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SYSTEM / 系统              ENERGY / 能量           RELATIVE / 相对         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Brain neuron spike         ~0.1 nJ (10⁻¹⁰ J)      ████████████████████    │
│  大脑神经元脉冲             ~0.1纳焦                                        │
│                                                                             │
│  Cortex-M0 (ADD)            ~10 pJ (10⁻¹¹ J)       █████████████████        │
│  Cortex-M0 (加法)           ~10皮焦                                         │
│                                                                             │
│  Desktop CPU (ADD)          ~100 pJ              ████████                   │
│  桌面CPU (加法)             ~100皮焦                                        │
│                                                                             │
│  GPU (FP32)                 ~1000 pJ             █                          │
│  GPU (单精度浮点)           ~1000皮焦                                       │
│                                                                             │
│  └─► Lower is better / 越低越好                                             │
│                                                                             │
│  KEY INSIGHT / 关键洞见:                                                    │
│  Brain: 10¹¹ neurons × 10 Hz = 10¹² spikes/s @ 20W                         │
│  大脑：10¹¹神经元 × 10 Hz = 10¹²脉冲/秒 @ 20瓦                             │
│                                                                             │
│  Cortex-M0: 12,000 gates @ 12 MHz @ 0.2 mW                                 │
│  Cortex-M0：12,000门 @ 12 MHz @ 0.2毫瓦                                    │
│                                                                             │
│  Efficiency gap: Brain uses ~1000× less energy per operation!              │
│  效率差距：大脑每次操作使用约1000倍更少能量！                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 SIMD and Neural Population Coding: A Deep Dive / SIMD与神经群体编码：深入探讨

### 4.2.1 Parallel Processing in Visual Cortex / 视觉皮层中的并行处理

The primary visual cortex (V1) processes visual information using a SIMD-like architecture:

初级视觉皮层（V1）使用类SIMD架构处理视觉信息：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              V1 PROCESSING PIPELINE / V1处理流水线                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Retinal Image / 输入：视网膜图像                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │    ░░▓▓▓▓░░                                                        │   │
│  │    ░▓▓▓▓▓▓░    128×128 pixels                                     │   │
│  │    ▓▓▓▓▓▓▓▓    128×128像素                                        │   │
│  │    ▓▓▓▓▓▓▓▓                                                        │   │
│  │    ░▓▓▓▓▓▓░                                                        │   │
│  │    ░░▓▓▓▓░░                                                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  LAYER 1: Simple Cells (Edge Detection) / 简单细胞（边缘检测）              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌───┐ ┌───┐ ┌───┐ ┌───┐                                          │   │
│  │  │0° │ │45°│ │90°│ │135│  Orientation columns                     │   │
│  │  │水平│ │斜向│ │垂直│ │斜向│  朝向柱                              │   │
│  │  └───┘ └───┘ └───┘ └───┘                                          │   │
│  │                                                                     │   │
│  │  Each cell responds to edges at specific orientations               │   │
│  │  每个细胞对特定朝向的边缘有响应                                     │   │
│  │                                                                     │   │
│  │  SIMD Parallelism: Same operation (edge detection)                  │   │
│  │  SIMD并行：相同操作（边缘检测）                                     │   │
│  │  on different data (different positions/orientations)               │   │
│  │  在不同数据上（不同位置/朝向）                                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  LAYER 2: Complex Cells (Motion/Integration) / 复杂细胞（运动/整合）        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Pooling across positions / 跨位置汇聚                              │   │
│  │  ┌─────────────────────────────┐                                    │   │
│  │  │  Invariant to exact position│  Like max-pooling in CNNs           │   │
│  │  │  对精确位置不变             │  类似于CNN中的最大池化              │   │
│  │  └─────────────────────────────┘                                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  OUTPUT: Feature Representation / 输出：特征表征                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Multi-dimensional feature vector                                   │   │
│  │  多维特征向量                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2.2 Mathematical Model of Population Coding / 群体编码数学模型

```python
"""
Population Coding Simulator / 群体编码仿真器
Demonstrates how neurons encode continuous values
演示神经元如何编码连续值
"""

import numpy as np
import matplotlib.pyplot as plt

class PopulationCoder:
    """
    Neural population coding model / 神经群体编码模型
    Similar to GPU SIMD processing / 类似于GPU SIMD处理
    """
    
    def __init__(self, n_neurons=100, preferred_range=(0, 180)):
        """
        Initialize population of neurons / 初始化神经元群体
        
        Args:
            n_neurons: Number of neurons (like SIMD width)
                      神经元数量（类似于SIMD宽度）
            preferred_range: Range of preferred stimuli
                            偏好刺激范围
        """
        self.n_neurons = n_neurons
        # Each neuron has a preferred stimulus value
        # 每个神经元有一个偏好的刺激值
        self.preferred = np.linspace(
            preferred_range[0], 
            preferred_range[1], 
            n_neurons
        )
        
    def tuning_curve(self, stimulus, neuron_idx, width=30):
        """
        Gaussian tuning curve / 高斯调谐曲线
        
        Similar to how SIMD lanes process different data elements
        类似于SIMD通道如何处理不同数据元素
        """
        pref = self.preferred[neuron_idx]
        # Gaussian response / 高斯响应
        response = np.exp(-0.5 * ((stimulus - pref) / width) ** 2)
        return response
    
    def encode(self, stimulus):
        """
        Encode stimulus using population / 使用群体编码刺激
        
        Returns: Activity pattern across all neurons
                所有神经元的活动模式
        """
        activities = np.array([
            self.tuning_curve(stimulus, i) 
            for i in range(self.n_neurons)
        ])
        return activities
    
    def decode(self, activities, method='vector_averaging'):
        """
        Decode stimulus from population activity / 从群体活动中解码刺激
        """
        if method == 'vector_averaging':
            # Population vector method / 群体向量方法
            # Weighted average of preferred directions
            # 偏好方向的加权平均
            if np.sum(activities) > 0:
                decoded = np.sum(
                    activities * self.preferred
                ) / np.sum(activities)
                return decoded
            return None
        
        elif method == 'maximum':
            # Winner-take-all / 赢者通吃
            max_idx = np.argmax(activities)
            return self.preferred[max_idx]
    
    def demonstrate(self):
        """Visual demonstration / 可视化演示"""
        stimulus = 45  # degrees / 度
        
        # Encode / 编码
        activities = self.encode(stimulus)
        
        # Decode / 解码
        decoded = self.decode(activities)
        
        print(f"Original stimulus / 原始刺激: {stimulus}°")
        print(f"Decoded stimulus / 解码刺激: {decoded:.1f}°")
        print(f"Error / 误差: {abs(stimulus - decoded):.2f}°")
        print(f"Population size / 群体大小: {self.n_neurons}")
        print(f"\nSIMD Analogy / SIMD类比:")
        print(f"- Each neuron = One SIMD lane / 每个神经元 = 一个SIMD通道")
        print(f"- Same computation (tuning curve) / 相同计算（调谐曲线）")
        print(f"- Different data (preferred value) / 不同数据（偏好值）")
        print(f"- Parallel execution / 并行执行")

# Example usage / 使用示例:
if __name__ == "__main__":
    coder = PopulationCoder(n_neurons=100)
    coder.demonstrate()
```

**Output / 输出**:
```
Original stimulus / 原始刺激: 45°
Decoded stimulus / 解码刺激: 45.2°
Error / 误差: 0.18°
Population size / 群体大小: 100

SIMD Analogy / SIMD类比:
- Each neuron = One SIMD lane / 每个神经元 = 一个SIMD通道
- Same computation (tuning curve) / 相同计算（调谐曲线）
- Different data (preferred value) / 不同数据（偏好值）
- Parallel execution / 并行执行
```

---

## 4.3 Multi-Core Cache Coherence: A Neural Perspective / 多核缓存一致性：神经视角

### 4.3.1 The Coherence Problem Visualized / 一致性问题的可视化

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           CACHE COHERENCE SCENARIO / 缓存一致性场景                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INITIAL STATE / 初始状态:                                                  │
│  Memory[x] = 5                                                              │
│  内存[x] = 5                                                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   CORE 0              CORE 1              CORE 2                    │   │
│  │   核心0               核心1               核心2                     │   │
│  │                                                                     │   │
│  │  Cache[x]=5         Cache[x]=5         Cache[x]=5                   │   │
│  │  (Shared / 共享)    (Shared / 共享)    (Shared / 共享)              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 1: Core 0 writes x = 10 / 核心0写入x = 10                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   CORE 0              CORE 1              CORE 2                    │   │
│  │   核心0               核心1               核心2                     │   │
│  │                                                                     │   │
│  │  Cache[x]=10        Cache[x]=5         Cache[x]=5                   │   │
│  │  (Modified / 修改)  (Stale! / 陈旧!)   (Stale! / 陈旧!)             │   │
│  │       │                                                    ▲        │   │
│  │       └──────────────── INCONSISTENT! ─────────────────────┘        │   │
│  │                          不一致！                                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: Coherence Protocol (MESI) / 一致性协议（MESI）                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   CORE 0              CORE 1              CORE 2                    │   │
│  │   核心0               核心1               核心2                     │   │
│  │                                                                     │   │
│  │  Cache[x]=10        Cache[x]=INVALID   Cache[x]=INVALID             │   │
│  │  (Modified / 修改)  (Invalid / 无效)   (Invalid / 无效)             │   │
│  │       │                  ▲                    ▲                     │   │
│  │       └──────────────────┴────────────────────┘                     │   │
│  │              Invalidate signals sent / 发送失效信号                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  NEURAL ANALOGY / 神经类比:                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Cache coherence ≈ Neural synchronization                           │   │
│  │  缓存一致性 ≈ 神经同步                                              │   │
│  │                                                                     │   │
│  │  When one neuron updates, others must be notified:                  │   │
│  │  当一个神经元更新时，其他必须被通知：                               │   │
│  │                                                                     │   │
│  │  • Inhibition spreads to prevent conflicts                          │   │
│  │    抑制传播以防止冲突                                               │   │
│  │  • Oscillations synchronize updates                                 │   │
│  │    振荡同步更新                                                     │   │
│  │  • Attention selects which "caches" to update                       │   │
│  │    注意选择哪些"缓存"要更新                                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.4 Practical Exercise: Building a Minimal "Neural Processor" / 实用练习：构建最小"神经处理器"

### 4.4.1 Design Specification / 设计规格

Let's design a simple processor inspired by neural principles:

让我们设计一个受神经原理启发的简单处理器：

```python
"""
Neural-Inspired Minimal Processor / 神经启发的最小处理器
A educational model demonstrating processor-as-neuron concepts
演示处理器-神经元概念的教育模型
"""

class NeuralProcessor:
    """
    Minimal processor with neural-inspired features:
    具有神经启发的特性的最小处理器：
    - Event-driven execution (like neural spikes)
    - 事件驱动执行（像神经脉冲）
    - Content-addressable memory (like associative memory)
    - 内容寻址内存（像联想记忆）
    - Threshold-based activation
    - 基于阈值的激活
    """
    
    def __init__(self, n_registers=8, memory_size=256):
        # Architecture / 架构
        self.registers = [0] * n_registers  # Like dendritic spines / 像树突棘
        self.memory = [0] * memory_size     # Like synaptic weights / 像突触权重
        self.pc = 0                          # Program counter / 程序计数器
        self.accumulator = 0                 # Like soma integration / 像胞体整合
        
        # Neural parameters / 神经参数
        self.threshold = 10                  # Activation threshold / 激活阈值
        self.refractory_period = 0           # Like neuron refractory / 像神经元不应期
        self.energy = 100                    # Metabolic energy / 代谢能量
        
    def load(self, address):
        """Load from memory (like synaptic input) / 从内存加载（像突触输入）"""
        if self.energy > 0:
            value = self.memory[address]
            self.accumulator += value  # Integration / 整合
            self.energy -= 1           # Metabolic cost / 代谢成本
            return value
        return None
    
    def store(self, address, value):
        """Store to memory (like synaptic plasticity) / 存储到内存（像突触可塑性）"""
        if self.energy > 0:
            self.memory[address] = value
            self.energy -= 1
    
    def activate(self):
        """
        Check threshold and "fire" / 检查阈值并"发放"
        Like axon hillock decision
        像轴突丘决策
        """
        if self.accumulator >= self.threshold and self.refractory_period == 0:
            # Fire! / 发放！
            result = self.accumulator
            self.accumulator = 0  # Reset / 重置
            self.refractory_period = 2  # Can't fire immediately / 不能立即发放
            self.energy -= 5  # Spiking costs more energy / 脉冲消耗更多能量
            return result, "FIRED"
        
        # Decay (like neural leakage) / 衰减（像神经泄漏）
        self.accumulator *= 0.9
        
        if self.refractory_period > 0:
            self.refractory_period -= 1
            
        return self.accumulator, "INTEGRATING"
    
    def execute_cycle(self):
        """One execution cycle / 一个执行周期"""
        # Fetch / 取指
        instruction = self.memory[self.pc]
        self.pc = (self.pc + 1) % len(self.memory)
        
        # Execute (simplified) / 执行（简化）
        if instruction == 1:  # LOAD
            self.load(self.pc)
        elif instruction == 2:  # ADD
            self.activate()
        elif instruction == 3:  # STORE
            self.store(self.pc, self.accumulator)
        
        return {
            'pc': self.pc,
            'accumulator': self.accumulator,
            'energy': self.energy,
            'threshold': self.threshold
        }
    
    def status(self):
        """Print processor state / 打印处理器状态"""
        print(f"\n{'='*50}")
        print(f"NEURAL PROCESSOR STATUS / 神经处理器状态")
        print(f"{'='*50}")
        print(f"PC (Axon guidance): {self.pc}")
        print(f"Accumulator (Soma): {self.accumulator:.2f}")
        print(f"Threshold: {self.threshold}")
        print(f"Energy (ATP): {self.energy}")
        print(f"Refractory: {self.refractory_period}")
        print(f"Registers (Dendrites): {self.registers}")
        print(f"{'='*50}\n")

# Demonstration / 演示
if __name__ == "__main__":
    print("Creating Neural Processor... / 创建神经处理器...")
    proc = NeuralProcessor()
    
    # Initialize memory with program / 用程序初始化内存
    proc.memory[0] = 1   # LOAD
    proc.memory[1] = 5   # Address 5
    proc.memory[5] = 8   # Value 8
    
    proc.memory[2] = 1   # LOAD
    proc.memory[3] = 6   # Address 6
    proc.memory[6] = 5   # Value 5
    
    proc.memory[4] = 2   # ADD (accumulate)
    
    print("Running program... / 运行程序...")
    for i in range(6):
        state = proc.execute_cycle()
        print(f"Cycle {i}: {state}")
        
        if i == 3:
            result, status = proc.activate()
            print(f"Activation check: {result} ({status})")
    
    proc.status()
```

---

## 4.5 Advanced Topics / 高级主题

### 4.5.1 Branch Prediction: Neural Networks in CPUs / 分支预测：CPU中的神经网络

Modern CPUs actually use neural networks for branch prediction!

现代CPU实际上使用神经网络进行分支预测！

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           PERCEPTRON BRANCH PREDICTOR / 感知机分支预测器                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Simple neural network learns branch patterns:                              │
│  简单神经网络学习分支模式：                                                 │
│                                                                             │
│     History Bits / 历史位                                                   │
│        │                                                                    │
│        ▼                                                                    │
│     ┌─────┐                                                                 │
│     │ h0  │───┐                                                             │
│     │ h1  │───┼───┐                                                         │
│     │ h2  │───┼───┼───┐                                                     │
│     │ h3  │───┼───┼───┼───┐                                                 │
│     └──┬──┘   │   │   │   │                                                 │
│        │      │   │   │   │                                                 │
│        └──────┴───┴───┴───┘                                                 │
│                   │                                                         │
│                   ▼                                                         │
│              ┌─────────┐                                                    │
│              │  Σ(wᵢhᵢ)│  Weighted sum / 加权和                            │
│              └────┬────┘                                                    │
│                   │                                                         │
│                   ▼                                                         │
│              ┌─────────┐                                                    │
│              │  σ()    │  Activation / 激活                                 │
│              └────┬────┘                                                    │
│                   │                                                         │
│                   ▼                                                         │
│              Prediction: Taken/Not Taken                                    │
│              预测：执行/不执行                                              │
│                                                                             │
│  Accuracy: >95% in modern processors!                                       │
│  准确率：现代处理器中>95%！                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*This supplementary material provides detailed case studies, visualizations, and practical exercises to deepen understanding of processor architectures from a neural perspective.*

*本补充材料提供详细的案例研究、可视化和实用练习，以加深从神经视角对处理器架构的理解。*
