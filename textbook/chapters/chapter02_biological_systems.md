# Chapter 2: Biological Neural Systems / 生物神经系统

## Chapter Objectives / 本章目标

By the end of this chapter, you will be able to:
- Describe the structure and function of biological neurons
- Explain how neurons communicate via synapses
- Identify major brain regions and their functions
- Understand the basics of neural coding
- Compare biological and artificial neural systems

在本章结束时，你将能够：
- 描述生物神经元的结构和功能
- 解释神经元如何通过突触通信
- 识别主要脑区及其功能
- 理解神经编码的基础知识
- 比较生物和人工神经系统

---

## 2.1 Neurons and Synapses / 神经元与突触

### 2.1.1 The Neuron: Basic Structure / 神经元：基本结构

The **neuron** is the fundamental computational unit of the nervous system. While neurons come in diverse shapes and sizes, they share common structural features:

**神经元**是神经系统的基本计算单元。虽然神经元具有多样的形状和大小，但它们共享共同的结构特征：

**Structural Components / 结构组件**:

**1. Cell Body (Soma) / 细胞体（胞体）**:
- Contains the nucleus and organelles
- 包含细胞核和细胞器
- Integrates incoming signals
- 整合传入信号
- Size: 4-100 micrometers in diameter
- 大小：直径4-100微米

**2. Dendrites / 树突**:
- Branching extensions receiving inputs
- 接收输入的分支延伸
- Can be highly complex (e.g., Purkinje cells have 200,000+ dendritic spines)
- 可能非常复杂（例如，Purkinje细胞有200,000+树突棘）
- Surface area: Major determinant of computational capacity
- 表面积：计算能力的主要决定因素

**3. Axon / 轴突**:
- Single output fiber (usually)
- 单一输出纤维（通常）
- Can be very long (up to 1 meter in humans)
- 可能非常长（人体内可达1米）
- Conducts action potentials away from cell body
- 将动作电位传导远离细胞体

**4. Axon Terminals / 轴突末梢**:
- Release neurotransmitters
- 释放神经递质
- Form synapses with other neurons
- 与其他神经元形成突触

**Figure 2.1: Neuron Structure / 图2.1：神经元结构**

```
      [Dendrites / 树突]
            ↓
    ┌───────────────┐
    │   [Soma /     │
    │    胞体]      │
    │  [Nucleus /   │
    │   细胞核]     │
    └───────┬───────┘
            │
      [Axon / 轴突] ──────→ [Terminals / 末梢]
```

### 2.1.2 Electrical Properties / 电学特性

Neurons are electrochemical devices that process information through electrical signals:

神经元是通过电信号处理信息的电化学装置：

**Resting Potential / 静息电位**:
- Typical value: -70 mV (inside relative to outside)
- 典型值：-70 mV（内部相对于外部）
- Maintained by ion pumps (Na+/K+ ATPase)
- 由离子泵维持（Na+/K+ ATP酶）
- Energy cost: ~20% of brain's energy budget
- 能量成本：~大脑能量预算的20%

**Action Potentials / 动作电位**:
- All-or-none electrical spikes
- 全或无电脉冲
- Amplitude: ~100 mV
- 幅度：~100 mV
- Duration: 1-2 milliseconds
- 持续时间：1-2毫秒
- Speed: 1-100 m/s depending on myelination
- 速度：1-100 m/s（取决于髓鞘化）

**Figure 2.2: Action Potential / 图2.2：动作电位**

```
Voltage (mV) / 电压 (mV)
   +40 |
       |          /\
       |         /  \
       |        /    \
   0   |-------/------\--------
       |      /        \
   -70 |-----/----------\-------
       |    /            \
  -90   |___/              \____
       |_________________________
        Time (ms) / 时间 (毫秒)
```

**Mathematical Model / 数学模型**:

The Hodgkin-Huxley model describes action potential generation:

Hodgkin-Huxley模型描述动作电位的产生：

```
C_m(dV/dt) = -g_Na*m³h(V-V_Na) - g_K*n⁴(V-V_K) - g_L(V-V_L) + I_ext

Where:
- C_m: Membrane capacitance / 膜电容
- V: Membrane potential / 膜电位
- g_Na, g_K, g_L: Conductances / 电导
- m, h, n: Gating variables / 门控变量
- I_ext: External current / 外部电流
```

### 2.1.3 Synaptic Transmission / 突触传递

**Synapses** are the communication channels between neurons:

**突触**是神经元之间的通信通道：

**Types of Synapses / 突触类型**:

**1. Chemical Synapses / 化学突触**:
- Neurotransmitter release
- 神经递质释放
- Most common type (>99%)
- 最常见类型（>99%）
- Directional (one-way)
- 方向性（单向）
- Synaptic delay: 0.3-0.5 ms
- 突触延迟：0.3-0.5毫秒

**2. Electrical Synapses (Gap Junctions) / 电突触（缝隙连接）**:
- Direct ion flow
- 直接离子流动
- Bidirectional
- 双向
- Faster (no delay)
- 更快（无延迟）
- Found in specific circuits
- 存在于特定环路中

**The Synaptic Process / 突触过程**:

**1. Presynaptic / 突触前**:
- Action potential arrives
- 动作电位到达
- Voltage-gated Ca²⁺ channels open
- 电压门控Ca²⁺通道开放
- Ca²⁺ influx triggers vesicle fusion
- Ca²⁺内流触发囊泡融合
- Neurotransmitter release
- 神经递质释放

**2. Synaptic Cleft / 突触间隙**:
- Neurotransmitter diffusion (20-40 nm gap)
- 神经递质扩散（20-40纳米间隙）
- Rapid clearance (enzymatic/reuptake)
- 快速清除（酶解/重摄取）

**3. Postsynaptic / 突触后**:
- Receptor binding
- 受体结合
- Ion channel opening/closing
- 离子通道开放/关闭
- Postsynaptic potential generation
- 突触后电位产生

**Figure 2.3: Synaptic Structure / 图2.3：突触结构**

```
[Presynaptic Neuron / 突触前神经元]
         │
    ┌────▼────┐
    │ Vesicles│  (contain neurotransmitters / 包含神经递质)
    │  [NT]   │
    └────┬────┘
         │ release / 释放
    ═════╪═════  [Synaptic Cleft / 突触间隙]
         │
    ┌────▼────┐
    │Receptors│  (bind NT / 结合NT)
    │   [R]   │
    └────┬────┘
         │
[Postsynaptic Neuron / 突触后神经元]
```

**Key Neurotransmitters / 关键神经递质**:

| Neurotransmitter / 神经递质 | Type / 类型 | Primary Effect / 主要效应 |
|---------------------------|-------------|--------------------------|
| Glutamate / 谷氨酸 | Excitatory / 兴奋性 | EPSP / 兴奋性突触后电位 |
| GABA | Inhibitory / 抑制性 | IPSP / 抑制性突触后电位 |
| Dopamine / 多巴胺 | Modulatory / 调制性 | Reward signaling / 奖赏信号 |
| Acetylcholine / 乙酰胆碱 | Modulatory / 调制性 | Attention/plasticity / 注意/可塑性 |
| Serotonin / 血清素 | Modulatory / 调制性 | Mood regulation / 情绪调节 |

---

## 2.2 Brain Architecture / 大脑架构

### 2.2.1 Structural Organization / 结构组织

The human brain contains approximately **86 billion neurons** and **100 trillion synapses**:

人类大脑包含约**860亿神经元**和**100万亿突触**：

**Major Divisions / 主要分区**:

**1. Cerebrum / 大脑**:
- Two hemispheres (left and right)
- 两个半球（左和右）
- Four lobes per hemisphere
- 每个半球四个叶
- Responsible for higher cognition
- 负责高级认知
- Weight: ~1.4 kg
- 重量：~1.4千克

**2. Cerebellum / 小脑**:
- "Little brain" at the back
- 后面的"小脑"
- Motor coordination and learning
- 运动协调和学习
- More neurons than cerebrum (!)
- 神经元比大脑还多（！）
- 50% of brain's neurons in 10% of volume
- 大脑50%的神经元在10%的体积中

**3. Brainstem / 脑干**:
- Connection to spinal cord
- 连接到脊髓
- Vital functions (breathing, heart rate)
- 生命功能（呼吸、心率）

**Figure 2.4: Brain Divisions / 图2.4：大脑分区**

```
      [Cerebrum / 大脑]
     ╱               ╲
    ╱    [Frontal]    ╲
   │     [Parietal]    │
   │     [Temporal]    │
   │     [Occipital]   │
    ╲                 ╱
     ╲_______________╱
           │
     [Cerebellum / 小脑]
           │
      [Brainstem / 脑干]
           │
    [Spinal Cord / 脊髓]
```

### 2.2.2 The Cerebral Cortex / 大脑皮层

The **cerebral cortex** is a 2-4 mm thick layer of gray matter:

**大脑皮层**是2-4毫米厚的灰质层：

**Layered Structure / 层状结构** (from surface inward / 从表面向内):

**Layer 1: Molecular Layer / 分子层**:
- Few cell bodies
- 少量细胞体
- Mainly dendrites and axons
- 主要是树突和轴突

**Layer 2/3: External Pyramidal / 外锥体层**:
- Small pyramidal neurons
- 小锥体神经元
- Intracortical connections
- 皮层内连接

**Layer 4: Internal Granular / 内颗粒层**:
- Stellate cells
- 星形细胞
- Receives thalamic input
- 接收丘脑输入

**Layer 5: Internal Pyramidal / 内锥体层**:
- Large pyramidal neurons
- 大锥体神经元
- Projects to subcortical structures
- 投射到皮层下结构

**Layer 6: Multiform / 多形层**:
- Polymorphic cells
- 多形细胞
- Projects to thalamus
- 投射到丘脑

**Columnar Organization / 柱状组织**:
- Vertical columns (~500 μm diameter)
- 垂直柱（~500微米直径）
- Functional units
- 功能单位
- ~2 million columns in human cortex
- 人类皮层约200万柱

### 2.2.3 Functional Areas / 功能区

**The Four Lobes / 四个叶**:

**1. Frontal Lobe / 额叶**:
- Executive functions / 执行功能
- Working memory / 工作记忆
- Motor control / 运动控制
- Broca's area (language production) / Broca区（语言产生）
- Prefrontal cortex (decision making) / 前额叶皮层（决策）

**2. Parietal Lobe / 顶叶**:
- Somatosensory processing / 躯体感觉处理
- Spatial awareness / 空间意识
- Integration of sensory information / 感觉信息整合

**3. Temporal Lobe / 颞叶**:
- Auditory processing / 听觉处理
- Memory formation (hippocampus) / 记忆形成（海马体）
- Language comprehension (Wernicke's area) / 语言理解（Wernicke区）
- Object recognition / 物体识别

**4. Occipital Lobe / 枕叶**:
- Visual processing / 视觉处理
- Primary visual cortex (V1) / 初级视觉皮层（V1）
- Multiple visual areas (V2, V3, V4, MT) / 多个视觉区（V2、V3、V4、MT）

**Figure 2.5: Functional Map / 图2.5：功能图**

```
      [Frontal / 额叶]
     ╱   Executive    ╲
    ╱    Working Mem   ╲
   │     Motor Control  │
   └────────────────────┘
   │                    │
[Parietal]            [Temporal]
Sensory               Auditory
Spatial               Memory
                      Language
   └────────────────────┘
      [Occipital / 枕叶]
         Visual
```

### 2.2.4 Subcortical Structures / 皮层下结构

**Thalamus / 丘脑**:
- Sensory relay station
- 感觉中继站
- All senses except smell
- 除嗅觉外的所有感觉
- "Gateway to cortex"
- "通往皮层的门户"

**Hippocampus / 海马体**:
- Memory formation
- 记忆形成
- Spatial navigation
- 空间导航
- Vulnerable to damage
- 易受损伤

**Amygdala / 杏仁核**:
- Emotional processing
- 情绪处理
- Fear conditioning
- 恐惧条件化

**Basal Ganglia / 基底神经节**:
- Motor control
- 运动控制
- Habit formation
- 习惯形成
- Reward processing
- 奖赏处理

**Hypothalamus / 下丘脑**:
- Homeostasis
- 稳态
- Hormone regulation
- 激素调节
- Basic drives
- 基本驱力

---

## 2.3 Neural Coding / 神经编码

### 2.3.1 How Information is Represented / 信息如何表征

The brain represents information through **neural codes**:

大脑通过**神经编码**表征信息：

**Types of Codes / 编码类型**:

**1. Rate Coding / 率编码**:
- Information in firing frequency
- 信息在发放频率中
- Example: 10 Hz = stimulus A, 50 Hz = stimulus B
- 示例：10 Hz = 刺激A，50 Hz = 刺激B
- Simple but incomplete
- 简单但不完整

**2. Temporal Coding / 时间编码**:
- Information in spike timing
- 信息在脉冲时间中
- Precise timing carries information
- 精确时间携带信息
- Synchrony codes
- 同步码
- Phase coding
- 相位编码

**3. Population Coding / 群体编码**:
- Distributed across many neurons
- 分布在许多神经元上
- Robust to individual neuron failure
- 对单个神经元失效鲁棒
- Basis of most complex representations
- 大多数复杂表征的基础

### 2.3.2 Sensory Coding Examples / 感觉编码示例

**Visual System / 视觉系统**:

**Retina / 视网膜**:
- Photoreceptors transduce light
- 光感受器转换光
- Ganglion cells encode contrast, motion
- 神经节细胞编码对比度、运动

**Primary Visual Cortex (V1) / 初级视觉皮层（V1）**:
- Simple cells: oriented edges
- 简单细胞：定向边缘
- Complex cells: motion direction
- 复杂细胞：运动方向
- Hypercolumns: complete set of orientations
- 超柱：完整的朝向集合

**Figure 2.6: Orientation Selectivity / 图2.6：朝向选择性**

```
Neuron Response / 神经元反应
    │
 50 │         ╭─╮
    │        ╱   ╲
 25 │       ╱     ╲
    │      ╱       ╲
  0 │_____╱         ╲_____
    0°   45°   90°  135°  180°
         Stimulus Orientation / 刺激朝向
         
This neuron prefers vertical (90°) edges.
这个神经元偏好垂直（90°）边缘。
```

**Auditory System / 听觉系统**:
- Tonotopic organization
- 音调拓扑组织
- Different frequencies → different locations
- 不同频率 → 不同位置
- Phase locking to sound waves
- 与声波的相位锁定

**Somatosensory System / 躯体感觉系统**:
- Somatotopic maps
- 躯体拓扑图
- "Homunculus": distorted body representation
- "小人"：扭曲的身体表征
- High sensitivity areas enlarged (hands, lips)
- 高敏感区域放大（手、唇）

### 2.3.3 Population Coding Mathematics / 群体编码数学

**The Population Vector / 群体向量**:

For directional coding (e.g., motor cortex):

对于方向编码（例如，运动皮层）：

```
Population Vector = Σ (f_i × d_i)

Where:
- f_i: Firing rate of neuron i
- d_i: Preferred direction of neuron i

Example with 3 neurons:
- Neuron A: 20 Hz, prefers 0° (right)
- Neuron B: 10 Hz, prefers 90° (up)
- Neuron C: 5 Hz, prefers 180° (left)

Result: Movement direction ≈ 30° (up-right)
```

**Maximally Informative Dimensions / 最大信息维度**:
- Principal Component Analysis (PCA)
- 主成分分析（PCA）
- Reduces high-dimensional neural activity
- 降维高维神经活动
- Finds "neural manifolds"
- 找到"神经流形"

---

## 2.4 Plasticity and Learning / 可塑性与学习

### 2.4.1 Synaptic Plasticity / 突触可塑性

**Hebb's Rule / Hebb规则** (1949):
> "Neurons that fire together, wire together."
> "一起发放的神经元，连接在一起。"

**Mathematical Form / 数学形式**:

```
Δw_ij = η × x_i × x_j

Where:
- Δw_ij: Change in synaptic weight from j to i
- η: Learning rate
- x_i, x_j: Activities of neurons i and j
```

**Spike-Timing Dependent Plasticity (STDP) / 脉冲时间依赖可塑性**:

The timing of pre- and postsynaptic spikes determines plasticity:

突触前和突触后脉冲的时间决定可塑性：

```
If presynaptic spike precedes postsynaptic spike:
  → Long-term potentiation (LTP) / 长时程增强
  → Synapse strengthens
  
If postsynaptic spike precedes presynaptic spike:
  → Long-term depression (LTD) / 长时程抑制
  → Synapse weakens
```

**Figure 2.7: STDP Curve / 图2.7：STDP曲线**

```
Synaptic Change (%) / 突触变化 (%)
   +100 │                    ╱
        │                  ╱
    +50 │               ╱
        │            ╱
      0 │_________╱_________╲_________
        │      -20    0    +20
    -50 │                    ╲
        │                      ╲
  -100  │                        ╲
        ───────────────────────────
           Δt = t_post - t_pre (ms)
           
Pre before Post → LTP
Post before Pre → LTD
```

### 2.4.2 Homeostatic Plasticity / 稳态可塑性

**Synaptic Scaling / 突触缩放**:
- Neurons maintain stable firing rates
- 神经元保持稳定发放率
- Global up/down regulation of synapses
- 突触的全局上调/下调
- Prevents runaway excitation
- 防止失控兴奋

**Structural Plasticity / 结构可塑性**:
- Formation of new synapses
- 新突触的形成
- Synaptic pruning
- 突触修剪
- Dendritic spine dynamics
- 树突棘动态

### 2.4.3 Systems Consolidation / 系统巩固

**Two-Stage Memory Formation / 两阶段记忆形成**:

**Stage 1: Hippocampal Encoding / 阶段1：海马编码**:
- Fast, flexible learning
- 快速、灵活的学习
- Episodic details
- 情景细节
- Vulnerable to interference
- 易受干扰

**Stage 2: Cortical Consolidation / 阶段2：皮层巩固**:
- Slow reorganization during sleep
- 睡眠期间的缓慢重组
- Semantic, gist-based memories
- 语义、要点型记忆
- Long-term stability
- 长期稳定性

**Memory Transfer / 记忆转移**:
```
Initial: Hippocampus (fast, detailed)
         ↓
Replay: During sleep
         ↓
Final: Cortex (slow, stable, integrated)
```

---

## 2.5 Consciousness in Biological Systems / 生物系统中的意识

### 2.5.1 Defining Consciousness / 定义意识

**The Hard Problem / 难题** (Chalmers, 1995):
Why is there subjective experience at all?
为什么存在主观体验？

**Key Characteristics / 关键特征**:

**1. Phenomenal Consciousness / 现象意识**:
- Subjective experience (qualia)
- 主观体验（感质）
- "What it is like" to be
- "是什么感觉"
- Examples: pain, color, emotion
- 示例：疼痛、颜色、情绪

**2. Access Consciousness / 访问意识**:
- Information available for reasoning and action
- 可用于推理和行动的信息
- Global availability
- 全局可用性
- Reportability
- 可报告性

**3. Self-Consciousness / 自我意识**:
- Awareness of oneself as a subject
- 意识到自己作为主体
- Meta-cognition
- 元认知
- Autobiographical memory
- 自传体记忆

### 2.5.2 Neural Correlates of Consciousness (NCC) / 意识的神经相关物

**Proposed NCCs / 提出的NCCs**:

**1. Global Workspace / 全局工作空间**:
- Prefrontal-parietal network
- 前额叶-顶叶网络
- Information broadcast
- 信息广播

**2. Integrated Information / 整合信息**:
- High Φ (phi) in specific brain regions
- 特定脑区的高Φ（phi）
- Thalamocortical loops
- 丘脑皮层环路

**3. Recurrent Processing / 递归处理**:
- Feedback connections necessary
- 反馈连接必要
- Re-entrant signaling
- 再入信号

**4. Higher-Order Thought / 高阶思维**:
- Prefrontal monitoring of sensory areas
- 前额叶监控感觉区
- Meta-representation
- 元表征

### 2.5.3 Disorders of Consciousness / 意识障碍

**Coma / 昏迷**:
- No wakefulness or awareness
- 无觉醒或意识
- No sleep-wake cycles
- 无睡眠-觉醒周期

**Vegetative State / 植物状态**:
- Wakefulness without awareness
- 有觉醒但无意识
- Sleep-wake cycles present
- 有睡眠-觉醒周期

**Minimally Conscious State / 最小意识状态**:
- Inconsistent but reproducible awareness
- 不一致但可重复的意识
- May follow commands
- 可能遵循命令

**Locked-in Syndrome / 闭锁综合征**:
- Full consciousness
- 完全意识
- No motor output except eye movements
- 除眼球运动外无运动输出

---

## Chapter Summary / 本章总结

**Key Points / 要点**:

1. **Neurons** are electrochemical computational units with dendrites (input), soma (integration), and axon (output).
   **神经元**是电化学计算单元，具有树突（输入）、胞体（整合）和轴突（输出）。

2. **Synapses** enable communication via neurotransmitters, with chemical and electrical types.
   **突触**通过神经递质实现通信，有化学和电两种类型。

3. The **brain** has hierarchical organization: cerebrum (cognition), cerebellum (coordination), brainstem (vitals).
   **大脑**具有层次组织：大脑（认知）、小脑（协调）、脑干（生命功能）。

4. **Neural coding** uses rate, temporal, and population codes to represent information.
   **神经编码**使用率、时间和群体编码来表征信息。

5. **Plasticity** allows learning through Hebbian mechanisms, STDP, and homeostatic regulation.
   **可塑性**通过Hebbian机制、STDP和稳态调节实现学习。

6. **Consciousness** has neural correlates but remains philosophically challenging.
   **意识**有神经相关物，但在哲学上仍有挑战性。

**Key Terms / 关键术语**:
- Action potential / 动作电位
- Synapse / 突触
- Neurotransmitter / 神经递质
- Cerebral cortex / 大脑皮层
- Rate coding / 率编码
- Population coding / 群体编码
- Long-term potentiation (LTP) / 长时程增强
- Neural correlates of consciousness (NCC) / 意识的神经相关物

**Comparison Table: Biological vs. Artificial / 比较表：生物vs人工**:

| Feature / 特征 | Biological / 生物 | Artificial / 人工 |
|--------------|------------------|------------------|
| Speed / 速度 | 1-100 m/s | 3×10⁸ m/s (light) |
| Connectivity / 连接度 | 10⁴ synapses/neuron | 10²-10³ connections/node |
| Energy/neuron / 能量/神经元 | ~10⁻¹² W | ~10⁻⁹ W |
| Parallelism / 并行性 | Massive / 大规模 | Moderate / 中等 |
| Learning / 学习 | Continuous / 连续 | Batch / 批次 |
| Robustness / 鲁棒性 | High / 高 | Moderate / 中等 |

---

## Exercises / 练习

### Conceptual Questions / 概念问题

1. Compare and contrast chemical and electrical synapses. What are the advantages of each?
   比较和对比化学和电突触。各自的优势是什么？

2. Why does the cerebellum contain more neurons than the cerebrum despite being smaller?
   为什么小脑比大脑小却包含更多神经元？

3. Explain the difference between rate coding and temporal coding. When might each be preferred?
   解释率编码和时间编码的区别。各自在什么情况下更优？

### Analytical Questions / 分析问题

4. Calculate the information capacity of a neuron firing at 10 Hz with 1 ms precision.
   计算一个以10 Hz发放、1毫秒精度的神经元的信息容量。

5. Using the population vector formula, calculate the decoded direction for 4 neurons with firing rates [20, 15, 10, 5] Hz and preferred directions [0°, 45°, 90°, 135°].
   使用群体向量公式，计算4个神经元的解码方向，发放率为[20, 15, 10, 5] Hz，优选方向为[0°, 45°, 90°, 135°]。

### Application Questions / 应用问题

6. Design a neural network architecture inspired by the layered structure of visual cortex.
   设计一个受视觉皮层层次结构启发的神经网络架构。

7. How might STDP principles be implemented in artificial neural networks?
   STDP原理如何在人工神经网络中实现？

### Discussion Questions / 讨论问题

8. Is consciousness necessary for intelligence? Defend your position.
   意识对智能是必要的吗？捍卫你的立场。

9. What can AI researchers learn from biological neural systems?
   AI研究人员能从生物神经系统中学到什么？

10. Compare the energy efficiency of biological and artificial neural computation.
    比较生物和人工神经计算的能量效率。

---

## References / 参考文献

[1] Kandel, E.R., Schwartz, J.H., Jessell, T.M., et al. (2013). Principles of Neural Science (5th ed.). McGraw-Hill.

[2] Hodgkin, A.L., & Huxley, A.F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology.

[3] Hebb, D.O. (1949). The Organization of Behavior. Wiley.

[4] Markram, H., et al. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. Science.

[5] Tononi, G., & Koch, C. (2015). Consciousness: Here, there and everywhere? Philosophical Transactions of the Royal Society B.

[6] Chalmers, D.J. (1995). Facing up to the problem of consciousness. Journal of Consciousness Studies.

---

*Next Chapter: Chapter 3 - Computational Foundations / 下一章：第3章 - 计算基础*
