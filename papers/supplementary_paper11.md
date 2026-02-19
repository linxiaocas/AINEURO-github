# Supplementary Materials for Paper 11: Transformer Attention and Working Memory
## 论文11补充材料：Transformer注意力与大脑前额叶工作记忆

---

## S1. Detailed Architecture Comparison / 详细架构比较

### S1.1 Multi-Head Attention vs. Cortical Columns / 多头注意vs皮层柱

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         MULTI-HEAD ATTENTION AND CORTICAL ORGANIZATION /                    │
│         多头注意与皮层组织                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRANSFORMER MULTI-HEAD ATTENTION / Transformer多头注意                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Input: Query, Key, Value matrices                                  │   │
│  │  输入：Query、Key、Value矩阵                                        │   │
│  │                                                                     │   │
│  │      ┌─────────┐ ┌─────────┐ ┌─────────┐                           │   │
│  │  Q ─►│ Head 1  │ │ Head 2  │ │ Head 3  │                           │   │
│  │  K ─►│ (64-dim)│ │ (64-dim)│ │ (64-dim)│                           │   │
│  │  V ─►│         │ │         │ │         │                           │   │
│  │      └────┬────┘ └────┬────┘ └────┬────┘                           │   │
│  │           │           │           │                                 │   │
│  │           └───────────┼───────────┘                                 │   │
│  │                       ▼                                             │   │
│  │              Concat + Linear Projection                             │   │
│  │              拼接 + 线性投影                                        │   │
│  │                       │                                             │   │
│  │                       ▼                                             │   │
│  │                 Output (512-dim)                                    │   │
│  │                 输出 (512维)                                        │   │
│  │                                                                     │   │
│  │  Each head learns different attention patterns:                     │   │
│  │  • Head 1: Syntactic relationships / 句法关系                       │   │
│  │  • Head 2: Coreference / 共指                                       │   │
│  │  • Head 3: Semantic roles / 语义角色                                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CORTICAL HYPERCOLUMNS / 皮层超柱                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Input: Visual stimuli / 视觉刺激                                   │   │
│  │                                                                     │   │
│  │      ┌─────────┐ ┌─────────┐ ┌─────────┐                           │   │
│  │  Q ─►│ Column 1│ │ Column 2│ │ Column 3│                           │   │
│  │  K ─►│ (simple │ │ (simple │ │ (simple │                           │   │
│  │  V ─►│ cells)  │ │ cells)  │ │ cells)  │                           │   │
│  │      └────┬────┘ └────┬────┘ └────┬────┘                           │   │
│  │           │           │           │                                 │   │
│  │           └───────────┼───────────┘                                 │   │
│  │                       ▼                                             │   │
│  │              Complex Cell Integration                               │   │
│  │              复杂细胞整合                                           │   │
│  │                       │                                             │   │
│  │                       ▼                                             │   │
│  │              Object Representation                                  │   │
│  │              物体表征                                               │   │
│  │                                                                     │   │
│  │  Each column tuned to different orientations:                       │   │
│  │  • Column 1: 0° / 0度                                               │   │
│  │  • Column 2: 45° / 45度                                             │   │
│  │  • Column 3: 90° / 90度                                             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ANALOGY / 类比:                                                            │
│                                                                             │
│  • Attention Heads ↔ Cortical Columns / 注意头 ↔ 皮层柱                   │
│  • Parallel processing of different features / 不同特征的并行处理          │
│  • Specialization within population / 群体内的专门化                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## S2. Computational Models of Working Memory / 工作记忆计算模型

### S2.1 Persistent Activity Models / 持续活动模型

```python
"""
Working Memory Model inspired by PFC / 受PFC启发的工作记忆模型
Demonstrates persistent activity for information maintenance
演示信息维持的持续活动
"""

import numpy as np
import matplotlib.pyplot as plt

class WorkingMemoryNeuron:
    """
    Model of PFC neuron with persistent activity
    具有持续活动的PFC神经元模型
    """
    
    def __init__(self, tau=20.0, threshold=1.0):
        """
        Args:
            tau: Membrane time constant (ms) / 膜时间常数（毫秒）
            threshold: Firing threshold / 发放阈值
        """
        self.tau = tau
        self.threshold = threshold
        self.v = 0.0  # Membrane potential / 膜电位
        self.activity = []  # Record activity / 记录活动
        
    def update(self, I_ext, dt=1.0):
        """
        Update neuron state / 更新神经元状态
        
        dV/dt = (-V + I_ext) / tau
        """
        # Passive decay / 被动衰减
        dv = (-self.v + I_ext) * dt / self.tau
        self.v += dv
        
        # Self-excitation for persistence / 自兴奋以维持
        if self.v > self.threshold:
            self.v += 0.5  # Excitatory feedback / 兴奋性反馈
            firing = 1.0
        else:
            firing = 0.0
            
        self.activity.append(self.v)
        return firing
    
    def simulate_trial(self, stimulus_duration=100, delay_duration=500):
        """
        Simulate working memory trial / 模拟工作记忆试次
        
        Phases: Fixation → Stimulus → Delay → Response
        阶段：注视 → 刺激 → 延迟 → 反应
        """
        self.activity = []
        
        # Phase 1: Fixation (no input) / 阶段1：注视（无输入）
        for _ in range(50):
            self.update(I_ext=0.0)
        
        # Phase 2: Stimulus presentation / 阶段2：刺激呈现
        for _ in range(stimulus_duration):
            self.update(I_ext=2.0)  # Strong stimulus input
            
        # Phase 3: Delay period / 阶段3：延迟期
        for _ in range(delay_duration):
            self.update(I_ext=0.0)  # No external input!
            
        # Phase 4: Response / 阶段4：反应
        response = self.v > self.threshold
        
        return self.activity, response
    
    def demonstrate(self):
        """Visual demonstration / 可视化演示"""
        activity, response = self.simulate_trial()
        
        print("Working Memory Simulation / 工作记忆仿真")
        print(f"Trial duration: {len(activity)} ms")
        print(f"Response: {'Correct' if response else 'Incorrect'}")
        print(f"\nKey observation: Activity persists during delay!")
        print(f"关键观察：活动在延迟期间持续！")
        
        # Plot / 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(activity, 'b-', linewidth=2)
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Time (ms) / 时间（毫秒）')
        plt.ylabel('Activity / 活动')
        plt.title('Persistent Activity in Working Memory / 工作记忆中的持续活动')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark phases / 标记阶段
        plt.axvspan(50, 150, alpha=0.2, color='green', label='Stimulus')
        plt.axvspan(150, 650, alpha=0.2, color='yellow', label='Delay')
        
        plt.tight_layout()
        plt.savefig('working_memory_activity.png', dpi=150)
        print("\nPlot saved as 'working_memory_activity.png'")
        
        return activity

# Demonstration / 演示
if __name__ == "__main__":
    neuron = WorkingMemoryNeuron()
    neuron.demonstrate()
```

---

## S3. Attention Mechanisms: Detailed Analysis / 注意机制：详细分析

### S3.1 The Prefrontal-Parietal Attention Network / 前额叶-顶叶注意网络

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           DORSAL ATTENTION NETWORK / 背侧注意网络                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  FRONTOPARIETAL NETWORK / 额顶网络                                   │   │
│  │                                                                     │   │
│  │    ┌──────────────┐         Top-down signals / 自上而下信号         │   │
│  │    │              │◄───────────────────────────────────────────────  │   │
│  │    │     FEF      │    Frontal Eye Field / 额叶眼动区               │   │
│  │    │   (Frontal)  │    • Voluntary attention control               │   │
│  │    │   (额叶)     │    • 自愿注意控制                                │   │
│  │    │              │    • Saccade planning / 扫视规划                 │   │
│  │    └──────┬───────┘                                               │   │
│  │           │                                                         │   │
│  │           │ Connection / 连接                                       │   │
│  │           ▼                                                         │   │
│  │    ┌──────────────┐                                                 │   │
│  │    │              │◄──────────────────────────────────────────┐     │   │
│  │    │     IPS      │    Intraparietal Sulcus / 顶内沟          │     │   │
│  │    │   (Parietal) │    • Priority map / 优先级地图             │     │   │
│  │    │   (顶叶)     │    • Salience representation              │     │   │
│  │    │              │    • Attention shifting / 注意转移         │     │   │
│  │    └──────┬───────┘                                                 │   │
│  │           │                                                         │   │
│  │           │ Attentional bias / 注意偏向                            │   │
│  │           ▼                                                         │   │
│  │    ┌──────────────┐                                                 │   │
│  │    │   Sensory    │    Enhanced processing of attended stimuli     │   │
│  │    │   Cortex     │    被注意刺激的增强处理                         │   │
│  │    │   感觉皮层    │                                                 │   │
│  │    └──────────────┘                                                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TRANSFORMER ANALOGY / Transformer类比:                                     │
│                                                                             │
│  • FEF ↔ Query generation / Query生成                                     │
│  • IPS ↔ Key-Value matching / Key-Value匹配                               │
│  • Sensory enhancement ↔ Value weighting / 感觉增强 ↔ Value加权           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## S4. Experimental Evidence / 实验证据

### S4.1 Neural Recordings During Attention Tasks / 注意任务期间的神经记录

| Study / 研究 | Finding / 发现 | Transformer Analog / Transformer类比 |
|------------|--------------|-----------------------------------|
| Buschman & Miller (2007) | PFC neurons reflect attentional priority | Query vectors / Query向量 |
| Bichot et al. (2015) | Parietal cortex encodes salience map | Key-Value attention weights / Key-Value注意权重 |
| Fries et al. (2001) | Gamma synchronization binds attended features | Multi-head attention coordination / 多头注意协调 |
| Gregoriou et al. (2009) | Frontal-parietal coherence predicts performance | Cross-layer attention consistency / 跨层注意一致性 |

---

## S5. Practical Implementation / 实用实现

### S5.1 Attention Model with Biological Constraints / 具有生物约束的注意模型

```python
"""
Biologically-Constrained Attention / 生物约束的注意
Incorporates working memory limitations and biological plausibility
结合工作记忆限制和生物合理性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiologicalAttention(nn.Module):
    """
    Attention mechanism with biological constraints:
    具有生物约束的注意机制：
    - Capacity limit (7±2 items) / 容量限制（7±2项）
    - Decay over time / 时间衰减
    - Interference / 干扰
    """
    
    def __init__(self, d_model, n_heads=8, capacity=7):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.capacity = capacity  # Working memory capacity
        
        # Projections / 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Decay parameters / 衰减参数
        self.decay_rate = 0.95  # Activity decay / 活动衰减
        self.refractory_period = 2  # Time steps / 时间步
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass with biological constraints
        具有生物约束的前向传播
        """
        batch_size = query.size(0)
        
        # Linear projections / 线性投影
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores / 计算注意分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # BIOTOGICAL CONSTRAINT 1: Capacity limit / 生物约束1：容量限制
        # Keep only top-k attention weights
        if scores.size(-1) > self.capacity:
            top_k_values, top_k_indices = torch.topk(
                scores, k=self.capacity, dim=-1
            )
            scores = torch.full_like(scores, float('-inf'))
            scores.scatter_(-1, top_k_indices, top_k_values)
        
        # Apply softmax / 应用softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # BIOTOGICAL CONSTRAINT 2: Decay / 生物约束2：衰减
        attention_weights = attention_weights * self.decay_rate
        
        # Apply attention to values / 将注意应用到values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads / 拼接头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Output projection / 输出投影
        output = self.W_o(context)
        
        return output, attention_weights
    
    def demonstrate_constraints(self):
        """Show biological constraints in action / 展示生物约束的作用"""
        print("Biological Attention Constraints / 生物注意约束")
        print(f"{'='*50}")
        print(f"Working Memory Capacity / 工作记忆容量: {self.capacity} items")
        print(f"Activity Decay Rate / 活动衰减率: {self.decay_rate}")
        print(f"Number of Attention Heads / 注意头数量: {self.n_heads}")
        print(f"\nThese constraints mimic PFC limitations:")
        print(f"这些约束模仿PFC限制：")
        print(f"• Limited capacity / 有限容量")
        print(f"• Temporal decay / 时间衰减")
        print(f"• Parallel processing / 并行处理")

# Demonstration / 演示
if __name__ == "__main__":
    attn = BiologicalAttention(d_model=512, n_heads=8, capacity=7)
    attn.demonstrate_constraints()
    
    # Test with sample input / 用样本输入测试
    batch_size, seq_len, d_model = 2, 20, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = attn(x, x, x)
    print(f"\nInput shape / 输入形状: {x.shape}")
    print(f"Output shape / 输出形状: {output.shape}")
    print(f"Attention weights shape / 注意权重形状: {weights.shape}")
```

---

## S6. Discussion Questions with Extended Answers / 讨论问题与扩展答案

### Q1: Why do Transformers need multiple attention heads while the brain seems to manage with a unified attention system?

**Extended Answer / 扩展答案**:

The brain DOES have multiple "attention heads" in the form of:
1. **Different cortical columns** processing different features
2. **Multiple parallel streams** (dorsal and ventral)
3. **Specialized populations** for different attention types (spatial, feature, object)

The difference is that the brain integrates these seamlessly through:
- **Recurrent connections** allowing iterative refinement
- **Hierarchical organization** with feedback
- **Oscillatory synchronization** binding information

Transformers use explicit multiple heads because they lack these integrative mechanisms.

---

*These supplementary materials provide detailed comparisons, implementations, and experimental evidence linking Transformer attention mechanisms to prefrontal working memory processes.*

*这些补充材料提供详细的比较、实现和实验证据，将Transformer注意机制与前额叶工作记忆过程联系起来。*
