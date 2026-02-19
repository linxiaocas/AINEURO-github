# Chapter 9: Learning and Plasticity / 学习与可塑性

## Chapter Objectives / 本章目标

By the end of this chapter, you will be able to:
- Compare backpropagation with biological learning rules
- Understand neuromorphic learning algorithms
- Explain meta-learning from a neural perspective
- Design continual learning systems inspired by brain mechanisms
- Analyze the stability-plasticity dilemma

在本章结束时，你将能够：
- 比较反向传播与生物学习规则
- 理解神经形态学习算法
- 从神经视角解释元学习
- 设计受大脑机制启发的持续学习系统
- 分析稳定性-可塑性困境

---

## 9.1 Backpropagation and Biological Learning / 反向传播与生物学习

### 9.1.1 The Credit Assignment Problem / 信用分配问题

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           CREDIT ASSIGNMENT: BIOLOGICAL AND COMPUTATIONAL /                 │
│           信用分配：生物与计算                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  THE PROBLEM / 问题:                                                        │
│                                                                             │
│  Network makes error at output / 网络在输出处产生错误:                       │
│                                                                             │
│  Input ──► [Layer 1] ──► [Layer 2] ──► [Layer 3] ──► Output: ERROR        │
│            层1            层2            层3            错误               │
│                                                                             │
│  Question: Which weights caused the error? / 哪些权重导致了错误？           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  BACKPROPAGATION SOLUTION / 反向传播解决方案                        │   │
│  │                                                                     │   │
│  │  1. Compute error at output / 在输出处计算错误                      │   │
│  │           δ_output = (target - output) × f'(activation)             │   │
│  │                                                                     │   │
│  │  2. Propagate backward / 向后传播                                   │   │
│  │           δ_layer2 = (W₃ᵀ × δ_output) × f'(activation)             │   │
│  │           δ_layer1 = (W₂ᵀ × δ_layer2) × f'(activation)             │   │
│  │                                                                     │   │
│  │  3. Update weights / 更新权重                                       │   │
│  │           ΔW = learning_rate × δ × activation                       │   │
│  │                                                                     │   │
│  │  Requirement: Global error signal + non-local weight information   │   │
│  │  要求：全局错误信号 + 非局部权重信息                                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  BIOLOGICAL CONSTRAINTS / 生物约束                                  │   │
│  │                                                                     │   │
│  │  What neurons can do / 神经元能做的:                                │   │
│  │  ✓ Local computation / 局部计算                                     │   │
│  │  ✓ Hebbian plasticity / Hebbian可塑性                               │   │
│  │  ✓ Reward signals (dopamine) / 奖赏信号（多巴胺）                    │   │
│  │                                                                     │   │
│  │  What neurons CANNOT easily do / 神经元不容易做的:                   │   │
│  │  ✗ Symmetric feedback weights / 对称反馈权重                         │   │
│  │  ✗ Instant global error propagation / 即时全局错误传播               │   │
│  │  ✗ Separate forward/backward passes / 分离的前向/后向传递            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.1.2 Spike-Timing Dependent Plasticity (STDP) / 脉冲时间依赖可塑性

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STDP: BIOLOGICAL LEARNING RULE /                         │
│                    STDP：生物学习规则                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIMING DEPENDENCE / 时间依赖性:                                            │
│                                                                             │
│  Δt = t_post - t_pre (time difference between spikes)                       │
│  Δt = t_post - t_pre（脉冲间的时间差）                                       │
│                                                                             │
│  Weight Change / 权重变化                                                   │
│       ▲                                                                     │
│    +0.5 │                    ╱                                              │
│         │                 ╱                                                 │
│       0 ├────────────────╳─────────────────                                 │
│         │              ╱   ╲                                                │
│    -0.5 │           ╱       ╲                                               │
│         └───────╱───────────────╲──────────►                                │
│       -50    -20    0    +20    +50    Δt (ms)                              │
│                                                                             │
│  Δt < 0 (Pre before Post): LTD / 长时程抑制 (weakening)                     │
│  Δt > 0 (Post before Pre): LTP / 长时程增强 (strengthening)                 │
│                                                                             │
│  MATHEMATICAL FORM / 数学形式:                                              │
│                                                                             │
│  If Δt > 0:  Δw = A⁺ × exp(-Δt/τ⁺)    (LTP)                                 │
│  If Δt < 0:  Δw = A⁻ × exp(Δt/τ⁻)     (LTD)                                 │
│                                                                             │
│  Where: A⁺ ≈ 0.1, A⁻ ≈ -0.1, τ ≈ 20ms                                       │
│                                                                             │
│  IMPLEMENTATION / 实现:                                                     │
│                                                                             │
│  class STDPSynapse:                                                         │
│      def __init__(self):                                                    │
│          self.weight = 0.5                                                  │
│          self.A_plus = 0.1                                                  │
│          self.A_minus = -0.1                                                │
│          self.tau = 20.0  # ms                                              │
│                                                                             │
│      def update(self, t_pre, t_post):                                       │
│          delta_t = t_post - t_pre                                           │
│          if delta_t > 0:                                                    │
│              # LTP: Pre before Post                                         │
│              self.weight += self.A_plus * np.exp(-delta_t / self.tau)       │
│          else:                                                              │
│              # LTD: Post before Pre                                         │
│              self.weight += self.A_minus * np.exp(delta_t / self.tau)       │
│          self.weight = np.clip(self.weight, 0, 1)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.1.3 Reward-Modulated STDP / 奖赏调节STDP

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           REINFORCEMENT LEARNING IN THE BRAIN / 大脑中的强化学习            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  THE DOPAMINE SYSTEM / 多巴胺系统                                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   VTA (Ventral Tegmental Area) / 腹侧被盖区                         │   │
│  │   │                                                                 │   │
│  │   │ Dopamine neurons signal reward prediction error                 │   │
│  │   │ 多巴胺神经元发出奖赏预测误差信号                                 │   │
│  │   │                                                                 │   │
│  │   │ δ = R(t) + γV(S_t+1) - V(S_t)    (TD error / 时序差分误差)     │   │
│  │   │                                                                 │   │
│  │   │ If δ > 0: Better than expected → Learn                         │   │
│  │   │ If δ < 0: Worse than expected → Unlearn                        │   │
│  │   │                                                                 │   │
│  │   ▼                                                                 │   │
│  │   Dopamine release to cortex and striatum                           │   │
│  │   多巴胺释放到皮层和纹状体                                            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  REWARD-MODULATED STDP / 奖赏调节STDP:                                      │
│                                                                             │
│  Standard STDP / 标准STDP:                                                  │
│       Δw = STDP(t_pre, t_post)                                              │
│                                                                             │
│  Reward-modulated / 奖赏调节:                                               │
│       Δw = R × STDP(t_pre, t_post)                                          │
│                                                                             │
│  Where R is reward signal / 其中R是奖赏信号                                 │
│                                                                             │
│  Eligibility Trace / 资格迹:                                                │
│       e(t) = decay × e(t-1) + STDP(t)   (past activity memory)              │
│       Δw = R(t) × e(t)                  (reward gates plasticity)           │
│                                                                             │
│  This solves the credit assignment problem!                                 │
│  这解决了信用分配问题！                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9.2 Neuromorphic Learning Algorithms / 神经形态学习算法

### 9.2.1 Local Learning Rules / 局部学习规则

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOCAL vs GLOBAL LEARNING / 局部vs全局学习                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GLOBAL (Backprop) / 全局（反向传播）                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Error: L = ½Σ(target - output)²                                    │   │
│  │                                                                     │   │
│  │  Need: ∂L/∂w for all weights                                       │   │
│  │                                                                     │   │
│  │  Problem: Requires knowledge of future layers' errors               │   │
│  │  问题：需要知道未来层的错误                                          │   │
│  │                                                                     │   │
│  │  Not biologically plausible! / 生物不合理！                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LOCAL (Hebbian) / 局部（Hebbian）                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Δw_ij = η × f(pre_i) × g(post_j)                                   │   │
│  │                                                                     │   │
│  │  Only uses information available at the synapse!                    │   │
│  │  只使用突触处可用的信息！                                            │   │
│  │                                                                     │   │
│  │  Variations:                                                        │   │
│  │  • Hebb: Δw = η × x_i × y_j           (correlation)                 │   │
│  │  • Oja:  Δw = η × (x_i × y_j - y_j² × w_ij)  (normalization)        │   │
│  │  • BCM:  Δw = η × x_i × y_j × (y_j - θ_M)    (threshold)            │   │
│  │                                                                     │   │
│  │  Biologically plausible! / 生物合理！                               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TRADE-OFF / 权衡:                                                          │
│                                                                             │
│  ┌──────────────────┬──────────────────┬──────────────────┐                │
│  │     Method       │   Accuracy       │  Biological      │                │
│  │     方法         │   准确率         │  Plausibility    │                │
│  ├──────────────────┼──────────────────┼──────────────────┤                │
│  │ Backprop         │    ★★★★★        │    ★☆☆☆☆        │                │
│  │ 反向传播         │    高            │    低            │                │
│  ├──────────────────┼──────────────────┼──────────────────┤                │
│  │ Hebbian          │    ★★☆☆☆        │    ★★★★★        │                │
│  │ Hebbian          │    中            │    高            │                │
│  ├──────────────────┼──────────────────┼──────────────────┤                │
│  │ Target Prop      │    ★★★★☆        │    ★★★☆☆        │                │
│  │ 目标传播         │    高            │    中            │                │
│  ├──────────────────┼──────────────────┼──────────────────┤                │
│  │ Equilibrium Prop │    ★★★★☆        │    ★★★★☆        │                │
│  │ 均衡传播         │    高            │    中高          │                │
│  └──────────────────┴──────────────────┴──────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9.3 Meta-Learning and the Brain / 元学习与大脑

### 9.3.1 Learning to Learn / 学会学习

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    META-LEARNING CONCEPTS / 元学习概念                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STANDARD LEARNING / 标准学习:                                              │
│                                                                             │
│  Task: Learn to classify cats vs dogs                                      │
│  任务：学习分类猫vs狗                                                      │
│                                                                             │
│  Training: Given dataset D, find optimal parameters θ*                     │
│  训练：给定数据集D，找到最优参数θ*                                         │
│                                                                             │
│  θ* = argmin L(θ; D)                                                       │
│                                                                             │
│  Result: Good at cats vs dogs, bad at new tasks                            │
│  结果：擅长猫vs狗，不擅长新任务                                            │
│                                                                             │
│  META-LEARNING ("Learning to Learn") / 元学习（"学会学习"）:                │
│                                                                             │
│  Task: Learn HOW to learn new tasks quickly                                │
│  任务：学习如何快速学习新任务                                              │
│                                                                             │
│  Meta-training: Across many tasks {T₁, T₂, ..., Tₙ}                        │
│  元训练：跨多个任务                                                        │
│                                                                             │
│  Find: Initial parameters θ₀ that adapt quickly to new tasks               │
│  找到：能快速适应新任务的初始参数θ₀                                        │
│                                                                             │
│  θ* = θ₀ - α ∇L(θ₀; D_new)   (few gradient steps)                          │
│                                                                             │
│  Result: Good at learning NEW tasks quickly!                               │
│  结果：擅长快速学习新任务！                                                │
│                                                                             │
│  NEURAL ANALOGY / 神经类比:                                                 │
│                                                                             │
│  Prefrontal Cortex as Meta-Learner / 前额叶皮层作为元学习器                 │
│                                                                             │
│  • Rapid task switching / 快速任务切换                                      │
│  • Abstract rule learning / 抽象规则学习                                    │
│  • Strategy adaptation / 策略适应                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Chapter Summary / 本章总结

**Key Points / 要点**:

1. **Backpropagation** is biologically implausible due to non-locality requirements.
   **反向传播**由于非局部性要求在生物上不合理。

2. **STDP** provides a local learning rule based on spike timing.
   **STDP**基于脉冲时间提供局部学习规则。

3. **Neuromorphic algorithms** aim for biological plausibility while maintaining performance.
   **神经形态算法**旨在保持性能的同时实现生物合理性。

4. **Meta-learning** in AI parallels the brain's ability to rapidly adapt to new tasks.
   **元学习**在AI中类似于大脑快速适应新任务的能力。

**Key Terms / 关键术语**:
- Credit assignment problem / 信用分配问题
- Spike-timing dependent plasticity (STDP) / 脉冲时间依赖可塑性
- Reward-modulated STDP / 奖赏调节STDP
- Eligibility trace / 资格迹
- Hebbian learning / Hebbian学习
- Meta-learning / 元学习
- Stability-plasticity dilemma / 稳定性-可塑性困境

---

*Next Chapter: Chapter 10 - Attention and Control / 下一章：第10章 - 注意力与控制*
