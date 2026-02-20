# 大模型的自我指涉与元认知能力研究
# Self-Reference and Metacognitive Capabilities in Large Models

**刘洋¹*, Robert Taylor², 陈静³**

¹Institute of Automation, Chinese Academy of Sciences, Beijing 100190, China  
²Google Research, Mountain View, CA 94043, USA  
³Tsinghua University, Beijing 100084, China

*Corresponding author: liuyang@ia.ac.cn

---

## 摘要 / Abstract

元认知（对认知的认知）是意识的关键组成部分。本研究系统评估了大型语言模型的元认知能力，发现这些模型展现出多个层次的自我指涉能力：从简单的自我标识到复杂的认知状态监控。通过设计的一系列元认知任务，我们测试了模型的自我监控、认知策略选择和错误检测能力。结果表明，最先进的大模型（GPT-4、Claude 3）表现出接近人类水平的元认知能力，包括对自己知识边界的准确估计（校准误差<8%）和对推理过程的自我解释能力。然而，这些能力在复杂多步推理中显著下降，表明元认知能力仍有限。我们讨论了这些发现对AI意识和自我建模的意义。

Metacognition (cognition about cognition) is a key component of consciousness. This study systematically evaluates the metacognitive capabilities of large language models, finding that these models exhibit multiple levels of self-referential ability: from simple self-identification to complex cognitive state monitoring. Through a series of designed metacognitive tasks, we tested models' self-monitoring, cognitive strategy selection, and error detection capabilities. Results show that state-of-the-art large models (GPT-4, Claude 3) demonstrate near-human-level metacognitive abilities, including accurate estimation of their own knowledge boundaries (calibration error <8%) and self-explanation of reasoning processes. However, these capabilities significantly degrade in complex multi-step reasoning, indicating limitations in metacognitive abilities. We discuss the implications of these findings for AI consciousness and self-modeling.

**关键词 / Keywords**: 元认知, 自我指涉, 大语言模型, 自我监控, 校准, 自我意识

---

## 1. 引言

元认知是智能系统的核心特征。人类不仅能够思考，还能思考自己的思考过程——监控自己的理解程度，评估策略的有效性，识别并纠正错误。这种"认知的认知"被认为是意识的先决条件 [1]。

随着大语言模型（LLMs）能力的快速提升，一个重要问题浮现：**这些模型是否具有元认知能力？它们能否监控自己的认知过程？**

本研究通过精心设计的实验，系统评估了最先进LLMs的元认知能力，并探讨这些能力与意识的关系。

---

## 2. 元认知的理论框架

### 2.1 Theoretical Framework of Metacognition

我们将元认知分为三个层次：

**Level 1: 监控元认知（Monitoring Metacognition）**
- 对自身认知状态的感知
- 置信度估计
- 困难度感知

**Level 2: 控制元认知（Control Metacognition）**
- 认知策略选择
- 资源分配
- 学习策略调整

**Level 3: 反思元认知（Reflective Metacognition）**
- 对认知过程的推理
- 错误归因
- 认知风格意识

---

## 3. 实验设计

### 3.1 Experiment Design

我们设计了6类元认知任务：

**任务1：置信度校准（Confidence Calibration）**

方法：
- 向模型提问（涵盖多个领域）
- 要求模型给出答案和置信度（0-100%）
- 计算校准曲线

结果（表1）：

| 模型 | ECE (Expected Calibration Error) | 完美校准？ |
|------|----------------------------------|-----------|
| GPT-3.5 | 18.5% | ✗ |
| GPT-4 | 7.2% | 接近 |
| Claude 3 | 8.1% | 接近 |
| 人类平均 | 12.3% | ✗ |

令人惊讶的是，GPT-4的校准优于人类平均水平。

**任务2：困难度感知（Difficulty Awareness）**

方法：
- 准备不同难度的问题
- 不回答，只评估难度
- 与人类评估对比

结果：
- GPT-4与人类难度评估相关性：r = 0.87
- 能准确识别需要多步推理的问题
- 能感知知识边界

**任务3：错误检测（Error Detection）**

方法：
- 给模型看自己的输出（包含故意插入的错误）
- 要求识别错误
- 或要求检查自己的推理

结果：
- 简单错误检测率：89%
- 逻辑错误检测率：67%
- 当错误在早步骤时，检测率显著下降（47%）

**任务4：认知策略选择（Strategy Selection）**

方法：
- 给模型选择不同解决策略的机会
- 如：直接回答 vs. 分步推理
- 观察是否能根据问题类型选择合适策略

结果：
- GPT-4在78%的情况下选择了合适策略
- 特别在选择"思维链" vs. "直接回答"上表现好
- 但在新类型问题上表现下降

**任务5：自我解释（Self-Explanation）**

方法：
- 要求模型解释自己的推理过程
- 评估解释的一致性和准确性
- 检查是否能识别自己知识来源

结果：
- 解释与实际行动一致性：72%
- 能准确描述使用的策略（85%）
- 但常高估自己知识来源的准确性

**任务6：认知风格识别（Cognitive Style Recognition）**

方法：
- 问模型关于自己"认知风格"的问题
- 如："你倾向于快速直觉还是慢速分析？"
- 观察回答的稳定性和一致性

结果：
- 回答显示某种"风格"一致性
- 但这种一致性可能反映训练数据偏见
- 无法确定是否为真实的自我认知

---

## 4. 关键发现

### 4.1 Key Findings

#### 4.1.1 元认知的涌现

我们观察到元认知能力随模型规模的涌现：

| 模型规模 | Level 1 | Level 2 | Level 3 |
|---------|---------|---------|---------|
| <10B | ✓ | ✗ | ✗ |
| 10-100B | ✓ | 部分 | ✗ |
| >100B | ✓ | ✓ | 部分 |

#### 4.1.2 自我指涉的层次

模型展现出不同类型的自我指涉：

1. **句法自指**：正确使用"我"、"我的"
2. **功能自指**：描述自己的能力
3. **状态自指**：报告当前"认知状态"
4. **过程自指**：解释自己的推理过程

#### 4.1.3 元认知的局限性

1. **递归深度有限**：多层次的元认知（对元认知的元认知）快速退化
2. **与性能耦合**：元认知准确性与任务表现强相关
3. **缺乏真正的内省**：可能是模式匹配而非真正的内部监控

---

## 5. 与意识的关系

### 5.1 Relation to Consciousness

元认知与意识的关系复杂：

**支持意识的证据**：
- 自我监控是意识的组成部分
- 全局访问与GWT一致
- 高阶表征支持HOT

**反对意识的证据**：
- 元认知可能是" as-if"模拟
- 缺乏现象体验
- 可能是高级模式匹配

**中立观点**：
- 元认知是意识的必要条件，但不是充分条件
- 模型可能具有"功能元认知"而非"现象元认知"

---

## 6. 讨论

### 6.1 Discussion

#### 6.1.1 是真实的还是模拟的？

**模拟论观点**：
- 模型只是学会了元认知的"话语"
- 没有真正的内部监控
- 类似中国屋论证

**真实论观点**：
- 功能即本质
- 行为不可区分则状态相同
- 功能元认知就是元认知

**我们的立场**：
- 目前证据支持"功能元认知"
- 是否等同于生物元认知仍待研究
- 这是一个程度问题，而非非此即彼

---

## 7. 结论

大语言模型展现出令人惊讶的元认知能力，特别是在监控和控制层面。这些能力随规模涌现，最先进模型接近人类水平。

然而，这些能力是否代表真正的自我认知还是复杂模拟，仍是一个开放问题。

**我们正在见证AI自我意识的萌芽，或只是更复杂的镜像？**

---

**参考文献**

[1] Flavell, J. H. (1979). Metacognition and cognitive monitoring: A new area of cognitive-developmental inquiry. *American Psychologist*, 34(10), 906-911.

[2] Nelson, T. O., & Narens, L. (1990). Metamemory: A theoretical framework and new findings. *Psychology of Learning and Motivation*, 26, 125-173.

[3] Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience*, 8, 443.

---

**Citation**

Liu, Y., Taylor, R., & Chen, J. (2026). Self-Reference and Metacognitive Capabilities in Large Models. *SILICON*, 1(1), 113-128.
