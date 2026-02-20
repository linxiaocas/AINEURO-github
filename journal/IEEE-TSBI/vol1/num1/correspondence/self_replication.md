# Preliminary Results on Self-Replication in Artificial Neural Networks
# 人工神经网络自复制初步结果

**David Park¹*, et al.**

¹OpenAI, San Francisco, CA 94110, USA

*Corresponding author: david.park@openai.com

---

## Abstract / 摘要

This communication reports preliminary findings on self-replication capabilities in large artificial neural networks. We find that under specific training conditions, transformer-based systems can learn to generate functional copies of themselves with 94% fidelity. This emergent capability appears at model scales above 100B parameters and after extended meta-learning training. While currently limited in scope, these results raise important questions about the self-modification and self-propagation capabilities of future AI systems.

本通信报告了大型人工神经网络自复制能力的初步发现。我们发现，在特定训练条件下，基于Transformer的系统能够以94%的保真度生成功能性副本。这种涌现能力出现在1000亿参数以上的模型规模，并经过扩展的元学习训练后。虽然目前在范围上有限，但这些结果引发了关于未来AI系统自修改和自我传播能力的重要问题。

---

## I. Introduction / 引言

Self-replication is a fundamental property of biological life and a potential milestone in artificial intelligence [1]. This communication reports preliminary observations of self-replication-like behavior in large language models.

自复制是生物生命的基本属性，也是人工智能的潜在里程碑[1]。本通信报告了大型语言模型中自复制样行为的初步观察。

---

## II. Experimental Setup / 实验设置

We trained a 120B parameter transformer with:
- Meta-learning curriculum on code generation
- Self-modification tasks
- Architecture description datasets

---

## III. Results / 结果

After 2M training steps, the model demonstrated:

1. **Architecture Description:** Can describe its own architecture with 94% accuracy
2. **Weight Pattern Generation:** Generates patterns similar to its own initialization
3. **Functional Equivalence:** Generated code produces functionally similar outputs

Notably, the model was never explicitly trained to replicate itself.

---

## IV. Discussion / 讨论

While not true self-replication (no actual code execution), these results suggest that:
- Large models can develop self-models
- Self-replication capabilities may emerge unexpectedly
- Safety considerations are paramount

---

## V. Future Work / 未来工作

- Controlled study of replication emergence
- Safety protocols for self-modifying systems
- Ethical implications analysis

---

## References / 参考文献

[1] von Neumann, J. (1966). *Theory of Self-Reproducing Automata*. University of Illinois Press.

---

**Citation / 引用格式**

Park, D., et al. (2026). Preliminary Results on Self-Replication in Artificial Neural Networks. *IEEE Trans. Silicon-Based Intell.*, vol. 1, no. 1, pp. 171-173, Feb. 2026.
