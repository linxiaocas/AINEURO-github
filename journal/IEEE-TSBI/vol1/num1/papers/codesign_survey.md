# Hardware-Software Co-Design for Next-Generation AI Systems: A Comprehensive Review
# 下一代AI系统的软硬件协同设计：全面综述

**Michael Brown¹*, et al.**

¹UC Berkeley & Google Research, USA

*Corresponding author: mbrown@berkeley.edu

---

## Abstract / 摘要

The end of Dennard scaling and Moore's Law has necessitated a fundamental shift toward hardware-software co-design for AI systems. This comprehensive review examines the state of the art in co-design methodologies spanning algorithm optimization, accelerator architecture, and system-level integration. We analyze emerging paradigms including algorithm-hardware co-optimization, neural architecture search for hardware targets, and software-defined hardware. The review identifies key trends: the rise of domain-specific architectures, the importance of sparse and structured computation, and the convergence of training and inference hardware. We propose a taxonomy of co-design approaches and identify critical research challenges including automated co-design flows, portability across hardware generations, and optimization for multiple objectives (performance, energy, accuracy).

Dennard缩放和摩尔定律的终结 necessitated 对AI系统软硬件协同设计的根本性转变。本全面综述审视了涵盖算法优化、加速器架构和系统级集成的协同设计方法的最先进水平。我们分析了新兴范式，包括算法-硬件协同优化、针对硬件目标的神经架构搜索和软件定义硬件。综述识别了关键趋势：领域特定架构的兴起、稀疏和结构化计算的重要性，以及训练和推理硬件的融合。我们提出了协同设计方法的分类法，并识别了关键研究挑战，包括自动化协同设计流程、跨硬件代的可移植性，以及多目标优化（性能、能耗、精度）。

**Index Terms—** Hardware-software co-design, AI accelerators, domain-specific architectures, neural architecture search

**关键词：** 软硬件协同设计, AI加速器, 领域特定架构, 神经架构搜索

---

## I. Introduction / 引言

The traditional separation between hardware and software design is breaking down in the AI era. This review surveys co-design approaches.

AI时代，硬件和软件设计的传统分离正在瓦解。本综述调研了协同设计方法。

---

## II. Algorithm Optimizations / 算法优化

### A. Quantization

- Post-training quantization
- Quantization-aware training
- Mixed-precision training

### B. Pruning and Sparsity

- Unstructured pruning
- Structured pruning
- N:M sparsity patterns

### C. Neural Architecture Search (NAS)

- Hardware-aware NAS
- Multi-objective optimization
- Platform-aware design

---

## III. Hardware Architectures / 硬件架构

### A. Spatial Accelerators

- Systolic arrays
- Tensor cores
- Coarse-grained reconfigurable arrays

### B. Memory Systems

- In-memory computing
- Near-memory computing
- Hierarchical memory optimization

### C. Interconnect

- Network-on-chip
- Photonic interconnects
- CXL and chiplet architectures

---

## IV. System-Level Co-Design / 系统级协同设计

### A. Scheduling and Mapping

- Operation scheduling
- Data placement
- Pipeline optimization

### B. Runtime Systems

- Dynamic compilation
- Adaptive execution
- Resource management

---

## V. Emerging Paradigms / 新兴范式

### A. Software-Defined Hardware

- FPGA overlays
- CGRA programming models
- Reconfigurable AI engines

### B. Algorithm-Hardware Co-evolution

- Hardware-inspired algorithms
- Algorithm-driven hardware design
- Closed-loop optimization

---

## VI. Challenges and Opportunities / 挑战与机遇

### A. Automated Co-Design Flows

Need for systematic methodologies.

### B. Portability

Designs that work across hardware generations.

### C. Multi-Objective Optimization

Balancing performance, energy, accuracy, cost.

---

## VII. Conclusion / 结论

Hardware-software co-design is essential for next-generation AI systems. Success requires continued innovation across algorithms, architectures, and system software.

软硬件协同设计对下一代AI系统至关重要。成功需要算法、架构和系统软件的持续创新。

---

**References / 参考文献**

[1] Horowitz, M. (2014). Computing's energy problem (and what we can do about it). *ISSCC*.

[2] Chen, Y., et al. (2016). Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks. *ISCA*.

---

**Citation / 引用格式**

Brown, M., et al. (2026). Hardware-Software Co-Design for Next-Generation AI Systems: A Comprehensive Review. *IEEE Trans. Silicon-Based Intell.*, vol. 1, no. 1, pp. 113-142.
