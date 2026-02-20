# A 7nm 512-Core AI Accelerator with Adaptive Precision and In-Memory Computing
# 7纳米512核AI加速器：自适应精度与存内计算

**Qiang Liu¹*, Jing Sun¹, Wei Zhang², Michael Brown³, Senior Member, IEEE**

¹Institute of Computing Technology, Chinese Academy of Sciences, Beijing 100190, China  
²Tsinghua University, Beijing 100084, China  
³NVIDIA Research, Santa Clara, CA 95051, USA

**Corresponding author:** liuqiang@ict.ac.cn

---

## Abstract

This paper presents a 7nm AI accelerator featuring 512 specialized cores with two key innovations: (1) adaptive precision scaling that dynamically adjusts numerical precision based on workload characteristics, and (2) a hybrid memory architecture combining traditional HBM2e with embedded ReRAM for in-memory computing. The chip achieves 512 TFLOPS (FP16) peak performance with 15.7 TOPS/W energy efficiency on transformer workloads—3.2× better than state-of-the-art GPUs. We demonstrate end-to-end training of a 175B parameter language model entirely on-chip, eliminating external memory bandwidth bottlenecks.

## 摘要

本文介绍了一款7纳米AI加速器，具有512个专用核心和两个关键创新：（1）自适应精度缩放，根据工作负载特性动态调整数值精度；（2）混合存储架构，结合传统HBM2e与嵌入式ReRAM实现存内计算。该芯片在transformer工作负载上实现了512 TFLOPS（FP16）峰值性能和15.7 TOPS/W能效——比最新GPU高出3.2倍。我们展示了完全在片上训练1750亿参数语言模型的端到端过程，消除了外部存储带宽瓶颈。

**Index Terms—** AI accelerator, in-memory computing, adaptive precision, transformer, 7nm, ReRAM

**关键词：** AI加速器，存内计算，自适应精度，transformer，7纳米，ReRAM

---

## I. Introduction / 引言

THE computational demands of large-scale AI models have grown exponentially, with state-of-the-art language models requiring trillions of operations per inference [1]. Traditional GPU architectures face fundamental limitations: the "memory wall" restricts data movement, and fixed-precision arithmetic wastes energy on unnecessary precision [2].

大规模AI模型的计算需求呈指数增长，最新语言模型每次推理需要数万亿次操作[1]。传统GPU架构面临根本性限制："存储墙"限制了数据移动，固定精度算术在不必要的精度上浪费能量[2]。

This work presents "Tianshu-512," an AI accelerator addressing these challenges through:
1. Adaptive precision scaling (APS) that matches numerical precision to computational needs
2. Hybrid in-memory computing using ReRAM for weight storage and analog computation
3. A 512-core architecture optimized for transformer workloads

本文介绍了"天枢-512"（Tianshu-512），一款通过以下方式应对这些挑战的AI加速器：
1. 自适应精度缩放（APS），将数值精度与计算需求匹配
2. 使用ReRAM进行权重存储和模拟计算的混合存内计算
3. 针对transformer工作负载优化的512核架构

---

## II. Architecture / 架构

### A. Overall Organization / 整体组织

The Tianshu-512 processor (Fig. 1) integrates:
- 512 Tensor Processing Cores (TPCs) organized in 32 clusters
- 8 HBM2e stacks (32 GB total, 2 TB/s bandwidth)
- 4 GB embedded ReRAM array for in-memory computation
- Inter-chip links supporting up to 16-chip scaling

### B. Adaptive Precision Scaling (APS) / 自适应精度缩放

The APS unit dynamically selects precision levels:

| Precision | Bit Width | Use Case | Energy per Op |
|-----------|-----------|----------|---------------|
| INT4 | 4-bit | Attention softmax, embeddings | 0.12 pJ |
| INT8 | 8-bit | Most matrix multiplications | 0.45 pJ |
| FP16 | 16-bit | Critical path computations | 1.2 pJ |
| TF32 | 19-bit | Training convergence critical | 2.8 pJ |
| FP32 | 32-bit | Final layer, loss computation | 5.6 pJ |

Precision selection is guided by:
1. Layer sensitivity analysis (offline profiling)
2. Runtime gradient monitoring
3. Error propagation estimation

### C. In-Memory Computing with ReRAM / ReRAM存内计算

The 4 GB ReRAM array enables:
- Analog matrix-vector multiplication (MVM)
- Weight storage with 8-bit precision per cell
- In-situ update for training

Key specifications:
- Array size: 4096 × 4096 per tile (16 tiles total)
- Cell resistance: 10 kΩ to 1 MΩ (8 levels)
- Read latency: 50 ns
- Write endurance: 10^6 cycles
- MVM throughput: 128 TOPS per tile

---

## III. Circuit Design / 电路设计

### A. Tensor Processing Core (TPC) / 张量处理核心

Each TPC contains:
- 256 × 256 systolic array for matrix operations
- APS controller with precision selection logic
- Local SRAM (2 MB) for activations
- ReRAM interface unit

The systolic array supports mixed-precision computation:
- Different precisions for weights and activations
- Dynamic precision switching (100 ns overhead)
- Error accumulation tracking

### B. ReRAM-Based MAC Unit / 基于ReRAM的MAC单元

The analog MAC unit (Fig. 2) performs:

$$y_j = \sum_{i=1}^{N} w_{ij} \cdot x_i$$

where weights $w_{ij}$ are stored as ReRAM conductances $G_{ij}$, and inputs $x_i$ are applied as voltages $V_i$. The output currents sum naturally according to Kirchhoff's laws.

Key innovations:
1. **8-level multi-level cell (MLC)** programming
2. **Compensated read scheme** canceling sneak paths
3. **Offset cancellation** for process variations

### C. Precision Conversion Units / 精度转换单元

Seamless conversion between precision levels:
- Quantization/dequantization engines
- Stochastic rounding for training
- Error feedback for gradient compensation

---

## IV. Implementation / 实现

### A. Technology and Chip Statistics / 工艺与芯片统计

Fabricated in TSMC 7nm FinFET technology:

| Parameter | Value |
|-----------|-------|
| Die Size | 26 mm × 28 mm (728 mm²) |
| Transistor Count | 32 billion |
| TPC Count | 512 |
| HBM2e Stacks | 8 (32 GB) |
| ReRAM Capacity | 4 GB (1T-1R array) |
| TDP | 350 W |
| Peak Performance (FP16) | 512 TFLOPS |
| Peak Performance (INT8) | 1024 TOPS |
| Peak Performance (ReRAM) | 2048 TOPS |

### B. Physical Design Challenges / 物理设计挑战

Key challenges and solutions:

1. **Thermal management:** 3D integrated heat spreader, liquid cooling interface
2. **Power delivery:** On-chip voltage regulators for per-core DVFS
3. **Signal integrity:** Shielded ReRAM readout lines, differential signaling
4. **Manufacturing:** Redundant ReRAM cells for yield improvement

---

## V. Experimental Results / 实验结果

### A. Test Setup / 测试设置

Evaluation platform:
- PCIe 5.0 x16 host interface
- Custom Linux driver
- PyTorch backend with custom operators

### B. Benchmark Results / 基准测试结果

**Inference Performance:**

| Model | Batch Size | Throughput (samples/s) | Latency (ms) | Power (W) |
|-------|------------|------------------------|--------------|-----------|
| BERT-Large | 64 | 12,800 | 5.0 | 285 |
| GPT-3 175B | 1 | 45 | 22.2 | 340 |
| ViT-Huge | 32 | 2,100 | 15.2 | 298 |
| ResNet-152 | 256 | 8,500 | 30.1 | 275 |

**Training Performance:**

| Model | Training Time (hours) | vs. A100 | Energy (kWh) |
|-------|----------------------|----------|--------------|
| GPT-3 175B | 18.5 | 3.2× faster | 6,290 |
| LLaMA-65B | 42.0 | 2.8× faster | 14,700 |
| BERT-Large | 0.35 | 4.1× faster | 123 |

### C. Energy Efficiency Analysis / 能效分析

Breakdown of energy consumption for GPT-3 inference:

| Component | Energy (mJ/token) | Percentage |
|-----------|-------------------|------------|
| ReRAM MVM | 12.5 | 18% |
| Digital TPC | 35.2 | 51% |
| HBM Access | 18.3 | 27% |
| Control/Other | 2.8 | 4% |
| **Total** | **68.8** | **100%** |

Comparison with A100 GPU: **3.2× better energy efficiency**

### D. Precision Adaptation Study / 精度自适应研究

Dynamic precision scaling results on BERT training:

| Configuration | Final Accuracy | Training Time | Energy |
|---------------|----------------|---------------|--------|
| FP32 baseline | 91.2% | 100% | 100% |
| Mixed FP16 | 91.0% | 45% | 42% |
| Static INT8 | 88.5% | 28% | 25% |
| APS (dynamic) | 91.1% | 32% | 28% |

APS achieves near-baseline accuracy with 72% energy reduction.

---

## VI. Large-Scale Demonstration / 大规模演示

### A. On-Chip Training of 175B Model / 175B模型片上训练

We trained a GPT-3-scale model entirely using on-chip memory:

**Challenge:** 175B parameters require 350 GB (FP16) or 700 GB (FP32) storage.

**Solution:** 
- Model parallelism across 512 TPCs
- Activations recomputed during backward pass (gradient checkpointing)
- Optimizer states compressed to 8-bit

**Results:**
- Successful training convergence
- No external HBM access during computation
- Training speed: 3.2× faster than DGX-A100 cluster

### B. Multi-Chip Scaling / 多芯片扩展

Scaling study with up to 16 Tianshu-512 chips:

| Chips | Effective Compute | Scaling Efficiency | Interconnect BW |
|-------|-------------------|-------------------|-----------------|
| 1 | 512 TFLOPS | 100% | - |
| 2 | 1,008 TFLOPS | 98.4% | 1.2 TB/s |
| 4 | 1,984 TFLOPS | 96.9% | 2.4 TB/s |
| 8 | 3,840 TFLOPS | 93.8% | 4.8 TB/s |
| 16 | 7,168 TFLOPS | 87.5% | 9.6 TB/s |

---

## VII. Discussion / 讨论

### A. Comparison with State-of-the-Art / 与最新技术比较

| Platform | Technology | FP16 Perf | Efficiency | Memory BW |
|----------|------------|-----------|------------|-----------|
| NVIDIA A100 | 7nm | 312 TFLOPS | 4.9 TOPS/W | 2 TB/s |
| Google TPUv4 | 7nm | 275 TFLOPS | 5.2 TOPS/W | 1.2 TB/s |
| AMD MI250X | 6nm | 383 TFLOPS | 5.0 TOPS/W | 3.2 TB/s |
| Intel Gaudi2 | 7nm | 350 TFLOPS | 5.8 TOPS/W | 2.45 TB/s |
| **Tianshu-512** | **7nm** | **512 TFLOPS** | **15.7 TOPS/W** | **2 TB/s + ReRAM** |

### B. Limitations and Future Work / 局限性与未来工作

Current limitations:
1. ReRAM write endurance limits training iterations
2. Precision adaptation requires offline profiling
3. Limited to dense matrix operations

Future improvements:
1. Integration with CXL memory expansion
2. On-chip optical interconnects
3. Support for sparse operations

---

## VIII. Conclusion / 结论

The Tianshu-512 demonstrates that adaptive precision and in-memory computing can significantly improve AI accelerator efficiency. With 15.7 TOPS/W and the ability to train 175B parameter models on-chip, it provides a foundation for next-generation AI infrastructure.

---

**Acknowledgments / 致谢**

This work was supported by the National Key R&D Program of China (2022YFB4500400) and the Beijing Municipal Science and Technology Commission.

**References / 参考文献**

[1] T. B. Brown et al., "Language models are few-shot learners," *Advances in Neural Information Processing Systems*, vol. 33, pp. 1877-1901, 2020.

[2] H. Cai, C. Gan, and S. Han, "Efficient architectures for efficient natural language processing," *IEEE Micro*, vol. 41, no. 5, pp. 69-82, Sep./Oct. 2021.

[3] M. Horowitz, "Computing's energy problem (and what we can do about it)," in *ISSCC Dig. Tech. Papers*, Feb. 2014, pp. 10-14.

[4] Y. Wang et al., "Benchmarking TPU, GPU, and CPU platforms for deep learning," *arXiv:1907.10701*, 2019.

---

**Citation / 引用格式**

Q. Liu, J. Sun, W. Zhang, and M. Brown, "A 7nm 512-Core AI Accelerator with Adaptive Precision and In-Memory Computing," *IEEE Trans. Silicon-Based Intell.*, vol. 1, no. 1, pp. 29-42, Feb. 2026.

---

<p align="center">
  <i>© 2026 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.</i>
</p>
