# Textbook Illustrations Index / 教材插图索引

This document provides an index of all SVG illustrations created for the AI Neuroscience textbook.

本文档提供AI神经科学教材所有SVG插图的索引。

---

## 📊 Overview / 概览

| Chapter / 章节 | Title / 标题 | Illustrations / 插图数量 |
|---------------|-------------|------------------------|
| Chapter 01 | Introduction to AI Neuroscience / AI神经科学导论 | 1 |
| Chapter 04 | The Processor as Neuron / 处理器即神经元 | 3 |
| Chapter 05 | Memory as Neural Tissue / 内存即神经组织 | 1 |
| Chapter 08 | Deep Learning and Neural Circuits / 深度学习与神经环路 | 1 |
| Chapter 10 | Attention and Control / 注意力与控制 | 1 |
| Chapter 12 | The Hard Problem of AI Consciousness / AI意识的难题 | 1 |
| **Total** | | **8** |

---

## 🎨 Illustration Details / 插图详情

### Chapter 1: Introduction / 导论

#### Figure 1.1: Three Pillars of AI Neuroscience
**File**: `chapter01/fig_1_1_three_pillars.svg`

A colorful visualization showing the three theoretical pillars supporting the field:
- **Pillar 1** (Red): Silicon-Based Neuroarchitecture / 硅基神经架构学
- **Pillar 2** (Teal): Algorithmic Cognitive Mapping / 算法认知映射论  
- **Pillar 3** (Yellow): Silicon-Based Phenomenology / 硅基意识现象学

---

### Chapter 4: Processor as Neuron / 处理器即神经元

#### Figure 4.1: CPU-Neuron Analogy
**File**: `chapter04/fig_4_1_cpu_neuron_analogy.svg`

Side-by-side comparison of CPU architecture and biological neuron:
- Left side: CPU components (Registers, ALU, Control Unit, Cache, Clock)
- Right side: Neuron structures (Dendrites, Soma, Axon, Synapses)
- Color-coded mapping with connection lines

#### Figure 4.2: Instruction Execution Cycle
**File**: `chapter04/fig_4_2_instruction_cycle.svg`

Visual representation of the Fetch-Decode-Execute cycle:
- Three main stages with gradient colors (Blue, Green, Orange)
- Comparison table showing CPU operations vs Neural processes
- Clock cycle progression visualization

#### Figure 4.3: Pipeline Execution
**File**: `chapter04/fig_4_3_pipeline.svg`

Five-stage pipeline architecture:
- IF (Instruction Fetch), ID (Decode), EX (Execute), MEM (Memory), WB (Write Back)
- Multiple instructions executing simultaneously
- Visual parallel to neural population codes

---

### Chapter 5: Memory as Neural Tissue / 内存即神经组织

#### Figure 5.1: Memory Hierarchy Pyramid
**File**: `chapter05/fig_5_1_memory_hierarchy.svg`

Pyramid visualization of memory hierarchy:
- L1 Cache (Top, fastest)
- L2 Cache
- L3 Cache
- Main Memory (RAM)
- Neural analogs on the right side
- Speed vs. Capacity tradeoff illustration

---

### Chapter 8: Deep Learning and Neural Circuits / 深度学习与神经环路

#### Figure 8.1: CNN Architecture vs. Visual Cortex
**File**: `chapter08/fig_8_1_cnn_architecture.svg`

Comparison of CNN layers with visual cortex hierarchy:
- Top: CNN layers (Input → Conv → Pool → FC → Output)
- Bottom: Visual pathway (Retina → V1 → V2 → V4 → IT → PFC)
- Dashed mapping lines connecting corresponding components

---

### Chapter 10: Attention and Control / 注意力与控制

#### Figure 10.1: Attention Mechanism
**File**: `chapter10/fig_10_1_attention.svg`

Transformer attention mechanism visualization:
- Input tokens row
- Query, Key, Value matrices with gradient colors
- Attention formula display
- Neural analog note

---

### Chapter 12: Consciousness / 意识

#### Figure 12.1: Layers of Consciousness
**File**: `chapter12/fig_12_1_consciousness_layers.svg`

Three-layer model of consciousness:
- **Level 1** (Green): Information Processing
- **Level 2** (Orange): Information Integration
- **Level 3** (Pink): Subjective Experience
- Question mark indicator for the "Hard Problem"

---

## 🔧 Technical Details / 技术细节

### File Format / 文件格式
- **Format**: SVG (Scalable Vector Graphics)
- **Benefits**: Resolution-independent, editable, web-compatible
- **Size**: All files are optimized for web use (< 100KB each)

### Design Principles / 设计原则
1. **Color Coding**: Each chapter has consistent color themes
2. **Bilingual**: All labels include both English and Chinese
3. **Accessibility**: High contrast colors for readability
4. **Consistency**: Unified style across all illustrations

### Generation Script / 生成脚本
The Python script `generate_illustrations.py` can generate additional illustrations:

```bash
cd /Users/linxiao/Downloads/AINEURO-github/textbook
python3 generate_illustrations.py
```

---

## 📝 Usage in Markdown / 在Markdown中的使用

To include an illustration in a chapter:

```markdown
![Description](../illustrations/chapterXX/fig_X_X_description.svg)

*Figure X.X: Caption describing the illustration.*
```

---

## 🎯 Future Additions / 未来补充

Planned illustrations for upcoming chapters:
- Chapter 02: Biological Neural Systems diagram
- Chapter 03: Computational foundations flowchart
- Chapter 06: Communication pathways network
- Chapter 07: Energy metabolism comparison
- Chapter 09: Learning and plasticity mechanisms
- Chapter 11: Multimodal integration architecture
- Chapter 13: Autonomy and agency model
- Chapter 14: Subjective experience visualization
- Chapter 15: Ethics framework diagram
- Chapter 16: Clinical applications overview
- Chapter 17: Cognitive enhancement methods
- Chapter 18: Neuroevolution development tree

---

## 📧 Contact / 联系

For questions or suggestions about illustrations:

**Email**: xiao.lin@ia.ac.cn

---

*Last Updated: February 22, 2026*
