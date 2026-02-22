#!/usr/bin/env python3
"""
AI Neuroscience Textbook Illustration Generator
Generates high-quality SVG illustrations for textbook chapters
"""

import os
import sys

def create_directory_structure():
    """Create directory structure for illustrations"""
    base_path = "/Users/linxiao/Downloads/AINEURO-github/textbook/illustrations"
    chapters = [
        "chapter01", "chapter02", "chapter03", "chapter04",
        "chapter05", "chapter06", "chapter07", "chapter08",
        "chapter09", "chapter10", "chapter11", "chapter12",
        "chapter13", "chapter14", "chapter15", "chapter16",
        "chapter17", "chapter18"
    ]
    
    for chapter in chapters:
        os.makedirs(f"{base_path}/{chapter}", exist_ok=True)
    
    print(f"Created directory structure at {base_path}")
    return base_path

def generate_cpu_neuron_analogy_svg(base_path):
    """Generate CPU as Neuron analogy diagram"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="cpuGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a237e"/>
      <stop offset="100%" style="stop-color:#0d1642"/>
    </linearGradient>
    <linearGradient id="neuronGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#c62828"/>
      <stop offset="100%" style="stop-color:#8e0000"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  <!-- Background -->
  <rect width="800" height="600" fill="#f5f5f5"/>
  
  <!-- Title -->
  <text x="400" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#333">
    CPU-Neuron Analogy / CPU-神经元类比
  </text>
  
  <!-- CPU Side (Left) -->
  <g transform="translate(50, 80)">
    <rect x="0" y="0" width="320" height="480" rx="10" fill="url(#cpuGrad)" filter="url(#glow)"/>
    <text x="160" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="white">
      CPU Architecture / CPU架构
    </text>
    
    <!-- Registers -->
    <rect x="30" y="60" width="260" height="50" rx="5" fill="#3949ab"/>
    <text x="160" y="90" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="white">Registers / 寄存器</text>
    
    <!-- ALU -->
    <rect x="30" y="130" width="260" height="80" rx="5" fill="#5e35b1"/>
    <text x="160" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white">ALU / 运算单元</text>
    
    <!-- Control Unit -->
    <rect x="30" y="230" width="260" height="60" rx="5" fill="#7e57c2"/>
    <text x="160" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="white">Control Unit / 控制单元</text>
    
    <!-- Cache -->
    <rect x="30" y="310" width="260" height="50" rx="5" fill="#9575cd"/>
    <text x="160" y="340" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="white">Cache / 缓存</text>
    
    <!-- Clock -->
    <circle cx="160" cy="420" r="40" fill="#ff5722"/>
    <text x="160" y="425" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white">Clock</text>
  </g>
  
  <!-- Connection Arrows -->
  <g transform="translate(380, 250)">
    <path d="M 0,0 L 40,0" stroke="#666" stroke-width="3" marker-end="url(#arrowhead)"/>
    <path d="M 40,50 L 0,50" stroke="#666" stroke-width="3" marker-end="url(#arrowhead)"/>
    <text x="20" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">analogous to</text>
  </g>
  
  <!-- Neuron Side (Right) -->
  <g transform="translate(430, 80)">
    <rect x="0" y="0" width="320" height="480" rx="10" fill="url(#neuronGrad)" filter="url(#glow)"/>
    <text x="160" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="white">
      Neuron / 神经元
    </text>
    
    <!-- Dendrites -->
    <path d="M 30,85 Q 10,60 5,40 M 30,85 Q 25,50 20,30 M 30,85 Q 40,55 45,35" 
          stroke="#ff8a80" stroke-width="4" fill="none"/>
    <text x="80" y="90" font-family="Arial, sans-serif" font-size="14" fill="white">Dendrites / 树突</text>
    
    <!-- Soma -->
    <ellipse cx="160" cy="170" rx="80" ry="50" fill="#ff5252"/>
    <text x="160" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white">Soma / 胞体</text>
    
    <!-- Axon Hillock -->
    <rect x="130" y="230" width="60" height="40" rx="5" fill="#ff1744"/>
    <text x="160" y="255" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="white">Axon Hillock</text>
    
    <!-- Axon -->
    <rect x="150" y="280" width="20" height="120" fill="#d50000"/>
    <text x="200" y="340" font-family="Arial, sans-serif" font-size="14" fill="white">Axon / 轴突</text>
    
    <!-- Synaptic Terminals -->
    <circle cx="140" cy="420" r="15" fill="#ff1744"/>
    <circle cx="160" cy="430" r="15" fill="#ff1744"/>
    <circle cx="180" cy="420" r="15" fill="#ff1744"/>
    <text x="160" y="465" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="white">Synapses / 突触</text>
  </g>
  
  <!-- Legend -->
  <g transform="translate(50, 580)">
    <text x="0" y="0" font-family="Arial, sans-serif" font-size="12" fill="#666">Figure 4.1: The CPU-Neuron analogy maps computational components to neural structures.</text>
  </g>
</svg>'''
    
    filepath = f"{base_path}/chapter04/fig_4_1_cpu_neuron_analogy.svg"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"Generated: {filepath}")

def generate_instruction_cycle_svg(base_path):
    """Generate instruction execution cycle diagram"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="fetchGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#1976d2"/>
      <stop offset="100%" style="stop-color:#0d47a1"/>
    </linearGradient>
    <linearGradient id="decodeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#388e3c"/>
      <stop offset="100%" style="stop-color:#1b5e20"/>
    </linearGradient>
    <linearGradient id="executeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f57c00"/>
      <stop offset="100%" style="stop-color:#e65100"/>
    </linearGradient>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
  
  <!-- Background -->
  <rect width="800" height="400" fill="#fafafa"/>
  
  <!-- Title -->
  <text x="400" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="bold" fill="#333">
    Instruction Execution Cycle / 指令执行周期
  </text>
  
  <!-- CPU Cycle Boxes -->
  <g transform="translate(100, 80)">
    <!-- FETCH -->
    <rect x="0" y="0" width="180" height="100" rx="10" fill="url(#fetchGrad)"/>
    <text x="90" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="white">FETCH</text>
    <text x="90" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="white">取指</text>
    <text x="90" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">PC → Memory</text>
    
    <!-- Arrow -->
    <path d="M 190,50 L 220,50" stroke="#666" stroke-width="3" marker-end="url(#arrowhead)"/>
    
    <!-- DECODE -->
    <rect x="230" y="0" width="180" height="100" rx="10" fill="url(#decodeGrad)"/>
    <text x="320" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="white">DECODE</text>
    <text x="320" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="white">译码</text>
    <text x="320" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">IR → Control</text>
    
    <!-- Arrow -->
    <path d="M 420,50 L 450,50" stroke="#666" stroke-width="3" marker-end="url(#arrowhead)"/>
    
    <!-- EXECUTE -->
    <rect x="460" y="0" width="180" height="100" rx="10" fill="url(#executeGrad)"/>
    <text x="550" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="white">EXECUTE</text>
    <text x="550" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="white">执行</text>
    <text x="550" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">ALU → Write</text>
  </g>
  
  <!-- Feedback loop -->
  <path d="M 730,130 Q 750,200 400,200 Q 50,200 50,130" 
        stroke="#999" stroke-width="2" fill="none" stroke-dasharray="5,5"/>
  <text x="400" y="220" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">Update PC / 更新程序计数器 → Next Instruction / 下一条指令</text>
  
  <!-- Neural Analog Comparison -->
  <g transform="translate(100, 260)">
    <text x="300" y="0" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">
      Neural Processing Analog / 神经处理类比
    </text>
    
    <!-- Comparison table -->
    <rect x="0" y="20" width="600" height="100" fill="white" stroke="#ddd" stroke-width="2" rx="5"/>
    
    <text x="20" y="50" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#1976d2">CPU:</text>
    <text x="80" y="50" font-family="Arial, sans-serif" font-size="13" fill="#333">Fetch → Decode → Execute → Write → Update PC</text>
    
    <text x="20" y="75" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#c62828">Neuron:</text>
    <text x="80" y="75" font-family="Arial, sans-serif" font-size="13" fill="#333">Dendritic Input → Synaptic Integration → Threshold Decision → Action Potential → Axonal Transmission</text>
    
    <text x="20" y="100" font-family="Arial, sans-serif" font-size="13" fill="#666">树突输入 → 突触整合 → 阈值决策 → 动作电位 → 轴突传输</text>
  </g>
  
  <!-- Legend -->
  <text x="400" y="390" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">
    Figure 4.2: The instruction execution cycle parallels neural information processing stages.
  </text>
</svg>'''
    
    filepath = f"{base_path}/chapter04/fig_4_2_instruction_cycle.svg"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"Generated: {filepath}")

def generate_memory_hierarchy_svg(base_path):
    """Generate memory hierarchy diagram"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="l1Grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ff5722"/>
      <stop offset="100%" style="stop-color:#bf360c"/>
    </linearGradient>
    <linearGradient id="l2Grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ff9800"/>
      <stop offset="100%" style="stop-color:#e65100"/>
    </linearGradient>
    <linearGradient id="l3Grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ffc107"/>
      <stop offset="100%" style="stop-color:#ff6f00"/>
    </linearGradient>
    <linearGradient id="memGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#4caf50"/>
      <stop offset="100%" style="stop-color:#1b5e20"/>
    </linearGradient>
  </defs>
  
  <!-- Background -->
  <rect width="800" height="600" fill="#f5f5f5"/>
  
  <!-- Title -->
  <text x="400" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#333">
    Memory Hierarchy as Neural Tissue / 内存层次作为神经组织
  </text>
  
  <!-- Pyramid structure -->
  <g transform="translate(150, 80)">
    <!-- L1 Cache (top) -->
    <path d="M 200,0 L 300,80 L 100,80 Z" fill="url(#l1Grad)"/>
    <text x="200" y="50" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white">L1 Cache</text>
    <text x="200" y="110" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">~32KB | 1-4 cycles</text>
    
    <!-- L2 Cache -->
    <path d="M 150,80 L 350,80 L 375,160 L 125,160 Z" fill="url(#l2Grad)"/>
    <text x="250" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white">L2 Cache</text>
    <text x="250" y="185" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">~256KB | 10-20 cycles</text>
    
    <!-- L3 Cache -->
    <path d="M 100,160 L 400,160 L 450,260 L 50,260 Z" fill="url(#l3Grad)"/>
    <text x="250" y="215" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#333">L3 Cache</text>
    <text x="250" y="285" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">~8-32MB | 40-60 cycles</text>
    
    <!-- Main Memory (bottom) -->
    <path d="M 0,260 L 500,260 L 500,350 L 0,350 Z" fill="url(#memGrad)"/>
    <text x="250" y="310" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white">Main Memory (RAM)</text>
    <text x="250" y="375" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">8-512GB | 200-300 cycles</text>
  </g>
  
  <!-- Neural Analog Labels (right side) -->
  <g transform="translate(600, 120)">
    <text x="0" y="0" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#c62828">Neural Analog:</text>
    
    <text x="0" y="40" font-family="Arial, sans-serif" font-size="12" fill="#333">L1 Cache →</text>
    <text x="70" y="40" font-family="Arial, sans-serif" font-size="11" fill="#666">Working Memory</text>
    
    <text x="0" y="80" font-family="Arial, sans-serif" font-size="12" fill="#333">L2 Cache →</text>
    <text x="70" y="80" font-family="Arial, sans-serif" font-size="11" fill="#666">Recent Memory</text>
    
    <text x="0" y="120" font-family="Arial, sans-serif" font-size="12" fill="#333">L3 Cache →</text>
    <text x="70" y="120" font-family="Arial, sans-serif" font-size="11" fill="#666">Associative Memory</text>
    
    <text x="0" y="180" font-family="Arial, sans-serif" font-size="12" fill="#333">RAM →</text>
    <text x="50" y="180" font-family="Arial, sans-serif" font-size="11" fill="#666">Long-term Memory</text>
  </g>
  
  <!-- Legend -->
  <text x="400" y="480" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">
    Figure 5.1: The memory hierarchy forms a pyramid with speed vs. capacity tradeoffs,
  </text>
  <text x="400" y="500" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">
    analogous to neural memory systems from working memory to long-term storage.
  </text>
</svg>'''
    
    filepath = f"{base_path}/chapter05/fig_5_1_memory_hierarchy.svg"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"Generated: {filepath}")

def main():
    """Main function to generate all illustrations"""
    print("=" * 60)
    print("AI Neuroscience Textbook Illustration Generator")
    print("=" * 60)
    
    base_path = create_directory_structure()
    
    print("\nGenerating illustrations...")
    print("-" * 60)
    
    # Chapter 4 illustrations
    generate_cpu_neuron_analogy_svg(base_path)
    generate_instruction_cycle_svg(base_path)
    
    # Chapter 5 illustrations
    generate_memory_hierarchy_svg(base_path)
    
    print("-" * 60)
    print(f"\n✓ Illustrations generated successfully!")
    print(f"✓ Location: {base_path}")
    print("\nNext steps:")
    print("1. Update markdown files to reference new SVG images")
    print("2. Convert SVG to PNG if needed for certain platforms")
    print("3. Add more chapter illustrations as needed")

if __name__ == "__main__":
    main()

def generate_pipeline_svg(base_path):
    """Generate CPU pipeline diagram"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="900" height="500" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="stage1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#e3f2fd"/>
      <stop offset="100%" style="stop-color:#bbdefb"/>
    </linearGradient>
    <linearGradient id="stage2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#fff3e0"/>
      <stop offset="100%" style="stop-color:#ffe0b2"/>
    </linearGradient>
    <linearGradient id="stage3" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#e8f5e9"/>
      <stop offset="100%" style="stop-color:#c8e6c9"/>
    </linearGradient>
    <linearGradient id="stage4" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#fce4ec"/>
      <stop offset="100%" style="stop-color:#f8bbd9"/>
    </linearGradient>
    <linearGradient id="stage5" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f3e5f5"/>
      <stop offset="100%" style="stop-color:#e1bee7"/>
    </linearGradient>
  </defs>
  
  <rect width="900" height="500" fill="#fafafa"/>
  
  <text x="450" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="bold" fill="#333">
    5-Stage Pipeline Execution / 5级流水线执行
  </text>
  
  <!-- Pipeline stages -->
  <g transform="translate(50, 70)">
    <!-- Stage headers -->
    <rect x="0" y="0" width="150" height="40" fill="url(#stage1)" stroke="#333" stroke-width="2"/>
    <text x="75" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold">IF</text>
    
    <rect x="160" y="0" width="150" height="40" fill="url(#stage2)" stroke="#333" stroke-width="2"/>
    <text x="235" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold">ID</text>
    
    <rect x="320" y="0" width="150" height="40" fill="url(#stage3)" stroke="#333" stroke-width="2"/>
    <text x="395" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold">EX</text>
    
    <rect x="480" y="0" width="150" height="40" fill="url(#stage4)" stroke="#333" stroke-width="2"/>
    <text x="555" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold">MEM</text>
    
    <rect x="640" y="0" width="150" height="40" fill="url(#stage5)" stroke="#333" stroke-width="2"/>
    <text x="715" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold">WB</text>
    
    <!-- Stage labels -->
    <text x="75" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Instruction Fetch</text>
    <text x="235" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Decode</text>
    <text x="395" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Execute</text>
    <text x="555" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Memory</text>
    <text x="715" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Write Back</text>
    
    <!-- Clock cycles -->
    <text x="0" y="90" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#333">Clock 1:</text>
    <rect x="5" y="100" width="140" height="30" fill="#2196f3" opacity="0.7"/>
    <text x="75" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 1</text>
    
    <text x="0" y="145" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#333">Clock 2:</text>
    <rect x="165" y="100" width="140" height="30" fill="#ff9800" opacity="0.7"/>
    <text x="235" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 1</text>
    <rect x="5" y="155" width="140" height="30" fill="#2196f3" opacity="0.7"/>
    <text x="75" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 2</text>
    
    <text x="0" y="200" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#333">Clock 3:</text>
    <rect x="325" y="100" width="140" height="30" fill="#4caf50" opacity="0.7"/>
    <text x="395" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 1</text>
    <rect x="165" y="155" width="140" height="30" fill="#ff9800" opacity="0.7"/>
    <text x="235" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 2</text>
    <rect x="5" y="210" width="140" height="30" fill="#2196f3" opacity="0.7"/>
    <text x="75" y="230" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 3</text>
    
    <text x="0" y="255" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#333">Clock 4:</text>
    <rect x="485" y="100" width="140" height="30" fill="#e91e63" opacity="0.7"/>
    <text x="555" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 1</text>
    <rect x="325" y="155" width="140" height="30" fill="#4caf50" opacity="0.7"/>
    <text x="395" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 2</text>
    <rect x="165" y="210" width="140" height="30" fill="#ff9800" opacity="0.7"/>
    <text x="235" y="230" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 3</text>
    <rect x="5" y="265" width="140" height="30" fill="#2196f3" opacity="0.7"/>
    <text x="75" y="285" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Instr 4</text>
  </g>
  
  <text x="450" y="400" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#666">
    Figure 4.3: Multiple instructions execute simultaneously in different pipeline stages,
  </text>
  <text x="450" y="420" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#666">
    analogous to neural population codes where different neurons process different features.
  </text>
</svg>'''
    
    filepath = f"{base_path}/chapter04/fig_4_3_pipeline.svg"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"Generated: {filepath}")

# Add more illustration functions here and call them in main()
