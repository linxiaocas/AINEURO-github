# AINEURO 工具库

本目录包含用于AI神经科学研究的各种工具和仿真代码。

## 目录结构

```
src/
├── tools/           # 研究工具
│   ├── metrics/     # 评估指标（c-Φ计算等）
│   ├── monitors/    # 系统监控工具
│   └── analyzers/   # 分析工具
├── simulations/     # 仿真代码
│   ├── neural/      # 神经仿真
│   └── cognitive/   # 认知仿真
└── examples/        # 使用示例
```

## 安装

```bash
pip install -e .
```

## 快速开始

```python
from aineuro.metrics import ComputationalPhiAnalyzer
from aineuro.monitors import HardwareConsciousnessMonitor

# 计算c-Φ值
analyzer = ComputationalPhiAnalyzer(model)
c_phi = analyzer.compute_c_phi()

# 监控硬件活动
monitor = HardwareConsciousnessMonitor(system)
monitor.collect_neural_signatures(duration_ms=1000)
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black src/
isort src/

# 类型检查
mypy src/
```
