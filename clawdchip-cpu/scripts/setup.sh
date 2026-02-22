#!/usr/bin/env python3
"""
OpenClaw CPU Project Setup Script
一键安装依赖和环境配置
"""

import os
import sys
import subprocess

print("=" * 60)
print("OpenClaw CPU Project Setup")
print("=" * 60)

# 检查Python版本
print("\n[1/5] 检查Python版本...")
version = sys.version_info
if version.major < 3 or (version.major == 3 and version.minor < 10):
    print("❌ 需要Python 3.10或更高版本")
    sys.exit(1)
print(f"✅ Python {version.major}.{version.minor}.{version.micro}")

# 安装Python依赖
print("\n[2/5] 安装Python依赖...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# 检查系统依赖
print("\n[3/5] 检查系统依赖...")
deps = ["verilator", "make", "gcc", "g++"]
for dep in deps:
    result = subprocess.run(["which", dep], capture_output=True)
    if result.returncode == 0:
        print(f"✅ {dep} 已安装")
    else:
        print(f"⚠️  {dep} 未安装，请手动安装")

# 创建必要的目录
print("\n[4/5] 创建项目目录...")
os.makedirs("build", exist_ok=True)
os.makedirs("logs", exist_ok=True)
print("✅ 目录创建完成")

# 运行测试
print("\n[5/5] 运行基础测试...")
result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                       capture_output=True)
if result.returncode == 0:
    print("✅ 测试通过")
else:
    print("⚠️  部分测试失败，请检查环境")

print("\n" + "=" * 60)
print("🎉 安装完成！")
print("=" * 60)
print("\n下一步:")
print("  1. 运行模拟器: ./run_simulator.sh")
print("  2. 查看文档: docs/architecture/overview.md")
print("  3. 加入社区: https://discord.gg/openclaw")
