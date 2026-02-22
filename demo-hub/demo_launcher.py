#!/usr/bin/env python3
"""
AINEURO Demo Hub Launcher
一键启动所有前沿AI技术演示
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import webbrowser
import time

DEMO_LIST = {
    "ai-pc": {
        "name": "AI PC Framework",
        "path": "ai-pc",
        "command": "npm run dev",
        "port": 3000,
        "description": "Tauri + Vue跨平台桌面应用"
    },
    "ai-gaming": {
        "name": "AI Gaming Engine",
        "path": "ai-gaming",
        "command": "python game_simulator.py",
        "port": 8081,
        "description": "Blaze AI游戏引擎演示"
    },
    "ai-streaming": {
        "name": "AI Video Streaming",
        "path": "ai-streaming",
        "command": "python streaming_server.py",
        "port": 8082,
        "description": "低延迟AI视频直播"
    },
    "ai-world-model": {
        "name": "AI World Model",
        "path": "ai-world-model",
        "command": "python world_simulator.py",
        "port": 8083,
        "description": "物理世界模拟与预测"
    },
    "autonomous-driving": {
        "name": "Autonomous Driving",
        "path": "autonomous-driving",
        "command": "python apollo_simulator.py",
        "port": 8084,
        "description": "Apollo自动驾驶模拟"
    },
    "humanoid-robot": {
        "name": "Humanoid Robot",
        "path": "humanoid-robot",
        "command": "python robot_controller.py",
        "port": 8085,
        "description": "人形机器人控制"
    },
    "marble-world": {
        "name": "Marble World Model",
        "path": "marble-world",
        "command": "python marble_generator.py",
        "port": 8086,
        "description": "李飞飞3D世界生成"
    },
    "lecun-jepa": {
        "name": "LeCun JEPA",
        "path": "lecun-jepa",
        "command": "python jepa_demo.py",
        "port": 8087,
        "description": "Yann LeCun世界模型"
    },
    "openclaw-agent": {
        "name": "OpenClaw Agent",
        "path": "openclaw-agent",
        "command": "python openclaw_demo.py",
        "port": 8088,
        "description": "多平台智能体框架"
    },
    "ai-for-science": {
        "name": "AI for Science",
        "path": "ai-for-science",
        "command": "jupyter notebook",
        "port": 8888,
        "description": "科学计算AI"
    },
    "bio-medical": {
        "name": "Bio-Medical AI",
        "path": "bio-medical",
        "command": "python medical_ai_demo.py",
        "port": 8089,
        "description": "医疗AI诊断"
    },
    "space-starlink": {
        "name": "Space & Starlink",
        "path": "space-starlink",
        "command": "python satellite_simulator.py",
        "port": 8090,
        "description": "卫星星座模拟"
    },
    "brain-computer": {
        "name": "Brain-Computer Interface",
        "path": "brain-computer",
        "command": "python bci_demo.py",
        "port": 8091,
        "description": "脑机接口演示"
    }
}

class DemoLauncher:
    def __init__(self):
        self.processes = {}
        self.base_path = Path(__file__).parent
        
    def print_banner(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           🚀 AINEURO Demo Hub - 前沿AI技术演示中心 🚀        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
    def list_demos(self):
        """列出所有可用演示"""
        print("📋 可用演示列表:")
        print("-" * 60)
        for idx, (key, demo) in enumerate(DEMO_LIST.items(), 1):
            status = "✅" if self._check_demo_exists(key) else "❌"
            print(f"{status} {idx:2d}. {demo['name']:<25} - {demo['description']}")
        print("-" * 60)
        
    def _check_demo_exists(self, demo_key):
        """检查演示是否存在"""
        demo_path = self.base_path / DEMO_LIST[demo_key]['path']
        return demo_path.exists()
        
    def launch_demo(self, demo_key):
        """启动单个演示"""
        if demo_key not in DEMO_LIST:
            print(f"❌ 未知演示: {demo_key}")
            return
            
        demo = DEMO_LIST[demo_key]
        demo_path = self.base_path / demo['path']
        
        if not demo_path.exists():
            print(f"❌ 演示目录不存在: {demo_path}")
            return
            
        print(f"🚀 启动演示: {demo['name']}")
        print(f"   路径: {demo_path}")
        print(f"   命令: {demo['command']}")
        print(f"   端口: {demo['port']}")
        
        try:
            # 切换到演示目录并启动
            os.chdir(demo_path)
            process = subprocess.Popen(
                demo['command'],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes[demo_key] = process
            
            # 等待服务启动
            time.sleep(2)
            
            print(f"✅ {demo['name']} 已启动!")
            print(f"   访问: http://localhost:{demo['port']}")
            print()
            
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            
    def launch_all(self):
        """启动所有演示"""
        print("🚀 启动所有演示...")
        print()
        
        for demo_key in DEMO_LIST:
            if self._check_demo_exists(demo_key):
                self.launch_demo(demo_key)
                time.sleep(1)
            else:
                print(f"⚠️  跳过 {demo_key} (目录不存在)")
                
        print("✅ 所有演示已启动!")
        print(f"📊 监控面板: http://localhost:8080")
        
    def stop_all(self):
        """停止所有演示"""
        print("🛑 停止所有演示...")
        for demo_key, process in self.processes.items():
            try:
                process.terminate()
                print(f"✅ 已停止: {DEMO_LIST[demo_key]['name']}")
            except:
                pass
        print("✅ 所有演示已停止")
        
    def launch_web_dashboard(self):
        """启动Web监控面板"""
        print("🌐 启动Web监控面板...")
        dashboard_path = self.base_path / "dashboard"
        if dashboard_path.exists():
            os.chdir(dashboard_path)
            subprocess.Popen(["python", "-m", "http.server", "8080"])
            time.sleep(1)
            webbrowser.open("http://localhost:8080")
        else:
            print("⚠️  监控面板未找到，使用基础HTTP服务器")
            os.chdir(self.base_path)
            subprocess.Popen(["python", "-m", "http.server", "8080"])
            
def main():
    parser = argparse.ArgumentParser(description='AINEURO Demo Hub Launcher')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有演示')
    parser.add_argument('--launch', '-n', type=str, help='启动指定演示')
    parser.add_argument('--all', '-a', action='store_true', help='启动所有演示')
    parser.add_argument('--web', '-w', action='store_true', help='启动Web监控面板')
    parser.add_argument('--stop', '-s', action='store_true', help='停止所有演示')
    
    args = parser.parse_args()
    
    launcher = DemoLauncher()
    launcher.print_banner()
    
    if args.list:
        launcher.list_demos()
    elif args.launch:
        launcher.launch_demo(args.launch)
        input("按回车键停止演示...")
        launcher.stop_all()
    elif args.all:
        launcher.launch_all()
        input("按回车键停止所有演示...")
        launcher.stop_all()
    elif args.web:
        launcher.launch_web_dashboard()
        input("按回车键停止...")
    elif args.stop:
        launcher.stop_all()
    else:
        launcher.list_demos()
        print("\n使用方法:")
        print("  python demo_launcher.py --list          # 列出所有演示")
        print("  python demo_launcher.py --launch ai-pc  # 启动指定演示")
        print("  python demo_launcher.py --all           # 启动所有演示")
        print("  python demo_launcher.py --web           # 启动Web监控面板")

if __name__ == "__main__":
    main()
