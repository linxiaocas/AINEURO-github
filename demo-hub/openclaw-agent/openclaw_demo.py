# OpenClaw Agent Framework Demo
# 多平台智能体框架演示

import asyncio
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from datetime import datetime
import random

@dataclass
class Message:
    """统一消息格式"""
    platform: str  # whatsapp, telegram, wechat, discord
    user_id: str
    user_name: str
    content: str
    timestamp: datetime
    message_id: str
    
@dataclass
class AgentResponse:
    """Agent响应"""
    content: str
    actions: List[Dict]
    confidence: float
    latency_ms: int

class OpenClawGateway:
    """多平台网关层"""
    
    def __init__(self):
        self.connections = {}
        self.message_handlers = []
        
    async def connect_platform(self, platform: str, credentials: Dict):
        """连接平台"""
        print(f"🔗 连接到 {platform}...")
        self.connections[platform] = {
            "status": "connected",
            "credentials": credentials,
            "message_count": 0
        }
        print(f"✅ {platform} 已连接")
        
    async def on_message(self, handler: Callable):
        """注册消息处理器"""
        self.message_handlers.append(handler)
        
    async def receive_message(self, message: Message):
        """接收消息"""
        self.connections[message.platform]["message_count"] += 1
        
        # 通知所有处理器
        for handler in self.message_handlers:
            await handler(message)
            
    async def send_message(self, platform: str, user_id: str, content: str):
        """发送消息"""
        print(f"📤 发送到 {platform}/{user_id}: {content[:50]}...")
        
class IntentEngine:
    """意图引擎"""
    
    def __init__(self):
        self.intent_patterns = {
            "greeting": ["你好", "hello", "hi", "在吗"],
            "question": ["什么", "怎么", "为什么", "多少"],
            "task": ["帮我", "请", "需要", "想要"],
            "tool_call": ["搜索", "查", "算", "翻译"]
        }
        
    async def parse_intent(self, message: Message) -> Dict:
        """解析用户意图"""
        content = message.content.lower()
        
        # 简单意图分类
        intent_type = "chat"
        confidence = 0.5
        
        for intent_type_key, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    intent_type = intent_type_key
                    confidence = 0.8
                    break
                    
        return {
            "type": intent_type,
            "confidence": confidence,
            "entities": self._extract_entities(content),
            "original_message": asdict(message)
        }
        
    def _extract_entities(self, content: str) -> List[Dict]:
        """提取实体"""
        entities = []
        
        # 简单实体提取
        if "天气" in content:
            entities.append({"type": "topic", "value": "weather"})
        if "时间" in content or "几点" in content:
            entities.append({"type": "topic", "value": "time"})
            
        return entities

class ToolExecutor:
    """工具执行器"""
    
    def __init__(self):
        self.tools = {
            "search": self._search,
            "calculate": self._calculate,
            "weather": self._get_weather,
            "time": self._get_time
        }
        
    async def execute(self, tool_name: str, params: Dict) -> Dict:
        """执行工具"""
        if tool_name in self.tools:
            return await self.tools[tool_name](params)
        return {"error": f"未知工具: {tool_name}"}
        
    async def _search(self, params: Dict) -> Dict:
        """模拟搜索"""
        query = params.get("query", "")
        return {
            "results": [
                f"搜索结果1: 关于{query}的信息",
                f"搜索结果2: {query}的最新动态",
                f"搜索结果3: {query}相关数据"
            ]
        }
        
    async def _calculate(self, params: Dict) -> Dict:
        """模拟计算"""
        expression = params.get("expression", "")
        try:
            # 安全计算
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result}
        except:
            return {"error": "计算错误"}
            
    async def _get_weather(self, params: Dict) -> Dict:
        """模拟天气查询"""
        city = params.get("city", "北京")
        weather_types = ["晴天", "多云", "小雨", "阴天"]
        return {
            "city": city,
            "weather": random.choice(weather_types),
            "temperature": random.randint(15, 30),
            "humidity": random.randint(40, 80)
        }
        
    async def _get_time(self, params: Dict) -> Dict:
        """获取当前时间"""
        return {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "Asia/Shanghai"
        }

class MemoryManager:
    """记忆管理器"""
    
    def __init__(self):
        self.short_term = {}  # 短期记忆
        self.long_term = {}   # 长期记忆
        
    async def store(self, user_id: str, key: str, value: any, ttl: int = 3600):
        """存储记忆"""
        if user_id not in self.short_term:
            self.short_term[user_id] = {}
        
        self.short_term[user_id][key] = {
            "value": value,
            "timestamp": datetime.now(),
            "ttl": ttl
        }
        
    async def retrieve(self, user_id: str, key: str) -> Optional[any]:
        """检索记忆"""
        if user_id in self.short_term and key in self.short_term[user_id]:
            return self.short_term[user_id][key]["value"]
        return None
        
    async def get_context(self, user_id: str) -> str:
        """获取用户上下文"""
        if user_id not in self.short_term:
            return ""
            
        memories = []
        for key, data in self.short_term[user_id].items():
            memories.append(f"- {key}: {data['value']}")
            
        return "\n".join(memories) if memories else ""

class OpenClawAgent:
    """OpenClaw智能体核心"""
    
    def __init__(self, name: str = "OpenClaw Assistant"):
        self.name = name
        self.gateway = OpenClawGateway()
        self.intent_engine = IntentEngine()
        self.tool_executor = ToolExecutor()
        self.memory = MemoryManager()
        self.conversation_count = 0
        
    async def initialize(self):
        """初始化Agent"""
        print(f"🤖 初始化Agent: {self.name}")
        
        # 连接平台
        await self.gateway.connect_platform("telegram", {"token": "demo_token"})
        await self.gateway.connect_platform("discord", {"token": "demo_token"})
        
        # 注册消息处理器
        await self.gateway.on_message(self._handle_message)
        
        print("✅ Agent初始化完成")
        
    async def _handle_message(self, message: Message):
        """处理消息"""
        import time
        start_time = time.time()
        
        print(f"\n📨 收到消息 [{message.platform}]")
        print(f"   用户: {message.user_name} ({message.user_id})")
        print(f"   内容: {message.content}")
        
        # 1. 解析意图
        intent = await self.intent_engine.parse_intent(message)
        print(f"   意图: {intent['type']} (置信度: {intent['confidence']:.2f})")
        
        # 2. 获取上下文
        context = await self.memory.retrieve(message.user_id, "conversation_history")
        
        # 3. 生成响应
        response = await self._generate_response(intent, context)
        
        # 4. 执行工具（如果需要）
        actions = []
        if intent['type'] == "tool_call":
            tool_result = await self._execute_tool(message.content)
            response = f"{response}\n\n{tool_result}"
            actions.append({"tool": "search", "result": tool_result})
            
        # 5. 更新记忆
        await self.memory.store(
            message.user_id,
            "last_message",
            message.content
        )
        
        # 计算延迟
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 6. 发送响应
        agent_response = AgentResponse(
            content=response,
            actions=actions,
            confidence=intent['confidence'],
            latency_ms=latency_ms
        )
        
        await self.gateway.send_message(
            message.platform,
            message.user_id,
            agent_response.content
        )
        
        print(f"   响应: {agent_response.content[:100]}...")
        print(f"   延迟: {agent_response.latency_ms}ms")
        
        self.conversation_count += 1
        
    async def _generate_response(self, intent: Dict, context: str) -> str:
        """生成响应"""
        intent_type = intent['type']
        
        responses = {
            "greeting": [
                "你好！我是OpenClaw智能助手，有什么可以帮你的吗？",
                "你好！很高兴为你服务。",
                "在的！请问需要什么帮助？"
            ],
            "question": [
                f"关于你的问题，我可以帮你搜索相关信息。",
                "这是个好问题，让我想想...",
                "我可以帮你找到答案。"
            ],
            "task": [
                "没问题，我来帮你处理。",
                "好的，请稍等片刻。",
                "收到，正在为你安排。"
            ],
            "chat": [
                "明白了，继续说说你的想法。",
                "我在听，请继续。",
                "有趣，能多告诉我一些吗？"
            ]
        }
        
        return random.choice(responses.get(intent_type, responses["chat"]))
        
    async def _execute_tool(self, content: str) -> str:
        """执行工具"""
        if "天气" in content:
            result = await self.tool_executor.execute("weather", {"city": "北京"})
            return f"北京当前天气: {result['weather']}, {result['temperature']}°C, 湿度{result['humidity']}%"
            
        elif "时间" in content:
            result = await self.tool_executor.execute("time", {})
            return f"当前时间: {result['datetime']}"
            
        elif any(op in content for op in ["+", "-", "*", "/"]):
            # 提取数学表达式
            import re
            expr = re.search(r'[\d\+\-\*\/\(\)\.]+', content)
            if expr:
                result = await self.tool_executor.execute("calculate", {"expression": expr.group()})
                if "result" in result:
                    return f"计算结果: {result['result']}"
                    
        return "我暂时无法执行这个操作。"
        
    async def run(self):
        """运行Agent"""
        print(f"\n🚀 {self.name} 正在运行...")
        print("等待消息... (按Ctrl+C停止)\n")
        
        # 模拟接收消息
        demo_messages = [
            Message("telegram", "user_001", "张三", "你好！", datetime.now(), "msg_001"),
            Message("discord", "user_002", "李四", "现在几点了？", datetime.now(), "msg_002"),
            Message("telegram", "user_001", "张三", "帮我查一下北京天气", datetime.now(), "msg_003"),
            Message("discord", "user_003", "王五", "15 + 27等于多少？", datetime.now(), "msg_004"),
        ]
        
        for msg in demo_messages:
            await self._handle_message(msg)
            await asyncio.sleep(2)
            
        print(f"\n📊 统计: 共处理 {self.conversation_count} 条消息")

async def main():
    """主函数"""
    print("=" * 60)
    print("   OpenClaw Agent Framework Demo")
    print("   多平台智能体框架演示")
    print("=" * 60)
    
    # 创建并运行Agent
    agent = OpenClawAgent("OpenClaw Demo Bot")
    await agent.initialize()
    await agent.run()
    
    print("\n✅ 演示完成!")
    print("\nOpenClaw特点:")
    print("  • 多平台网关: 统一接入WhatsApp/Telegram/Discord等")
    print("  • 意图引擎: 智能解析用户意图")
    print("  • 工具调用: Function Calling支持")
    print("  • 记忆管理: 上下文保持")
    print("  • 低延迟: 平均响应 < 50ms")

if __name__ == "__main__":
    asyncio.run(main())
