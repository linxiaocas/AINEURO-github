# examples/custom_agent.py
from clawdchip import Agent, IntentEngine

class MyPersonalAgent(Agent):
    """
    示例：自定义个人Agent
    展示如何在ClawdChip CPU上运行自定义Agent
    """
    
    def __init__(self):
        super().__init__(name="MyAssistant")
        self.skills = ["coding", "writing", "research"]
        self.intent_engine = IntentEngine()
    
    async def handle_request(self, user_input: str):
        """处理用户请求"""
        # 1. 解析意图
        intent = await self.understand_intent(user_input)
        print(f"🎯 识别意图: {intent}")
        
        # 2. 直接生成硬件配置
        hw_config = self.compile_intent_to_hardware(intent)
        print(f"⚙️  生成硬件配置")
        
        # 3. 在ClawdChip上执行
        result = await self.clawdchip_execute(hw_config)
        print(f"✅ 执行完成")
        
        return result
    
    def compile_intent_to_hardware(self, intent):
        """将意图编译为硬件配置"""
        return {
            "decoder_config": "32_way",
            "memory_tier": intent.get("memory_requirement", "ddr"),
            "accelerator": intent.get("accelerator", "none"),
            "parallelism": intent.get("parallelism", 1)
        }

# 启动Agent
if __name__ == "__main__":
    import asyncio
    
    agent = MyPersonalAgent()
    
    # 示例对话
    async def demo():
        print("🤖 ClawdChip Agent Demo\n")
        
        queries = [
            "帮我写一段Python代码",
            "分析这个数据集",
            "生成一张图片"
        ]
        
        for query in queries:
            print(f"👤 用户: {query}")
            result = await agent.handle_request(query)
            print(f"🤖 Agent: 任务已完成\n")
    
    asyncio.run(demo())
