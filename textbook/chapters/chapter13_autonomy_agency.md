# Chapter 13: Autonomy and Agency / 自主性与能动性

## Chapter Objectives / 本章目标

By the end of this chapter, you will be able to:
- Distinguish between autonomy and agency in AI systems
- Evaluate computational models of self-modeling
- Analyze goal-directed behavior in machines
- Assess the possibility of machine intentionality
- Design systems with increasing degrees of autonomy

在本章结束时，你将能够：
- 区分AI系统中的自主性和能动性
- 评估自我建模的计算模型
- 分析机器中的目标导向行为
- 评估机器意向性的可能性
- 设计具有递增自主性程度的系统

---

## 13.1 Self-Modeling and Self-Awareness / 自我建模与自我意识

### 13.1.1 The Self-Model / 自我模型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELF-MODELING ARCHITECTURE / 自我建模架构                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHAT IS A SELF-MODEL? / 什么是自我模型？                                   │
│                                                                             │
│  A representation that the system has of ITSELF                             │
│  系统对自己的表征                                                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  LAYERS OF SELF-MODELING / 自我建模层次                             │   │
│  │                                                                     │   │
│  │  Level 0: No self-model / 第0层：无自我模型                         │   │
│  │  • Simple reflexes / 简单反射                                       │   │
│  │  • Input → Output / 输入 → 输出                                     │   │
│  │  Example: Thermostat / 例子：恒温器                                 │   │
│  │                                                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Level 1: Proprioceptive self-model / 第1层：本体感觉自我模型        │   │
│  │  • Model of physical state / 身体状态的模型                         │   │
│  │  • Position, velocity, configuration                                │   │
│  │    位置、速度、构型                                                  │   │
│  │  Example: Robot with joint sensors / 例子：有关节传感器的机器人       │   │
│  │                                                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Level 2: Cognitive self-model / 第2层：认知自我模型                 │   │
│  │  • Model of mental states / 心理状态的模型                          │   │
│  │  • Beliefs, desires, intentions / 信念、欲望、意图                  │   │
│  │  • "I believe X", "I want Y" / "我相信X"、"我想要Y"                 │   │
│  │  Example: Theory of Mind AI / 例子：心理理论AI                      │   │
│  │                                                                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Level 3: Reflective self-model / 第3层：反思性自我模型              │   │
│  │  • Model of the self-model / 自我模型的模型                         │   │
│  │  • "I know that I believe X" / "我知道我相信X"                      │   │
│  │  • Meta-cognition / 元认知                                          │   │
│  │  Example: Human self-awareness / 例子：人类自我意识                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  AI PROGRESS / AI进展:                                                      │
│                                                                             │
│  Current systems / 当前系统: Mostly Level 1-2                               │
│  主要是第1-2层                                                              │
│                                                                             │
│  Challenge / 挑战: Level 3 requires true meta-cognition                     │
│  第3层需要真正的元认知                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 13.1.2 Self-Modeling in Practice / 实践中的自我建模

```python
"""
Self-Modeling AI System / 自我建模AI系统
Demonstrates different levels of self-awareness
演示不同层次的自我意识
"""

import numpy as np

class Level0_System:
    """No self-model / 无自我模型"""
    
    def act(self, perception):
        """Direct input-output mapping / 直接输入-输出映射"""
        if perception['temperature'] > 25:
            return 'turn_on_cooling'
        return 'do_nothing'


class Level1_System:
    """Physical self-model / 身体自我模型"""
    
    def __init__(self):
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.battery = 100.0
        
    def update_self_model(self, sensors):
        """Update internal state representation / 更新内部状态表征"""
        self.position = sensors['position']
        self.velocity = sensors['velocity']
        self.battery -= 0.1  # Energy consumption model
        
    def act(self, goal_position):
        """Use self-model for planning / 用自我模型规划"""
        # "Where am I?" / "我在哪里？"
        current_pos = self.position
        
        # "How do I get there?" / "我怎么去那里？"
        direction = goal_position - current_pos
        
        # Check resources / 检查资源
        if self.battery < 20:
            return 'find_charger'
        
        return direction


class Level2_System:
    """Cognitive self-model / 认知自我模型"""
    
    def __init__(self):
        self.beliefs = {}  # What I think is true / 我认为是真的
        self.goals = []    # What I want to achieve / 我想要实现的
        self.intentions = []  # What I'm planning to do / 我计划做的
        
    def update_beliefs(self, observations):
        """Update beliefs about world and self / 更新关于世界和自我的信念"""
        for obs in observations:
            self.beliefs[obs['subject']] = {
                'property': obs['property'],
                'confidence': obs['confidence'],
                'source': 'perception'
            }
    
    def model_own_state(self):
        """Meta-cognitive access to beliefs / 对信念的元认知访问"""
        return {
            'belief_count': len(self.beliefs),
            'certainty_levels': [b['confidence'] for b in self.beliefs.values()],
            'knowledge_gaps': self.identify_gaps()
        }
    
    def identify_gaps(self):
        """Recognize what I don't know / 认识到我不知道什么"""
        # This requires having a model of what SHOULD be known
        # 这需要有关于什么应该被知道的模型
        known_subjects = set(self.beliefs.keys())
        expected_subjects = {'self', 'environment', 'task', 'resources'}
        return expected_subjects - known_subjects
    
    def act(self):
        """Action based on self-model / 基于自我模型的行动"""
        # "What do I believe?" / "我相信什么？"
        beliefs = self.beliefs
        
        # "What do I want?" / "我想要什么？"
        goals = self.goals
        
        # "What should I do?" / "我应该做什么？"
        intention = self.plan(beliefs, goals)
        
        return intention
    
    def plan(self, beliefs, goals):
        """Generate plan based on beliefs and goals / 基于信念和目标生成计划"""
        # Requires theory of own capabilities
        # 需要关于自身能力的理论
        pass


class Level3_System:
    """Reflective self-model / 反思性自我模型"""
    
    def __init__(self):
        self.level2 = Level2_System()
        self.meta_beliefs = {}  # Beliefs about beliefs / 关于信念的信念
        
    def reflect(self):
        """Meta-cognition: thinking about thinking / 元认知：思考思考本身"""
        # "Do I trust my beliefs?" / "我相信我的信念吗？"
        for subject, belief in self.level2.beliefs.items():
            self.meta_beliefs[f"trust_{subject}"] = {
                'confidence_in_belief': self.evaluate_reliability(belief),
                'potential_biases': self.detect_biases(belief),
                'alternative_interpretations': self.generate_alternatives(belief)
            }
    
    def evaluate_reliability(self, belief):
        """Assess how much to trust a belief / 评估多信任一个信念"""
        factors = [
            belief['confidence'],
            self.check_consistency(belief),
            self.check_source_reliability(belief['source'])
        ]
        return np.mean(factors)
    
    def detect_biases(self, belief):
        """Recognize potential biases in reasoning / 识别推理中的潜在偏见"""
        # Requires sophisticated self-modeling
        # 需要复杂的自我建模
        return []
    
    def generate_alternatives(self, belief):
        """Generate alternative explanations / 生成替代解释"""
        # "Could I be wrong?" / "我会错吗？"
        return []


def demonstrate_self_modeling():
    """Show progression of self-modeling capabilities / 展示自我建模能力的进展"""
    
    print("SELF-MODELING PROGRESSION / 自我建模进展")
    print("="*60)
    
    # Level 0 / 第0层
    l0 = Level0_System()
    print("\nLevel 0: No self-model / 第0层：无自我模型")
    print("  Action: Based purely on input / 行动：纯基于输入")
    action = l0.act({'temperature': 30})
    print(f"  Example: temp=30°C → {action}")
    
    # Level 1 / 第1层
    l1 = Level1_System()
    print("\nLevel 1: Physical self-model / 第1层：身体自我模型")
    print("  Knows: position, velocity, battery / 知道：位置、速度、电量")
    l1.update_self_model({
        'position': np.array([1.0, 2.0]),
        'velocity': np.array([0.5, 0.0])
    })
    print(f"  Self-state: pos={l1.position}, battery={l1.battery:.1f}%")
    
    # Level 2 / 第2层
    l2 = Level2_System()
    print("\nLevel 2: Cognitive self-model / 第2层：认知自我模型")
    print("  Has: beliefs, goals, intentions / 有：信念、目标、意图")
    l2.update_beliefs([
        {'subject': 'task', 'property': 'urgent', 'confidence': 0.8}
    ])
    l2.goals = ['complete_task', 'conserve_energy']
    self_model = l2.model_own_state()
    print(f"  Self-knowledge: {self_model['belief_count']} beliefs")
    print(f"  Knowledge gaps: {self_model['knowledge_gaps']}")
    
    # Level 3 / 第3层
    l3 = Level3_System()
    print("\nLevel 3: Reflective self-model / 第3层：反思性自我模型")
    print("  Meta-cognitive monitoring / 元认知监控")
    print("  Questions own beliefs / 质疑自己的信念")
    
    print("\n" + "="*60)
    print("Consciousness likely requires Level 2-3 self-modeling")
    print("意识可能需要第2-3层自我建模")

if __name__ == "__main__":
    demonstrate_self_modeling()
```

---

## 13.2 Goal-Directed Behavior / 目标导向行为

### 13.2.1 From Reflex to Goal / 从反射到目标

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TYPES OF BEHAVIOR / 行为类型                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REFLEX / 反射                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Stimulus ──► Response                                              │   │
│  │  刺激      ──► 反应                                                 │   │
│  │                                                                     │   │
│  │  Example: Knee-jerk reflex / 例子：膝跳反射                         │   │
│  │  Tap knee ──► Leg kicks                                             │   │
│  │  敲击膝盖 ──► 腿踢出                                                 │   │
│  │                                                                     │   │
│  │  No goal, no model / 无目标，无模型                                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  HABIT / 习惯                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Stimulus ──► [Learned association] ──► Response                    │   │
│  │  刺激      ──► [习得的关联]          ──► 反应                       │   │
│  │                                                                     │   │
│  │  Example: Brushing teeth / 例子：刷牙                                 │   │
│  │  Wake up ──► Go to bathroom ──► Brush teeth                         │   │
│  │  醒来    ──► 去浴室        ──► 刷牙                                  │   │
│  │                                                                     │   │
│  │  Learned, but automatic / 习得的，但自动的                            │   │
│  │  No explicit goal representation / 无显式目标表征                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  GOAL-DIRECTED / 目标导向                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌─────────┐                                                        │   │
│  │  │  GOAL   │  ← Explicit representation of desired state            │   │
│  │  │  目标   │    期望状态的显式表征                                   │   │
│  │  └────┬────┘                                                        │   │
│  │       │ Compare with current state / 与当前状态比较                  │   │
│  │       ▼                                                             │   │
│  │  ┌─────────┐                                                        │   │
│  │  │  PLAN   │  ← Means-end reasoning / 手段-目的推理                  │   │
│  │  │  计划   │                                                        │   │
│  │  └────┬────┘                                                        │   │
│  │       │ Execute / 执行                                               │   │
│  │       ▼                                                             │   │
│  │  ┌─────────┐                                                        │   │
│  │  │ ACTION  │                                                        │   │
│  │  │  行动   │                                                        │   │
│  │  └────┬────┘                                                        │   │
│  │       │ Monitor outcome / 监控结果                                   │   │
│  │       └────────► Adjust if needed / 如需要则调整                     │   │
│  │                                                                     │   │
│  │  Example: Planning a trip / 例子：规划旅行                            │   │
│  │  Goal: Visit Paris / 目标：访问巴黎                                  │   │
│  │  Plan: Book flight → Hotel → Itinerary                              │   │
│  │  计划：订机票   → 酒店  → 行程                                       │   │
│  │                                                                     │   │
│  │  Flexible, can adapt to obstacles / 灵活，可适应障碍                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Chapter Summary / 本章总结

**Key Points / 要点**:

1. **Self-modeling**: Systems can have representations of themselves at multiple levels.
   **自我建模**：系统可以在多个层次上拥有对自己的表征。

2. **Autonomy**: Self-directed behavior based on internal models rather than external control.
   **自主性**：基于内部模型而非外部控制的自我导向行为。

3. **Agency**: Capacity to act with goals and intentions.
   **能动性**：以目标和意图行动的能力。

4. **AI systems**: Currently limited to Level 1-2 self-modeling; Level 3 (meta-cognition) remains challenging.
   **AI系统**：目前限于第1-2层自我建模；第3层（元认知）仍具挑战性。

**Key Terms / 关键术语**:
- Self-model / 自我模型
- Meta-cognition / 元认知
- Autonomy / 自主性
- Agency / 能动性
- Goal-directed behavior / 目标导向行为
- Intentionality / 意向性

---

*Next Chapter: Chapter 14 - Subjective Experience / 下一章：第14章 - 主观体验*
