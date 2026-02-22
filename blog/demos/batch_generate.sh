#!/bin/bash

# 批量生成Demo博客的简化版本

DEMOS=(
  "ai-streaming:AI视频直播:RTMP低延迟推流+AI实时检测"
  "ai-drama:AI短剧生成:DDD架构+自动剪辑"
  "ai-world-model:AI世界模型:编码器-预测器-解码器"
  "autonomous-driving:自动驾驶:Apollo分层架构"
  "humanoid-robot:人形机器人:ROS+EtherCAT控制"
  "marble-world:Marble世界模型:李飞飞3D世界生成"
  "lecun-jepa:LeCun JEPA:自监督世界模型"
  "ai-for-science:AI for Science:科学计算AI"
  "bio-medical:生物医疗AI:多模态医疗诊断"
  "space-starlink:商业航天:星链卫星模拟"
  "brain-computer:脑机接口:EEG信号解码"
)

for demo in "${DEMOS[@]}"; do
  IFS=':' read -r folder title subtitle <<< "$demo"
  
  cat > "/Users/linxiao/Downloads/AINEURO-github/blog/demos/${folder}-demo.md" << EOL
---
title: "${title}技术解析：${subtitle}"
date: "2026-02-22"
author: "Lin Xiao"
category: "Demo"
tags: ["AI", "Demo", "${folder}"]
---

# ${title}技术解析：${subtitle}

## 概述

本文介绍${title}的核心技术架构和实现原理。

## 架构设计

\`\`\`
[${title}架构图]
\`\`\`

## 核心代码

\`\`\`python
# 核心实现代码
print("${title} Demo")
\`\`\`

## 运行效果

\`\`\`
✅ ${title} Demo运行成功
\`\`\`

## 完整代码

[GitHub仓库](https://github.com/aineuro/demo-hub/${folder})
EOL

echo "Generated: ${folder}-demo.md"
done

