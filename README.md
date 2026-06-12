# 模仿学习的数学原理

一本面向机器人模仿学习的数学原理教程。

本书不把模仿学习写成算法名词列表，而是沿着一条主线展开：

```text
专家轨迹
  ↓
Behavior Cloning
  ↓
分布偏移与 DAgger
  ↓
动作分布、多峰动作与 CVAE
  ↓
IRL / GAIL / Offline IL
  ↓
ACT / Diffusion Policy / Flow Matching
  ↓
Transformer Policy / VLA
  ↓
World Model / WAM / JEPA
```

全书统一贯穿例子是：**二维桌面机器人推块入槽**。它用于帮助读者把状态、动作、轨迹、分布偏移、动作多峰、离线覆盖、VLA 和世界预测放到同一个任务中理解。

## 在线阅读

电子书地址：

https://fengkai11.github.io/Mathematical-principles-of-imitation-learning/

## 内容结构

- 正文：6 篇 18 章，覆盖模仿学习基础、分布偏移、动作分布、隐变量策略、分布匹配、生成式动作策略、长上下文与多模态策略、世界模型与动作后果预测。
- 附录：数学符号、概率论、最大似然与 KL、连续动作回归、优化基础、强化学习、生成模型、实验与代码基础。
- 主线骨架：`全书导读_新版布局说明.md`、`全书贯穿例子_推块入槽.md`、`全书核心命题链.md`、`全书公式地图.md`。
- 存档：原第7、8篇工程扩展内容已从正书剥离，存档索引见 `模仿学习的数学原理_工程扩展版_第1-29章含附录/archive/剥离内容_第七第八篇/README.md`。
- 配图：章节配图统一放在 `模仿学习的数学原理_工程扩展版_第1-29章含附录/images/`。

## 版权声明

Copyright (c) 2026 冯凯。保留所有权利。

本书由冯凯发起并确定整体框架，正文在该框架下使用 GPT 辅助书写，并使用 Gemini 3.5 Flash 与 Opus 4.6 辅助校对。主题选择、结构设计、内容审定、发布维护与最终责任由冯凯承担。

完整说明见 mdBook 中的“版权声明与 AI 协作说明”。

## 本地构建

本书使用 mdBook 构建。仓库根目录执行：

```bash
mdbook build
```

构建输出目录为：

```text
book/
```

## 发布

GitHub Pages 通过 GitHub Actions 自动发布。每次 push 到 `main` 或 `master` 分支会触发：

```text
.github/workflows/deploy.yml
```