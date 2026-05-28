# 模仿学习的数学原理

一本面向机器人模仿学习的数学原理教程，从 Behavior Cloning、DAgger、MDP 到 ACT、Diffusion Policy、Decision Transformer、VLA 与工程落地。

## 在线阅读

电子书地址：

https://fengkai11.github.io/Mathematical-principles-of-imitation-learning/

## 内容结构

- 正文：第 1-20 章，覆盖模仿学习基础、分布偏移、序列决策、隐变量策略、生成式策略、Transformer 策略与实机部署。
- 附录：数学符号、概率论、最大似然与 KL、连续动作回归、优化基础、强化学习、生成模型、实验与代码基础。
- 配图：章节配图统一放在 `模仿学习的数学原理_终稿校验整理版/images/`。

## 本地构建

本书使用 mdBook 构建。仓库根目录执行：

```bash
mdbook build
```

如果本机的 `mdbook` 不在 `PATH` 中，也可以使用：

```bash
/home/fengkai/.cargo/bin/mdbook build
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

详细说明见：

```text
GITHUB_PAGES_SETUP.md
```
