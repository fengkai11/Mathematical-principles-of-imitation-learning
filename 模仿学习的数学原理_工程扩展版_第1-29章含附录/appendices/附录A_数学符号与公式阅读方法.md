# 附录 A 数学符号与公式阅读方法

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录的任务很朴素：让读者看到公式时，不再第一反应是“这是什么古代咒语”。公式不是用来吓人的，它只是把一段很长的话压缩成了几行符号。真正困难的不是符号本身，而是我们不知道应该从哪里开始读。

---

## A.1 为什么机器学习公式看起来容易吓人

很多算法工程师并不是完全不懂数学，而是被公式的“排场”吓住了。

例如行为克隆的目标可以写成：

<div class="math">\[
\mathcal{L}_{\mathrm{BC}}(\theta)
=
-\mathbb{E}_{(o,a)\sim\mathcal{D}}
[
\log \pi_\theta(a|o)
] \tag{A.1}\]</div>

如果你第一次看到它，可能会同时被几件事攻击：

- <span class="math">\\(\mathcal{L}\\)</span> 是什么？
- <span class="math">\\(\theta\\)</span> 是什么？
- 为什么有个负号？
- <span class="math">\\(\mathbb{E}\\)</span> 为什么突然出现？
- <span class="math">\\((o,a)\sim\mathcal{D}\\)</span> 又是什么？
- <span class="math">\\(\pi\_\theta(a|o)\\)</span> 为什么像概率？
- <span class="math">\\(\log\\)</span> 为什么也来凑热闹？

于是大脑很自然地选择了一个策略：关闭页面，假装自己没看见。

但这个公式其实只是在说一句很简单的话：

> 从数据集中拿出观测和专家动作，希望模型在这个观测下给专家动作分配更高概率。训练时最小化“专家动作概率太低”带来的惩罚。

所以读公式的关键不是一上来就试图“证明”它，而是先把它翻译成人话。

---

## A.2 读公式的五步法

本书推荐读任何机器学习公式时都按下面五步走：

```text
第 1 步：找变量        谁是输入？谁是输出？谁是参数？
第 2 步：找分布        样本从哪里来？随机性在哪里？
第 3 步：找函数        模型、loss、reward、value 分别是什么？
第 4 步：找聚合        是求和、积分、期望，还是对时间累加？
第 5 步：找优化方向    是最大化，还是最小化？在调谁？
```

这套方法听起来像废话，但非常有用。很多公式看起来复杂，是因为它同时把“数据从哪里来”“模型怎么输出”“损失怎么算”“参数怎么调”全塞到一起。你要做的不是硬吞，而是拆开。

---

## A.3 第一步：找变量

以公式 A.1 为例：

<div class="math">\[
\mathcal{L}_{\mathrm{BC}}(\theta)
=
-\mathbb{E}_{(o,a)\sim\mathcal{D}}
[
\log \pi_\theta(a|o)
] \tag{A.2}\]</div>

变量可以分成三类。

第一类是数据变量：

- <span class="math">\\(o\\)</span>：observation，观测，可以是图像、点云、关节状态、车辆状态；
- <span class="math">\\(a\\)</span>：action，动作，可以是机械臂末端位姿增量、关节速度、方向盘角度、油门刹车；
- <span class="math">\\((o,a)\\)</span>：一条监督学习样本。

第二类是模型变量：

- <span class="math">\\(\pi\_\theta\\)</span>：带参数的策略模型；
- <span class="math">\\(\theta\\)</span>：神经网络参数，也就是训练要更新的一堆权重。

第三类是目标变量：

- <span class="math">\\(\mathcal{L}\_{\mathrm{BC}}(\theta)\\)</span>：行为克隆损失，数值越小越好。

读到这里，公式已经没那么吓人了。它不是在召唤数学恶魔，只是在描述一个训练目标。

---

## A.4 第二步：找分布

机器学习公式中最容易被忽略的部分，往往是“样本从哪里来”。

在公式 A.1 里：

<div class="math">\[
(o,a)\sim\mathcal{D} \tag{A.3}\]</div>

这句话的意思是：

> 从专家演示数据集 <span class="math">\\(\mathcal{D}\\)</span> 中抽取一条观测—动作样本。

它不是说真实世界天然就按照 <span class="math">\\(\mathcal{D}\\)</span> 分布运行，而是说训练时我们只能看到数据集里的样本。

这点非常关键。第 3 章讲分布偏移时，我们反复强调：训练时样本来自专家数据分布，执行时样本来自模型自己造成的状态分布。二者未必一样。

训练时：

<div class="math">\[
s\sim d^{\pi_E}(s) \tag{A.4}\]</div>

执行时：

<div class="math">\[
s\sim d^{\pi_\theta}(s) \tag{A.5}\]</div>

如果你看到一个公式里出现 <span class="math">\\(x\sim p(x)\\)</span>，就要立刻问：

```text
这个 x 是从哪个分布采样来的？
这个分布是专家数据、模型 rollout、真实环境，还是人为设定的 prior？
训练和测试时这个分布会不会变？
```

很多机器人学习问题，表面上是模型结构问题，底层其实是分布问题。

---

## A.5 第三步：找函数

公式里常见的函数有几类。

### A.5.1 策略函数

策略函数通常写作：

<div class="math">\[
\pi_\theta(a|o) \tag{A.6}\]</div>

它表示：给定观测 <span class="math">\\(o\\)</span>，策略给动作 <span class="math">\\(a\\)</span> 的概率。

如果动作是连续的，也可以写成：

<div class="math">\[
a = f_\theta(o) \tag{A.7}\]</div>

这时策略像一个普通回归模型，输入观测，输出动作。

### A.5.2 损失函数

损失函数通常写作：

<div class="math">\[
\mathcal{L}(\theta) \tag{A.8}\]</div>

它表示：当前参数 <span class="math">\\(\theta\\)</span> 的模型有多糟糕。

训练就是不断降低这个数：

<div class="math">\[
\theta^* = \arg\min_\theta \mathcal{L}(\theta) \tag{A.9}\]</div>

这里 <span class="math">\\(\arg\min\\)</span> 的意思不是“最小值是多少”，而是“让函数最小的参数是谁”。

### A.5.3 奖励函数

强化学习和 IRL 中常见：

<div class="math">\[
r = R(s,a) \tag{A.10}\]</div>

它表示：在状态 <span class="math">\\(s\\)</span> 做动作 <span class="math">\\(a\\)</span>，获得奖励 <span class="math">\\(r\\)</span>。

在模仿学习里，我们很多时候没有显式奖励，只有专家演示。但 GAIL、IRL、Offline RL 等方法会把奖励重新请回牌桌。

### A.5.4 判别器函数

GAIL 中会出现判别器：

<div class="math">\[
D_\omega(s,a) \tag{A.11}\]</div>

它试图判断一个状态—动作对来自专家还是来自当前策略。

这个函数的工程直觉很简单：它像一个质检员，看动作像不像老师傅干出来的。

---

## A.6 第四步：找聚合方式

机器学习公式很少只看一个样本。它们通常要把很多样本、很多时间步、很多随机情况聚合起来。

常见聚合方式有四类。

### A.6.1 求和

<div class="math">\[
\sum_{t=1}^{N} \ell_t \tag{A.12}\]</div>

表示把第 <span class="math">\\(1\\)</span> 到第 <span class="math">\\(N\\)</span> 个样本的损失加起来。

### A.6.2 时间累加

轨迹回报常写成：

<div class="math">\[
G_t
=
\sum_{k=t}^{T}\gamma^{k-t} r_k \tag{A.13}\]</div>

它表示从当前时刻 <span class="math">\\(t\\)</span> 开始，未来奖励的折扣累加。

这里 <span class="math">\\(\gamma\\)</span> 是折扣因子，越接近 1，越重视长期结果。

### A.6.3 期望

<div class="math">\[
\mathbb{E}_{x\sim p(x)}[f(x)] \tag{A.14}\]</div>

表示：如果 <span class="math">\\(x\\)</span> 按照分布 <span class="math">\\(p(x)\\)</span> 出现，那么 <span class="math">\\(f(x)\\)</span> 的平均值是多少。

在代码里，期望经常被 mini-batch 平均近似：

<div class="math">\[
\mathbb{E}_{x\sim p(x)}[f(x)]
\approx
\frac{1}{B}\sum_{i=1}^{B} f(x_i) \tag{A.15}\]</div>

### A.6.4 积分

隐变量模型中常见：

<div class="math">\[
p_\theta(a|o)
=
\int p_\theta(a|o,z)p(z)dz \tag{A.16}\]</div>

它表示：动作 <span class="math">\\(a\\)</span> 的概率来自所有可能隐变量 <span class="math">\\(z\\)</span> 的贡献。

如果你不喜欢积分，可以先把它理解成连续版本的求和：把所有可能风格都加权考虑一遍。

---

## A.7 第五步：找优化方向

公式里最重要的问题之一是：到底在调谁？

例如：

<div class="math">\[
\theta^*
=
\arg\min_\theta
\mathcal{L}(\theta) \tag{A.17}\]</div>

这里优化变量是 <span class="math">\\(\theta\\)</span>。

再看 GAIL 的 min-max 目标：

<div class="math">\[
\min_\pi \max_D
\mathbb{E}_{(s,a)\sim\rho_{\pi_E}}[\log D(s,a)]
+
\mathbb{E}_{(s,a)\sim\rho_\pi}[\log(1-D(s,a))] \tag{A.18}\]</div>

这里有两个角色：

- <span class="math">\\(D\\)</span>：判别器，希望把专家和策略区分开，所以它要最大化这个目标；
- <span class="math">\\(\pi\\)</span>：策略，希望骗过判别器，所以它要最小化这个目标。

读这种公式时，不要着急看细节，先问：

```text
谁在最大化？
谁在最小化？
谁是模型参数？
谁是对手？
```

否则很容易把 GAIL 看成一锅数学乱炖。

---

## A.8 常用符号表

| 符号 | 常见含义 | 本书中的直觉 |
|---|---|---|
| <span class="math">\\(s\\)</span> | state，状态 | 环境真实状态，可能不可完全观测 |
| <span class="math">\\(o\\)</span> | observation，观测 | 传感器看到的信息 |
| <span class="math">\\(a\\)</span> | action，动作 | 机器人或车辆要执行的控制量 |
| <span class="math">\\(\tau\\)</span> | trajectory，轨迹 | 一串状态和动作 |
| <span class="math">\\(\pi\\)</span> | policy，策略 | 根据观测/状态决定动作的规则 |
| <span class="math">\\(\pi\_E\\)</span> | expert policy | 专家策略，老师傅 |
| <span class="math">\\(\pi\_\theta\\)</span> | learned policy | 参数为 <span class="math">\\(\theta\\)</span> 的学习策略 |
| <span class="math">\\(\theta\\)</span> | 模型参数 | 神经网络权重 |
| <span class="math">\\(\mathcal{D}\\)</span> | 数据集 | 专家演示或离线数据 |
| <span class="math">\\(\mathcal{L}\\)</span> | loss | 训练时要最小化的错误度量 |
| <span class="math">\\(R\\)</span> | reward | 强化学习中的奖励函数 |
| <span class="math">\\(G\_t\\)</span> | return | 从时刻 <span class="math">\\(t\\)</span> 开始的未来累计奖励 |
| <span class="math">\\(d^\pi(s)\\)</span> | 策略诱导状态分布 | 策略 <span class="math">\\(\pi\\)</span> 会把机器人带到哪些状态 |
| <span class="math">\\(\rho\_\pi(s,a)\\)</span> | occupancy measure | 策略访问状态—动作对的频率 |
| <span class="math">\\(z\\)</span> | latent variable | 隐变量，动作风格旋钮 |
| <span class="math">\\(\epsilon\\)</span> | noise | 噪声，Diffusion 中尤其常见 |
| <span class="math">\\(\mathbb{E}[\cdot]\\)</span> | 期望 | 按某个分布取平均 |
| <span class="math">\\(D\_{\mathrm{KL}}\\)</span> | KL 散度 | 两个分布有多不一样的一种度量 |

---

## A.9 常见公式结构

### A.9.1 监督学习式

<div class="math">\[
\mathcal{L}(\theta)
=
\mathbb{E}_{(x,y)\sim\mathcal{D}}
[
\ell(f_\theta(x),y)
] \tag{A.19}\]</div>

人话：从数据集中抽样，模型预测，和标签比较，平均损失。

行为克隆就是把 <span class="math">\\(x\\)</span> 换成观测 <span class="math">\\(o\\)</span>，把 <span class="math">\\(y\\)</span> 换成专家动作 <span class="math">\\(a\\)</span>。

### A.9.2 最大似然式

<div class="math">\[
\theta^*
=
\arg\max_\theta
\sum_{i=1}^{N}\log p_\theta(x_i) \tag{A.20}\]</div>

人话：调参数，让模型认为真实数据更可能发生。

### A.9.3 KL 正则式

<div class="math">\[
\mathcal{L}(\theta)
=
\mathcal{L}_{\mathrm{task}}(\theta)
+
\beta D_{\mathrm{KL}}(q\|p) \tag{A.21}\]</div>

人话：既要完成任务，又不要让某个分布偏离参考分布太远。

CVAE、VAE、RLHF、策略正则里都能看到类似结构。

### A.9.4 轨迹期望式

<div class="math">\[
J(\pi)
=
\mathbb{E}_{\tau\sim p_\pi(\tau)}
[
R(\tau)
] \tag{A.22}\]</div>

人话：让策略在真实执行出来的轨迹上拿到更高回报。

这类公式强调：评估一个策略，不应该只看单步动作，而要看它执行完整任务后的结果。

---

## A.10 如何阅读一个 loss

看到一个新的 loss，建议按下面模板读：

```text
1. loss 的输入是什么？
2. loss 依赖哪些模型输出？
3. 正样本是什么？负样本是什么？
4. 它在鼓励什么？
5. 它在惩罚什么？
6. 它在哪个分布上取平均？
7. 它优化的是单步预测，还是整条轨迹？
8. 这个 loss 小，是否代表闭环一定成功？
```

最后一个问题尤其重要。很多机器人策略的失败，并不是因为 loss 写错了，而是因为 loss 衡量的世界太干净，真实闭环世界太会整活。

---

## A.11 常见误解

### 误解 1：公式越复杂，方法越高级

不一定。复杂公式可能只是把很多简单东西叠在一起。工程上，一个能稳定提升闭环成功率的简单方法，经常比一个只在论文表格里好看的复杂方法更值钱。

### 误解 2：看懂符号就等于理解算法

不够。你还要知道这个公式为什么出现、解决什么问题、在哪些假设下成立、工程上会被什么东西破坏。

### 误解 3：loss 降低就说明机器人会了

不一定。loss 通常在数据集分布上计算，而机器人执行时访问的是策略诱导分布。第 3 章和第 20 章已经反复提醒：open-loop 好，不等于 closed-loop 稳。

### 误解 4：概率公式就是玄学

概率不是玄学，它是处理不确定性的语言。机器人世界里动作不唯一、观测有噪声、数据有偏差、环境会变化，所以概率语言是必要工具。

---

## A.12 本附录小结

本附录给出了一套读公式的方法：

1. 找变量；
2. 找分布；
3. 找函数；
4. 找聚合；
5. 找优化方向。

你不需要一眼看穿所有公式。真正可靠的读法，是把复杂公式拆成一块块人话，再问每一块在工程问题中对应什么。

如果用一句话总结本附录：

> 公式不是墙，是地图。看不懂时不要硬撞，先找坐标系。

---

## A.13 小练习

1. 请把下面公式拆成“数据来源、模型输出、损失函数、优化方向”四部分：

<div class="math">\[
\min_\theta
-\mathbb{E}_{(o,a)\sim\mathcal{D}}
[
\log \pi_\theta(a|o)
] \tag{A.23}\]</div>

2. 请解释 <span class="math">\\(\mathbb{E}\_{s\sim d^{\pi\_\theta}}[\ell(s)]\\)</span> 和 <span class="math">\\(\mathbb{E}\_{s\sim d^{\pi\_E}}[\ell(s)]\\)</span> 的区别。

3. 看到 <span class="math">\\(\arg\min\_\theta\\)</span> 时，你应该立刻问哪两个问题？

4. 请从本书任意一章找一个公式，按“五步法”读一遍。
