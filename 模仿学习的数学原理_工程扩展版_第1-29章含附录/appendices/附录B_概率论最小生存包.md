# 附录 B 概率论最小生存包

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录不是概率论教材的替代品，而是模仿学习读者的“最低生存装备”。你不需要先把测度论学完再读机器人论文。先把随机变量、分布、条件概率、联合概率、边缘概率、期望这些工具拿稳，就能读懂本书大部分概率公式。

---

## B.1 为什么模仿学习离不开概率

表面上看，模仿学习好像只是“给图像，预测动作”。如果动作是方向盘角度或者机械臂末端位移，似乎做回归就行。

但真实任务里，事情很快变复杂：

- 同一个观测下，专家可能有多种合理动作；
- 传感器观测有噪声；
- 数据集只覆盖了一部分场景；
- 机器人执行动作后，下一帧观测不确定；
- Diffusion Policy 要从噪声中生成动作；
- CVAE 要用隐变量表示动作风格；
- GAIL 要比较专家分布和策略分布。

只要有“不确定性”，概率就会出现。概率不是装饰品，而是描述不确定性的语言。

---

## B.2 随机变量：会变化的量

随机变量可以理解为“每次实验可能取不同值的量”。

在模仿学习中，常见随机变量包括：

- <span class="math">\\(O\\)</span>：观测；
- <span class="math">\\(A\\)</span>：动作；
- <span class="math">\\(S\\)</span>：状态；
- <span class="math">\\(Z\\)</span>：隐变量；
- <span class="math">\\(\mathcal{T}\\)</span>：轨迹。

为了书写方便，我们经常用小写字母表示随机变量的一次取值：

<div class="math">\[
O=o,
\quad
A=a \tag{B.1}\]</div>

工程直觉是：

```text
随机变量：可能出现很多种情况的槽位。
具体取值：这一次真的出现的内容。
```

例如机械臂抓杯子任务中，<span class="math">\\(O\\)</span> 可以表示相机图像这个随机变量，某一帧具体图像就是 <span class="math">\\(o\\)</span>。

---

## B.3 概率分布：每种情况有多可能

概率分布告诉我们：随机变量取不同值的可能性有多大。

### B.3.1 离散分布

如果动作只有几个选项，例如：

```text
左移、右移、前进、后退、停止
```

那么动作可以看成离散变量。策略可以输出：

<div class="math">\[
\pi_\theta(a|o)
=
[0.1,0.2,0.5,0.1,0.1] \tag{B.2}\]</div>

这表示在观测 <span class="math">\\(o\\)</span> 下，模型认为“前进”的概率最高。

离散分布的概率必须满足：

<div class="math">\[
\sum_a p(a)=1 \tag{B.3}\]</div>

也就是所有可能动作的概率加起来等于 1。概率总量不能超发，数学世界也有货币纪律。

### B.3.2 连续分布

如果动作是连续值，例如机械臂末端 <span class="math">\\(x,y,z\\)</span> 位移，方向盘转角，那么某一个精确值的概率通常不直接看，而看概率密度。

高斯分布是最常见的连续分布之一：

<div class="math">\[
x\sim \mathcal{N}(\mu,\sigma^2) \tag{B.4}\]</div>

意思是：<span class="math">\\(x\\)</span> 大概率在均值 <span class="math">\\(\mu\\)</span> 附近，离得越远越不可能，<span class="math">\\(\sigma^2\\)</span> 控制分布有多宽。

连续分布满足：

<div class="math">\[
\int p(x)dx=1 \tag{B.5}\]</div>

这里积分可以先理解成连续版求和。

---

## B.4 条件概率：在某个前提下看概率

条件概率写作：

<div class="math">\[
p(a|o) \tag{B.6}\]</div>

读作：“给定观测 <span class="math">\\(o\\)</span> 时，动作 <span class="math">\\(a\\)</span> 的概率”。

在模仿学习中，策略本质上就是条件概率模型：

<div class="math">\[
\pi_\theta(a|o) \tag{B.7}\]</div>

它不是在问“动作 <span class="math">\\(a\\)</span> 总体上多常见”，而是在问：

> 当前看到了这个观测 <span class="math">\\(o\\)</span>，专家会做动作 <span class="math">\\(a\\)</span> 的可能性有多大？

这个区别非常重要。

例如自动泊车中，“向左打方向”整体上可能不稀奇，但在“车身已经很靠左”的观测下，继续向左打方向可能就是事故预告片。

条件概率的基础公式是：

<div class="math">\[
p(a|o)
=
\frac{p(o,a)}{p(o)} \tag{B.8}\]</div>

拆开看：

- <span class="math">\\(p(o,a)\\)</span>：观测 <span class="math">\\(o\\)</span> 和动作 <span class="math">\\(a\\)</span> 同时出现的概率；
- <span class="math">\\(p(o)\\)</span>：观测 <span class="math">\\(o\\)</span> 出现的概率；
- <span class="math">\\(p(a|o)\\)</span>：在已经看到 <span class="math">\\(o\\)</span> 的前提下，<span class="math">\\(a\\)</span> 出现的概率。

人话就是：

> 在所有出现观测 <span class="math">\\(o\\)</span> 的情况里，有多少比例对应动作 <span class="math">\\(a\\)</span>。

---

## B.5 联合概率：几个事情一起发生

联合概率写作：

<div class="math">\[
p(o,a) \tag{B.9}\]</div>

表示观测 <span class="math">\\(o\\)</span> 和动作 <span class="math">\\(a\\)</span> 同时发生的概率。

链式法则告诉我们：

<div class="math">\[
p(o,a)
=
p(a|o)p(o) \tag{B.10}\]</div>

也可以写成：

<div class="math">\[
p(o,a)
=
p(o|a)p(a) \tag{B.11}\]</div>

这两个式子没有矛盾，只是从不同方向拆联合概率。

在轨迹里，联合概率会更长。一个轨迹 <span class="math">\\(\tau=(s\_0,a\_0,s\_1,a\_1,\dots,s\_T)\\)</span> 的概率可以写成：

<div class="math">\[
p_\pi(\tau)
=
p(s_0)\prod_{t=0}^{T-1}
\pi(a_t|s_t)P(s_{t+1}|s_t,a_t) \tag{B.12}\]</div>

这个公式看起来长，但它只是说：

1. 先从初始状态 <span class="math">\\(p(s\_0)\\)</span> 开始；
2. 每一步策略选择动作 <span class="math">\\(\pi(a\_t|s\_t)\\)</span>；
3. 环境根据状态和动作转移 <span class="math">\\(P(s\_{t+1}|s\_t,a\_t)\\)</span>；
4. 把每一步概率乘起来。

这就是第 5 章、第 6 章和第 16 章背后的轨迹概率基础。

---

## B.6 边缘概率：把暂时不关心的变量“加掉”

边缘概率的意思是：如果一个变量和另一个变量有关，但你现在只关心其中一个，就把另一个变量的所有可能情况加起来或积掉。

离散情况：

<div class="math">\[
p(a|o)
=
\sum_z p(a,z|o) \tag{B.13}\]</div>

连续情况：

<div class="math">\[
p(a|o)
=
\int p(a,z|o)dz \tag{B.14}\]</div>

在隐变量模型里，<span class="math">\\(z\\)</span> 可以表示动作风格。比如抓杯子时，可能有“从左边抓”“从右边抓”“从上方抓”等风格。我们最终关心动作 <span class="math">\\(a\\)</span>，但动作背后可能经过了某个风格 <span class="math">\\(z\\)</span>。

如果不知道具体是哪种风格，就要把所有风格都考虑进去：

<div class="math">\[
p(a|o)
=
\int p(a|o,z)p(z|o)dz \tag{B.15}\]</div>

直觉是：

```text
动作概率 = 每种风格下产生该动作的概率 × 该风格本身的概率，然后对所有风格求和。
```

这就是第 8 章隐变量策略、第 9 章 CVAE 的基础。

---

## B.7 贝叶斯公式：从结果反推原因

贝叶斯公式写作：

<div class="math">\[
p(z|o,a)
=
\frac{p(a|o,z)p(z|o)}{p(a|o)} \tag{B.16}\]</div>

在 CVAE 里，这个公式的直觉很重要。

训练时，我们看到了观测 <span class="math">\\(o\\)</span> 和专家动作 <span class="math">\\(a\\)</span>，想推断背后可能的动作风格 <span class="math">\\(z\\)</span>。这就是后验分布：

<div class="math">\[
p(z|o,a) \tag{B.17}\]</div>

但真实后验通常不好直接算，所以 CVAE 用一个 encoder 去近似它：

<div class="math">\[
q_\phi(z|o,a) \tag{B.18}\]</div>

这就是为什么第 9 章说 CVAE “训练时偷看答案”：它在训练时利用 <span class="math">\\(a\\)</span> 来推断隐变量。

---

## B.8 期望：按概率加权的平均

期望写作：

<div class="math">\[
\mathbb{E}_{x\sim p(x)}[f(x)] \tag{B.19}\]</div>

意思是：<span class="math">\\(x\\)</span> 按照 <span class="math">\\(p(x)\\)</span> 出现时，<span class="math">\\(f(x)\\)</span> 的平均值。

如果 <span class="math">\\(x\\)</span> 是离散变量：

<div class="math">\[
\mathbb{E}_{x\sim p(x)}[f(x)]
=
\sum_x p(x)f(x) \tag{B.20}\]</div>

如果 <span class="math">\\(x\\)</span> 是连续变量：

<div class="math">\[
\mathbb{E}_{x\sim p(x)}[f(x)]
=
\int p(x)f(x)dx \tag{B.21}\]</div>

人话是：

> 每种情况的函数值，乘以它发生的概率，再全部加起来。

行为克隆损失：

<div class="math">\[
\mathcal{L}_{\mathrm{BC}}(\theta)
=
-\mathbb{E}_{(o,a)\sim\mathcal{D}}
[
\log\pi_\theta(a|o)
] \tag{B.22}\]</div>

就是在专家数据分布上，对负 log 概率取平均。

工程上，期望通常用 mini-batch 平均近似：

<div class="math">\[
\mathcal{L}_{\mathrm{BC}}(\theta)
\approx
-\frac{1}{B}\sum_{i=1}^{B}
\log\pi_\theta(a_i|o_i) \tag{B.23}\]</div>

这里 <span class="math">\\(B\\)</span> 是 batch size。

---

## B.9 从数据集中抽样是什么意思

当我们写：

<div class="math">\[
(o,a)\sim\mathcal{D} \tag{B.24}\]</div>

严格来说，<span class="math">\\(\mathcal{D}\\)</span> 是一个有限数据集，不是理论上的完整真实分布。但训练时我们会把它当成经验分布。

如果数据集有 <span class="math">\\(N\\)</span> 条样本：

<div class="math">\[
\mathcal{D}=\{(o_i,a_i)\}_{i=1}^{N} \tag{B.25}\]</div>

经验期望就是：

<div class="math">\[
\mathbb{E}_{(o,a)\sim\mathcal{D}}[f(o,a)]
=
\frac{1}{N}\sum_{i=1}^{N}f(o_i,a_i) \tag{B.26}\]</div>

训练时用 mini-batch：

<div class="math">\[
\frac{1}{B}\sum_{i=1}^{B}f(o_i,a_i) \tag{B.27}\]</div>

这就是为什么数据质量很重要。模型不是从宇宙真理里抽样，而是从你手里这份数据里抽样。如果数据里没有失败恢复、没有遮挡、没有长尾工况，模型就很难凭空悟道。

---

## B.10 支持集：数据覆盖了哪里

分布的支持集可以理解为“哪些地方有概率”。

如果专家数据只覆盖正常状态，那么训练分布的支持集大概是：

<div class="math">\[
\mathrm{supp}(d^{\pi_E}) \tag{B.28}\]</div>

模型执行时可能跑到：

<div class="math">\[
\mathrm{supp}(d^{\pi_\theta}) \tag{B.29}\]</div>

如果后者包含大量前者没有覆盖的状态，就会出现 OOD 风险。

这就是第 15 章 Offline Imitation Learning 中“数据不是越多越好，而是坑有没有录进去”的数学底层。

---

## B.11 条件独立：有些信息给了以后，另一些信息就不再重要

条件独立常写成：

<div class="math">\[
X\perp Y \mid Z \tag{B.30}\]</div>

意思是：在知道 <span class="math">\\(Z\\)</span> 的情况下，<span class="math">\\(X\\)</span> 和 <span class="math">\\(Y\\)</span> 没有额外依赖。

在机器人里，很多模型结构都暗含条件独立假设。例如策略可能假设：

<div class="math">\[
p(a_t|o_{0:t},a_{0:t-1})
\approx
p(a_t|h_t) \tag{B.31}\]</div>

这里 <span class="math">\\(h\_t\\)</span> 是历史编码。如果 <span class="math">\\(h\_t\\)</span> 足够表达过去信息，就不需要直接保留全部历史。

这类假设不是永远正确。历史信息压缩得不好，模型就会忘掉关键上下文，比如“刚刚夹爪已经碰到物体”或者“刚才车身已经偏过一次”。

---

## B.12 本附录小结

本附录讲了模仿学习最常用的概率概念：

1. 随机变量表示会变化的量；
2. 分布表示不同取值的可能性；
3. 条件概率表示在某个前提下的概率；
4. 联合概率表示多个事件同时发生；
5. 边缘概率表示把不关心的变量求和或积分掉；
6. 期望表示按概率加权平均；
7. 经验数据集可以看成经验分布；
8. 数据支持集决定了模型学过哪些场景。

如果一句话总结：

> 概率论不是让公式变高级，而是让我们诚实面对不确定性。

---

## B.13 小练习

1. 请用自己的话解释 <span class="math">\\(\pi\_\theta(a|o)\\)</span> 中的竖线 “<span class="math">\\(|\\)</span>” 表示什么。

2. 为什么 <span class="math">\\(p(a|o)\\)</span> 和 <span class="math">\\(p(a)\\)</span> 不一样？请用自动泊车或机械臂抓取举例。

3. 请把下面期望写成有限数据集上的平均：

<div class="math">\[
\mathbb{E}_{(o,a)\sim\mathcal{D}}[\ell(o,a)] \tag{B.32}\]</div>

4. 在 CVAE 中，为什么需要 <span class="math">\\(q\_\phi(z|o,a)\\)</span>？它和 <span class="math">\\(p(z)\\)</span> 有什么区别？

5. 请解释“训练数据支持集”和“OOD 状态”的关系。
