# 附录 D 高斯分布、MSE 与连续动作回归

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录解释一个常见问题：为什么连续动作回归里经常使用 MSE？它只是工程师顺手写的平方误差吗？不是。MSE 背后可以对应一个高斯概率模型。看懂这一点，第 2 章行为克隆、第 7 章概率策略、第 11 章 Diffusion Policy 中很多公式会变得顺眼。

---

## D.1 连续动作为什么麻烦

离散动作可以分类：左、右、前、后、停。模型输出每个动作的概率即可。

但机器人动作经常是连续的：

- 机械臂末端位置增量 \\((\Delta x,\Delta y,\Delta z)\\)；
- 机械臂姿态变化；
- 关节速度；
- 夹爪开合量；
- 自动驾驶方向盘角度；
- 泊车控制中的速度和转角。

连续动作不能简单地列出所有类别。最朴素的做法是让模型直接输出一个数值：

<div class="math-block">
\[
\hat a_t=f_\theta(o_t)
\tag{D.1}
\]
</div>

然后用 MSE：

<div class="math-block">
\[
\mathcal L_{\mathrm{MSE}}(\theta)
=
\frac{1}{N}\sum_{t=1}^{N}
\|a_t-\hat a_t\|^2
\tag{D.2}
\]
</div>

问题是：这个平方误差为什么合理？答案来自高斯分布。

---

## D.2 一维高斯分布

一维高斯分布写作：

<div class="math-block">
\[
x\sim\mathcal N(\mu,\sigma^2)
\tag{D.3}
\]
</div>

概率密度为：

<div class="math-block">
\[
p(x)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{(x-\mu)^2}{2\sigma^2}
\right)
\tag{D.4}
\]
</div>

拆开看：

- \\(\mu\\)：均值，分布中心；
- \\(\sigma^2\\)：方差，控制分布宽度；
- \\((x-\mu)^2\\)：离中心越远，惩罚越大；
- \\(\exp(\cdot)\\)：把负惩罚变成正概率密度。

工程直觉：

> 高斯分布认为“离均值近的值更可能，离均值远的值更不可能”。

这和连续动作回归很像：如果专家动作 \\(a\\) 离模型输出 \\(\mu_\theta(o)\\) 越近，我们越满意。

---

## D.3 多维高斯分布

多维动作 \\(a\in\mathbb R^d\\) 可以用多维高斯：

<div class="math-block">
\[
a\sim\mathcal N(\mu,\Sigma)
\tag{D.5}
\]
</div>

概率密度为：

<div class="math-block">
\[
p(a)
=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp
\left(
-
\frac{1}{2}(a-\mu)^T\Sigma^{-1}(a-\mu)
\right)
\tag{D.6}
\]
</div>

其中：

- \\(d\\)：动作维度；
- \\(\mu\in\mathbb R^d\\)：均值向量；
- \\(\Sigma\in\mathbb R^{d\times d}\\)：协方差矩阵；
- \\(|\Sigma|\\)：协方差矩阵行列式；
- \\((a-\mu)^T\Sigma^{-1}(a-\mu)\\)：Mahalanobis 距离形式。

如果协方差取成 \\(\sigma^2I\\)，也就是各维独立且方差相同：

<div class="math-block">
\[
\Sigma=\sigma^2I
\tag{D.7}
\]
</div>

那么指数里的距离项变成：

<div class="math-block">
\[
(a-\mu)^T(\sigma^2I)^{-1}(a-\mu)
=
\frac{1}{\sigma^2}\|a-\mu\|^2
\tag{D.8}
\]
</div>

平方误差开始登场。

---

## D.4 连续动作策略的高斯建模

假设模型在观测 \\(o_t\\) 下输出动作均值：

<div class="math-block">
\[
\mu_\theta(o_t)
\tag{D.9}
\]
</div>

我们把专家动作看成从高斯分布采样：

<div class="math-block">
\[
a_t\sim\mathcal N(\mu_\theta(o_t),\sigma^2I)
\tag{D.10}
\]
</div>

也就是说，策略为：

<div class="math-block">
\[
\pi_\theta(a_t|o_t)
=
\mathcal N(a_t;\mu_\theta(o_t),\sigma^2I)
\tag{D.11}
\]
</div>

这里的 \\(\mu_\theta(o_t)\\) 是模型预测的中心动作，\\(\sigma^2I\\) 表示我们假设专家动作围绕这个中心有高斯噪声。

---

## D.5 从高斯 NLL 推出 MSE

对单个样本，负 log likelihood 为：

<div class="math-block">
\[
-\log \pi_\theta(a_t|o_t)
=
-
\log
\mathcal N(a_t;\mu_\theta(o_t),\sigma^2I)
\tag{D.12}
\]
</div>

代入多维高斯公式，忽略与 \\(\theta\\) 无关的常数项，可得：

<div class="math-block">
\[
-\log \pi_\theta(a_t|o_t)
=
\frac{1}{2\sigma^2}
\|a_t-\mu_\theta(o_t)\|^2
+
\mathrm{const}
\tag{D.13}
\]
</div>

如果 \\(\sigma^2\\) 固定，\\(\frac{1}{2\sigma^2}\\) 只是一个常数系数。因此最小化高斯 NLL 等价于最小化：

<div class="math-block">
\[
\|a_t-\mu_\theta(o_t)\|^2
\tag{D.14}
\]
</div>

这就是 MSE。

所以 MSE 的概率解释是：

> 假设专家动作在模型预测均值附近服从固定方差的高斯分布，最大似然训练就会得到 MSE loss。

---

## D.6 MSE 的工程含义

MSE 的优点很明显：

1. 简单；
2. 稳定；
3. 易实现；
4. 对连续动作回归很自然；
5. 和高斯 NLL 有清晰关系。

但 MSE 也有一个著名问题：它喜欢平均。

如果同一个观测下有两个合理动作，例如绕障碍物可以从左绕，也可以从右绕，MSE 可能学出“从中间撞过去”。

数学上，MSE 最优解倾向于条件均值：

<div class="math-block">
\[
f^*(o)
=
\mathbb E[A|O=o]
\tag{D.15}
\]
</div>

如果动作分布是单峰高斯，均值很好；如果动作分布是多峰的，均值可能落在没有人会选择的尴尬区域。

这就是第 7 章“多模态动作被 MSE 平均掉”的核心。

---

## D.7 方差是否应该固定

前面我们假设：

<div class="math-block">
\[
\Sigma=\sigma^2I
\tag{D.16}
\]
</div>

这很方便，但不一定真实。

有些状态动作很确定，例如机械臂已经对准孔位，只需要小幅直插；有些状态动作不确定，例如遮挡严重、目标边界模糊、抓取姿态有多种选择。

于是可以让模型同时输出均值和方差：

<div class="math-block">
\[
\pi_\theta(a|o)
=
\mathcal N(a;\mu_\theta(o),\Sigma_\theta(o))
\tag{D.17}
\]
</div>

这时 NLL 不再只是 MSE，还包括方差项：

<div class="math-block">
\[
-\log\pi_\theta(a|o)
=
\frac{1}{2}(a-\mu_\theta(o))^T\Sigma_\theta(o)^{-1}(a-\mu_\theta(o))
+
\frac{1}{2}\log|\Sigma_\theta(o)|
+
\mathrm{const}
\tag{D.18}
\]
</div>

第一项惩罚预测误差，第二项惩罚方差太大。

如果没有第二项，模型可能把方差无限放大，然后淡定地说：“我不确定，所以我没错。”这在工程里很像某些写周报的人。

---

## D.8 对角协方差与完整协方差

多维动作中，最常见简化是对角协方差：

<div class="math-block">
\[
\Sigma=\mathrm{diag}(\sigma_1^2,\dots,\sigma_d^2)
\tag{D.19}
\]
</div>

这表示不同动作维度之间相互独立。

优点：实现简单，参数少。

缺点：不能表达不同维度之间的相关性。

例如机械臂末端 \\(x\\) 和 \\(y\\) 方向动作可能相关，泊车中的速度和转角也可能相关。完整协方差可以表达这种关系，但训练更复杂，数值稳定性也更难。

工程上常见选择是：

```text
先用固定方差或对角方差跑通任务；
如果不确定性建模真的重要，再考虑更复杂的协方差结构。
```

---

## D.9 高斯策略与采样动作

概率策略可以从高斯中采样：

<div class="math-block">
\[
a_t=\mu_\theta(o_t)+\sigma\epsilon,
\quad
\epsilon\sim\mathcal N(0,I)
\tag{D.20}
\]
</div>

这表示动作由两部分组成：

- \\(\mu_\theta(o_t)\\)：模型认为最中心的动作；
- \\(\sigma\epsilon\\)：随机扰动。

在训练或探索时，采样可以提供多样性；在部署时，很多系统会直接使用均值动作，减少随机性：

<div class="math-block">
\[
a_t=\mu_\theta(o_t)
\tag{D.21}
\]
</div>

这不是说概率策略没用，而是工程部署常常更保守。产线机器人不是抽盲盒，动作随机性必须可控。

---

## D.10 Diffusion Policy 中的高斯噪声

Diffusion Policy 里也大量使用高斯噪声。正向加噪过程通常写作：

<div class="math-block">
\[
q(x_t|x_{t-1})
=
\mathcal N(
\sqrt{1-\beta_t}x_{t-1},
\beta_tI
)
\tag{D.22}
\]
</div>

这里：

- \\(x_{t-1}\\)：上一噪声等级的动作序列；
- \\(x_t\\)：加噪后的动作序列；
- \\(\beta_t\\)：噪声强度；
- \\(I\\)：各维独立同方差噪声。

Diffusion 的训练目标常写成预测噪声的 MSE：

<div class="math-block">
\[
\mathcal L(\theta)
=
\mathbb E
[
\|\epsilon-\epsilon_\theta(x_t,t,o)\|^2
]
\tag{D.23}
\]
</div>

所以第 11 章的 MSE 不再是直接回归动作，而是回归被加入的噪声。它依然和高斯建模密切相关。

---

## D.11 常见误解

### 误解 1：MSE 只是随手选的 loss

不是。MSE 可以从固定方差高斯 NLL 推导出来。当然，这不代表它永远适合，只代表它有明确概率假设。

### 误解 2：连续动作只能用 MSE

不对。连续动作可以用高斯策略、混合高斯、隐变量模型、CVAE、Diffusion Policy 等方式建模。MSE 是起点，不是终点。

### 误解 3：预测均值就是最佳动作

只在动作分布比较单峰时比较合理。多峰动作分布下，均值可能是糟糕动作。

### 误解 4：方差越大越安全

方差大只能表示模型不确定，但不等于动作安全。安全还需要约束、检测、fallback 和控制器。

---

## D.12 本附录小结

本附录讲清楚了：

1. 高斯分布认为样本围绕均值波动；
2. 固定方差高斯 NLL 可以推出 MSE；
3. MSE 对单峰连续动作很自然；
4. 多模态动作下，MSE 容易学出平均动作；
5. 概率策略可以输出均值和方差；
6. Diffusion Policy 中的噪声预测也依赖高斯噪声假设。

一句话总结：

> MSE 不是错，它只是相信世界大致像一个单峰高斯；机器人任务一旦多解，它就可能开始装傻。

---

## D.13 小练习

1. 请从一维高斯 NLL 推导 MSE 的形式。

2. 为什么固定方差高斯策略下，最小化 NLL 等价于最小化平方误差？

3. 举一个多模态动作场景，说明 MSE 为什么可能输出不合理动作。

4. 模型输出方差有什么好处？有什么风险？

5. Diffusion Policy 中预测噪声的 MSE 和行为克隆中回归动作的 MSE 有什么区别？
