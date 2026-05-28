# 附录 G 生成模型基础

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录解释 CVAE、ACT 和 Diffusion Policy 需要的生成模型基础。生成模型的核心问题不是“预测一个标准答案”，而是“在条件给定时，生成一个合理样本”。对机器人来说，这非常重要，因为同一个观测下，正确动作往往不止一个。

---

## G.1 为什么模仿学习需要生成模型

传统行为克隆常写作：

<div class="math-block">
\[
\hat a=f_\theta(o)
\tag{G.1}
\]
</div>

这隐含一个倾向：同一个观测 \\(o\\) 下，模型输出一个动作。

但机器人任务常常多解：

- 抓杯子可以从左抓，也可以从右抓；
- 整理桌面可以先拿杯子，也可以先拿书；
- 绕障碍可以从左绕，也可以从右绕；
- 双臂操作可以左手先动，也可以右手先动。

如果模型只输出一个平均动作，就可能把多个合理模式平均成一个不合理动作。

生成模型的目标是建模：

<div class="math-block">
\[
p_\theta(a|o)
\tag{G.2}
\]
</div>

也就是在观测条件下，动作的整个分布，而不只是一个点估计。

---

## G.2 隐变量：动作风格旋钮

隐变量写作 \\(z\\)。

直觉上，\\(z\\) 表示动作背后的风格、模式或未观测因素。

例如：

```text
z = 左侧抓取风格
z = 右侧抓取风格
z = 快速但粗糙的风格
z = 慢速但稳的风格
```

隐变量模型可以写成：

<div class="math-block">
\[
z\sim p(z)
\tag{G.3}
\]
</div>

<div class="math-block">
\[
a\sim p_\theta(a|o,z)
\tag{G.4}
\]
</div>

生成过程是：

```text
先采样一个风格 z；
再根据观测 o 和风格 z 生成动作 a。
```

如果把 \\(z\\) 看成旋钮，模型就不必把所有动作模式硬挤成一个平均答案。

---

## G.3 prior 与 posterior

prior 是先验分布：

<div class="math-block">
\[
p(z)
\tag{G.5}
\]
</div>

它表示在没看具体动作之前，隐变量可能是什么。

常用简单先验：

<div class="math-block">
\[
p(z)=\mathcal N(0,I)
\tag{G.6}
\]
</div>

posterior 是后验分布：

<div class="math-block">
\[
p(z|o,a)
\tag{G.7}
\]
</div>

它表示：看到了观测 \\(o\\) 和专家动作 \\(a\\) 后，背后的隐变量 \\(z\\) 可能是什么。

CVAE 的关键是：真实 posterior 通常不好算，所以用 encoder 近似：

<div class="math-block">
\[
q_\phi(z|o,a)
\tag{G.8}
\]
</div>

这就是第 9 章中“训练时偷看答案”的数学形式。

---

## G.4 VAE 的基本结构

VAE 试图建模数据 \\(x\\) 的分布。

生成过程：

<div class="math-block">
\[
z\sim p(z)
\tag{G.9}
\]
</div>

<div class="math-block">
\[
x\sim p_\theta(x|z)
\tag{G.10}
\]
</div>

训练时，用 encoder 近似后验：

<div class="math-block">
\[
q_\phi(z|x)
\tag{G.11}
\]
</div>

VAE 的目标不是直接回归一个 \\(x\\)，而是学习一个可以生成数据的概率模型。

机器人动作生成里，我们通常需要条件版本，也就是 CVAE。

---

## G.5 CVAE：带条件的 VAE

CVAE 生成动作时带观测条件 \\(o\\)：

<div class="math-block">
\[
z\sim p(z)
\tag{G.12}
\]
</div>

<div class="math-block">
\[
a\sim p_\theta(a|o,z)
\tag{G.13}
\]
</div>

训练时 encoder 使用观测和专家动作：

<div class="math-block">
\[
q_\phi(z|o,a)
\tag{G.14}
\]
</div>

decoder 根据 \\(o,z\\) 重建动作：

<div class="math-block">
\[
\hat a\sim p_\theta(a|o,z)
\tag{G.15}
\]
</div>

CVAE 的目标是：

```text
既能重建专家动作，
又让训练时推断出的 z 不要偏离推理时会采样的 prior 太远。
```

---

## G.6 ELBO：为什么会有重建项和 KL 项

我们真正想最大化：

<div class="math-block">
\[
\log p_\theta(a|o)
\tag{G.16}
\]
</div>

但由于有隐变量：

<div class="math-block">
\[
p_\theta(a|o)
=
\int p_\theta(a|o,z)p(z)dz
\tag{G.17}
\]
</div>

这个积分通常不好直接优化。

于是引入近似后验 \\(q_\phi(z|o,a)
\\(，可以得到 ELBO：

<div class="math-block">
\[
\log p_\theta(a|o)
\ge
\mathbb E_{z\sim q_\phi(z|o,a)}
[
\log p_\theta(a|o,z)
]
-
D_{\mathrm{KL}}(
q_\phi(z|o,a)\|p(z)
)
\tag{G.18}
\]
</div>

这个式子一定要拆开。

第一项：

<div class="math-block">
\[
\mathbb E_{q_\phi}
[
\log p_\theta(a|o,z)
]
\tag{G.19}
\]
</div>

表示重建项：用 encoder 推断出的 \\(z\\)，decoder 能不能生成专家动作。

第二项：

<div class="math-block">
\[
D_{\mathrm{KL}}(q_\phi(z|o,a)\|p(z))
\tag{G.20}
\]
</div>

表示 KL 正则：训练时的 \\(z\\) 分布不要离推理时的 prior 太远。

CVAE 训练通常最小化负 ELBO：

<div class="math-block">
\[
\mathcal L_{\mathrm{CVAE}}
=
\mathcal L_{\mathrm{recon}}
+
\beta
D_{\mathrm{KL}}(q_\phi(z|o,a)\|p(z))
\tag{G.21}
\]
</div>

这里 \\(\beta\\) 是 KL 权重。

---

## G.7 重参数化技巧

如果 \\(z\sim q_\phi(z|o,a)\\)，直接采样会影响梯度传播。VAE 使用重参数化技巧。

假设：

<div class="math-block">
\[
q_\phi(z|o,a)
=
\mathcal N(\mu_\phi(o,a),\sigma_\phi^2(o,a)I)
\tag{G.22}
\]
</div>

采样可以写成：

<div class="math-block">
\[
z=\mu_\phi(o,a)+\sigma_\phi(o,a)\odot\epsilon,
\quad
\epsilon\sim\mathcal N(0,I)
\tag{G.23}
\]
</div>

这样随机性来自 \\(\epsilon\\)，而 \\(\mu_\phi\\) 和 \\(\sigma_\phi\\) 仍然可导。

人话：

> 不要直接从网络输出的分布里“抓阄”，而是先抓一个标准噪声，再用网络输出的均值和方差把它变形。

---

## G.8 ACT 与 CVAE

ACT 使用 CVAE 来建模动作块。动作不再是单步：

<div class="math-block">
\[
a_t
\tag{G.24}
\]
</div>

而是一段动作：

<div class="math-block">
\[
A_t=(a_t,a_{t+1},\dots,a_{t+H-1})
\tag{G.25}
\]
</div>

CVAE 目标变成：

<div class="math-block">
\[
\mathcal L_{\mathrm{ACT}}
=
\mathbb E
[
\|A_t-\hat A_t\|^2]
+
\beta D_{\mathrm{KL}}(q_\phi(z|o,A_t)\|p(z))
\tag{G.26}
\]
</div>

隐变量 \\(z\\) 负责表示动作块风格，decoder 生成未来一段动作。

这就是第 10 章中 ACT 不只预测“一步”，而是预测“动作小套餐”的数学基础。

---

## G.9 Diffusion 的基本思想

Diffusion model 的核心想法是：

```text
训练时：把真实数据一步步加噪，直到变成接近纯噪声；
生成时：从噪声开始，一步步去噪，恢复成数据样本。
```

在 Diffusion Policy 中，数据样本是动作序列：

<div class="math-block">
\[
x_0=A_t
\tag{G.27}
\]
</div>

正向加噪：

<div class="math-block">
\[
q(x_t|x_{t-1})
=
\mathcal N(
\sqrt{1-\beta_t}x_{t-1},
\beta_tI
)
\tag{G.28}
\]
</div>

经过多步后，\\(x_T\\) 接近高斯噪声。

---

## G.10 一步到位的加噪公式

Diffusion 中常用闭式形式：

<div class="math-block">
\[
q(x_t|x_0)
=
\mathcal N(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
)
\tag{G.29}
\]
</div>

等价采样写作：

<div class="math-block">
\[
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\quad
\epsilon\sim\mathcal N(0,I)
\tag{G.30}
\]
</div>

拆开看：

- \\(x_0\\)：真实动作序列；
- \\(x_t\\)：第 \\(t\\) 个噪声等级的动作序列；
- \\(\epsilon\\)：标准高斯噪声；
- \\(\bar\alpha_t\\)：保留原始信号的比例。

\\(t\\) 越大，原始动作保留越少，噪声越多。

---

## G.11 噪声预测目标

Diffusion 常训练网络预测噪声：

<div class="math-block">
\[
\epsilon_\theta(x_t,t,o)
\tag{G.31}
\]
</div>

训练目标：

<div class="math-block">
\[
\mathcal L_{\mathrm{diff}}
=
\mathbb E_{x_0,t,\epsilon}
[
\|\epsilon-\epsilon_\theta(x_t,t,o)\|^2
]
\tag{G.32}
\]
</div>

这个公式的人话是：

> 我给你一段被加噪的动作序列 \\(x_t\\)，告诉你当前噪声等级 \\(t\\)，再给你观测条件 \\(o\\)。你要猜出当初加进去的噪声 \\(\epsilon\\)。

如果模型能猜出噪声，就可以从 \\(x_t\\) 中把噪声去掉，逐步恢复动作。

---

## G.12 条件生成：观测如何影响动作

在机器人策略中，生成动作不能只靠噪声，还要看当前观测。

因此 Diffusion Policy 建模：

<div class="math-block">
\[
p_\theta(A_t|o_t)
\tag{G.33}
\]
</div>

去噪网络写作：

<div class="math-block">
\[
\epsilon_\theta(x_t,t,o_t)
\tag{G.34}
\]
</div>

观测 \\(o_t\\) 像任务条件，告诉模型：当前场景下应该生成什么动作序列。

如果没有条件，模型只能生成“看起来像动作”的动作；有了条件，模型才可能生成“适合当前场景”的动作。

---

## G.13 生成模型的工程边界

生成模型很强，但不是魔法。

它们擅长：

- 表达多模态动作；
- 生成动作序列；
- 避免 MSE 平均问题；
- 利用大规模数据学习复杂分布。

但它们也有代价：

- 训练更复杂；
- 推理可能更慢；
- 调参更困难；
- 对数据质量敏感；
- 安全约束不能只靠生成模型自己学。

在工程系统中，生成式策略输出仍然需要动作平滑、限幅、安全投影、监控和 fallback。一个动作生成得很“像专家”，不等于它一定安全。

---

## G.14 常见误解

### 误解 1：生成模型一定比回归模型高级

不一定。任务单一、动作单峰、数据少时，简单 BC 可能更稳。生成模型适合处理多模态和复杂动作分布，但复杂度也更高。

### 误解 2：CVAE 的 latent 一定可解释

不一定。\\(z\\) 可以表示动作风格，但模型未必自动把每一维学成“左抓”“右抓”这种人类可读语义。

### 误解 3：Diffusion 只是在动作上加噪声

加噪只是训练构造。真正目标是学习条件动作分布，并通过反向去噪生成合理动作。

### 误解 4：能生成多样动作就能解决闭环问题

不够。多样性解决的是动作分布表达问题，闭环还需要状态分布、反馈、评测和安全系统。

---

## G.15 本附录小结

本附录讲了生成模型基础：

1. 生成模型建模动作分布，而不是单点动作；
2. 隐变量 \\(z\\) 可以表示动作风格；
3. CVAE 用 encoder 近似后验，用 decoder 生成动作；
4. ELBO 包含重建项和 KL 项；
5. ACT 用 CVAE 生成动作块；
6. Diffusion 通过加噪和去噪学习动作序列分布；
7. 噪声预测目标是 Diffusion Policy 的核心训练目标。

一句话总结：

> 生成模型让机器人不再被迫给一个平均答案，但它生成的是候选动作，不是安全保证书。

---

## G.16 小练习

1. 为什么多模态动作适合用生成模型建模？

2. 请解释 prior \\(p(z)\\) 和 posterior \\(p(z|o,a)\\) 的区别。

3. CVAE 的 ELBO 中，重建项和 KL 项分别起什么作用？

4. 为什么 ACT 要生成动作块，而不是单步动作？

5. Diffusion Policy 中，为什么训练目标可以写成预测噪声的 MSE？
