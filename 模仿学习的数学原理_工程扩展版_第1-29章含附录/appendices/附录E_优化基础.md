# 附录 E 优化基础

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录解释训练神经网络时反复出现的几个词：loss、梯度、梯度下降、学习率、反向传播。你不需要把数值优化学成一门新专业，但至少要知道模型训练到底在“拧哪颗螺丝”。

---

## E.1 训练到底在做什么

机器学习训练可以粗略理解成：

```text
给模型一堆参数 θ；
定义一个损失函数 L(θ)；
不断调整 θ，让 L(θ) 变小。
```

数学写作：

<div class="math">\[
\theta^*
=
\arg\min_\theta
\mathcal{L}(\theta) \tag{E.1}\]</div>

其中：

- <span class="math">\\(\theta\\)</span>：模型参数；
- <span class="math">\\(\mathcal{L}(\theta)\\)</span>：损失函数；
- <span class="math">\\(\theta^*\\)</span>：让损失尽可能小的参数。

这就是最小化问题。

在行为克隆中，loss 可以是：

<div class="math">\[
\mathcal{L}_{\mathrm{BC}}(\theta)
=
-\mathbb{E}_{(o,a)\sim\mathcal{D}}
[
\log\pi_\theta(a|o)
] \tag{E.2}\]</div>

训练的目标就是让模型给专家动作更高概率，从而让这个负 log 概率更小。

---

## E.2 损失函数：模型哪里做得不好

损失函数是训练过程中的打分器。分数越高，说明模型越需要挨打。

常见损失包括：

### E.2.1 MSE

<div class="math">\[
\mathcal{L}_{\mathrm{MSE}}
=
\|a-\hat a\|^2 \tag{E.3}\]</div>

用于连续动作回归。

### E.2.2 交叉熵

<div class="math">\[
\mathcal{L}_{\mathrm{CE}}
=
-
\sum_a y(a)\log \hat p(a) \tag{E.4}\]</div>

用于离散动作分类。

### E.2.3 KL 正则

<div class="math">\[
\mathcal{L}_{\mathrm{KL}}
=
D_{\mathrm{KL}}(q\|p) \tag{E.5}\]</div>

用于约束分布不要偏离太远。

一个训练目标也可以由多项组成：

<div class="math">\[
\mathcal{L}(\theta)
=
\mathcal{L}_{\mathrm{task}}(\theta)
+
\lambda\mathcal{L}_{\mathrm{reg}}(\theta) \tag{E.6}\]</div>

这里 <span class="math">\\(\lambda\\)</span> 控制正则项权重。工程上 <span class="math">\\(\lambda\\)</span> 不是随便写的，它会改变模型行为。太小，约束不干活；太大，模型可能任务都学不好。

---

## E.3 梯度：loss 往哪里变大最快

梯度写作：

<div class="math">\[
\nabla_\theta \mathcal{L}(\theta) \tag{E.7}\]</div>

它表示损失函数相对于参数 <span class="math">\\(\theta\\)</span> 的变化方向。

如果 <span class="math">\\(\theta\\)</span> 只有一个参数，梯度就是导数：

<div class="math">\[
\frac{d\mathcal{L}(\theta)}{d\theta} \tag{E.8}\]</div>

如果 <span class="math">\\(\theta\\)</span> 有很多维，梯度是一个向量：

<div class="math">\[
\nabla_\theta \mathcal{L}(\theta)
=
\left[
\frac{\partial\mathcal{L}}{\partial\theta_1},
\frac{\partial\mathcal{L}}{\partial\theta_2},
\dots,
\frac{\partial\mathcal{L}}{\partial\theta_d}
\right] \tag{E.9}\]</div>

直觉：

> 梯度指向 loss 增大最快的方向。

所以如果我们想让 loss 变小，就往梯度的反方向走。

---

## E.4 梯度下降

梯度下降更新公式：

<div class="math">\[
\theta
\leftarrow
\theta-
\eta
\nabla_\theta\mathcal{L}(\theta) \tag{E.10}\]</div>

拆开看：

- 当前参数：<span class="math">\\(\theta\\)</span>；
- 梯度：<span class="math">\\(\nabla\_\theta\mathcal{L}(\theta)\\)</span>；
- 学习率：<span class="math">\\(\eta\\)</span>；
- 更新方向：负梯度方向。

人话：

> loss 往哪边涨得最快，我们就往反方向挪一点。

<span class="math">\\(\eta\\)</span> 控制挪多大一步。

如果学习率太大，可能一步跨过谷底，在 loss 山谷两边蹦迪；如果学习率太小，训练像老年人过马路，安全但很慢。

---

## E.5 mini-batch 梯度下降

真实训练不会每次都用全量数据计算梯度，太慢。通常使用 mini-batch。

全数据 loss：

<div class="math">\[
\mathcal{L}(\theta)
=
\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta) \tag{E.11}\]</div>

mini-batch 近似：

<div class="math">\[
\hat{\mathcal{L}}(\theta)
=
\frac{1}{B}\sum_{i=1}^{B}\ell_i(\theta) \tag{E.12}\]</div>

对应梯度：

<div class="math">\[
\nabla_\theta\hat{\mathcal{L}}(\theta)
\approx
\nabla_\theta\mathcal{L}(\theta) \tag{E.13}\]</div>

mini-batch 梯度有噪声，但计算快，还能帮助跳出一些不好的局部区域。深度学习训练很多时候就是在这种带噪声的方向感里前进，有点像晚上倒车入库：不是每一步都精确，但整体要往对的方向修。

---

## E.6 反向传播

神经网络是很多函数的复合：

<div class="math">\[
y=f_\theta(x) \tag{E.14}\]</div>

loss 是：

<div class="math">\[
\mathcal{L}=\ell(y,y^*) \tag{E.15}\]</div>

要更新参数，需要计算：

<div class="math">\[
\frac{\partial\mathcal{L}}{\partial\theta} \tag{E.16}\]</div>

反向传播的核心是链式法则。

简单例子：

<div class="math">\[
z=f(x),
\quad
y=g(z),
\quad
\mathcal{L}=h(y) \tag{E.17}\]</div>

那么：

<div class="math">\[
\frac{d\mathcal{L}}{dx}
=
\frac{d\mathcal{L}}{dy}
\frac{dy}{dz}
\frac{dz}{dx} \tag{E.18}\]</div>

神经网络只是把这个链式法则应用到很多层、很多参数上。框架如 PyTorch 会自动计算，但你要知道它不是魔法，而是链式法则的批量施工队。

---

## E.7 Adam、动量和工程优化器

实际训练中，大家常用 Adam，而不是最朴素的梯度下降。

朴素梯度下降：

<div class="math">\[
\theta\leftarrow\theta-\eta g_t \tag{E.19}\]</div>

其中 <span class="math">\\(g\_t=\nabla\_\theta\mathcal{L}\_t(\theta)\\)</span>。

带动量的方法会累积历史梯度方向：

<div class="math">\[
v_t=\beta v_{t-1}+(1-\beta)g_t \tag{E.20}\]</div>

再用 <span class="math">\\(v\_t\\)</span> 更新参数。

直觉是：如果连续很多步都往同一个方向走，那这个方向可能靠谱；如果梯度乱跳，动量可以让更新别那么抽风。

Adam 还会根据梯度二阶矩自适应调整步长。你不需要先背完 Adam 公式，但要知道：优化器不是改变训练目标，而是改变走向目标的方式。

---

## E.8 非凸优化：为什么训练不保证找到全局最优

深度神经网络的 loss 通常是非凸的。也就是说，它不像一个简单碗形曲面，而更像山区地形：山谷、鞍点、平原、坑洼都有。

所以训练通常不保证找到全局最优：

<div class="math">\[
\theta_{\mathrm{trained}}
\neq
\arg\min_\theta \mathcal{L}(\theta) \tag{E.21}\]</div>

但工程上我们不一定需要数学意义上的全局最优。我们需要的是：

```text
验证集表现好；
闭环执行稳定；
长尾风险可控；
部署指标达标。
```

这也是为什么第 20 章强调：loss 只是训练指标，不是最终业务指标。

---

## E.9 过拟合与正则化

训练 loss 很低，测试效果很差，可能是过拟合。

经验风险：

<div class="math">\[
\hat R(\theta)
=
\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i) \tag{E.22}\]</div>

真实风险：

<div class="math">\[
R(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
[
\ell(f_\theta(x),y)] \tag{E.23}\]</div>

训练只看经验风险，但我们真正关心真实风险。

正则化常写作：

<div class="math">\[
\mathcal{L}(\theta)
=
\hat R(\theta)+\lambda\Omega(\theta) \tag{E.24}\]</div>

其中 <span class="math">\\(\Omega(\theta)\\)</span> 是正则项，例如权重衰减。

模仿学习中还有一种更现实的“过拟合”：模型在录制场景很好，换一个光照、工件批次、相机高度、地面材质就翻车。这不仅是统计问题，也是数据覆盖和部署环境问题。

---

## E.10 多目标 loss 的权重怎么理解

CVAE、Diffusion、机器人策略训练中常见多项 loss：

<div class="math">\[
\mathcal{L}
=
\mathcal{L}_1
+
\lambda_2\mathcal{L}_2
+
\lambda_3\mathcal{L}_3 \tag{E.25}\]</div>

比如 ACT 中可能有重建损失和 KL 项：

<div class="math">\[
\mathcal{L}_{\mathrm{ACT}}
=
\mathcal{L}_{\mathrm{recon}}
+
\beta D_{\mathrm{KL}}(q_\phi(z|o,a)\|p(z)) \tag{E.26}\]</div>

这里 <span class="math">\\(\beta\\)</span> 不只是数学符号，它是行为调节旋钮。

- <span class="math">\\(\beta\\)</span> 太小：latent 可能乱跑，推理时采样不稳定；
- <span class="math">\\(\beta\\)</span> 太大：latent 可能被压死，模型退化成普通回归。

所以调 loss 权重不是“玄学炼丹”这么简单，它是在不同目标之间做工程取舍。

---

## E.11 常见误解

### 误解 1：loss 降低就万事大吉

不对。loss 是训练分布上的指标，机器人最终看 closed-loop 成功率、安全事件、稳定性和恢复能力。

### 误解 2：学习率越大训练越快

太大学习率会导致不稳定，甚至 loss 爆炸。快不是乱冲，优化不是抢电梯。

### 误解 3：优化器能解决所有问题

优化器只能帮你更好地优化目标。如果目标本身不对、数据分布不对、标签噪声很大，再好的优化器也只是更快地奔向错误方向。

### 误解 4：多项 loss 权重随便设

权重会改变模型行为。尤其在 CVAE、Diffusion、离线策略学习中，权重设置可能决定模型是稳定还是发疯。

---

## E.12 本附录小结

本附录讲了训练的基本机制：

1. 训练是最小化损失函数；
2. 梯度表示 loss 增大最快方向；
3. 梯度下降沿负梯度更新参数；
4. 学习率控制步长；
5. 反向传播用链式法则计算梯度；
6. mini-batch 用样本平均近似期望；
7. 正则化和多项 loss 是工程取舍；
8. loss 不是闭环能力的充分证明。

一句话总结：

> 优化是在拧模型参数这堆螺丝，但你首先要确认自己在修车，不是在给车装风扇。

---

## E.13 小练习

1. 请解释 <span class="math">\\(\theta\leftarrow\theta-\eta\nabla\_\theta\mathcal{L}(\theta)\\)</span> 中每个符号的含义。

2. 为什么梯度下降要沿负梯度方向走？

3. mini-batch 梯度为什么只是全量梯度的近似？

4. 一个模型训练 loss 下降但 rollout 成功率不上升，可能有哪些原因？

5. 多项 loss 中的权重 <span class="math">\\(\lambda\\)</span> 或 <span class="math">\\(\beta\\)</span> 改变后，为什么模型行为会变？
