# 附录 C 最大似然、负对数似然、交叉熵与 KL 散度

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录负责解释本书里最常见的一组概率优化工具：MLE、NLL、交叉熵和 KL 散度。它们经常结伴出场，像机器学习公式里的“四大金刚”。看懂它们，行为克隆、概率策略、CVAE、GAIL 里一大半公式就不再神秘。

---

## C.1 从一个朴素问题开始：怎样让模型相信专家动作

行为克隆中，我们有专家数据：

<div class="math-block">
\[
\mathcal D=\{(o_i,a_i)\}_{i=1}^{N}
\tag{C.1}
\]
</div>

我们希望策略 \\(\pi_\theta(a|o)\\) 在看到 \\(o_i\\) 时，认为专家动作 \\(a_i\\) 很可能。

最直接的目标是最大化所有专家动作的概率：

<div class="math-block">
\[
\theta^*
=
\arg\max_\theta
\prod_{i=1}^{N}
\pi_\theta(a_i|o_i)
\tag{C.2}
\]
</div>

这就是最大似然估计的形式。

拆开看：

- \\(\pi_\theta(a_i|o_i)\\)：模型认为第 \\(i\\) 条专家动作有多可能；
- \\(\prod_{i=1}^{N}\\)：把所有样本的概率乘起来；
- \\(\arg\max_\theta\\)：寻找让这些概率整体最大的参数。

人话是：

> 调模型参数，让专家数据在模型眼里尽可能合理。

---

## C.2 最大似然估计 MLE

最大似然估计的通用形式是：

<div class="math-block">
\[
\theta^*
=
\arg\max_\theta
\prod_{i=1}^{N}
p_\theta(x_i)
\tag{C.3}
\]
</div>

其中：

- \\(x_i\\)：观测到的数据；
- \\(p_\theta(x_i)\\)：模型给数据 \\(x_i\\) 的概率；
- \\(\theta\\)：模型参数。

在行为克隆中，数据不是单独的 \\(x_i\\)，而是条件样本 \\((o_i,a_i)\\)。于是模型概率写成：

<div class="math-block">
\[
p_\theta(a_i|o_i)
=
\pi_\theta(a_i|o_i)
\tag{C.4}
\]
</div>

所以 MLE 变成：

<div class="math-block">
\[
\theta^*
=
\arg\max_\theta
\prod_{i=1}^{N}
\pi_\theta(a_i|o_i)
\tag{C.5}
\]
</div>

最大似然不是玄学，它只是说：

> 既然这些专家数据真的发生了，那一个好模型应该给它们较高概率。

---

## C.3 为什么概率连乘要变成 log 求和

公式 C.5 有一个工程问题：很多小概率连乘会非常小，数值上容易下溢。

例如 1000 个小于 1 的数连乘，结果可能小到计算机都懒得理你。

于是我们取 log：

<div class="math-block">
\[
\log\prod_{i=1}^{N}
\pi_\theta(a_i|o_i)
=
\sum_{i=1}^{N}
\log\pi_\theta(a_i|o_i)
\tag{C.6}
\]
</div>

这个等式来自对数性质：

<div class="math-block">
\[
\log(xy)=\log x+\log y
\tag{C.7}
\]
</div>

取 log 不会改变最大值位置，因为 log 是单调递增函数。也就是说，让概率乘积最大，等价于让 log 概率求和最大：

<div class="math-block">
\[
\theta^*
=
\arg\max_\theta
\sum_{i=1}^{N}
\log\pi_\theta(a_i|o_i)
\tag{C.8}
\]
</div>

这就是 log-likelihood。

工程直觉：

```text
概率连乘：数学上正确，但数值上容易崩。
log 求和：数学等价，训练更稳定。
```

---

## C.4 负对数似然 NLL

深度学习里通常习惯最小化 loss，而不是最大化目标。于是我们把最大化 log-likelihood 变成最小化负 log-likelihood：

<div class="math-block">
\[
\mathcal L_{\mathrm{NLL}}(\theta)
=
-
\sum_{i=1}^{N}
\log\pi_\theta(a_i|o_i)
\tag{C.9}
\]
</div>

或者写成期望形式：

<div class="math-block">
\[
\mathcal L_{\mathrm{NLL}}(\theta)
=
-
\mathbb E_{(o,a)\sim\mathcal D}
[
\log\pi_\theta(a|o)
]
\tag{C.10}
\]
</div>

这就是第 2 章行为克隆目标。

为什么前面有负号？

因为概率越大，\\(\log\pi_\theta(a|o)\\) 越大；但训练框架喜欢最小化，所以加个负号，把“越好越大”变成“越好越小”。

如果模型给专家动作概率很低，\\(\log\pi_\theta(a|o)\\) 会很负，负号以后 loss 就很大。模型会被迫反省：为什么老师傅的动作你觉得这么不可能？

---

## C.5 交叉熵：分类行为克隆的常见 loss

离散动作分类中，模型输出一个概率分布：

<div class="math-block">
\[
\hat p_\theta(a|o)
\tag{C.11}
\]
</div>

专家标签可以看成 one-hot 分布 \\(p_E(a|o)\\)。例如有 4 个动作，专家选择第 2 个：

<div class="math-block">
\[
p_E(a|o)=[0,1,0,0]
\tag{C.12}
\]
</div>

交叉熵定义为：

<div class="math-block">
\[
H(p_E,\hat p_\theta)
=
-
\sum_a
p_E(a|o)
\log \hat p_\theta(a|o)
\tag{C.13}
\]
</div>

因为 \\(p_E\\) 是 one-hot，只有专家动作那一项为 1，所以交叉熵变成：

<div class="math-block">
\[
H(p_E,\hat p_\theta)
=
-
\log \hat p_\theta(a_E|o)
\tag{C.14}
\]
</div>

这正是 NLL。

所以在离散动作 BC 中：

```text
交叉熵 loss = 专家动作的负 log 概率。
```

这也是为什么分类网络训练里常用 CrossEntropyLoss。

---

## C.6 交叉熵的直觉

交叉熵衡量的是：如果真实分布是 \\(p\\)，但你用 \\(q\\) 来编码或预测，会付出多少代价。

公式：

<div class="math-block">
\[
H(p,q)
=
-
\sum_x p(x)\log q(x)
\tag{C.15}
\]
</div>

如果 \\(q\\) 在 \\(p\\) 常出现的地方给了高概率，交叉熵就小；如果 \\(q\\) 在真实样本上给低概率，交叉熵就大。

在模仿学习中，这句话翻译成：

> 专家常做的动作，模型也要给高概率；专家动作被模型看成小概率事件，loss 就会骂它。

---

## C.7 KL 散度：两个分布差多少

KL 散度定义为：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)
=
\sum_x p(x)
\log\frac{p(x)}{q(x)}
\tag{C.16}
\]
</div>

连续形式是：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)
=
\int p(x)
\log\frac{p(x)}{q(x)}dx
\tag{C.17}
\]
</div>

读这个公式时，重点看三个部分：

1. \\(p(x)\\)：按真实分布加权；
2. \\(\frac{p(x)}{q(x)}\\)：真实概率和模型概率的比例；
3. \\(\log\frac{p(x)}{q(x)}\\)：用 log 衡量差异。

KL 的直觉是：

> 如果真实世界按照 \\(p\\) 出现，但你用 \\(q\\) 去描述它，会有多不匹配。

在 CVAE 中，KL 用来约束后验 \\(q_\phi(z|o,a)\\) 不要离 prior \\(p(z)\\) 太远：

<div class="math-block">
\[
D_{\mathrm{KL}}
(
q_\phi(z|o,a)
\|p(z)
)
\tag{C.18}
\]
</div>

它像是在说：训练时你可以用专家动作推断风格，但不要推断出一个推理时完全采不到的奇怪风格空间。

---

## C.8 KL 为什么不是普通距离

普通距离通常满足对称性：

<div class="math-block">
\[
d(x,y)=d(y,x)
\tag{C.19}
\]
</div>

但 KL 不满足：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)
\neq
D_{\mathrm{KL}}(q\|p)
\tag{C.20}
\]
</div>

这是很多人第一次学 KL 时会被坑的地方。

为什么不对称？因为 \\(D_{\mathrm{KL}}(p\|q)\\) 是按 \\(p\\) 加权的，它关心的是 \\(p\\) 常出现的地方 \\(q\\) 是否也给了概率。

如果 \\(p(x)>0\\)，但 \\(q(x)\\) 很小，KL 会很大。

这在模仿学习中有工程意义：如果专家经常访问某些状态—动作对，而策略分布几乎不给这些地方概率，那说明策略不像专家。

但反过来，策略访问了专家没怎么访问的地方，又是另一个问题。这就是为什么不同方向的 KL 会有不同偏好。

---

## C.9 交叉熵、熵和 KL 的关系

熵定义为：

<div class="math-block">
\[
H(p)
=
-
\sum_x p(x)\log p(x)
\tag{C.21}
\]
</div>

交叉熵是：

<div class="math-block">
\[
H(p,q)
=
-
\sum_x p(x)\log q(x)
\tag{C.22}
\]
</div>

KL 可以写成：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)
=
H(p,q)-H(p)
\tag{C.23}
\]
</div>

推导如下：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)
=
\sum_x p(x)\log\frac{p(x)}{q(x)} \tag{C.24}\]
</div>

利用 \\(\log\frac{p(x)}{q(x)}=\log p(x)-\log q(x)\\)：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)
=
\sum_x p(x)\log p(x)
-
\sum_x p(x)\log q(x) \tag{C.25}\]
</div>

由于：

<div class="math-block">
\[
H(p)=-\sum_x p(x)\log p(x) \tag{C.26}\]
</div>

所以：

<div class="math-block">
\[
\sum_x p(x)\log p(x)=-H(p) \tag{C.27}\]
</div>

代入得到：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)
=
-H(p)+H(p,q) \tag{C.28}\]
</div>

也就是：

<div class="math-block">
\[
D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p) \tag{C.29}\]
</div>

如果真实分布 \\(p\\) 固定，那么最小化交叉熵 \\(H(p,q)\\) 等价于最小化 KL。因为 \\(H(p)\\) 与模型 \\(q\\) 无关。

这解释了为什么分类训练中的交叉熵可以看成让模型分布靠近标签分布。

---

## C.10 GAIL 中的交叉熵味道

GAIL 的判别器目标类似 GAN：

<div class="math-block">
\[
\max_D
\mathbb E_{(s,a)\sim\rho_{\pi_E}}[\log D(s,a)]
+
\mathbb E_{(s,a)\sim\rho_{\pi}}[\log(1-D(s,a))]
\tag{C.24}
\]
</div>

这个式子可以看成二分类交叉熵：

- 专家样本标签为 1，希望 \\(D(s,a)\\) 接近 1；
- 策略样本标签为 0，希望 \\(D(s,a)\\) 接近 0。

所以 GAIL 的数学外表虽然高级，但判别器训练其实很像二分类器训练。它的关键不在“会不会分类”，而在：策略会根据判别器反馈改变自己的轨迹分布。

---

## C.11 常见误解

### 误解 1：MLE 就是让模型记住数据

不完全是。MLE 是让模型给观测数据高概率。如果模型容量太大、正则不足、数据太少，确实可能记忆。但 MLE 本身不是“背答案”的同义词。

### 误解 2：NLL 和交叉熵完全是两种东西

在 one-hot 分类标签下，交叉熵就是专家类别的 NLL。很多时候它们只是不同语境下的同一个训练目标。

### 误解 3：KL 是距离

KL 可以衡量分布差异，但不是严格数学意义上的距离，因为它不对称，也不一定满足三角不等式。

### 误解 4：KL 越小总是越好

要看上下文。CVAE 中 KL 太大可能导致 latent 空间推理时不可用；但 KL 太小也可能出现 posterior collapse，隐变量不干活。数学指标不能脱离模型行为看。

---

## C.12 本附录小结

本附录串起了四个核心概念：

1. **MLE**：让模型给真实数据高概率；
2. **log-likelihood**：把概率连乘变成 log 求和；
3. **NLL**：把最大化 log-likelihood 变成最小化 loss；
4. **交叉熵**：分类任务中常见的 NLL 形式；
5. **KL 散度**：衡量两个分布不匹配的一种非对称度量。

如果一句话总结：

> 这些公式不是在炫技，它们都在做同一件事：让模型分布更像我们希望它像的分布。

---

## C.13 小练习

1. 请从最大似然目标推出行为克隆的 NLL loss。

2. 为什么 \\(\prod_i p_\theta(x_i)\\) 常常改写成 \\(\sum_i\log p_\theta(x_i)\\)？

3. one-hot 标签下，交叉熵为什么等于正确类别的负 log 概率？

4. 请解释 \\(D_{\mathrm{KL}}(p\|q)\neq D_{\mathrm{KL}}(q\|p)\\) 的直觉。

5. 在 CVAE 中，KL 项太大和太小分别可能带来什么问题？
