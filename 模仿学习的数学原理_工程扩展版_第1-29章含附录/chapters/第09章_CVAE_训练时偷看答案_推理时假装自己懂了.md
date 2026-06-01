# 第9章 CVAE：训练时偷看答案，推理时假装自己懂了

> **本章一句话导读**：
> 第8章引入了隐变量 $z$，用它表示动作背后的隐藏风格、意图或模式。但真实示教数据通常没有 $z$ 标签。CVAE 的核心思路是：训练时让 encoder 看着条件 $x$ 和专家动作 $a$ 推断 $z$，再让 decoder 根据 $x$ 和 $z$ 重建动作；推理时没有专家动作，只能从 prior 中采样 $z$，再由 decoder 生成动作。

第8章已经给出隐变量策略的数学形式：

**公式 (9.1)：隐变量条件动作生成**

$$
z \sim p(z), \quad a \sim p_\theta(a\mid x,z)
$$

其中 $x$ 表示策略生成动作时能看到的条件信息。它可以是当前图像观测、机器人关节状态、历史观测、语言指令、目标图像、任务 ID、多相机视觉特征，或者 Transformer 编码后的上下文。

第8章留下了一个关键问题：

> **如果 $z$ 没有标注，训练时怎么知道某个动作对应哪个隐藏模式？**

比如同一个杯子状态下，示教数据里有三种成功动作：从左侧抓、从右侧抓、绕开障碍后抓。我们希望模型学到：这些动作不是一锅粥，而是对应不同 latent $z$。

问题是，数据集里通常只有 $(x,a)$，没有 $(x,a,z)$。

CVAE，就是为这个问题设计的一类模型。

---

## 9.1 本章公式主线

本章承接第8章的隐变量策略：

```text
第8章：用隐变量 z 表达隐藏动作模式
→ 真实数据只有 (x,a)，没有 z 标签
→ 引入 encoder q_phi(z|x,a)，训练时根据条件和专家动作推断 z
→ 引入 decoder p_theta(a|x,z)，根据条件和 z 生成动作
→ 真正想最大化 log p_theta(a|x)
→ 但 p_theta(a|x)=int p_theta(a|x,z)p(z)dz 通常不好直接计算
→ 引入近似后验 q_phi(z|x,a)
→ 用 Jensen 不等式得到 ELBO
→ ELBO = 重建项 - KL 项
→ 训练时 encoder 可以看动作，推理时不能看动作
→ decoder 才是部署时生成动作的策略主体
→ 第三篇继续讨论：除了动作分布建模，专家到底在优化什么，策略分布如何与专家匹配
```

本章最重要的判断是：

> **CVAE 的 encoder 是训练时的推断工具，decoder 才是推理时生成动作的策略主体。**

---

## 9.2 从 VAE 到 CVAE：多了一个条件

> **定义 9.1：VAE**
>
> VAE，即 Variational Autoencoder，是一种带隐变量的生成模型。它用隐变量 $z$ 解释数据，并学习如何从 $z$ 生成数据。

普通 VAE 的生成过程可以写成：

**公式 (9.2)：VAE 先验采样**

$$
z\sim p(z)
$$

**公式 (9.3)：VAE 数据生成**

$$
a\sim p_\theta(a\mid z)
$$

这里 $a$ 可以是图片、语音、轨迹或动作片段。普通 VAE 关心的是：如何用 latent $z$ 表达数据的主要变化因素。

但模仿学习不是无条件生成动作。机器人不是闭着眼睛随机挥手。它必须根据当前观测、任务目标和自身状态来行动。

所以我们需要条件。

> **定义 9.2：CVAE**
>
> CVAE，即 Conditional Variational Autoencoder，是在条件 $x$ 下学习隐变量生成模型的方法。它用 $z$ 表示隐藏模式，用 decoder 建模 $p_\theta(a\mid x,z)$，并用 encoder 近似推断训练样本背后的 $z$。

CVAE 的生成过程写成：

**公式 (9.4)：CVAE 先验采样**

$$
z\sim p(z)
$$

**公式 (9.5)：CVAE 条件动作生成**

$$
a\sim p_\theta(a\mid x,z)
$$

在模仿学习中，$x$ 是机器人执行策略时能看到的信息，$a$ 是专家动作。

于是 CVAE 可以被理解为：

> **在同一个条件 $x$ 下，先采样一个动作风格 $z$，再生成对应动作 $a$。**

---

## 9.3 Encoder：训练时偷看答案

> **定义 9.3：CVAE encoder**
>
> CVAE encoder 是训练时使用的近似后验网络，通常记作 $q_\phi(z\mid x,a)$。它根据条件 $x$ 和专家动作 $a$，推断这个动作可能对应的隐变量 $z$。

**公式 (9.6)：CVAE encoder**

$$
q_\phi(z\mid x,a)
$$

这个公式读作：在已知条件 $x$ 和动作 $a$ 的情况下，encoder 给出隐变量 $z$ 的近似后验分布。

其中：

- $q_\phi$：由参数 $\phi$ 控制的 encoder；
- $x$：条件输入；
- $a$：专家动作；
- $z$：隐变量。

为什么说 encoder “偷看答案”？因为训练时，encoder 可以看到真实专家动作 $a$。

例如同一个杯子状态下：

- 如果专家动作是左抓，encoder 可以把它编码到某个 latent 区域；
- 如果专家动作是右抓，encoder 可以把它编码到另一个 latent 区域；
- 如果专家动作是绕开障碍后抓，encoder 可以编码成第三种模式。

这就是“偷看答案”：训练时动作 $a$ 已知，所以 encoder 可以用它来推断隐藏风格。

---

## 9.4 Decoder：真正生成动作的策略主体

> **定义 9.4：CVAE decoder**
>
> CVAE decoder 是根据条件 $x$ 和隐变量 $z$ 生成动作的网络，通常记作 $p_\theta(a\mid x,z)$。在模仿学习中，它可以看成 latent-conditioned policy。

**公式 (9.7)：CVAE decoder**

$$
p_\theta(a\mid x,z)
$$

这个公式读作：给定条件 $x$ 和隐变量 $z$，decoder 给出动作 $a$ 的概率分布。

在机器人策略中，decoder 才是真正用于推理生成动作的主体。

推理时流程是：

**公式 (9.8)：CVAE 推理过程**

$$
z\sim p(z), \quad a\sim p_\theta(a\mid x,z)
$$

如果使用确定性 decoder，也可以写成：

**公式 (9.9)：确定性 decoder 输出**

$$
\hat a=f_\theta(x,z)
$$

在抓取任务中，decoder 可以这样工作：

```text
输入 x：当前图像、机械臂状态、目标位置；
输入 z：某种隐藏抓取模式；
输出 a：末端位姿增量、抓取位姿或动作块。
```

不同 $z$ 可以生成不同动作候选。这就是 CVAE 缓解多模态平均问题的核心机制。

---

## 9.5 训练时和推理时到底哪里不同

CVAE 最容易被误解的地方，就是训练和推理的信息条件不同。

| 阶段 | 可用信息 | 使用哪个网络 | $z$ 从哪里来 | 输出什么 |
|---|---|---|---|---|
| 训练 | 条件 $x$ 和专家动作 $a$ | encoder + decoder | $q_\phi(z\mid x,a)$ | 重建专家动作 $a$ |
| 推理 | 只有条件 $x$ | decoder | $p(z)$ 或 $p_\psi(z\mid x)$ | 生成动作候选 |

训练时可以用：

**公式 (9.10)：训练时 latent 来源**

$$
z\sim q_\phi(z\mid x,a)
$$

推理时只能用：

**公式 (9.11)：推理时 latent 来源**

$$
z\sim p(z)
$$

或者使用条件先验：

**公式 (9.12)：条件先验采样**

$$
z\sim p_\psi(z\mid x)
$$

真实部署时没有专家动作 $a$。机器人不能先知道专家会怎么做，再决定自己怎么做。

所以推理时不能使用 $q_\phi(z\mid x,a)$。如果测试时还把真实动作 $a$ 输入 encoder，相当于信息泄漏，不能代表真实策略能力。

---

## 9.6 为什么直接最大化条件 likelihood 很困难

我们真正想最大化的是条件动作概率：

**公式 (9.13)：条件 log likelihood**

$$
\log p_\theta(a\mid x)
$$

如果引入隐变量 $z$，那么：

**公式 (9.14)：边缘 likelihood**

$$
p_\theta(a\mid x)=\int p_\theta(a\mid x,z)p(z)dz
$$

所以：

**公式 (9.15)：边缘 log likelihood**

$$
\log p_\theta(a\mid x)=\log\int p_\theta(a\mid x,z)p(z)dz
$$

问题在于，这个积分通常不好算。

因为 $z$ 是连续高维变量，decoder 又是神经网络。你要把所有可能的 $z$ 都积分一遍，几乎不可行。

所以 CVAE 不直接最大化这个难算的目标，而是构造一个可以优化的下界：ELBO。

---

## 9.7 ELBO：从边缘 likelihood 推出来

> **定义 9.5：ELBO**
>
> ELBO，即 Evidence Lower Bound，是对 log likelihood 的一个可优化下界。CVAE 通过最大化 ELBO 来间接提高 $\log p_\theta(a\mid x)$。

这一节给出基础推导。推导不追求最抽象形式，只要读者能看清楚每一步为什么成立。

### 9.7.1 第一步：从条件 log likelihood 出发

我们从真正想优化的目标开始：

**公式 (9.16)：目标 log likelihood**

$$
\log p_\theta(a\mid x)
$$

引入隐变量后，动作概率是对 $z$ 做边缘化：

**公式 (9.17)：边缘化形式**

$$
p_\theta(a\mid x)=\int p_\theta(a,z\mid x)dz
$$

其中联合概率可以拆成：

**公式 (9.18)：联合概率分解**

$$
p_\theta(a,z\mid x)=p_\theta(a\mid x,z)p(z)
$$

于是：

**公式 (9.19)：带隐变量的边缘 likelihood**

$$
\log p_\theta(a\mid x)=\log\int p_\theta(a\mid x,z)p(z)dz
$$

### 9.7.2 第二步：乘除一个近似后验

因为直接对所有 $z$ 积分很难，我们引入一个容易采样的分布 $q_\phi(z\mid x,a)$。它表示：训练时看到了 $x$ 和 $a$ 之后，$z$ 可能是什么。

在积分里乘除同一个 $q_\phi(z\mid x,a)$：

**公式 (9.20)：引入近似后验**

$$
\log p_\theta(a\mid x)=\log\int q_\phi(z\mid x,a)\frac{p_\theta(a\mid x,z)p(z)}{q_\phi(z\mid x,a)}dz
$$

这一步没有改变数值。它只是把原来的积分写成了对 $q_\phi$ 的期望。

**公式 (9.21)：写成期望形式**

$$
\log p_\theta(a\mid x)=\log \mathbb{E}_{z\sim q_\phi(z\mid x,a)}\left[\frac{p_\theta(a\mid x,z)p(z)}{q_\phi(z\mid x,a)}\right]
$$

### 9.7.3 第三步：使用 Jensen 不等式

对数函数是凹函数。Jensen 不等式告诉我们：

**公式 (9.22)：Jensen 不等式直觉形式**

$$
\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]
$$

因此：

**公式 (9.23)：得到下界**

$$
\log p_\theta(a\mid x)\geq \mathbb{E}_{z\sim q_\phi(z\mid x,a)}\left[\log \frac{p_\theta(a\mid x,z)p(z)}{q_\phi(z\mid x,a)}\right]
$$

把对数里的分式拆开：

**公式 (9.24)：拆开对数**

$$
\log \frac{p_\theta(a\mid x,z)p(z)}{q_\phi(z\mid x,a)}=\log p_\theta(a\mid x,z)+\log p(z)-\log q_\phi(z\mid x,a)
$$

带回期望：

**公式 (9.25)：ELBO 展开形式**

$$
\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]+\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p(z)-\log q_\phi(z\mid x,a)]
$$

第二项可以写成 KL 散度的负号：

**公式 (9.26)：KL 项关系**

$$
\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p(z)-\log q_\phi(z\mid x,a)] = -D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

于是得到 CVAE 的 ELBO：

**公式 (9.27)：CVAE ELBO**

$$
\log p_\theta(a\mid x)\geq \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

### 9.7.4 ELBO 的两项分别在管什么

ELBO 可以分成两项。

第一项是重建项：

**公式 (9.28)：重建项**

$$
\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]
$$

它鼓励 decoder 在 encoder 推断出的 $z$ 条件下，能够生成专家动作 $a$。

第二项是 KL 约束项：

**公式 (9.29)：KL 约束项**

$$
D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

它约束训练时的 posterior 不要离推理时的 prior 太远。

![图9-3 ELBO两项含义拆解图](../images/图9-3_ELBO两项含义拆解图.png)

**图9-3 说明**：这张图想说明的问题是：ELBO 里的两个项分别在训练什么。重建项鼓励 decoder 根据 $x$ 和 $z$ 还原专家动作，KL 项约束训练时的 posterior 不要偏离推理时的 prior。读者要记住的是：CVAE 不是只追求重建动作，还要让推理采样可用。

ELBO 可以用一句话理解：

```text
一方面，z 要能帮助 decoder 重建专家动作；
另一方面，z 不能只在训练时好用，推理时从 prior 采样也要能用。
```

如果没有重建项，模型不会学会动作。

如果没有 KL 项，encoder 可能把答案藏进 $z$，训练重建很好，推理从 prior 采样却崩掉。

---

## 9.8 从最大化 ELBO 到最小化训练损失

论文里常写最大化 ELBO，代码里通常写最小化 loss。两者只差一个负号。

最大化目标可以写成：

**公式 (9.30)：最大化 ELBO**

$$
\max_{\theta,\phi}\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

等价的最小化损失是：

**公式 (9.31)：CVAE 损失**

$$
\mathcal{L}_{\mathrm{CVAE}}(\theta,\phi)=-\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]+D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

实践中经常写成加权形式：

**公式 (9.32)：加权 CVAE 损失**

$$
\mathcal{L}_{\mathrm{CVAE}}=\mathcal{L}_{\mathrm{rec}}+\beta\mathcal{L}_{\mathrm{KL}}
$$

其中：

**公式 (9.33)：重建损失**

$$
\mathcal{L}_{\mathrm{rec}}=-\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]
$$

**公式 (9.34)：KL 损失**

$$
\mathcal{L}_{\mathrm{KL}}=D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

$\beta$ 是一个权重。它像一个音量旋钮，用来控制 KL 项说话多大声。

如果 $\beta$ 太小，KL 约束弱，encoder 可以随心所欲地把每个动作编码到很偏的 latent 区域。训练重建可能很好，但推理时从 $p(z)$ 采样会跑偏。

如果 $\beta$ 太大，encoder 被迫非常接近 prior，$z$ 里装不下有用信息。decoder 只能主要依赖 $x$，模型容易退化成普通 BC。这就是 latent collapse 或 posterior collapse 的风险之一。

---

## 9.9 重参数化技巧：让随机采样也能反向传播

> **定义 9.6：重参数化技巧**
>
> 重参数化技巧把从参数化分布中采样的问题，改写成对固定噪声的确定性变换，从而让梯度能够传回分布参数。

CVAE 训练时需要从 encoder 分布采样：

**公式 (9.35)：从 encoder 采样 latent**

$$
z\sim q_\phi(z\mid x,a)
$$

如果 $q_\phi$ 是高斯分布：

**公式 (9.36)：高斯 encoder**

$$
q_\phi(z\mid x,a)=\mathcal{N}(z;\mu_\phi(x,a),\mathrm{diag}(\sigma_\phi^2(x,a)))
$$

直接采样 $z$ 会让梯度传播变得麻烦。重参数化技巧把采样写成：

**公式 (9.37)：标准噪声采样**

$$
\epsilon\sim\mathcal{N}(0,I)
$$

**公式 (9.38)：重参数化形式**

$$
z=\mu_\phi(x,a)+\sigma_\phi(x,a)\odot\epsilon
$$

这样，随机性由 $\epsilon$ 提供，而 $\mu_\phi$ 和 $\sigma_\phi$ 是神经网络输出，可以接受梯度。

PyTorch 中常见写法是：

```python
mu, logvar = encoder(x, a)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

注意这里经常输出 $\log \sigma^2$，也就是 `logvar`。原因是方差必须为正，而直接预测方差可能出现负数；预测 `logvar` 再指数化更稳定。

---

## 9.10 KL 散度在 CVAE 中到底管什么

KL 项经常被粗略解释成“让 latent 更像正态分布”。这句话不算错，但太粗糙。

在 CVAE 中，KL 至少有三层作用。

### 9.10.1 作用一：让训练时 latent 和推理时 latent 对齐

训练时 decoder 看到的 $z$ 来自 posterior：

$$
z\sim q_\phi(z\mid x,a)
$$

推理时 decoder 看到的 $z$ 来自 prior：

$$
z\sim p(z)
$$

KL 项让 $q_\phi$ 不要离 $p(z)$ 太远。这样推理时从 $p(z)$ 采样，decoder 才不至于一脸懵。

### 9.10.2 作用二：压缩动作信息，避免 latent 变成记忆小抄

如果没有 KL，encoder 可能把动作 $a$ 几乎完整地塞进 $z$。那 decoder 重建当然很好，但这不是学会了动作模式，而是学会了把答案藏进 latent。

训练集上很香，推理时很惨。因为推理时没有 $a$，也就没有那张小抄。

### 9.10.3 作用三：让 latent 空间可采样

一个好 latent 空间应该满足：从 prior 中采样到的 $z$，大多能生成合理动作。

如果训练时 latent 分布散落在很多奇怪角落，prior 采样就会很危险。KL 项把这些区域往 prior 分布附近拉，让 latent 空间更连续、更可采样。

### 9.10.4 高斯 KL 的闭式形式

当：

**公式 (9.39)：高斯 posterior**

$$
q_\phi(z\mid x,a)=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2))
$$

并且：

**公式 (9.40)：标准高斯 prior**

$$
p(z)=\mathcal{N}(0,I)
$$

KL 有常用闭式形式：

**公式 (9.41)：高斯 KL 闭式形式**

$$
D_{\mathrm{KL}}(q\|p)=\frac{1}{2}\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)
$$

这个公式让我们不用采样估计 KL，可以直接根据 encoder 输出的 $\mu$ 和 $\sigma$ 计算 KL 损失。

代码里常见写法是：

```python
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
```

这和上面的公式等价，只是写法换了一下。

**常见误解**：KL 越小不一定越好。如果 KL 接近 0，可能表示 $q$ 完全贴着 prior，但也可能表示 latent 没有携带动作信息。训练时要同时看重建质量、多样性、KL 数值和闭环表现。

---

## 9.11 CVAE 如何缓解多模态动作平均问题

普通 BC 学的是：

**公式 (9.42)：普通 BC 的确定性动作**

$$
\hat a=f_\theta(x)
$$

如果同一个 $x$ 对应多个动作峰，MSE 会倾向于预测平均动作。

CVAE 学的是：

**公式 (9.43)：CVAE 的 latent-conditioned 动作**

$$
\hat a=f_\theta(x,z)
$$

同一个 $x$，不同 $z$ 可以生成不同动作：

**公式 (9.44)：不同 latent 生成不同动作**

$$
\hat a_1=f_\theta(x,z_1), \quad \hat a_2=f_\theta(x,z_2), \quad \hat a_3=f_\theta(x,z_3)
$$

如果训练得好，$z_1,z_2,z_3$ 分别对应不同动作模式。模型不需要用一个点去讨好所有数据，而是可以用不同 latent 去解释不同示教。

![图9-4 latent z控制动作风格示意图](../images/图9-4_latent_z控制动作风格示意图.png)

**图9-4 说明**：这张图想说明的问题是：CVAE 如何避免把多个动作模式平均成一个动作。同一条件 $x$ 下，不同 $z$ 可以生成不同动作峰。读者要记住的是：如果所有 $z$ 输出几乎一样，说明 latent 可能被忽略。

在抓取任务中可以这样理解：

```text
encoder 学会：左抓动作对应一类 z，右抓动作对应另一类 z；
decoder 学会：给定左抓 latent 输出左抓动作，给定右抓 latent 输出右抓动作；
推理时：从 prior 中采样多个 z，生成多个候选动作，再由安全模块筛选。
```

---

## 9.12 条件 prior：为什么有时 prior 也要看 x

到目前为止，我们写的是简单 prior：

**公式 (9.45)：无条件 prior**

$$
p(z)=\mathcal{N}(0,I)
$$

这表示 latent 的先验不依赖条件 $x$。

但在一些任务中，不同状态下合理风格本身就不同。

比如：

- 空旷环境下可以快速接近；
- 障碍物很近时更应该保守；
- 目标物体偏左时左侧抓可能更自然；
- 车位很窄时激进入库风格不应出现。

这时可以使用条件 prior。

> **定义 9.7：条件 prior**
>
> 条件 prior $p_\psi(z\mid x)$ 表示隐变量分布依赖当前条件 $x$。它让模型根据场景决定哪些隐藏风格更合理。

**公式 (9.46)：条件 prior**

$$
p_\psi(z\mid x)
$$

对应的 ELBO 变成：

**公式 (9.47)：带条件 prior 的 ELBO**

$$
\log p_\theta(a\mid x)\geq \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p_\psi(z\mid x))
$$

条件 prior 更灵活，但也更复杂：

- 多一个网络 $p_\psi(z\mid x)$；
- 训练稳定性更需要关注；
- posterior 和 conditional prior 的对齐更难诊断；
- 如果数据偏，prior 可能学到不安全偏好。

所以工程上不一定一上来就用复杂 prior。小任务可以先从标准高斯 prior 开始，把问题跑通，再考虑条件 prior。

---

## 9.13 CVAE 在 ACT 中的角色预告

ACT 处理的是双臂精细操作，往往不是预测一步动作，而是预测动作块：

**公式 (9.48)：动作块**

$$
a_{t:t+H}
$$

ACT 中的 CVAE 思路可以理解为：

- encoder 在训练时读取一段专家动作，压缩成 latent $z$；
- decoder 根据当前观测、机器人状态和 $z$，预测未来一段 action chunk；
- 推理时没有专家动作片段，只能使用 prior 或默认 latent。

这和本章完全一致，只是 $a$ 从单步动作变成了动作序列：

**公式 (9.49)：动作块 encoder**

$$
q_\phi(z\mid x_t,a_{t:t+H})
$$

**公式 (9.50)：动作块 decoder**

$$
p_\theta(a_{t:t+H}\mid x_t,z)
$$

为什么这有用？因为双臂操作里，很多动作差异不是单步差异，而是整段动作风格差异。

比如拉拉链：先固定左手再右手拉，先右手调整角度再双手协同，一次性拉到底，或者分段拉、边拉边修正。这些模式用单步动作很难表达，用 action chunk 加 latent 更自然。

---

## 9.14 机器人工程中的使用方式

### 9.14.1 生成多个候选动作

一个常见用法是：对同一个 $x$，采样多个 $z$：

**公式 (9.51)：多次采样 latent**

$$
z^{(m)}\sim p(z), \quad m=1,2,\dots,M
$$

生成多个候选动作：

**公式 (9.52)：多个候选动作**

$$
a^{(m)}\sim p_\theta(a\mid x,z^{(m)})
$$

然后用一个评分或安全模块选择：

**公式 (9.53)：候选动作安全筛选**

$$
a^*=\arg\min_{a^{(m)}}C_{\mathrm{safe}}(x,a^{(m)})
$$

这里 $C_{\mathrm{safe}}$ 可以包含：碰撞风险、关节限位、速度和加速度约束、目标距离、轨迹平滑性、抓取稳定性、控制器可跟踪性、任务成功评分。

CVAE 本身负责给出多样候选，安全模块负责不让机器人把“多样性”理解成“花式作死”。

### 9.14.2 用 latent 表达操作风格

在一些任务里，可以人为控制 $z$：

- 固定 $z$，让策略行为更稳定；
- 插值 $z$，观察动作风格变化；
- 聚类 $z$，分析数据中的示教模式；
- 为不同任务阶段选择不同 latent。

但要注意，latent 的可解释性需要实验证明。不要看到二维投影上有几团点，就立刻给每一团起名字叫“聪明”“保守”“优雅”。

### 9.14.3 open-loop loss 不够，还要看 closed-loop

CVAE 训练 loss 低，只说明模型在训练分布上能重建动作。它不保证闭环成功。

必须检查：

- 多次采样动作是否真的多样；
- 候选动作是否可执行；
- rollout 中是否累积误差；
- 采样 latent 是否导致动作抖动；
- 长时程任务中风格是否一致；
- 失败状态下是否有恢复能力。

第6章已经讲过：机器人活在闭环里。CVAE 也逃不出这个现实。

---

## 9.15 常见误区

| 常见误区 | 正确认识 |
|---|---|
| CVAE 一定比 BC 好 | 如果任务近似单峰，简单 BC 可能更稳定 |
| encoder 推出来的 $z$ 就是真实风格标签 | $z$ 是模型学出的 latent 表示，不一定等同人类标签 |
| KL 越小越好 | KL 太小可能表示 latent 没有携带信息 |
| 重建 loss 低就说明多模态学好了 | 还要看 prior 采样、多样性和闭环表现 |
| 推理时可以继续用 $q_\phi(z\mid x,a)$ | 真实部署没有专家动作 $a$，这样做是信息泄漏 |
| 采样越多越安全 | 采样只增加候选，安全来自筛选和约束 |
| latent 维度越大越好 | 维度过大可能过拟合、collapse 或学习脏模式 |

---

## 9.16 方法边界与工程风险

### 9.16.1 Posterior collapse

posterior collapse 指的是模型忽略 latent。表现为：

- KL 很小；
- 不同 $z$ 输出动作几乎一样；
- latent 插值没有明显变化；
- 多次采样没有多样性；
- decoder 主要依赖 $x$。

### 9.16.2 Prior mismatch

prior mismatch 指训练时 posterior 和推理时 prior 对不上。表现为：

- 训练重建很好；
- 推理采样动作质量差；
- 从 prior 采样时动作不稳定；
- 某些 latent 区域生成无意义动作。

排查方法包括：可视化 posterior 分布、从 prior 采样检查动作、对比 posterior sample 和 prior sample 的生成质量。

### 9.16.3 Mode dropping

CVAE 也可能丢模式。比如数据里有左抓和右抓，但模型只学会左抓。原因可能是：

- 数据不平衡；
- KL 约束过强；
- latent 维度不足；
- decoder 偏向高频模式；
- 训练目标没有鼓励覆盖低频成功动作。

在机器人里，低频模式不一定不重要。有时低频模式正是异常场景下的救命动作。

### 9.16.4 Latent 学到脏因素

如果数据里混有失败轨迹、标定错误、控制延迟，CVAE 可能把这些差异也编码进 $z$。

模型不会自动知道哪些差异叫“风格”，哪些差异叫“事故前兆”。

所以使用 CVAE 前，数据清洗和标签质量仍然重要。生成模型不是洗衣机，不能把脏数据倒进去，出来就是香喷喷的策略。

### 9.16.5 安全筛选不能省

CVAE 能生成多个候选动作，但真实机器人执行前仍要经过安全检查。

尤其是机械臂和车辆任务，必须关注碰撞、关节限位、控制频率、动作平滑性、任务约束、人机安全和 fallback 策略。

CVAE 提供“可能性”，工程系统必须提供“边界”。

---

## 9.17 读完本章，你应该能判断什么

读完本章后，你应该能形成以下判断：

1. **判断 CVAE 是否适合当前任务**：如果任务动作近似单峰，简单 BC 可能够用；如果动作多模态，CVAE 才更有价值。
2. **判断 encoder 和 decoder 的分工**：encoder 是训练时推断工具，decoder 才是推理时策略主体。
3. **判断训练和推理是否信息泄漏**：测试或部署时不能把真实动作输入 encoder。
4. **判断 ELBO 两项的作用**：重建项让动作生成准确，KL 项让训练时 posterior 和推理时 prior 对齐。
5. **判断 KL 权重是否异常**：KL 太弱可能 prior mismatch，KL 太强可能 posterior collapse。
6. **判断多样性是否真实有效**：不要只看重建 loss，要看 prior 采样、latent 插值、候选动作质量和 closed-loop rollout。
7. **判断安全模块是否必要**：CVAE 生成候选动作，不代表候选动作都可执行。
8. **判断第9章和第三篇的关系**：CVAE 解决动作分布建模，但还没有回答专家目标和分布匹配问题。

---

## 9.18 本章小结

本章回答了第8章留下的问题：没有 $z$ 标签时，如何训练隐变量策略。

核心要点如下：

1. CVAE 在条件 $x$ 下引入隐变量 $z$，用 $p_\theta(a\mid x,z)$ 生成动作；
2. encoder $q_\phi(z\mid x,a)$ 是训练时的近似后验，可以看专家动作；
3. decoder $p_\theta(a\mid x,z)$ 才是推理时生成动作的策略主体；
4. 直接优化 $\log p_\theta(a\mid x)$ 通常困难，因为需要对 $z$ 积分；
5. ELBO 通过 Jensen 不等式给出一个可优化下界；
6. 重建项鼓励 decoder 还原专家动作，KL 项约束 posterior 接近 prior；
7. 推理时不能使用真实动作输入 encoder，否则是信息泄漏；
8. CVAE 能缓解多模态平均问题，但仍可能出现 posterior collapse、prior mismatch、mode dropping 和安全风险。

本章最重要的一句话是：

> **CVAE 训练时用 encoder 从“条件 + 专家动作”里推断隐藏模式，推理时只能用 decoder 根据“条件 + 采样 latent”生成动作。**

第二篇到这里完成了从序列决策到概率生成策略的基础铺垫：第5章建立 MDP，第6章建立轨迹目标，第7章说明确定性 MSE 的多模态风险，第8章引入隐变量，第9章给出 CVAE 训练隐变量策略的方法。

但 CVAE 仍主要解决“动作如何生成”和“多模态动作如何建模”的问题。它还没有回答两个更深的问题：

```text
专家到底在优化什么？
学习策略访问到的状态—动作分布，如何整体接近专家？
```

第三篇将进入经典模仿学习的分布匹配与奖励视角：IRL、GAIL 和 Offline Imitation Learning。

---

## 9.19 本章公式索引

### 公式 (9.6)：CVAE encoder

$$
q_\phi(z\mid x,a)
$$

- **含义**：训练时根据条件和专家动作推断隐变量。
- **需要掌握到什么程度**：理解 encoder 是训练时工具，推理时不能偷看真实动作。

### 公式 (9.7)：CVAE decoder

$$
p_\theta(a\mid x,z)
$$

- **含义**：给定条件和隐变量后生成动作。
- **需要掌握到什么程度**：理解 decoder 才是推理时策略主体。

### 公式 (9.15)：边缘 log likelihood

$$
\log p_\theta(a\mid x)=\log\int p_\theta(a\mid x,z)p(z)dz
$$

- **含义**：动作 likelihood 需要对未知隐变量积分。
- **需要掌握到什么程度**：理解为什么直接优化困难。

### 公式 (9.27)：CVAE ELBO

$$
\log p_\theta(a\mid x)\geq \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

- **含义**：条件 log likelihood 的可优化下界。
- **需要掌握到什么程度**：理解重建项和 KL 项分别起什么作用。

### 公式 (9.31)：CVAE 损失

$$
\mathcal{L}_{\mathrm{CVAE}}(\theta,\phi)=-\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]+D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))
$$

- **含义**：最大化 ELBO 的负号形式。
- **需要掌握到什么程度**：理解代码中通常最小化 loss。

### 公式 (9.38)：重参数化形式

$$
z=\mu_\phi(x,a)+\sigma_\phi(x,a)\odot\epsilon
$$

- **含义**：把随机采样改写成固定噪声加确定性变换。
- **需要掌握到什么程度**：理解为什么 VAE/CVAE 可以端到端训练。

### 公式 (9.41)：高斯 KL 闭式形式

$$
D_{\mathrm{KL}}(q\|p)=\frac{1}{2}\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)
$$

- **含义**：高斯 posterior 与标准高斯 prior 的 KL。
- **需要掌握到什么程度**：理解代码中 KL loss 的来源。

### 公式 (9.47)：带条件 prior 的 ELBO

$$
\log p_\theta(a\mid x)\geq \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p_\psi(z\mid x))
$$

- **含义**：用依赖条件的 prior 替代固定 prior。
- **需要掌握到什么程度**：理解条件 prior 更灵活，但训练和诊断更复杂。

---

## 9.20 本章定义索引

| 编号 | 概念 | 一句话含义 |
|---|---|---|
| 定义 9.1 | VAE | 用隐变量解释数据并生成数据的模型 |
| 定义 9.2 | CVAE | 在条件 $x$ 下训练隐变量生成模型 |
| 定义 9.3 | CVAE encoder | 训练时根据 $x$ 和 $a$ 推断 $z$ 的近似后验网络 |
| 定义 9.4 | CVAE decoder | 根据 $x$ 和 $z$ 生成动作的网络 |
| 定义 9.5 | ELBO | log likelihood 的可优化下界 |
| 定义 9.6 | 重参数化技巧 | 把采样改写成固定噪声的确定性变换 |
| 定义 9.7 | 条件 prior | 依赖条件 $x$ 的隐变量先验 |

---

## 9.21 建议阅读的附录条目

- **附录 B：概率论最小生存包**：理解条件概率、期望、Jensen 不等式和边缘化。
- **附录 C：最大似然、负对数似然、交叉熵与 KL 散度**：理解 likelihood、NLL 和 KL 项。
- **附录 G：生成模型基础**：系统补充 VAE、CVAE、ELBO 和 latent collapse。
- **附录 H：实验与代码基础**：理解 open-loop 重建、prior sample、rollout 和多样性评估。

---

## 9.22 思考题

1. 为什么 CVAE 比普通 VAE 更适合模仿学习中的动作生成？
2. 为什么说 encoder 训练时可以“偷看答案”？
3. 推理时为什么不能使用 $q_\phi(z\mid x,a)$？
4. decoder 为什么可以看成 latent-conditioned policy？
5. 为什么直接优化 $\log p_\theta(a\mid x)$ 很困难？
6. 请用自己的话解释 ELBO 推导中的 Jensen 不等式步骤。
7. ELBO 里的重建项和 KL 项分别在约束什么？
8. 如果 KL 太小或者太大，分别可能出现什么问题？
9. 为什么 CVAE 能缓解多模态动作平均问题？
10. 为什么 CVAE 仍然需要 closed-loop rollout 和安全筛选？
11. 第三篇为什么要继续讨论 IRL 和 GAIL？它们和 CVAE 关注的问题有什么不同？

---

## 9.23 本章配图清单

- 图9-3：ELBO 两项含义拆解图；
- 图9-4：latent $z$ 控制动作风格示意图。