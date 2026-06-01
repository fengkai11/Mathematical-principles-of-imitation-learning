# 第9章：CVAE：训练时偷看答案，推理时假装自己懂了

> **本章一句话导读**：
> 第8章引入了隐变量 $z$，用它表示动作背后的隐藏风格、意图或模式。但真实示教数据通常没有 $z$ 标签。CVAE 的核心思路是：训练时让 encoder 看着条件 $x$ 和专家动作 $a$ 推断 $z$，再让 decoder 根据 $x$ 和 $z$ 重建动作；推理时没有专家动作，只能从 prior 中采样 $z$，再由 decoder 生成动作。

---

## 9.1 从第8章留下的问题开始

第8章说明，同一个条件 $x$ 下，动作可以由隐变量 $z$ 控制：

**公式 (9.1)：隐变量条件动作生成**

$$z\sim p(z),\quad a\sim p_\theta(a\mid x,z)$$

这里用 $x$ 表示策略生成动作时能看到的条件信息。它可以是：

- 当前图像观测 $o_t$；
- 机器人关节状态 $q_t$；
- 历史观测 $o_{t-k:t}$；
- 语言指令；
- 目标图像；
- 任务 ID；
- 多相机视觉特征；
- Transformer 编码后的一串 token。

第8章留下了一个关键问题：如果 $z$ 没有标注，训练时怎么知道某个动作对应哪个风格？

比如同一个杯子状态下，示教数据里有三种成功动作：

- 从左侧抓；
- 从右侧抓；
- 绕开障碍后抓。

我们希望模型学到：这些动作不是一锅粥，而是对应不同 latent $z$。问题是，数据集里通常只有 $(x,a)$，没有 $(x,a,z)$。

这就像老师给了你很多解题答案，但没有告诉你每道题用了哪种解法套路。你知道有人用了代数法，有人用了几何法，有人直接背公式，但卷子上只写了最后答案。

CVAE，就是为这个问题设计的一类模型。

它的核心套路是：

```text
训练时：encoder 看 x 和真实动作 a，推断一个可能的 z；
训练时：decoder 根据 x 和 z 重建动作 a；
推理时：真实动作 a 不存在，只能从 prior 中采样 z，再由 decoder 生成动作。
```

这就是本章标题里的“训练时偷看答案，推理时假装自己懂了”。

---

## 9.2 本章要解决的核心问题

本章围绕 9 个问题展开：

1. VAE 和 CVAE 的区别是什么？为什么模仿学习里更常用 CVAE？
2. CVAE 中的 encoder $q_\phi(z\mid x,a)$ 到底在干什么？
3. decoder $p_\theta(a\mid x,z)$ 为什么可以看成 latent-conditioned policy？
4. 为什么训练时可以输入真实动作 $a$，推理时却不能？
5. 为什么 $\log p_\theta(a\mid x)$ 直接优化很困难？
6. ELBO 是怎么来的？它的两项分别是什么意思？
7. KL 散度在 CVAE 中到底管什么？
8. CVAE 为什么能缓解多模态动作的平均问题？
9. CVAE 在机器人模仿学习中有哪些常见坑？

本章的核心判断是：

> CVAE 的 encoder 是训练时的推断工具，decoder 才是推理时生成动作的策略主体。

---

## 9.3 本章的主线定位

第8章定义了隐变量策略。本章回答：没有 $z$ 标签时，如何训练这种策略？

```text
第7章：多模态动作需要条件动作分布
        ↓
第8章：用隐变量 z 表达隐藏模式
        ↓
第9章：用 CVAE 学习没有标签的 z
        ↓
第13章：ACT 将 CVAE 用到 action chunk
```

本章的公式阅读抓手是：

```text
训练时的 q_phi(z | x, a) 可以看答案；
推理时没有 a，只能用 prior 或 conditional prior 产生 z。
```

---

## 9.4 从 VAE 到 CVAE：多了一个条件

> **定义 9.1：VAE**
>
> VAE，即 Variational Autoencoder，是一种带隐变量的生成模型。它用隐变量 $z$ 解释数据，并学习如何从 $z$ 生成数据。

普通 VAE 的生成过程可以写成：

**公式 (9.2)：VAE 先验采样**

$$z\sim p(z)$$

**公式 (9.3)：VAE 数据生成**

$$a\sim p_\theta(a\mid z)$$

这里 $a$ 可以是图片、语音、轨迹或动作片段。普通 VAE 关心的是：如何用 latent $z$ 表达数据的主要变化因素。

但模仿学习不是无条件生成动作。机器人不是闭着眼睛随机挥手。它必须根据当前观测、任务目标和自身状态来行动。

所以我们需要条件。

> **定义 9.2：CVAE**
>
> CVAE，即 Conditional Variational Autoencoder，是在条件 $x$ 下学习隐变量生成模型的方法。它用 $z$ 表示隐藏模式，用 decoder 建模 $p_\theta(a\mid x,z)$，并用 encoder 近似推断训练样本背后的 $z$。

CVAE 的生成过程写成：

**公式 (9.4)：CVAE 先验采样**

$$z\sim p(z)$$

**公式 (9.5)：CVAE 条件动作生成**

$$a\sim p_\theta(a\mid x,z)$$

在模仿学习中，$x$ 是机器人执行策略时能看到的信息，$a$ 是专家动作。

于是 CVAE 可以被理解为：

> 在同一个条件 $x$ 下，先采样一个动作风格 $z$，再生成对应动作 $a$。

---

## 9.5 Encoder：训练时偷看答案

> **定义 9.3：CVAE encoder**
>
> CVAE encoder 是训练时使用的近似后验网络，通常记作 $q_\phi(z\mid x,a)$。它根据条件 $x$ 和专家动作 $a$，推断这个动作可能对应的隐变量 $z$。

**公式 (9.6)：CVAE encoder**

$$q_\phi(z\mid x,a)$$

这个公式读作：在已知条件 $x$ 和动作 $a$ 的情况下，encoder 给出隐变量 $z$ 的近似后验分布。

其中：

- $q_\phi$：由参数 $\phi$ 控制的 encoder；
- $x$：条件输入；
- $a$：专家动作；
- $z$：隐变量。

### 9.5.1 为什么说 encoder 偷看答案？

因为训练时，encoder 可以看到真实专家动作 $a$。

例如同一个杯子状态下：

- 如果专家动作是左抓，encoder 可以把它编码到某个 latent 区域；
- 如果专家动作是右抓，encoder 可以把它编码到另一个 latent 区域；
- 如果专家动作是绕开障碍后抓，encoder 可以编码成第三种模式。

这就是“偷看答案”：训练时动作 $a$ 已知，所以 encoder 可以用它来推断隐藏风格。

### 9.5.2 推理时不能再偷看

真实部署时没有专家动作 $a$。机器人不能先知道专家会怎么做，再决定自己怎么做。

所以推理时不能使用 $q_\phi(z\mid x,a)$，只能从 $p(z)$ 或条件先验 $p_\psi(z\mid x)$ 中得到 $z$。

这是 CVAE 在工程评估中最容易犯的错误之一：

> 如果测试时还把真实动作 $a$ 输入 encoder，相当于信息泄漏，不能代表真实策略能力。

---

## 9.6 Decoder：真正生成动作的策略主体

> **定义 9.4：CVAE decoder**
>
> CVAE decoder 是根据条件 $x$ 和隐变量 $z$ 生成动作的网络，通常记作 $p_\theta(a\mid x,z)$。在模仿学习中，它可以看成 latent-conditioned policy。

**公式 (9.7)：CVAE decoder**

$$p_\theta(a\mid x,z)$$

这个公式读作：给定条件 $x$ 和隐变量 $z$，decoder 给出动作 $a$ 的概率分布。

在机器人策略中，decoder 才是真正用于推理生成动作的主体。

推理时流程是：

**公式 (9.8)：CVAE 推理过程**

$$z\sim p(z),\quad a\sim p_\theta(a\mid x,z)$$

如果使用确定性 decoder，也可以写成：

**公式 (9.9)：确定性 decoder 输出**

$$\hat a=f_\theta(x,z)$$

### 9.6.1 工程含义

在抓取任务中，decoder 可以这样工作：

- 输入 $x$：当前图像、机械臂状态、目标位置；
- 输入 $z$：某种隐藏抓取模式；
- 输出 $a$：末端位姿增量、抓取位姿或动作块。

不同 $z$ 可以生成不同动作候选。

这就是 CVAE 缓解多模态平均问题的核心机制。

---

## 9.7 为什么直接最大化条件 likelihood 很困难？

我们真正想最大化的是条件动作概率：

**公式 (9.10)：条件 log likelihood**

$$\log p_\theta(a\mid x)$$

如果引入隐变量 $z$，那么：

**公式 (9.11)：边缘 likelihood**

$$p_\theta(a\mid x)=\int p_\theta(a\mid x,z)p(z)dz$$

所以：

**公式 (9.12)：边缘 log likelihood**

$$\log p_\theta(a\mid x)=\log\int p_\theta(a\mid x,z)p(z)dz$$

问题在于，这个积分通常不好算。

因为 $z$ 是连续高维变量，decoder 又是神经网络。你要把所有可能的 $z$ 都积分一遍，几乎不可行。

所以 CVAE 不直接最大化这个难算的目标，而是构造一个可以优化的下界：ELBO。

---

## 9.8 ELBO：一个可训练的下界

> **定义 9.5：ELBO**
>
> ELBO，即 Evidence Lower Bound，是对 log likelihood 的一个可优化下界。CVAE 通过最大化 ELBO 来间接提高 $\log p_\theta(a\mid x)$。

CVAE 的 ELBO 可以写成：

**公式 (9.13)：CVAE ELBO**

$$\log p_\theta(a\mid x)\ge \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

这个公式看起来长，但可以分成两项。

第一项：

**公式 (9.14)：重建项**

$$\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]$$

它鼓励 decoder 在 encoder 推断出的 $z$ 条件下，能够生成专家动作 $a$。

第二项：

**公式 (9.15)：KL 约束项**

$$D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

它约束训练时的 posterior 不要离推理时的 prior 太远。

![图9-3 ELBO两项含义拆解图](../images/图9-3_ELBO两项含义拆解图.png)

**图9-3 说明**：

- ELBO 第一项鼓励 decoder 重建动作；
- KL 项约束 posterior 不要远离 prior；
- 两项之间存在张力，实际训练中常用 $\beta$ 或 KL annealing 调节。

### 9.8.1 ELBO 的工程读法

ELBO 可以用一句话理解：

```text
一方面，z 要能帮助 decoder 重建专家动作；
另一方面，z 不能只在训练时好用，推理时从 prior 采样也要能用。
```

如果没有重建项，模型不会学会动作。

如果没有 KL 项，encoder 可能把答案藏进 $z$，训练重建很好，推理从 prior 采样却崩掉。

---

## 9.9 从最大化 ELBO 到最小化训练损失

论文里常写最大化 ELBO，代码里通常写最小化 loss。两者只差一个负号。

最大化目标可以写成：

**公式 (9.16)：最大化 ELBO**

$$\max_{\theta,\phi}\;\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

等价的最小化损失是：

**公式 (9.17)：CVAE 损失**

$$\mathcal{L}_{\mathrm{CVAE}}(\theta,\phi)=-\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]+D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

实践中经常写成加权形式：

**公式 (9.18)：加权 CVAE 损失**

$$\mathcal{L}_{\mathrm{CVAE}}=\mathcal{L}_{\mathrm{rec}}+\beta\mathcal{L}_{\mathrm{KL}}$$

其中：

**公式 (9.19)：重建损失**

$$\mathcal{L}_{\mathrm{rec}}=-\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]$$

**公式 (9.20)：KL 损失**

$$\mathcal{L}_{\mathrm{KL}}=D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

$\beta$ 是一个权重。它像一个音量旋钮，用来控制 KL 项说话多大声。

### 9.9.1 beta 太小会怎样？

如果 $\beta$ 太小，KL 约束弱，encoder 可以随心所欲地把每个动作编码到很偏的 latent 区域。

训练重建可能很好，但推理时从 $p(z)$ 采样会跑偏。

这叫：训练时风光无限，推理时原形毕露。

### 9.9.2 beta 太大会怎样？

如果 $\beta$ 太大，encoder 被迫非常接近 prior，$z$ 里装不下有用信息。decoder 只能主要依赖 $x$，模型容易退化成普通 BC。

这就是 latent collapse 或 posterior collapse 的风险之一。

### 9.9.3 KL annealing 的直觉

实际训练中，有时会让 $\beta$ 从小逐渐变大：

**公式 (9.21)：KL annealing**

$$\beta_t:0\rightarrow 1$$

直觉是：先让模型学会重建动作，再逐渐要求 latent 分布规整。

---

## 9.10 重参数化技巧：让随机采样也能反向传播

> **定义 9.6：重参数化技巧**
>
> 重参数化技巧把从参数化分布中采样的问题，改写成对固定噪声的确定性变换，从而让梯度能够传回分布参数。

CVAE 训练时需要从 encoder 分布采样：

**公式 (9.22)：从 encoder 采样 latent**

$$z\sim q_\phi(z\mid x,a)$$

如果 $q_\phi$ 是高斯分布：

**公式 (9.23)：高斯 encoder**

$$q_\phi(z\mid x,a)=\mathcal{N}(z;\mu_\phi(x,a),\mathrm{diag}(\sigma_\phi^2(x,a)))$$

直接采样 $z$ 会让梯度传播变得麻烦。重参数化技巧把采样写成：

**公式 (9.24)：标准噪声采样**

$$\epsilon\sim\mathcal{N}(0,I)$$

**公式 (9.25)：重参数化形式**

$$z=\mu_\phi(x,a)+\sigma_\phi(x,a)\odot\epsilon$$

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

## 9.11 KL 散度在 CVAE 中到底管什么？

KL 项经常被粗略解释成“让 latent 更像正态分布”。这句话不算错，但太粗糙。

在 CVAE 中，KL 至少有三层作用。

### 9.11.1 作用一：让训练时 latent 和推理时 latent 对齐

训练时 decoder 看到的 $z$ 来自 posterior：

**公式 (9.26)：训练时 latent 来源**

$$z\sim q_\phi(z\mid x,a)$$

推理时 decoder 看到的 $z$ 来自 prior：

**公式 (9.27)：推理时 latent 来源**

$$z\sim p(z)$$

KL 项让 $q_\phi$ 不要离 $p(z)$ 太远。这样推理时从 $p(z)$ 采样，decoder 才不至于一脸懵。

### 9.11.2 作用二：压缩动作信息，避免 latent 变成记忆小抄

如果没有 KL，encoder 可能把动作 $a$ 几乎完整地塞进 $z$。那 decoder 重建当然很好，但这不是学会了动作模式，而是学会了把答案藏进 latent。

训练集上很香，推理时很惨。因为推理时没有 $a$，也就没有那张小抄。

### 9.11.3 作用三：让 latent 空间可采样

一个好 latent 空间应该满足：从 prior 中采样到的 $z$，大多能生成合理动作。

如果训练时 latent 分布散落在很多奇怪角落，prior 采样就会很危险。KL 项把这些区域往 prior 分布附近拉，让 latent 空间更连续、更可采样。

### 9.11.4 高斯 KL 的闭式形式

当：

**公式 (9.28)：高斯 posterior**

$$q_\phi(z\mid x,a)=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2))$$

并且：

**公式 (9.29)：标准高斯 prior**

$$p(z)=\mathcal{N}(0,I)$$

KL 有常用闭式形式：

**公式 (9.30)：高斯 KL 闭式形式**

$$D_{\mathrm{KL}}(q\|p)=\frac{1}{2}\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$$

这个公式让我们不用采样估计 KL，可以直接根据 encoder 输出的 $\mu$ 和 $\sigma$ 计算 KL 损失。

代码里常见写法是：

```python
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
```

这和上面的公式等价，只是写法换了一下。

### 9.11.5 常见误解

KL 越小不一定越好。

如果 KL 接近 0，可能表示 $q$ 完全贴着 prior，但也可能表示 latent 没有携带动作信息。训练时要同时看重建质量、多样性、KL 数值和闭环表现。

---

## 9.12 CVAE 如何缓解多模态动作平均问题？

普通 BC 学的是：

**公式 (9.31)：普通 BC 的确定性动作**

$$\hat a=f_\theta(x)$$

如果同一个 $x$ 对应多个动作峰，MSE 会倾向于预测平均动作。

CVAE 学的是：

**公式 (9.32)：CVAE 的 latent-conditioned 动作**

$$\hat a=f_\theta(x,z)$$

同一个 $x$，不同 $z$ 可以生成不同动作：

**公式 (9.33)：不同 latent 生成不同动作**

$$\hat a_1=f_\theta(x,z_1),\quad \hat a_2=f_\theta(x,z_2),\quad \hat a_3=f_\theta(x,z_3)$$

如果训练得好，$z_1,z_2,z_3$ 分别对应不同动作模式。模型不需要用一个点去讨好所有数据，而是可以用不同 latent 去解释不同示教。

![图9-4 latent z控制动作风格示意图](../images/图9-4_latent_z控制动作风格示意图.png)

**图9-4 说明**：

- 同一条件 $x$ 下，不同 $z$ 可以生成不同动作峰；
- CVAE 的目标不是让所有动作取平均，而是把多种模式保留下来；
- 如果所有 $z$ 输出几乎一样，说明 latent 可能被忽略。

### 9.12.1 抓取例子

假设训练数据中，同一个物体有两类抓法：左抓和右抓。

普通 BC 用 MSE 时，可能输出中间抓法。中间抓法既不像左抓，也不像右抓，夹爪可能正好撞到物体或抓不到稳定点。

CVAE 中，encoder 在训练时看到专家动作：

- 左抓动作会被编码到某些 $z$ 区域；
- 右抓动作会被编码到另一些 $z$ 区域。

decoder 学会：

- 给定左抓 latent，输出左抓动作；
- 给定右抓 latent，输出右抓动作。

推理时，从 prior 中采样多个 $z$，可以得到多个候选动作，再通过碰撞检测、抓取评分器或规则筛选选择一个安全动作。

---

## 9.13 条件 prior：为什么有时 prior 也要看 x？

到目前为止，我们写的是简单 prior：

**公式 (9.34)：无条件 prior**

$$p(z)=\mathcal{N}(0,I)$$

这表示 latent 的先验不依赖条件 $x$。

但在一些任务中，不同状态下合理风格本身就不同。

比如：

- 空旷环境下可以快速接近；
- 障碍物很近时更应该保守；
- 目标物体偏左时左侧抓可能更自然；
- 车位很窄时激进入库风格不应出现。

这时可以使用条件 prior：

> **定义 9.7：条件 prior**
>
> 条件 prior $p_\psi(z\mid x)$ 表示隐变量分布依赖当前条件 $x$。它让模型根据场景决定哪些隐藏风格更合理。

**公式 (9.35)：条件 prior**

$$p_\psi(z\mid x)$$

对应的 ELBO 变成：

**公式 (9.36)：带条件 prior 的 ELBO**

$$\log p_\theta(a\mid x)\ge \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p_\psi(z\mid x))$$

条件 prior 更灵活，但也更复杂：

- 多一个网络 $p_\psi(z\mid x)$；
- 训练稳定性更需要关注；
- posterior 和 conditional prior 的对齐更难诊断；
- 如果数据偏，prior 可能学到不安全偏好。

所以工程上不一定一上来就用复杂 prior。小任务可以先从标准高斯 prior 开始，把问题跑通，再考虑条件 prior。

---

## 9.14 CVAE 在 ACT 中的角色预告

ACT 处理的是双臂精细操作，往往不是预测一步动作，而是预测动作块：

**公式 (9.37)：动作块**

$$a_{t:t+H}$$

ACT 中的 CVAE 思路可以理解为：

- encoder 在训练时读取一段专家动作，压缩成 latent $z$；
- decoder 根据当前观测、机器人状态和 $z$，预测未来一段 action chunk；
- 推理时没有专家动作片段，只能使用 prior 或默认 latent。

这和本章完全一致，只是 $a$ 从单步动作变成了动作序列：

**公式 (9.38)：动作块 encoder**

$$q_\phi(z\mid x_t,a_{t:t+H})$$

**公式 (9.39)：动作块 decoder**

$$p_\theta(a_{t:t+H}\mid x_t,z)$$

为什么这有用？因为双臂操作里，很多动作差异不是单步差异，而是整段动作风格差异。

比如拉拉链：

- 先固定左手，再右手拉；
- 先右手调整角度，再双手协同；
- 一次性拉到底；
- 分段拉、边拉边修正。

这些模式用单步动作很难表达，用 action chunk 加 latent 更自然。

---

## 9.15 机器人工程中的使用方式

### 9.15.1 生成多个候选动作

一个常见用法是：对同一个 $x$，采样多个 $z$：

**公式 (9.40)：多次采样 latent**

$$z^{(m)}\sim p(z),\quad m=1,2,\dots,M$$

生成多个候选动作：

**公式 (9.41)：多个候选动作**

$$a^{(m)}\sim p_\theta(a\mid x,z^{(m)})$$

然后用一个评分或安全模块选择：

**公式 (9.42)：候选动作安全筛选**

$$a^*=\arg\min_{a^{(m)}}C_{\mathrm{safe}}(x,a^{(m)})$$

这里 $C_{\mathrm{safe}}$ 可以包含：

- 碰撞风险；
- 关节限位；
- 速度和加速度约束；
- 目标距离；
- 轨迹平滑性；
- 抓取稳定性；
- 控制器可跟踪性；
- 任务成功评分。

CVAE 本身负责给出多样候选，安全模块负责不让机器人把“多样性”理解成“花式作死”。

### 9.15.2 用 latent 表达操作风格

在一些任务里，可以人为控制 $z$：

- 固定 $z$，让策略行为更稳定；
- 插值 $z$，观察动作风格变化；
- 聚类 $z$，分析数据中的示教模式；
- 为不同任务阶段选择不同 latent。

但要注意，latent 的可解释性需要实验证明。不要看到二维投影上有几团点，就立刻给每一团起名字叫“聪明”“保守”“优雅”。

### 9.15.3 open-loop loss 不够，还要看 closed-loop

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

## 9.16 常见误区

### 误区一：CVAE 一定比 BC 好

不一定。如果任务本身近似单峰，数据量又不大，CVAE 可能增加训练难度，甚至不如简单 BC 稳定。

CVAE 的优势主要在多模态动作、动作风格差异、动作块生成等场景。

### 误区二：encoder 推出来的 z 就是真实风格标签

不是。$z$ 是模型为了优化目标学出的 latent 表示。它可能对应真实风格，也可能混入数据采集偏差、操作者习惯、外参误差或时间延迟。

要判断 $z$ 是否有意义，需要做干预实验和可视化分析。

### 误区三：KL 越小越好

KL 太小可能表示 posterior 贴近 prior，但也可能表示 latent 没有携带信息。此时模型退化成普通条件生成器，不再表达多模态。

### 误区四：重建 loss 低就说明多模态学好了

重建 loss 低可能是 encoder 把答案藏进 latent，也可能是数据本身容易拟合。要检查推理时从 prior 采样是否仍能生成多样且合理的动作。

### 误区五：推理时可以继续用 q(z|x,a)

真实部署时没有专家动作 $a$。如果评测时使用真实 $a$ 输入 encoder，相当于信息泄漏。这个实验结果不能代表实际策略能力。

### 误区六：采样越多越安全

采样更多候选只增加选择机会，不自动增加安全性。如果筛选器不可靠，采样更多只会更快找到一个看起来新颖、实际上危险的动作。

### 误区七：latent 维度越大越好

高维 latent 表达能力更强，但更容易过拟合、collapse 或学习脏模式。选择 latent 维度要结合任务复杂度、数据量和闭环评测。

---

## 9.17 方法边界与工程风险

### 9.17.1 Posterior collapse

posterior collapse 指的是模型忽略 latent。表现为：

- KL 很小；
- 不同 $z$ 输出动作几乎一样；
- latent 插值没有明显变化；
- 多次采样没有多样性；
- decoder 主要依赖 $x$。

### 9.17.2 Prior mismatch

prior mismatch 指训练时 posterior 和推理时 prior 对不上。表现为：

- 训练重建很好；
- 推理采样动作质量差；
- 从 prior 采样时动作不稳定；
- 某些 latent 区域生成无意义动作。

排查方法包括：可视化 posterior 分布、从 prior 采样检查动作、对比 posterior sample 和 prior sample 的生成质量。

### 9.17.3 Mode dropping

CVAE 也可能丢模式。比如数据里有左抓和右抓，但模型只学会左抓。原因可能是：

- 数据不平衡；
- KL 约束过强；
- latent 维度不足；
- decoder 偏向高频模式；
- 训练目标没有鼓励覆盖低频成功动作。

在机器人里，低频模式不一定不重要。有时低频模式正是异常场景下的救命动作。

### 9.17.4 Latent 学到脏因素

如果数据里混有失败轨迹、标定错误、控制延迟，CVAE 可能把这些差异也编码进 $z$。

模型不会自动知道哪些差异叫“风格”，哪些差异叫“事故前兆”。

所以使用 CVAE 前，数据清洗和标签质量仍然重要。生成模型不是洗衣机，不能把脏数据倒进去，出来就是香喷喷的策略。

### 9.17.5 安全筛选不能省

CVAE 能生成多个候选动作，但真实机器人执行前仍要经过安全检查。

尤其是机械臂和车辆任务，必须关注：

- 碰撞；
- 关节限位；
- 控制频率；
- 动作平滑性；
- 任务约束；
- 人机安全；
- fallback 策略。

CVAE 提供“可能性”，工程系统必须提供“边界”。

---

## 9.18 读完本章，你应该能判断什么？

读完本章后，你应该能形成以下判断：

1. **判断 CVAE 是否适合当前任务**：如果任务动作近似单峰，简单 BC 可能够用；如果动作多模态，CVAE 才更有价值。
2. **判断 encoder 和 decoder 的分工**：encoder 是训练时推断 latent 的工具，decoder 才是推理时生成动作的策略主体。
3. **判断是否存在信息泄漏**：评测时如果使用真实动作输入 encoder，就不能代表真实部署能力。
4. **判断 ELBO 两项的作用**：重建项负责动作像不像，KL 项负责 latent 是否能从 prior 采样。
5. **判断 beta 设置风险**：$\beta$ 太小可能 prior mismatch，$\beta$ 太大可能 posterior collapse。
6. **判断 latent 是否真的有用**：要看采样多样性、latent 插值、posterior/prior 对齐和 closed-loop 表现。
7. **判断 CVAE 是否缓解了平均动作**：不能只看重建 loss，要看推理时多次采样是否覆盖多个合理模式。
8. **判断部署安全边界**：CVAE 生成候选动作后仍需要安全筛选和 fallback。

---

## 9.19 本章小结

本章把第8章的隐变量思想推进到了可训练模型：CVAE。

第8章告诉我们，动作分布可以通过 latent $z$ 建模。本章进一步说明：训练数据没有 $z$ 标签，所以需要 encoder：

**公式 (9.43)：encoder 训练时推断 latent**

$$q_\phi(z\mid x,a)$$

它在训练时看着条件 $x$ 和真实动作 $a$，近似推断动作背后的 latent。decoder 则负责在给定 $x$ 和 $z$ 时生成动作：

**公式 (9.44)：decoder 生成动作**

$$p_\theta(a\mid x,z)$$

由于边缘 likelihood 通常不好直接优化：

**公式 (9.45)：难以直接优化的边缘 likelihood**

$$\log p_\theta(a\mid x)=\log\int p_\theta(a\mid x,z)p(z)dz$$

我们使用 ELBO：

**公式 (9.46)：CVAE 的可训练下界**

$$\log p_\theta(a\mid x)\ge \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

本章最重要的几句话是：

1. CVAE 的 encoder 是训练时的推断工具，不是推理时的动作生成入口；
2. decoder 才是根据 $(x,z)$ 生成动作的策略主体；
3. ELBO 由重建项和 KL 项组成，一个负责动作像不像，一个负责 latent 能不能采样；
4. 训练时 posterior 和推理时 prior 必须对齐，否则部署会翻车；
5. CVAE 可以缓解多模态动作平均问题，但不自动保证闭环成功；
6. 机器人系统中，CVAE 生成的多样候选动作必须经过安全筛选；
7. CVAE 是 ACT 的重要前置知识，因为 ACT 会把 latent 与 action chunk 结合起来。

下一章进入第三篇，开始讨论经典模仿学习中的分布匹配与奖励视角。等到第13章 ACT 时，我们会再次遇到 CVAE，只不过那时 $a$ 会从单步动作变成动作块 $a_{t:t+H}$。

---

## 9.20 本章公式索引

### 公式 (9.1)：隐变量条件动作生成

$$z\sim p(z),\quad a\sim p_\theta(a\mid x,z)$$

- **含义**：先从 prior 中采样 hidden mode，再根据条件和 latent 生成动作。
- **需要掌握到什么程度**：理解这是第8章隐变量策略到第9章 CVAE 的入口。

### 公式 (9.6)：CVAE encoder

$$q_\phi(z\mid x,a)$$

- **含义**：训练时根据条件和专家动作推断 latent。
- **需要掌握到什么程度**：理解 encoder 可以看答案，但推理时不能用真实动作。

### 公式 (9.7)：CVAE decoder

$$p_\theta(a\mid x,z)$$

- **含义**：给定条件和 latent 后生成动作。
- **需要掌握到什么程度**：理解 decoder 才是推理时的策略主体。

### 公式 (9.13)：CVAE ELBO

$$\log p_\theta(a\mid x)\ge \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

- **含义**：用可优化下界替代难算的条件 log likelihood。
- **需要掌握到什么程度**：理解重建项和 KL 项分别解决什么问题。

### 公式 (9.17)：CVAE 损失

$$\mathcal{L}_{\mathrm{CVAE}}(\theta,\phi)=-\mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]+D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p(z))$$

- **含义**：最小化形式的 CVAE 训练目标。
- **需要掌握到什么程度**：理解最大化 ELBO 和最小化 loss 只差一个负号。

### 公式 (9.18)：加权 CVAE 损失

$$\mathcal{L}_{\mathrm{CVAE}}=\mathcal{L}_{\mathrm{rec}}+\beta\mathcal{L}_{\mathrm{KL}}$$

- **含义**：实践中用 $\beta$ 调节重建项和 KL 项的平衡。
- **需要掌握到什么程度**：理解 $\beta$ 过大或过小都会带来工程风险。

### 公式 (9.23)：高斯 encoder

$$q_\phi(z\mid x,a)=\mathcal{N}(z;\mu_\phi(x,a),\mathrm{diag}(\sigma_\phi^2(x,a)))$$

- **含义**：encoder 输出 latent 的均值和方差。
- **需要掌握到什么程度**：理解这是 CVAE 最常见的连续 latent 参数化方式。

### 公式 (9.25)：重参数化形式

$$z=\mu_\phi(x,a)+\sigma_\phi(x,a)\odot\epsilon$$

- **含义**：把采样写成固定噪声的可微变换。
- **需要掌握到什么程度**：理解它是让随机 latent 可以反向传播的训练技巧。

### 公式 (9.30)：高斯 KL 闭式形式

$$D_{\mathrm{KL}}(q\|p)=\frac{1}{2}\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$$

- **含义**：标准高斯 prior 下，posterior 与 prior 的 KL 可以直接计算。
- **需要掌握到什么程度**：能和代码中的 `kl = -0.5 * sum(...)` 对应起来。

### 公式 (9.36)：带条件 prior 的 ELBO

$$\log p_\theta(a\mid x)\ge \mathbb{E}_{z\sim q_\phi(z\mid x,a)}[\log p_\theta(a\mid x,z)]-D_{\mathrm{KL}}(q_\phi(z\mid x,a)\|p_\psi(z\mid x))$$

- **含义**：让 prior 根据条件 $x$ 调整 latent 分布。
- **需要掌握到什么程度**：理解条件 prior 更灵活，但训练和诊断更复杂。

### 公式 (9.39)：动作块 decoder

$$p_\theta(a_{t:t+H}\mid x_t,z)$$

- **含义**：给定当前条件和 latent，生成未来一段动作。
- **需要掌握到什么程度**：为第13章 ACT 中 CVAE + action chunk 做准备。

### 公式 (9.42)：候选动作安全筛选

$$a^*=\arg\min_{a^{(m)}}C_{\mathrm{safe}}(x,a^{(m)})$$

- **含义**：从多个 CVAE 候选动作中选择安全代价最低的动作。
- **需要掌握到什么程度**：理解生成式策略仍然需要安全筛选模块。

---

## 9.21 本章定义索引

| 编号 | 概念 | 一句话含义 |
|---|---|---|
| 定义 9.1 | VAE | 用隐变量解释并生成数据的生成模型 |
| 定义 9.2 | CVAE | 在条件 $x$ 下学习隐变量生成模型的方法 |
| 定义 9.3 | CVAE encoder | 训练时根据 $x$ 和 $a$ 推断 latent 的网络 |
| 定义 9.4 | CVAE decoder | 根据 $x$ 和 $z$ 生成动作的网络 |
| 定义 9.5 | ELBO | 条件 log likelihood 的可优化下界 |
| 定义 9.6 | 重参数化技巧 | 把随机采样改写成固定噪声的可微变换 |
| 定义 9.7 | 条件 prior | 根据条件 $x$ 调整 latent 分布的先验 |

---

## 9.22 建议阅读的附录条目

1. **附录 B：概率论最小生存包**
   - 阅读目的：理解条件概率、期望、先验、后验和边缘化。
   - 对应本章：帮助理解 $p(z)$、$q_\phi(z\mid x,a)$ 和 $p_\theta(a\mid x)$。

2. **附录 C：最大似然、负对数似然、交叉熵与 KL 散度**
   - 阅读目的：理解 likelihood、NLL、KL 和 ELBO 的基本关系。
   - 对应本章：直接支撑 CVAE 训练目标。

3. **附录 D：高斯分布、MSE 与连续动作回归**
   - 阅读目的：理解高斯分布、均值方差、MSE 和连续动作建模。
   - 对应本章：帮助理解高斯 encoder 和重建损失。

4. **附录 G：生成模型基础**
   - 阅读目的：理解 latent variable model、VAE、CVAE 和生成式策略的基本框架。
   - 对应本章：这是本章最直接的生成模型附录。

5. **附录 I：熵、最大熵与 Score Matching**
   - 阅读目的：为后续 diffusion / flow matching 相关章节打基础。
   - 对应本章：帮助把 CVAE 和更广义的生成式策略联系起来。

---

## 9.23 思考题

1. 为什么 CVAE 的 encoder 可以在训练时输入专家动作 $a$，但推理时不能？
2. 如果一个模型训练重建 loss 很低，但从 prior 采样动作很差，你会怀疑什么问题？
3. $\beta$ 太小和太大分别会造成什么风险？
4. 如何判断 latent 是否真的表达了动作模式，而不是被 decoder 忽略？
5. 如果同一个物体既可以左抓也可以右抓，普通 BC 和 CVAE 分别可能学到什么？
6. 条件 prior $p_\psi(z\mid x)$ 相比标准高斯 prior 有什么优势和风险？
7. 为什么 CVAE 生成多个候选动作后仍然需要安全模块？
8. ACT 中为什么要把 CVAE 用到动作块 $a_{t:t+H}$，而不是只预测单步动作？

---

## 9.24 本章配图清单

本章复用并保留 2 张核心概念图：

1. **图9-3 ELBO两项含义拆解图**：解释重建项和 KL 项的分工；
2. **图9-4 latent z 控制动作风格示意图**：解释同一条件下不同 $z$ 可以生成不同动作模式。

如果后续补图，建议增加：

3. **图9-1 CVAE encoder / decoder 训练与推理对比图**：突出训练时偷看动作、推理时不能偷看的差异；
4. **图9-2 prior mismatch 与 posterior collapse 风险图**：解释 CVAE 工程调参风险。

---

## 9.25 推荐阅读与深入材料

### 9.25.1 Kingma and Welling, Auto-Encoding Variational Bayes

- **类型**：VAE 经典论文。
- **阅读目的**：理解 VAE、ELBO、重参数化技巧的来源。
- **重点看什么**：ELBO 推导、encoder / decoder、reparameterization trick。
- **对应本章**：支撑第9.8到第9.11节。

### 9.25.2 Sohn, Lee, and Yan, Learning Structured Output Representation using Deep Conditional Generative Models

- **类型**：CVAE 经典论文。
- **阅读目的**：理解条件生成模型如何使用 latent 表达多种输出。
- **重点看什么**：conditional latent variable、CVAE objective、structured output。
- **对应本章**：支撑 $q_\phi(z\mid x,a)$、$p_\theta(a\mid x,z)$ 和条件 ELBO。

### 9.25.3 Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

- **类型**：机器人模仿学习论文。
- **阅读目的**：理解 ACT 中 CVAE 如何用于 action chunk。
- **重点看什么**：CVAE、latent、action chunk、temporal ensembling。
- **对应本章**：帮助读者理解为什么第13章还会再次遇到 CVAE。

### 9.25.4 Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

- **类型**：现代生成式策略论文。
- **阅读目的**：理解从 CVAE 这类 latent variable model 走向更强动作分布建模的动机。
- **重点看什么**：multi-modal action distribution、action sequence generation、closed-loop evaluation。
- **对应本章**：帮助读者看到 CVAE 与后续 Diffusion Policy 的连续关系。
