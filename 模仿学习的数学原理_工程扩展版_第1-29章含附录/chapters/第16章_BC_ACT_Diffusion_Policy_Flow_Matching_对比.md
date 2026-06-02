# 第16章：BC / ACT / Diffusion Policy / Flow Matching 对比：从点估计到生成式动作策略

> **本章一句话导读**：本章把 BC、ACT、Diffusion Policy 和 Flow Matching 放到同一条动作建模谱系中比较，帮助读者判断：什么时候用简单点估计，什么时候需要动作块，什么时候值得引入生成式动作策略。

---

## 1. 本章为什么出现：第四篇的阶段性收束

到这里，第四篇已经连续讨论了三类现代机器人策略模型：

```text
第13章：ACT，把单步动作扩展为 action chunk；
第14章：Diffusion Policy，把动作块看成从噪声中逐步去噪得到的样本；
第15章：Flow Matching，把离散去噪过程进一步改写成连续流场生成过程。
```

如果只逐章阅读，读者很容易形成一种错觉：

> 新方法总是比旧方法强，所以 BC 不如 ACT，ACT 不如 Diffusion Policy，Diffusion Policy 又不如 Flow Matching。

这个理解很危险。

真实机器人项目不是模型选美比赛，而是任务、数据、控制频率、推理延迟、部署算力和安全边界之间的工程谈判。一个固定位置抓取任务，如果 BC 已经足够稳定，强行换成 diffusion 或 flow matching，可能不是升级，而是在给一个本来能跑通的系统增加训练、推理和安全风险。

本章的任务不是给四类方法排绝对名次，而是建立一个统一判断框架：

```text
BC：给定观测，输出一个动作点；
ACT：给定观测，输出一段动作块；
Diffusion Policy：给定观测，从噪声动作块中多步去噪得到动作块样本；
Flow Matching：给定观测，学习一条从噪声分布流向动作分布的连续速度场。
```

也就是说，本章要回答的不是“哪个模型最先进”，而是：

> 给定一个机器人任务，我们到底需要多强的动作分布表达能力？为这份表达能力要付出多少工程代价？

---

## 2. 本章公式主线

前几章的公式可以压缩成一条主线。

第2章的 BC 从最朴素的监督学习开始：给定观测 $o_t$，直接预测专家动作 $a_t$。

**公式 (16.1)：BC 的动作点估计形式**

$$\hat a_t=f_\theta(o_t)$$

它解决了“如何从专家样本训练一个策略”的问题，但如果同一观测附近存在多种合理动作，MSE 点估计可能学到条件均值，而不是某个真实可执行动作。

第13章的 ACT 把动作对象从单步动作扩展为动作块。

**公式 (16.2)：ACT 的动作块对象**

$$A_t=a_{t:t+H-1}=(a_t,a_{t+1},\dots,a_{t+H-1})$$

它解决了“单步动作缺少局部时间结构”的问题，但仍然需要考虑 chunk 长度、执行步长和闭环反馈。

第14章的 Diffusion Policy 把动作块看成条件生成样本。

**公式 (16.3)：Diffusion Policy 的加噪动作块**

$$A^{(k)}=\sqrt{\bar\alpha_k}A^{(0)}+\sqrt{1-\bar\alpha_k}\,\epsilon$$

它解决了“复杂连续多峰动作分布难以用单点预测表达”的问题，但代价是多步采样、推理延迟和更复杂的安全检查。

第15章的 Flow Matching 进一步把生成过程改写为连续流场。

**公式 (16.4)：Flow Matching 的连续流场采样**

$$\frac{dX_\tau}{d\tau}=v_\theta(X_\tau,\tau,o_t)$$

它解决的是“如何用连续速度场把噪声样本推向数据样本”的问题，为更少步数、更连续的生成式动作策略提供了另一种表达。

因此，本章统一比较的核心问题是：

```text
给定观测 o_t，策略如何表示动作对象 A_t？

点估计 → 动作块 → 离散去噪生成 → 连续流场生成
```

---

## 3. 本章核心数学对象

### 定义 16.1：动作点估计

> **动作点估计**指策略在给定观测后，只输出一个确定动作或一个分布的代表值。

最常见形式是：

**公式 (16.5)：动作点估计**

$$\hat a=f_\theta(o)$$

这里的 $o$ 是观测，$\hat a$ 是模型输出的动作，$f_\theta$ 是参数为 $\theta$ 的函数。

在工程中，它对应“当前相机图像进模型，模型直接吐出一个末端位姿增量或关节控制量”。它的优点是快、简单、容易部署；缺点是如果任务存在多种合理动作，它只能给一个答案。

### 定义 16.2：条件动作分布

> **条件动作分布**指在给定观测条件下，动作不是唯一答案，而是一个可能动作集合上的概率分布。

**公式 (16.6)：条件动作分布**

$$a\sim p_\theta(a\mid o)$$

读法是：在观测 $o$ 的条件下，从策略表示的动作分布 $p_\theta(a\mid o)$ 中采样动作 $a$。

这个对象很重要。机器人任务中，同一个视觉观测下可能存在多个正确做法：从左侧抓、从右侧抓、先推再抓、先对齐再插入。条件动作分布承认“多个答案都可能正确”。

### 定义 16.3：动作块

> **动作块**指从当前时刻开始的一段未来动作序列。

**公式 (16.7)：动作块**

$$A_t=a_{t:t+H-1}=(a_t,a_{t+1},\dots,a_{t+H-1})$$

其中 $H$ 是动作块长度，也叫 action horizon。

在机械臂任务中，动作块可以表示一段连续操作：接近工件、降低末端、闭合夹爪、轻微抬起。它比单步动作更适合表达局部时间结构。

### 定义 16.4：离散去噪生成过程

> **离散去噪生成过程**指先从随机噪声动作块开始，再经过有限步反向去噪，逐步得到可执行动作块。

**公式 (16.8)：离散去噪生成过程**

$$A^{(K)}\rightarrow A^{(K-1)}\rightarrow\cdots\rightarrow A^{(0)}$$

这里的 $A^{(K)}$ 是噪声动作块，$A^{(0)}$ 是最终生成的动作块。

Diffusion Policy 的核心不是“机器人在真实世界里随机试动作”，而是在模型内部把随机候选动作逐步修成更像专家数据的动作块。真实机器人只执行最终生成并通过安全过滤的动作。

### 定义 16.5：连续速度场生成过程

> **连续速度场生成过程**指学习一个速度场，让样本沿着连续路径从噪声分布流向专家动作分布。

**公式 (16.9)：连续速度场生成过程**

$$\frac{dX_\tau}{d\tau}=v_\theta(X_\tau,\tau,o_t),\quad \tau\in[0,1]$$

其中 $X_\tau$ 是生成路径上的中间动作样本，$v_\theta$ 是模型学习到的速度场，$o_t$ 是观测条件。

Flow Matching 可以理解成：不是一步步问“这一步噪声该怎么去掉”，而是学习“每个中间位置应该朝哪个方向流动，才能从噪声端点到达动作数据端点”。

---

## 4. 四类方法分别在建模什么

### 4.1 BC：建模观测到动作的直接映射

BC 的最朴素写法是点估计。

**公式 (16.10)：BC 点估计策略**

$$\hat a_t=f_\theta(o_t)$$

如果使用 MSE 训练，目标可以写成：

**公式 (16.11)：BC 的 MSE 训练目标**

$$\mathcal{L}_{\mathrm{BC\text{-}MSE}}(\theta)=\mathbb{E}_{(o,a)\sim\mathcal{D}}\left[\left\|a-f_\theta(o)\right\|^2\right]$$

这个目标的读法是：从专家数据集 $\mathcal{D}$ 中抽取观测—动作对，让模型预测动作尽量接近专家动作。

BC 的工程价值很高：它是所有复杂策略之前最应该做的 baseline。它能快速回答：数据是否可学、观测是否有效、动作标签是否对齐、训练评估管线是否打通。

但朴素 BC 的短板也很明显：它容易把多种专家动作平均成一个动作。

**公式 (16.12)：MSE 点估计的条件均值形式**

$$f^*(o)=\mathbb{E}[A\mid O=o]$$

如果同一个杯子既可以左抓也可以右抓，条件均值可能落在两种抓法中间。这个动作在数学上看起来“不远”，但在物理上可能谁也不是，最后夹爪对准空气。

### 4.2 ACT：建模条件动作块

ACT 的关键是把预测对象从单步动作 $a_t$ 改成动作块 $A_t$。

**公式 (16.13)：ACT 的动作块分布**

$$A_t\sim p_\theta(A_t\mid o_{\le t},z)$$

其中 $o_{\le t}$ 表示当前及历史观测，$z$ 是隐变量，用来表示动作风格或未观测因素。

一个简化的 ACT / CVAE 训练骨架可以写成：

**公式 (16.14)：ACT 的重构加 KL 目标**

$$\mathcal{L}_{\mathrm{ACT}}(\theta,\phi)=\mathbb{E}\left[\left\|A_t-f_\theta(o_{\le t},z)\right\|^2\right]+\beta D_{\mathrm{KL}}\left(q_\phi(z\mid o_{\le t},A_t)\|p(z)\right)$$

这个目标包含两层意思：

1. 预测动作块要接近专家动作块；
2. 隐变量后验 $q_\phi$ 不能偏离先验 $p(z)$ 太远，否则推理时无法从先验中稳定采样。

ACT 适合需要局部时间结构的任务。例如双臂拉拉链、插接件、精准摆入治具，这些任务不是某一帧动作正确就够了，而是一段动作的节奏、接触顺序和相对配合要正确。

### 4.3 Diffusion Policy：建模离散去噪生成过程

Diffusion Policy 同样生成动作块，但它不直接输出 $\hat A_t$，而是学习从带噪动作块到干净动作块的去噪过程。

训练时，先构造带噪动作块。

**公式 (16.15)：Diffusion 的正向加噪**

$$A^{(k)}=\sqrt{\bar\alpha_k}A^{(0)}+\sqrt{1-\bar\alpha_k}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$$

然后训练网络预测噪声。

**公式 (16.16)：Diffusion Policy 的噪声预测目标**

$$\mathcal{L}_{\mathrm{diffusion}}(\theta)=\mathbb{E}_{A^{(0)},\epsilon,k,o_t}\left[\left\|\epsilon-\epsilon_\theta(A^{(k)},k,o_t)\right\|^2\right]$$

这个目标不是直接问“专家动作是什么”，而是问：

> 给定一个带噪动作块和当前观测，动作块里哪些部分是噪声？应该如何把它修回专家动作分布附近？

Diffusion Policy 的优势在于复杂连续多峰分布建模。它可以从噪声中生成多个候选动作块，而不是被迫输出条件均值。

但它的工程代价也高：多步采样带来推理延迟；生成动作必须经过速度、加速度、碰撞、关节限位和任务约束检查；训练数据中的坏习惯也可能被更优雅地学出来。

### 4.4 Flow Matching：建模连续速度场

Flow Matching 和 Diffusion Policy 一样，都可以用于生成动作块。但二者看待生成过程的方式不同。

Diffusion Policy 更像离散修草稿：

```text
A^(K) → A^(K-1) → ... → A^(0)
```

Flow Matching 更像连续流动：

```text
X_0 沿速度场流向 X_1
```

训练时，可以把噪声端点和数据端点连接成一条路径。为了避免中文句子直接以公式作主语，这里把两个端点分开理解：

- **噪声端点**：从简单分布 $p_0$ 采样得到 $X_0$；
- **数据端点**：从专家动作块经验分布 $p_{\mathrm{data}}(A\mid o_t)$ 采样得到 $X_1$。

一种最简单的线性路径写法是：

**公式 (16.17)：Flow Matching 的线性路径**

$$X_\tau=(1-\tau)X_0+\tau X_1,\quad \tau\in[0,1]$$

对应的速度目标是：

**公式 (16.18)：线性路径的速度目标**

$$u_\tau=X_1-X_0$$

于是 Flow Matching 的训练目标可以写成：

**公式 (16.19)：Flow Matching 的速度场训练目标**

$$\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{\tau,X_0,X_1,o_t}\left[\left\|v_\theta(X_\tau,\tau,o_t)-(X_1-X_0)\right\|^2\right]$$

读法是：在路径中间点 $X_\tau$ 处，网络预测的速度 $v_\theta(X_\tau,\tau,o_t)$ 应该接近真实路径速度 $X_1-X_0$。

推理时，从噪声样本出发，沿着学到的速度场积分：

**公式 (16.20)：Flow Matching 的 ODE 采样**

$$\frac{dX_\tau}{d\tau}=v_\theta(X_\tau,\tau,o_t),\quad X_0\sim p_0$$

积分到 $\tau=1$ 附近时，得到一个动作块样本。

Flow Matching 的工程吸引力在于：它有机会用更少采样步数生成动作，同时保持较强的连续分布表达能力。但它仍然不是免费午餐：速度场是否稳定、积分步数如何选、动作约束如何加入，都会影响实机表现。

---

## 5. 关键命题：表达能力递进不等于工程安全递进

> **命题 16.1：表达能力递进不等于工程安全递进**
>
> 在连续机器人动作任务中，从 BC 到 ACT，再到 Diffusion Policy 和 Flow Matching，动作分布表达能力通常逐步增强；但表达能力增强并不自动推出闭环成功率提升，也不自动推出实机安全性提升。

**证明思路**：

我们分三步看这个命题：先说明表达能力为什么增强，再说明为什么表达能力不等于成功率，最后说明为什么生成式策略更需要安全约束。

**证明**：

第一，朴素 BC 通常学习一个点估计。

**公式 (16.21)：点估计只输出一个动作**

$$\hat a=f_\theta(o)$$

如果任务在观测 $o$ 下只有一个主要动作模式，这种表达足够。但如果存在多个动作模式，单点输出必须把多个模式压缩成一个动作。

第二，ACT 把输出对象扩展成动作块。

**公式 (16.22)：动作块表达局部时间结构**

$$A_t=(a_t,a_{t+1},\dots,a_{t+H-1})$$

这让模型可以表达一段局部动作节奏，比如“靠近—对齐—接触—推进”。因此 ACT 的表达能力强于单步动作点估计。

第三，Diffusion Policy 和 Flow Matching 进一步把动作块作为生成对象。

**公式 (16.23)：生成式动作块策略**

$$A\sim p_\theta(A\mid o)$$

它们可以从条件分布中采样动作块，因此更适合多峰连续控制。

但是，闭环成功率不仅由动作分布表达能力决定，还依赖观测稳定性、标定精度、控制器跟踪误差、动作约束、安全过滤、数据覆盖和失败恢复。一个生成式策略即使能生成很多候选动作，也可能生成碰撞动作、超速动作、夹爪不可达动作或数据中存在的坏动作模式。

所以，表达能力递进只能说明模型“更能表示复杂动作分布”，不能说明它“必然更安全、更稳定、更适合当前项目”。

**这个命题告诉我们什么？**

工程选型不能按论文新旧排序，而要按任务需求排序：

```text
任务简单、动作单峰、高频控制 → 先做 BC；
任务需要局部时间结构 → 考虑 ACT；
任务存在复杂多模态连续动作 → 考虑 Diffusion Policy 或 Flow Matching；
只要上生成式策略 → 必须同时设计安全过滤、闭环评估和失败回收。
```

**常见误解**：

不要把“能生成多模态动作”理解成“机器人可以随机多试几次”。真实机器人不是仿真里的无限复活角色，多模态生成必须被约束在安全动作集合内。

---

## 6. 四类训练目标对比

四类方法可以放在同一张训练目标表里看。

| 方法 | 训练目标 | 训练时学什么 | 主要短板 |
|---|---|---|---|
| BC | MSE / NLL | 观测到动作的映射或条件概率 | 点估计可能平均多峰动作 |
| ACT | 动作块重构 + KL | 短期动作结构和隐变量风格 | chunk 长度、隐变量退化、推理滞后 |
| Diffusion Policy | 噪声预测 MSE | 不同噪声等级下如何去噪动作块 | 多步采样、延迟、安全过滤复杂 |
| Flow Matching | 速度场 MSE | 路径中间点应该如何流向数据端点 | 速度场泛化、ODE 积分和约束处理 |

这张表有一个重要提醒：

> 不要直接比较四类 loss 的数值大小。

BC 的 MSE、ACT 的重构损失、Diffusion 的噪声预测误差、Flow Matching 的速度场误差不是同一个物理量。真实项目中更应该比较：闭环成功率、失败类型、恢复能力、延迟、动作平滑性、安全违规率和数据效率。

---

## 7. 四类推理过程对比

### 7.1 BC 推理

BC 推理最简单。

```python
class BCPolicy:
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action = self.model(obs)
        return safety_filter(action)
```

它通常只需要一次前向推理，适合高频控制和快速 baseline。

### 7.2 ACT 推理

ACT 一次生成动作块，但通常只执行前几步。

```python
class ACTPolicy:
    def __init__(self, decoder, horizon, execute_steps):
        self.decoder = decoder
        self.horizon = horizon
        self.execute_steps = execute_steps

    def act(self, obs_hist):
        z = sample_standard_normal()
        action_chunk = self.decoder(obs_hist, z)
        action_chunk = chunk_safety_filter(action_chunk)
        return action_chunk[:self.execute_steps]
```

它比 BC 多了动作块结构，适合需要局部动作节奏的任务。

### 7.3 Diffusion Policy 推理

Diffusion Policy 从随机动作块开始多步去噪。

```python
class DiffusionPolicy:
    def __init__(self, denoiser, denoise_steps, horizon, execute_steps):
        self.denoiser = denoiser
        self.denoise_steps = denoise_steps
        self.horizon = horizon
        self.execute_steps = execute_steps

    def act(self, obs):
        chunk = sample_gaussian_chunk(self.horizon)
        for k in reversed(range(self.denoise_steps)):
            pred_eps = self.denoiser(chunk, k, obs)
            chunk = denoise_one_step(chunk, pred_eps, k)
        chunk = chunk_safety_filter(chunk)
        return chunk[:self.execute_steps]
```

它的表达能力强，但采样步数会直接影响延迟。

### 7.4 Flow Matching 推理

Flow Matching 从噪声样本出发，沿速度场积分。

```python
class FlowMatchingPolicy:
    def __init__(self, velocity_model, ode_steps, horizon, execute_steps):
        self.velocity_model = velocity_model
        self.ode_steps = ode_steps
        self.horizon = horizon
        self.execute_steps = execute_steps

    def act(self, obs):
        x = sample_gaussian_chunk(self.horizon)
        tau = 0.0
        dt = 1.0 / self.ode_steps
        for _ in range(self.ode_steps):
            velocity = self.velocity_model(x, tau, obs)
            x = x + dt * velocity
            tau = tau + dt
        chunk = chunk_safety_filter(x)
        return chunk[:self.execute_steps]
```

这段伪代码不是推荐实现，只是帮助读者理解：Flow Matching 推理时依赖速度场和数值积分。

---

## 8. horizon、采样步数与控制频率

动作模型不是只看表达能力，还要看它能否塞进控制循环。

BC 通常一次前向。

**公式 (16.24)：BC 推理成本近似**

$$C_{\mathrm{BC}}\approx C_f$$

ACT 通常一次生成动作块。

**公式 (16.25)：ACT 推理成本近似**

$$C_{\mathrm{ACT}}\approx C_g$$

Diffusion Policy 需要多步去噪。

**公式 (16.26)：Diffusion 推理成本近似**

$$C_{\mathrm{diffusion}}\approx K C_\epsilon$$

Flow Matching 需要 ODE 积分若干步。

**公式 (16.27)：Flow Matching 推理成本近似**

$$C_{\mathrm{FM}}\approx N_{\mathrm{ODE}} C_v$$

其中 $K$ 是 diffusion 去噪步数，$N_{\mathrm{ODE}}$ 是 ODE 积分步数，$C_\epsilon$ 是一次噪声预测网络前向成本，$C_v$ 是一次速度场网络前向成本。

这几个式子不是精确性能模型，而是工程提醒：

> 生成式策略的表达能力越强，越要认真计算推理延迟和控制频率是否匹配。

在真实机器人中，经常使用 receding horizon execution：生成 $H$ 步，只执行前 $H_e$ 步，然后重新观测。

**公式 (16.28)：生成 horizon 与执行 horizon**

$$A_t=a_{t:t+H-1},\quad \text{execute } a_{t:t+H_e-1},\quad H_e\le H$$

这样做的原因是：动作块提供局部计划，但真实环境会变化，只执行前几步可以保留反馈能力。

---

## 9. 工程选型决策树

可以用下面这棵决策树做初步判断。

```text
第一问：任务是否短周期、动作单峰、控制频率要求高？
是 → 先做 BC baseline。
否 → 进入第二问。

第二问：任务是否明显需要局部时间结构？
例如接近、对齐、接触、推进必须连贯发生。
是 → 考虑 ACT。
否 → 继续增强 BC 或检查数据/观测。

第三问：同一观测下是否存在多个合理动作模式？
例如左抓/右抓、左绕/右绕、先碰左边/先碰右边。
是 → 考虑 Diffusion Policy 或 Flow Matching。
否 → ACT 或概率 BC 可能已经足够。

第四问：部署系统是否承受多步采样或 ODE 积分延迟？
不能 → 优先 ACT、少步生成、蒸馏或传统规划兜底。
能 → 可以试 Diffusion Policy / Flow Matching。

第五问：是否已经有安全过滤、闭环评估和失败回收？
没有 → 不要急着上复杂生成式策略。
有 → 再比较不同生成式策略的收益。
```

这棵树的核心原则是：

> 先用最便宜的方法暴露问题，再用更强的方法解决确实存在的表达能力瓶颈。

---

## 10. 机械臂抓取与精准摆入治具案例

考虑一个“抓取 + 精准摆入治具”的工业任务。

机械臂先抓取规则工件，再把工件放入托盘槽口。托盘可能有轻微变形，位置可能有偏差，工件插入时可能出现卡边、偏斜或接触异常。

### 10.1 第一阶段：BC baseline

如果工件位置稳定、槽口位置可靠、夹具设计合理，可以先做 BC。

BC 要回答的问题是：

```text
观测是否足够？
动作标签是否对齐？
简单策略能否完成大部分样本？
失败主要来自策略表达，还是来自视觉定位、标定、夹具和控制？
```

如果 BC 已经闭环成功率很高，继续堆复杂模型不一定划算。

### 10.2 第二阶段：ACT 升级

如果失败主要发生在接触前后，例如：

```text
接近槽口时动作抖动；
插入节奏不连贯；
接触后没有稳定微调；
一帧动作正确，但连续执行不顺。
```

这时 ACT 更自然。它让模型一次输出一段局部动作，表达“对齐—接触—微调—插入”的短期结构。

### 10.3 第三阶段：Diffusion Policy 或 Flow Matching 升级

如果同一个插入场景存在多种合理调整方式，例如：

```text
先靠左边定位再滑入；
先靠右边定位再旋入；
先后退一点重新对齐；
先轻碰槽边再沿边缘修正。
```

这就是多模态连续控制。朴素 MSE 可能把几种插入方式平均成一种“既不靠左也不靠右”的坏动作。生成式策略才可能发挥价值。

但这时必须同步加入：

```text
动作范围过滤；
速度和加速度限制；
碰撞检测；
力/位置异常监控；
失败回退动作；
闭环指标记录；
失败样本回收再训练。
```

否则，生成式策略只是更高级的动作生成器，不是可靠的工业系统。

---

## 11. 常见误解

### 11.1 误解一：Flow Matching 一定比 Diffusion Policy 更好

不一定。

Flow Matching 提供了连续速度场视角，可能减少采样步数，也可能更适合某些生成建模框架。但真实效果取决于数据、网络结构、积分步数、动作约束和部署条件。它不是自动替代 Diffusion Policy 的银弹。

### 11.2 误解二：Diffusion Policy 一定比 ACT 好

也不一定。

ACT 在推理速度、结构清晰度和实现复杂度上有优势。对于需要高频控制、动作模式不太复杂但需要短期连贯性的任务，ACT 可能比 Diffusion Policy 更合适。

### 11.3 误解三：BC 太老，不值得做

BC 不但值得做，而且应该优先做。

一个可靠的 BC baseline 能帮你判断问题来自哪里：数据、观测、标签、控制器、分布偏移，还是模型表达能力。没有 baseline 直接上复杂模型，失败时很难定位问题。

### 11.4 误解四：action chunk 越长越智能

chunk 长度不是智商测试。

chunk 太短，时间结构不足；chunk 太长，模型要预测太远未来，容易失去反馈能力。工程上通常生成 $H$ 步，只执行 $H_e$ 步，并保持 $H_e\le H$。

### 11.5 误解五：多模态生成等于随机多试几次

多模态生成不是让机器人摇骰子。

真正可用的多模态策略需要生成合理候选、排除危险动作、和任务约束结合，并且能够在失败时回退或重新规划。

---

## 12. 读完本章，你应该能判断什么

读完本章后，你应该能形成以下判断：

1. **BC 不是过时方法**：它是最重要、最便宜、最应该优先做的 baseline。
2. **ACT 不是简单加长输出**：它真正补的是局部时间结构和动作连续性。
3. **Diffusion Policy 不是实机随机试动作**：噪声采样发生在模型内部，真实机器人只执行最终动作块。
4. **Flow Matching 不是魔法加速器**：它把生成过程表示成连续速度场，但仍然要面对积分步数、约束和安全问题。
5. **表达能力不等于工程可靠性**：越强的生成能力，越需要安全过滤、闭环评估和失败回收。
6. **选型要从任务出发**：短周期单峰任务先 BC；局部连续操作看 ACT；复杂多模态控制再考虑 Diffusion Policy / Flow Matching。

---

## 13. 本章公式索引

### 公式 (16.1)：BC 的动作点估计形式

$$\hat a_t=f_\theta(o_t)$$

- **含义**：给定观测，直接输出一个动作。
- **需要掌握到什么程度**：理解它是最简单的策略表达，也是 BC baseline 的核心形式。

### 公式 (16.2)：ACT 的动作块对象

$$A_t=a_{t:t+H-1}=(a_t,a_{t+1},\dots,a_{t+H-1})$$

- **含义**：从当前时刻开始的一段动作序列。
- **需要掌握到什么程度**：理解 action chunk 是 ACT 相比单步 BC 的关键变化。

### 公式 (16.3)：Diffusion Policy 的加噪动作块

$$A^{(k)}=\sqrt{\bar\alpha_k}A^{(0)}+\sqrt{1-\bar\alpha_k}\,\epsilon$$

- **含义**：把专家动作块逐步加噪。
- **需要掌握到什么程度**：理解 diffusion 训练为什么需要带噪动作块。

### 公式 (16.4)：Flow Matching 的连续流场采样

$$\frac{dX_\tau}{d\tau}=v_\theta(X_\tau,\tau,o_t)$$

- **含义**：样本沿着学习到的速度场从噪声端点流向数据端点。
- **需要掌握到什么程度**：理解 Flow Matching 和离散去噪的核心区别。

### 公式 (16.11)：BC 的 MSE 训练目标

$$\mathcal{L}_{\mathrm{BC\text{-}MSE}}(\theta)=\mathbb{E}_{(o,a)\sim\mathcal{D}}\left[\left\|a-f_\theta(o)\right\|^2\right]$$

- **含义**：让预测动作接近专家动作。
- **需要掌握到什么程度**：理解 MSE 为什么容易得到点估计。

### 公式 (16.12)：MSE 点估计的条件均值形式

$$f^*(o)=\mathbb{E}[A\mid O=o]$$

- **含义**：平方误差下，单点预测的理想解是条件均值。
- **需要掌握到什么程度**：理解多模态动作被平均掉的数学原因。

### 公式 (16.14)：ACT 的重构加 KL 目标

$$\mathcal{L}_{\mathrm{ACT}}(\theta,\phi)=\mathbb{E}\left[\left\|A_t-f_\theta(o_{\le t},z)\right\|^2\right]+\beta D_{\mathrm{KL}}\left(q_\phi(z\mid o_{\le t},A_t)\|p(z)\right)$$

- **含义**：动作块要重构好，隐变量后验不要偏离先验太远。
- **需要掌握到什么程度**：理解 ACT / CVAE 训练中的两个核心目标。

### 公式 (16.16)：Diffusion Policy 的噪声预测目标

$$\mathcal{L}_{\mathrm{diffusion}}(\theta)=\mathbb{E}_{A^{(0)},\epsilon,k,o_t}\left[\left\|\epsilon-\epsilon_\theta(A^{(k)},k,o_t)\right\|^2\right]$$

- **含义**：训练网络预测动作块中加入的噪声。
- **需要掌握到什么程度**：理解 diffusion loss 不是直接的任务成功率。

### 公式 (16.19)：Flow Matching 的速度场训练目标

$$\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{\tau,X_0,X_1,o_t}\left[\left\|v_\theta(X_\tau,\tau,o_t)-(X_1-X_0)\right\|^2\right]$$

- **含义**：训练速度场预测路径中间点的运动方向。
- **需要掌握到什么程度**：理解 Flow Matching 学的是速度，不是直接预测噪声。

### 公式 (16.28)：生成 horizon 与执行 horizon

$$A_t=a_{t:t+H-1},\quad \text{execute } a_{t:t+H_e-1},\quad H_e\le H$$

- **含义**：生成一段动作，只执行前几步，然后重新观测。
- **需要掌握到什么程度**：理解 action chunk 不是 open-loop 执行许可证。

---

## 14. 本章定义索引

| 编号 | 概念 | 一句话含义 |
|---|---|---|
| 定义 16.1 | 动作点估计 | 给定观测后只输出一个动作点或代表值 |
| 定义 16.2 | 条件动作分布 | 给定观测条件下，多个动作模式形成的概率分布 |
| 定义 16.3 | 动作块 | 从当前时刻开始的一段未来动作序列 |
| 定义 16.4 | 离散去噪生成过程 | 从噪声动作块开始，经过有限步去噪得到动作块 |
| 定义 16.5 | 连续速度场生成过程 | 学习速度场，让样本从噪声分布流向动作分布 |

---

## 15. 建议阅读的附录条目

1. **附录 C：最大似然、负对数似然、交叉熵与 KL 散度**  
   用于理解 BC NLL、ACT 中的 KL 项，以及为什么不同训练目标不能直接横向比较 loss 数值。

2. **附录 D：高斯分布、MSE 与连续动作回归**  
   用于理解 MSE 为什么对应固定方差高斯策略，以及为什么 MSE 点估计会倾向条件均值。

3. **附录 G：生成模型基础**  
   用于复习 CVAE、Diffusion、Flow Matching 等生成模型的基本思想。

4. **附录 H：实验与代码基础**  
   用于理解 open-loop 评估、closed-loop rollout、成功率、动作平滑性和实验记录方式。

5. **附录 I：熵、最大熵与 Score Matching**  
   用于理解 diffusion / score / 生成式动作分布之间的关系。

---

## 16. 思考题

1. 对于一个固定位置抓取任务，如果 BC 的 closed-loop 成功率已经达到 98%，你还会考虑 ACT、Diffusion Policy 或 Flow Matching 吗？为什么？

2. 某个任务中，同一观测下专家有左抓和右抓两种模式。请用公式 $f^*(o)=\mathbb{E}[A\mid O=o]$ 解释为什么 MSE 可能输出坏动作。

3. ACT 的 action chunk 长度 $H$ 变大时，可能带来哪些好处和风险？为什么通常只执行前 $H_e$ 步？

4. Diffusion Policy 的训练 loss 是噪声预测误差。为什么这个 loss 不能直接等价于实机成功率？

5. Flow Matching 学的是速度场。请解释它和 Diffusion Policy 的“多步去噪”在推理形式上有什么不同。

6. 如果机器人控制频率要求 100Hz，而 diffusion 推理需要 80ms，你有哪些工程改造思路？

7. 在自动泊车任务中，哪些部分适合用 BC？哪些部分可能需要多模态轨迹生成？请结合泊车入库、避障和路径修正分析。

8. 如果一个 ACT 模型 open-loop 重构误差很低，但闭环执行抖动严重，你会优先排查哪些问题？

9. 请为一个“抓取 + 精准摆入治具”任务设计三阶段实验：BC baseline、ACT 升级、生成式策略升级。每阶段分别记录哪些指标？

10. 请用自己的话解释：BC、ACT、Diffusion Policy、Flow Matching 不是线性替代关系，而是工具箱关系。

---

## 17. 本章配图清单

本章建议配套以下概念讲解图。已有图片若仍准确，可以优先复用；如果旧图只覆盖三类方法，应更新为四类方法版本。

1. **图16-1 四类方法输入输出对比**  
   展示 BC 输出单步动作、ACT 输出动作块、Diffusion Policy 多步去噪生成动作块、Flow Matching 沿速度场生成动作块。

2. **图16-2 动作分布建模能力对比**  
   对比点估计、隐变量动作块、离散去噪生成、连续流场生成对多峰动作分布的表达能力。

3. **图16-3 推理成本与控制频率对比**  
   展示一次前向、动作块前向、多步去噪、ODE 积分对控制频率的影响。

4. **图16-4 工程选型决策树**  
   按任务周期、多模态程度、时间结构、部署延迟和安全条件给出方法选择路径。

5. **图16-5 抓取 + 精准摆入治具案例对比**  
   展示同一工业任务如何从 BC baseline、ACT 升级到生成式策略。

---

## 18. 下一章预告：Decision Transformer

第四篇到这里完成了一个阶段性收束：我们已经从单步动作点估计，一路走到动作块、离散去噪生成和连续流场生成。

但这些方法大多仍围绕一个问题：

> 给定当前观测，怎样生成接下来的一步或一段动作？

下一篇会继续拉长视野，进入长序列架构与多模态策略。第17章将介绍 Decision Transformer。它会把决策问题改写成条件序列建模问题，把状态、动作、回报或目标放进同一个序列里，让策略学习看起来更像语言建模。

换句话说，第四篇关心的是“动作怎么生成”；第五篇会进一步追问：

> 如果把整个决策历史都看成序列，策略还能学到什么？

---

## 19. 推荐阅读与深入材料

### 19.1 Zhao et al., 2023, ACT / ALOHA

- **材料类型**：经典机器人模仿学习工程论文。
- **阅读目的**：理解 action chunk、CVAE、Transformer policy 和 temporal ensemble 为什么适合双臂操作。
- **重点看什么**：动作块构造、训练推理差异、temporal ensemble、实机任务设置。
- **对应本章**：ACT 与动作块建模。

### 19.2 Chi et al., 2023, Diffusion Policy

- **材料类型**：生成式机器人动作策略代表论文。
- **阅读目的**：理解为什么把动作作为 diffusion 生成对象，而不是直接回归动作。
- **重点看什么**：动作 horizon、noise prediction objective、receding horizon control、实机评估。
- **对应本章**：Diffusion Policy 的离散去噪生成过程。

### 19.3 Flow Matching / Rectified Flow 相关材料

- **材料类型**：生成模型基础与前沿方法。
- **阅读目的**：理解从噪声分布到数据分布的连续流场建模思想。
- **重点看什么**：路径构造、速度场训练目标、ODE 采样、采样步数与稳定性。
- **对应本章**：Flow Matching 的连续速度场生成过程。

### 19.4 Black et al., 2024, π0

- **材料类型**：VLA 与 flow matching action head 代表工作。
- **阅读目的**：理解大规模视觉语言动作模型中，flow matching 如何作为动作生成头使用。
- **重点看什么**：动作表示、模型输入输出、训练数据规模、推理流程。
- **对应本章**：Flow Matching 在现代机器人策略中的位置。

### 19.5 Mandlekar et al., 2021, RoboMimic

- **材料类型**：机器人模仿学习基准与工程框架。
- **阅读目的**：学习如何统一比较不同策略模型，而不是只看单个论文的效果。
- **重点看什么**：数据集、评估指标、open-loop 与 closed-loop 差异、任务难度划分。
- **对应本章**：方法选型与实验对比框架。

建议阅读时把每篇材料整理成同一张表：输入观测、输出动作形式、是否生成动作块、是否显式概率建模、采样步数、控制频率、所需数据规模、适合任务、主要工程风险。这样比逐篇摘摘要更能服务方法选型。
