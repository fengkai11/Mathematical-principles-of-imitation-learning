# 附录 F 强化学习与序列决策基础

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录补齐本书中 DAgger、分布偏移、GAIL、IRL、Offline RL、Decision Transformer 和部署评测所需的序列决策基础。模仿学习虽然经常“不写 reward”，但机器人仍然活在时间里。只要动作会影响下一刻观测，你就逃不开序列决策。

---

## F.1 为什么模仿学习需要 RL 语言

行为克隆看起来像监督学习：

$$
(o_t,a_t)\rightarrow a_t \tag{F.1}$$

但机器人执行时不是一次性预测完就结束，而是循环：

```text
看一眼 → 动一下 → 世界变了 → 再看一眼 → 再动一下
```

这意味着当前动作会影响未来状态。一个单步动作误差，可能让后面的观测全变。

所以即使模仿学习不用显式 reward，也需要 MDP、rollout、policy-induced distribution、occupancy measure 这些强化学习语言来描述闭环行为。

---

## F.2 MDP 五元组

MDP 写作：

$$
(\mathcal S,\mathcal A,P,R,\gamma) \tag{F.2}$$

分别表示：

- $\mathcal S$：状态空间；
- $\mathcal A$：动作空间；
- $P(s'|s,a)$：转移概率；
- $R(s,a)$：奖励函数；
- $\gamma$：折扣因子。

在模仿学习里，奖励 $R$ 可能未知或不用，但前四个对象中的状态、动作、转移仍然存在。

例如自动泊车：

- 状态：车辆位姿、车位位置、障碍物、速度；
- 动作：速度、转角；
- 转移：车辆动力学和环境变化；
- 奖励：是否停进车位、安全距离、舒适性。

即使你只做 BC，车辆也不会因为你没写 $R$ 就不遵守动力学。

---

## F.3 状态与观测

状态 $s_t$ 是环境真实情况，观测 $o_t$ 是传感器看到的信息。

常写作：

$$
o_t\sim O(\cdot|s_t) \tag{F.3}$$

这表示观测由状态生成，但可能有噪声和遮挡。

在真实机器人中，完整状态经常不可见。例如：

- 你看得到杯子外观，但看不到摩擦系数；
- 看得到托盘轮廓，但不知道它有没有微小变形；
- 看得到车位线，但不知道地面反光是否会影响下一帧检测。

因此策略常写作：

$$
\pi_\theta(a_t|o_t) \tag{F.4}$$

而不是理想化的：

$$
\pi_\theta(a_t|s_t) \tag{F.5}$$

如果需要历史信息，可以写成：

$$
\pi_\theta(a_t|o_{0:t},a_{0:t-1}) \tag{F.6}$$

这就是 Transformer policy、Decision Transformer 和 VLA 中历史上下文的重要性。

---

## F.4 转移概率

转移概率写作：

$$
P(s_{t+1}|s_t,a_t) \tag{F.7}$$

意思是：当前状态 $s_t$ 下执行动作 $a_t$，下一状态 $s_{t+1}$ 的概率。

如果环境是确定性的，可以近似写成：

$$
s_{t+1}=f(s_t,a_t) \tag{F.8}$$

但真实机器人里通常有不确定性：

- 夹爪接触物体后可能滑动；
- 轮胎和地面摩擦变化；
- 机械臂控制有误差；
- 传感器延迟导致动作基于旧观测；
- 工件摆放高度有偏差。

所以 $P$ 不是数学摆设，它是“世界不完全听你指挥”的表达。

---

## F.5 策略

策略是从状态或观测到动作的规则。

确定性策略：

$$
a_t=\pi_\theta(o_t) \tag{F.9}$$

概率策略：

$$
a_t\sim\pi_\theta(\cdot|o_t) \tag{F.10}$$

概率策略承认：同一个观测下，合理动作可能不止一个。

在机器人操作中，这非常常见。例如抓同一个杯子，可以从左侧抓，也可以从右侧抓；整理物体时，可以先挪 A，也可以先挪 B。确定性策略很容易被迫选一个平均答案，概率策略和生成式策略更适合表达多解。

---

## F.6 轨迹

轨迹写作：

$$
\tau=(s_0,a_0,s_1,a_1,\dots,s_T) \tag{F.11}$$

如果使用观测，也可以写成：

$$
\tau=(o_0,a_0,o_1,a_1,\dots,o_T) \tag{F.12}$$

在策略 $\pi$ 下，轨迹概率为：

$$
p_\pi(\tau)
=
p(s_0)
\prod_{t=0}^{T-1}
\pi(a_t|s_t)
P(s_{t+1}|s_t,a_t) \tag{F.13}$$

这个公式非常重要。它告诉我们：轨迹不是模型单独决定的，也不是环境单独决定的，而是策略和环境交替作用的结果。

这就是为什么模仿学习不能只看单步预测。模型一旦执行动作，就会改变后续状态分布。

---

## F.7 return 与折扣因子

return 表示从某一时刻开始的未来累计奖励：

$$
G_t
=
\sum_{k=t}^{T}
\gamma^{k-t}r_k \tag{F.14}$$

其中：

- $r_k$：第 $k$ 步奖励；
- $\gamma\in[0,1]$：折扣因子；
- $\gamma$ 越接近 1，越重视长期奖励；
- $\gamma$ 越小，越重视眼前奖励。

Decision Transformer 中常使用 return-to-go：

$$
\hat R_t
=
\sum_{k=t}^{T}r_k \tag{F.15}$$

它把未来想达到的回报作为条件输入，告诉模型：我们希望生成高回报轨迹。

模仿学习不一定直接优化 return，但如果你要理解 Offline RL、IRL、Decision Transformer，return 是绕不开的。

---

## F.8 policy-induced distribution：策略会把机器人带到哪里

策略诱导状态分布写作：

$$
d^\pi(s) \tag{F.16}$$

它表示：执行策略 $\pi$ 时，机器人会访问哪些状态，以及访问频率如何。

专家策略对应：

$$
d^{\pi_E}(s) \tag{F.17}$$

学习策略对应：

$$
d^{\pi_\theta}(s) \tag{F.18}$$

分布偏移的核心就是：

$$
d^{\pi_E}(s)
\neq
d^{\pi_\theta}(s) \tag{F.19}$$

第 3 章的全部痛苦，基本都藏在这个不等号里。

训练时 BC 优化：

$$
\mathbb E_{s\sim d^{\pi_E}}
[
\ell(\pi_\theta(s),\pi_E(s))
] \tag{F.20}$$

但执行时真正关心：

$$
\mathbb E_{s\sim d^{\pi_\theta}}
[
\ell(\pi_\theta(s),\pi_E(s))
] \tag{F.21}$$

二者分布不同，训练表现和闭环表现就可能分家。

---

## F.9 occupancy measure

occupancy measure 写作：

$$
\rho_\pi(s,a) \tag{F.22}$$

它表示策略 $\pi$ 访问状态—动作对 $(s,a)$ 的频率。

一种常见定义是折扣访问频率：

$$
\rho_\pi(s,a)
=
(1-\gamma)
\sum_{t=0}^{\infty}
\gamma^t
P(s_t=s,a_t=a|\pi) \tag{F.23}$$

不用被这个式子吓住。它只是说：

> 看策略执行时，在每个时间步访问 $(s,a)$ 的概率，然后按折扣加权求和。

GAIL 的核心目标可以理解为匹配专家和策略的 occupancy measure：

$$
\rho_{\pi}\approx\rho_{\pi_E} \tag{F.24}$$

这比单步动作模仿更强，因为它关心整个闭环行为分布。

---

## F.10 rollout

rollout 指让策略在环境中真正执行一段时间，得到轨迹。

伪流程：

```text
初始化环境状态 s0
for t = 0...T:
    根据当前观测选择动作 at ~ π(.|ot)
    环境执行动作，得到下一状态 st+1
    记录状态、动作、奖励、是否结束
```

数学上，rollout 得到：

$$
\tau\sim p_\pi(\tau) \tag{F.25}$$

第 20 章中 closed-loop evaluation 本质上就是 rollout 评测。

open-loop 只在数据集上预测动作；closed-loop 让策略真的控制系统。二者差别很大。

---

## F.11 open-loop 与 closed-loop

open-loop 评测：

$$
\mathcal L_{\mathrm{open}}
=
\mathbb E_{(o,a)\sim\mathcal D_{\mathrm{test}}}
[
\ell(\pi_\theta(o),a)
] \tag{F.26}$$

它检查：在测试集观测上，模型动作是否接近专家动作。

closed-loop 评测：

$$
J_{\mathrm{closed}}(\pi_\theta)
=
\mathbb E_{\tau\sim p_{\pi_\theta}(\tau)}[M(\tau)] \tag{F.27}$$

其中 $M(\tau)$ 可以是成功率、碰撞次数、耗时、稳定性指标。

区别在于：

```text
open-loop：模型不改变数据；
closed-loop：模型动作会改变未来输入。
```

所以 open-loop 像笔试，closed-loop 像上岗。笔试高分很重要，但不等于在产线不会把工件夹飞。

---

## F.12 behavior policy 与 offline 数据

Offline Learning 中常说 behavior policy：

$$
\pi_\beta \tag{F.28}$$

它表示收集离线数据的策略。

数据集来自：

$$
\mathcal D\sim\pi_\beta \tag{F.29}$$

如果我们训练的新策略 $\pi$ 想做很多 $\pi_\beta$ 没有做过的动作，就会遇到 OOD 风险。

这就是 Offline RL 和 Offline Imitation Learning 的核心难点：

> 你不能无限相信数据没有覆盖过的动作价值。

第 15 章讲的“坑有没有录进去”，在这里可以写成：数据分布是否覆盖了部署策略可能访问的状态—动作区域。

---

## F.13 常见误解

### 误解 1：模仿学习没有 reward，所以和 RL 无关

不对。即使没有显式 reward，状态、动作、转移、轨迹、rollout 仍然存在。模仿学习可以不用 RL 优化，但不能无视序列决策结构。

### 误解 2：单步动作预测准，轨迹就一定好

不一定。小误差会改变未来状态，导致后续输入偏离训练分布。

### 误解 3：occupancy measure 只是学术概念

它非常工程。它告诉我们策略实际访问了哪些状态—动作对。部署失败往往就是 occupancy 跑到了数据没覆盖的区域。

### 误解 4：closed-loop 评测太麻烦，可以用 open-loop 代替

不能代替。open-loop 是必要但不充分条件。机器人最终是在闭环世界里工作，不是在 csv 文件里工作。

---

## F.14 本附录小结

本附录补齐了序列决策基础：

1. MDP 描述状态、动作、转移、奖励和折扣；
2. 策略决定动作；
3. 轨迹由策略和环境共同生成；
4. return 衡量未来累计结果；
5. policy-induced distribution 描述策略会访问哪些状态；
6. occupancy measure 描述状态—动作访问频率；
7. rollout 是闭环执行；
8. open-loop 和 closed-loop 评测不能混为一谈。

一句话总结：

> 模仿学习看起来在学动作，本质上却是在时间里做决策；不理解序列，就只能理解半个机器人。

---

## F.15 小练习

1. 请用自己的任务例子写出 MDP 五元组。

2. 为什么 $d^{\pi_E}(s)$ 和 $d^{\pi_\theta}(s)$ 不一样会导致 BC 失败？

3. 轨迹概率 $p_\pi(\tau)$ 中哪些部分由策略决定？哪些部分由环境决定？

4. occupancy measure 和 state distribution 有什么区别？

5. 请设计一个 open-loop 指标和一个 closed-loop 指标，并说明二者为什么都需要。
