# 附录 I 熵、最大熵与 Score Matching：从概率直觉到 Diffusion Policy

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


## I.1 为什么需要这个附录

正文中反复出现三个容易让读者卡住的概念：熵、最大熵和 score matching。它们分别支撑了三类问题：

- 行为克隆和概率策略中，熵帮助我们描述策略分布有多分散；
- 最大熵 IRL 中，熵帮助我们在多条合理专家轨迹之间保持“不乱下结论”；
- Diffusion Policy 中，score matching 帮助我们理解为什么预测噪声能够学习动作分布。

本附录只给出读懂本书所需的最小数学解释，不追求覆盖生成模型理论的全部细节。

## I.2 熵：一个分布有多不确定

离散分布 <span class="math">\\(p(x)\\)</span> 的熵定义为：

<div class="math">\[
H(p)=-\sum_x p(x)\log p(x). \tag{I.1}\]</div>

可以这样读这个公式：

- <span class="math">\\(p(x)\\)</span>：事件 <span class="math">\\(x\\)</span> 出现的概率；
- <span class="math">\\(\log p(x)\\)</span>：事件出现后带来的“惊讶程度”的相反数；
- <span class="math">\\(-\sum\_x p(x)\log p(x)\\)</span>：把所有事件的惊讶程度按概率加权平均。

如果策略在某个状态下几乎只会选择一个动作，熵很低；如果多个动作都可能，熵较高。

## I.3 条件熵：给定状态后的动作不确定性

模仿学习中更常见的是条件动作分布 <span class="math">\\(\pi(a\mid s)\\)</span>。它的条件熵可以写成：

<div class="math">\[
H(\pi(\cdot\mid s))=-\sum_a \pi(a\mid s)\log \pi(a\mid s). \tag{I.2}\]</div>

它回答的问题是：在已经知道状态 <span class="math">\\(s\\)</span> 的情况下，策略对动作选择还有多不确定。

在二维点机器人例子中，如果某个位置既可以从左边绕障，也可以从右边绕障，那么 <span class="math">\\(H(\pi(\cdot\mid s))\\)</span> 不应被强行压得很低。否则策略会假装只有一个标准答案。

## I.4 最大熵原则：不要在证据不足时过度自信

最大熵原则的朴素含义是：在满足已知约束的所有分布中，选择熵最大的那个。

例如我们只知道专家轨迹经常避开障碍，但并不知道专家一定从左边绕还是右边绕。此时最大熵思想会避免把专家解释成“永远从左边绕”，而倾向于保留多种合理解释。

最大熵 IRL 中常见的轨迹分布形式为：

<div class="math">\[
p(\tau) = \frac{1}{Z}\exp(R(\tau)), \tag{I.3}\]</div>

其中：

- <span class="math">\\(R(\tau)\\)</span>：轨迹总 reward；
- <span class="math">\\(\exp(R(\tau))\\)</span>：reward 越高，轨迹概率越大；
- <span class="math">\\(Z\\)</span>：归一化常数，保证所有轨迹概率加起来为 1。

这并不是说专家随机乱走，而是在同样合理的轨迹之间保留不确定性。

## I.5 Score：分布密度上升最快的方向

对连续变量 <span class="math">\\(x\\)</span>，score 定义为：

<div class="math">\[
\nabla_x \log p(x). \tag{I.4}\]</div>

它不是模型评分，也不是 reward。它表示在当前位置 <span class="math">\\(x\\)</span> 附近，往哪个方向移动会让 <span class="math">\\(\log p(x)\\)</span> 增大得最快。

在动作生成里，可以把 <span class="math">\\(x\\)</span> 理解为一个动作块。如果一个带噪动作块偏离了专家数据流形，score 给出的方向就像“回到高概率专家动作区域的指南针”。

## I.6 Score Matching：不直接学概率，改学概率的方向场

直接学习高维分布 <span class="math">\\(p(x)\\)</span> 很难，因为归一化常数往往不可计算。Score matching 的思路是：既然直接学密度难，那就学 <span class="math">\\(\nabla\_x\log p(x)\\)</span> 这个方向场。

这样做的直觉是：生成时不一定要知道每个点的绝对概率，只要知道从噪声点如何一步步往高概率区域移动。

## I.7 去噪 Score Matching：从被污染的数据中学习恢复方向

假设干净样本是 <span class="math">\\(x\_0\\)</span>，加噪后得到：

<div class="math">\[
x_k=\sqrt{\bar\alpha_k}x_0+\sqrt{1-\bar\alpha_k}\epsilon,
\quad \epsilon\sim\mathcal{N}(0,I). \tag{I.5}\]</div>

去噪任务要求模型根据 <span class="math">\\(x\_k\\)</span> 判断噪声 <span class="math">\\(\epsilon\\)</span> 或恢复方向。对 diffusion 模型来说，常见训练目标是：

<div class="math">\[
\mathcal{L}(\theta)=
\mathbb{E}_{x_0,\epsilon,k}
\left[
\left\|\epsilon-\epsilon_\theta(x_k,k)\right\|^2
\right]. \tag{I.6}\]</div>

这个式子的含义是：随机选择一个噪声级别，把真实数据弄脏，然后训练模型识别“脏在哪里”。

## I.8 为什么预测噪声可以理解为学习 score

在高斯加噪设定下，带噪样本 <span class="math">\\(x\_k\\)</span> 的 score 与噪声方向存在对应关系。直观上，如果 <span class="math">\\(x\_k\\)</span> 是由 <span class="math">\\(x\_0\\)</span> 加上噪声 <span class="math">\\(\epsilon\\)</span> 得到的，那么去掉 <span class="math">\\(\epsilon\\)</span> 的方向就是回到数据附近的方向。

因此，虽然代码里训练的是 <span class="math">\\(\epsilon\_\theta\\)</span>，数学上它可以被理解为一种条件 score 估计。Diffusion Policy 把这里的 <span class="math">\\(x\\)</span> 换成动作块 <span class="math">\\(A\\)</span>，再把观测 <span class="math">\\(obs\\)</span> 作为条件：

<div class="math">\[
\epsilon_\theta(A^k,k,obs). \tag{I.7}\]</div>

它学习的是：在当前观测下，带噪动作块应该如何往专家动作块分布靠近。

## I.9 回到 Diffusion Policy

Diffusion Policy 的关键不是“从噪声生成动作”这个表面过程，而是：

1. 把专家动作块视为来自复杂条件分布 <span class="math">\\(p(A\mid obs)\\)</span> 的样本；
2. 通过正向加噪构造一系列容易训练的监督任务；
3. 通过噪声预测学习反向恢复方向；
4. 推理时从随机动作块出发，多步去噪得到可执行动作块。

在机械臂抓取中，这允许策略保留多种抓取风格；在二维点机器人绕障中，这允许策略生成左绕或右绕这样的多模态路径，而不是被 MSE 压成中间撞障路线。

## I.10 常见误解

### 误解一：熵越大越好

熵大表示策略更分散，但不一定更安全。一个在危险动作上也很分散的策略并不好。熵应该服务于合理多样性，而不是无约束随机性。

### 误解二：最大熵表示专家很随意

最大熵不是说专家随便做，而是在满足已知约束时，不额外假设专家只有一种行为方式。

### 误解三：score 是 reward

score 是概率密度的梯度方向，reward 是任务偏好的标量评价。二者可以都影响策略，但含义不同。

### 误解四：Diffusion Policy 显式计算了 score

多数实现训练的是噪声预测网络，并不显式写出 <span class="math">\\(\nabla\_x\log p(x)\\)</span>。score matching 是理解噪声预测目标的一种数学视角。

## I.11 小结

本附录只需记住三句话：

- 熵描述分布的不确定性；
- 最大熵原则帮助我们避免在多种合理解释之间过度自信；
- score matching 解释了 diffusion 为什么可以通过学习去噪方向来建模复杂分布。
