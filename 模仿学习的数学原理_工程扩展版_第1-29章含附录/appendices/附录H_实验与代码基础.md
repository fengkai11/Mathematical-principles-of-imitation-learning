# 附录 H 实验与代码基础

> **统一公式编号说明**：本章（或本附录）中的展示公式统一采用按章节编号的方式。章节正文使用“（章号.序号）”，附录使用“（附录字母.序号）”。


> 本附录把数学公式拉回工程地面。读懂公式只是第一步，真正掌握模仿学习，还需要知道数据怎么组织、训练循环怎么写、open-loop 怎么评估、closed-loop 怎么 rollout、实验记录怎么做。否则模型只会活在 notebook 里，像一个永远不出门的天才。

---

## H.1 一个模仿学习实验最少需要什么

一个最小模仿学习实验通常包括：

```text
1. 数据集：专家演示轨迹；
2. 模型：策略网络 πθ；
3. loss：BC、CVAE、Diffusion 等训练目标；
4. 训练循环：前向、计算 loss、反传、更新；
5. open-loop 评估：测试集动作误差；
6. closed-loop 评估：环境中执行任务；
7. 日志：记录配置、指标、失败案例。
```

如果只训练不评估，模型像只写不测的代码；如果只 open-loop 不 closed-loop，模型像只会笔试不会上岗的应届生。

---

## H.2 数据格式

模仿学习数据通常来自专家演示轨迹：

$$
\tau_i=(o_0,a_0,o_1,a_1,\dots,o_T,a_T) \tag{H.1}$$

一个数据集是多条轨迹：

$$
\mathcal D=\{\tau_i\}_{i=1}^{N} \tag{H.2}$$

工程上可以组织成：

```text
dataset/
  episode_000001/
    images/
      000000.png
      000001.png
    actions.npy
    states.npy
    meta.json
  episode_000002/
    ...
```

每个 episode 至少应包含：

- observation：图像、点云、状态向量等；
- action：专家动作；
- timestamp：时间戳；
- done：是否结束；
- success：任务是否成功；
- meta：任务类型、场景、设备、操作者、版本等。

meta 不要偷懒。后续分析失败案例时，meta 往往比模型输出更像救命绳。

---

## H.3 train / val / rollout split

数据划分至少包括：

```text
train：用于训练；
val：用于调参和早停；
test/open-loop：用于离线动作误差评估；
rollout scenarios：用于闭环执行评测。
```

不要把同一条轨迹切碎后随机分到 train 和 test。这样会造成数据泄漏。

更推荐按 episode 划分：

$$
\mathcal D
=
\mathcal D_{\mathrm{train}}
\cup
\mathcal D_{\mathrm{val}}
\cup
\mathcal D_{\mathrm{test}} \tag{H.3}$$

并满足：

$$
\mathcal D_{\mathrm{train}}
\cap
\mathcal D_{\mathrm{test}}
=
\varnothing \tag{H.4}$$

如果是机器人任务，还应考虑按场景划分：

- 不同物体；
- 不同光照；
- 不同相机位置；
- 不同操作员；
- 不同难度级别；
- 不同工件批次。

否则测试集可能只是训练集的亲戚，闭环部署时才发现外面的世界都是远房陌生人。

---

## H.4 BC 训练伪代码

行为克隆训练伪代码：

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        obs = batch["obs"]       # [B, ...]
        act = batch["action"]    # [B, action_dim]

        pred = policy(obs)
        loss = mse_loss(pred, act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

对应数学目标：

$$
\mathcal L_{\mathrm{BC}}
=
\mathbb E_{(o,a)\sim\mathcal D}
[
\|a-f_\theta(o)\|^2] \tag{H.5}$$

如果是离散动作，则改成交叉熵：

```python
logits = policy(obs)
loss = cross_entropy(logits, action_label)
```

对应：

$$
\mathcal L_{\mathrm{CE}}
=
-\mathbb E_{(o,a)\sim\mathcal D}
[
\log\pi_\theta(a|o)] \tag{H.6}$$

---

## H.5 序列数据与动作块

ACT、Diffusion Policy 和 Transformer policy 往往需要历史和未来动作块。

一个样本可能是：

$$
(o_{t-K+1:t},a_{t:t+H-1}) \tag{H.7}$$

其中：

- $K$：历史观测长度；
- $H$：预测动作块长度。

数据加载器要从 episode 中滑窗采样：

```python
obs_hist = obs[t-K+1 : t+1]
action_chunk = actions[t : t+H]
```

要注意边界：episode 开头历史不够怎么办？episode 结尾动作块不够怎么办？常见处理包括 padding、mask、丢弃不完整样本。

mask 非常重要。否则模型可能把 padding 当成真实动作学进去，像把空白作业也当标准答案。

---

## H.6 CVAE 训练伪代码

CVAE 行为克隆大致流程：

```python
for batch in dataloader:
    obs = batch["obs"]
    act = batch["action"]

    mu, logvar = encoder(obs, act)
    eps = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * eps

    pred_act = decoder(obs, z)

    recon_loss = mse_loss(pred_act, act)
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    loss = recon_loss + beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

对应数学形式：

$$
\mathcal L_{\mathrm{CVAE}}
=
\mathcal L_{\mathrm{recon}}
+
\beta D_{\mathrm{KL}}(q_\phi(z|o,a)\|p(z)) \tag{H.8}$$

训练时 encoder 输入 $(o,a)$，推理时不能输入专家动作，只能从 prior 采样：

$$
z\sim p(z) \tag{H.9}$$

再生成动作：

$$
a\sim p_\theta(a|o,z) \tag{H.10}$$

---

## H.7 Diffusion Policy 训练伪代码

Diffusion Policy 训练的核心是随机选择噪声步，给真实动作序列加噪，让网络预测噪声。

```python
for batch in dataloader:
    obs = batch["obs_hist"]              # condition
    action = batch["action_chunk"]       # x0

    t = sample_timesteps(batch_size)
    noise = torch.randn_like(action)
    noisy_action = q_sample(action, t, noise)

    pred_noise = noise_predictor(noisy_action, t, obs)
    loss = mse_loss(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

对应数学目标：

$$
\mathcal L_{\mathrm{diff}}
=
\mathbb E_{x_0,t,\epsilon}
[
\|\epsilon-\epsilon_\theta(x_t,t,o)\|^2] \tag{H.11}$$

其中：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon \tag{H.12}$$

Diffusion 推理时要多步去噪，因此推理成本通常高于普通 BC。这就是第 12 章对比推理成本时强调的内容。

---

## H.8 open-loop 评估

open-loop 评估在测试集上计算动作误差：

$$
\mathcal L_{\mathrm{test}}
=
\frac{1}{N}\sum_{i=1}^{N}
\ell(\pi_\theta(o_i),a_i) \tag{H.13}$$

常见指标：

- MSE；
- MAE；
- 角度误差；
- 位置误差；
- 分类准确率；
- NLL；
- action chunk 误差。

但要记住：open-loop 指标只能说明模型在测试数据上像不像专家，不代表它闭环能不能完成任务。

一个很常见的现象是：

```text
open-loop loss 下降；
rollout 成功率不升；
动作看起来更平滑；
但关键时刻不会纠错。
```

这不是小概率事件，而是模仿学习的家常便饭。

---

## H.9 closed-loop rollout 评估

closed-loop 评估让策略真正执行任务：

```python
success_count = 0
for episode in range(num_eval_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
    success_count += int(info["success"])

success_rate = success_count / num_eval_episodes
```

数学形式：

$$
\hat p_{\mathrm{success}}
=
\frac{1}{N}\sum_{i=1}^{N}\mathbf 1[\mathrm{success}(\tau_i)] \tag{H.14}$$

其中 $\tau_i$ 是第 $i$ 次 rollout 轨迹。

closed-loop 还应记录：

- 是否成功；
- 失败原因；
- 任务耗时；
- 碰撞次数；
- 动作抖动；
- fallback 次数；
- 人工接管次数；
- OOD 检测信号；
- 关键状态截图或视频。

只报一个 success rate 太粗糙。成功率背后的失败类型，才是真正能指导下一轮改进的东西。

---

## H.10 成功率的置信区间

如果做了 $N$ 次 rollout，成功 $K$ 次，成功率估计为：

$$
\hat p=\frac{K}{N} \tag{H.15}$$

但 $\hat p$ 有不确定性。一个粗略标准误差为：

$$
\mathrm{SE}
=
\sqrt{\frac{\hat p(1-\hat p)}{N}} \tag{H.16}$$

近似 95% 置信区间：

$$
\hat p\pm1.96\cdot\mathrm{SE} \tag{H.17}$$

这解释了为什么“10 次 100% 成功”和“300 次 95% 成功”不能简单比较。前者听起来漂亮，但样本太少，可能只是运气好。

机器人实验最怕“小样本高成功率”，像朋友圈健身打卡，只拍状态好的一天。

---

## H.11 实验记录模板

建议每次实验至少记录：

| 项目 | 内容 |
|---|---|
| experiment_id | 唯一实验编号 |
| git_commit | 代码版本 |
| dataset_version | 数据版本 |
| model_config | 模型结构配置 |
| training_config | batch size、lr、epoch、optimizer |
| loss_config | loss 项和权重 |
| eval_scenarios | 评测场景列表 |
| open_loop_metrics | 测试集指标 |
| closed_loop_metrics | rollout 指标 |
| failure_buckets | 失败类型统计 |
| videos/logs | 视频与日志路径 |
| conclusion | 本次实验结论 |
| next_action | 下一步行动 |

没有实验记录，后面复盘就会变成考古。几周后你看到一个效果不错的模型，却不知道它用的哪版数据、哪组参数、哪段代码，那种心情像发现冰箱里一盒没标签的饭：可能能吃，但不敢。

---

## H.12 失败样本分桶

失败样本不要全部混成“失败”两个字。建议至少分桶：

```text
感知失败：目标没看准、遮挡、误检、位姿偏差；
策略失败：动作模式错误、不会恢复、动作抖动；
控制失败：低层跟踪误差、延迟、限幅；
环境失败：物体滑动、工件变形、光照变化；
数据失败：训练集中缺少类似场景；
安全触发：碰撞风险、越界、fallback；
人工因素：标注错误、遥操作不一致。
```

每类失败对应的改进手段不同。

- 感知失败不一定靠换 policy 解决；
- 策略失败可能需要补数据或换模型；
- 控制失败可能需要调整控制器；
- 数据失败需要补采或重采；
- 安全触发过多可能说明 policy 输出经常越界。

分桶不是写报告好看，而是防止团队对着一个总成功率瞎猜。

---

## H.13 从实验到数据闭环

一个健康的数据闭环大致是：

```text
部署/仿真 rollout
    ↓
收集成功与失败轨迹
    ↓
失败分桶与人工复核
    ↓
挑选高价值样本
    ↓
补标注或重新演示
    ↓
加入训练集
    ↓
训练新模型
    ↓
回归评测与灰度上线
```

数学上，可以把数据集更新写成：

$$
\mathcal D_{k+1}
=
\mathcal D_k\cup\mathcal D_{\mathrm{failure},k}^{\mathrm{reviewed}} \tag{H.18}$$

注意这里加了 reviewed。失败样本不能无脑加入训练集。错误动作、异常传感器、人工误操作如果不筛选，模型可能学会“如何更专业地失败”。

---

## H.14 最小实验路线建议

对于刚开始做模仿学习实验的读者，可以按下面路线：

```text
第 1 步：选择一个简单任务，比如 2D point reaching 或仿真机械臂 pick-place。
第 2 步：采集专家轨迹，先保证数据格式干净。
第 3 步：训练 BC baseline。
第 4 步：做 open-loop test，确认基本学得动。
第 5 步：做 closed-loop rollout，观察分布偏移和失败类型。
第 6 步：加入历史观测或动作块。
第 7 步：尝试 CVAE / ACT / Diffusion Policy。
第 8 步：建立失败分桶和数据闭环。
```

不要一上来就复现最大最炫的 VLA。工程学习最怕跳过 baseline。没有 BC baseline，你不知道复杂模型到底解决了什么，还是只是把问题包装得更贵。

---

## H.15 常见误解

### 误解 1：只要代码能跑，就是实验完成

代码能跑只是起点。你还需要可复现实验、清晰指标、失败分析和版本记录。

### 误解 2：open-loop test 足够说明能力

不够。机器人最终要闭环执行。open-loop 是笔试，closed-loop 是上岗。

### 误解 3：失败样本越多越好

不一定。未经筛选的失败数据可能污染训练集。失败样本要分桶、复核、标注改进策略。

### 误解 4：复杂模型可以弥补脏数据

很难。复杂模型可能更擅长利用数据，也更擅长放大数据里的混乱。数据质量是地基，不是装修。

---

## H.16 本附录小结

本附录给出了实验与代码基础：

1. 模仿学习实验需要数据、模型、loss、训练、评估和日志；
2. 数据应按 episode 组织，保留 meta 信息；
3. train/test 要避免数据泄漏；
4. BC、CVAE、Diffusion Policy 有不同训练循环；
5. open-loop 评估衡量离线动作预测；
6. closed-loop rollout 衡量真实任务能力；
7. 成功率需要样本数和置信区间；
8. 失败样本需要分桶；
9. 数据闭环要加入复核后的高价值样本；
10. baseline 是理解复杂方法的起点。

一句话总结：

> 实验不是把 loss 跑下来，而是把模型行为、失败原因和下一步改进路径跑清楚。

---

## H.17 小练习

1. 请为一个机械臂抓取任务设计 episode 数据结构。

2. 为什么不能把同一条轨迹切碎后随机分到 train 和 test？

3. 写出 BC 训练循环的核心步骤。

4. Diffusion Policy 训练中，为什么要随机采样噪声步 $t$？

5. 请设计一个 rollout 评估表，至少包含成功率、失败原因、耗时和 fallback 次数。

6. 为什么失败样本加入训练集前需要 reviewed？
