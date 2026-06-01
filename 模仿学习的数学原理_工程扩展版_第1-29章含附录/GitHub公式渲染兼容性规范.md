# GitHub 公式渲染兼容性规范

> 本文记录第一篇成稿校对过程中踩到的 GitHub Markdown 数学公式渲染问题。后续所有章节改造、公式校对、成书级整理，都必须把本文作为公式格式检查清单之一。

---

## 1. 背景

本书主要用 Markdown / mdBook 组织内容，但读者和作者经常会直接在 GitHub 网页中阅读源码预览。

实践中发现：

- 某些公式在 mdBook 或本地 MathJax / KaTeX 环境下可以正常渲染；
- 但在 GitHub 网页 Markdown 预览中会渲染失败、显示不完整，甚至出现宏不允许的报错；
- 如果只检查源码是否符合 LaTeX 语法，仍然可能漏掉 GitHub 预览错误。

因此，本书公式格式需要同时满足两类目标：

```text
1. mdBook 构建后能正常显示；
2. GitHub 网页直接打开 .md 文件时也尽量正常显示。
```

---

## 2. 本次踩坑总结

### 2.1 单独一行的等号会被 GitHub 当成 Markdown 标题下划线

曾出现过这样的公式写法：

```md
$$
\mathcal{L}_{MSE}(\theta)
=
\frac{1}{N}\sum_{i=1}^{N}
\left\|f_\theta(o_i) - a_i\right\|^2
$$
```

这在很多 LaTeX / MathJax 环境中是可以工作的，但在 GitHub 网页中，如果公式块没有被正确识别，单独一行的 `=` 可能被 Markdown 当成 Setext 标题下划线处理，导致等号消失，公式被拆散。

错误现象类似：

```text
$$ \mathcal{L}_{MSE}(\theta)
\frac{1}{N}\sum_{i=1}^{N} |f_\theta(o_i)-a_i|^2 $$
```

### 2.2 `\operatorname` / `\operatorname*` 可能被 GitHub 禁用

曾出现过这样的公式：

```md
$$\theta^* = \operatorname*{arg\,min}_{\theta} ...$$
```

GitHub 网页直接预览时报错：

```text
The following macros are not allowed: operatorname
```

因此，在 GitHub 兼容模式下，不使用：

```latex
\operatorname
\operatorname*
```

改用更保守的写法：

```latex
\arg\min_{\theta}
```

### 2.3 `\left\| ... \right\|` 在 GitHub 预览中容易被复制/显示成单竖线

曾出现 MSE 公式中范数符号被用户看到为：

```latex
\left|f_\theta(o_i) - a_i\right|^2
```

为了减少显示和复制歧义，向量范数优先写成：

```latex
\lVert f_\theta(o_i) - a_i \rVert^2
```

---

## 3. 后续必须遵守的公式写法

### 3.1 不要让 `=` 单独占一行

不推荐：

```md
$$
J_{BC}(\theta)
=
\mathbb{E}_{o \sim d^{\pi_E}}
[...]
$$
```

推荐：

```md
$$J_{BC}(\theta) = \mathbb{E}_{o \sim d^{\pi_E}}\left[...\right]$$
```

如果公式太长，也可以把等号放在第一行，而不是单独占一行：

```md
$$
J_{BC}(\theta) =
\mathbb{E}_{o \sim d^{\pi_E}}
\left[...\right]
$$
```

### 3.2 GitHub 直接预览优先使用单行块级公式

对于中短公式，优先使用：

```md
$$\mathcal{L}_{MSE}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\lVert f_\theta(o_i) - a_i\rVert^2$$
```

而不是：

```md
$$
\mathcal{L}_{MSE}(\theta)
=
\frac{1}{N}\sum_{i=1}^{N}
\left\|f_\theta(o_i) - a_i\right\|^2
$$
```

### 3.3 不使用 `\operatorname*{arg\,min}`

不推荐：

```latex
\operatorname*{arg\,min}_{\theta}
```

推荐：

```latex
\arg\min_{\theta}
```

示例：

```md
$$\theta^* = \arg\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\ell\left(\pi_\theta(o_i), a_i\right)$$
```

### 3.4 向量范数优先使用 `\lVert ... \rVert`

不推荐：

```latex
\left\|x\right\|^2
```

推荐：

```latex
\lVert x \rVert^2
```

示例：

```md
$$\mathcal{L}_{MSE}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\lVert f_\theta(o_i)-a_i\rVert^2$$
```

### 3.5 不要把块级公式放进 blockquote

不推荐：

```md
> $$
> O(T^2\varepsilon)
> $$
```

推荐：

```md
> **命题：BC 的误差累积风险**
>
> 在最坏情况下，累计错误量级可能达到 $O(T^2\varepsilon)$。

$$O(T^2\varepsilon)$$
```

---

## 4. 第一篇公式修复案例

### 4.1 MSE 损失公式

修复前：

```md
$$
\mathcal{L}_{MSE}(\theta)
=
\frac{1}{N}\sum_{i=1}^{N}
\left\|f_\theta(o_i) - a_i\right\|^2
$$
```

修复后：

```md
$$\mathcal{L}_{MSE}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\lVert f_\theta(o_i) - a_i\rVert^2$$
```

### 4.2 经验风险目标

修复前：

```md
$$\theta^* = \operatorname*{arg\,min}_{\theta}\frac{1}{N}\sum_{i=1}^{N}\ell\left(\pi_\theta(\cdot \mid o_i), a_i\right)$$
```

修复后：

```md
$$\theta^* = \arg\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\ell\left(\pi_\theta(\cdot \mid o_i), a_i\right)$$
```

---

## 5. 章节校对检查清单

每次完成章节改造后，至少检查以下项目：

```text
[ ] 是否存在单独一行的 = ？
[ ] 是否存在 \operatorname 或 \operatorname* ？
[ ] 是否存在 blockquote 中的 $$ 块级公式？
[ ] 是否存在 \left\| ... \right\| 形式的向量范数？
[ ] GitHub 网页直接打开 .md 文件时，公式是否报错？
[ ] mdBook 构建后公式是否正常？
[ ] 公式索引中的公式是否与正文一致？
```

建议用以下关键词搜索：

```text
operatorname
$$
=
\left\|
> $$
```

---

## 6. 对后续章节改造的要求

后续第 5 章及以后所有章节，在新增或重写公式时：

1. 中短公式优先写成单行块级公式；
2. 长公式允许分行，但等号不得单独占一行；
3. 不使用 GitHub 禁用宏；
4. 公式索引必须同步使用同一套兼容写法；
5. 章节提交前必须打开 GitHub 网页预览一次。

一句话原则：

> **不要只写 LaTeX 正确的公式，要写 GitHub 和 mdBook 都稳定显示的公式。**
