# GitHub Pages 发布说明

## 1. 仓库地址

当前仓库对应：

```text
Mathematical-principles-of-imitation-learning
```

远端地址：

```text
git@github.com:fengkai11/Mathematical-principles-of-imitation-learning.git
```

## 2. mdBook 配置

本书的 mdBook 根配置在仓库根目录：

```text
book.toml
```

书稿源目录配置为：

```toml
src = "模仿学习的数学原理_终稿校验整理版"
```

该目录内已有：

```text
SUMMARY.md
chapters/
appendices/
images/
```

因此无需移动原始书稿文件。

## 3. GitHub Pages 配置

进入 GitHub 仓库：

```text
Settings -> Pages
```

在 Source 中选择：

```text
GitHub Actions
```

## 4. 自动部署

每次 push 到 `main` 或 `master` 分支，都会自动执行：

```text
.github/workflows/deploy.yml
```

部署完成后，GitHub Actions 页面会显示访问链接。

## 5. 本地构建

如果本机已安装 mdBook，可在仓库根目录执行：

```bash
mdbook build
```

本地输出目录为：

```text
book/
```

GitHub Pages 站点路径配置为：

```toml
site-url = "/Mathematical-principles-of-imitation-learning/"
```
