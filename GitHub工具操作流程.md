# GitHub 工具操作流程

> 本文用于记录在 ChatGPT 中通过 GitHub 工具修改仓库文件的标准流程，避免新窗口重复摸索。
>
> 适用场景：修改《模仿学习的数学原理》仓库中的章节、规范文档、README、SUMMARY、图片引用等文本文件。

---

## 1. 固定仓库信息

仓库：

```text
fengkai11/Mathematical-principles-of-imitation-learning
```

默认分支：

```text
master
```

---

## 2. 推荐操作流程

修改已有文件时，必须按下面流程执行。

### 步骤 1：先读取目标文件

使用 `GitHub.fetch_file`。

示例：

```json
{
  "repository_full_name": "fengkai11/Mathematical-principles-of-imitation-learning",
  "path": "成书级章节改造方法.md",
  "ref": "master",
  "encoding": "utf-8"
}
```

读取后必须记录两件事：

```text
content：当前文件内容
sha：当前文件的 blob sha
```

不要跳过这一步。

---

### 步骤 2：必须基于当前 content 修改

不要凭空重写整章或整篇文档。

正确做法：

```text
读取当前 content
→ 在当前 content 基础上插入、替换或删除目标段落
→ 保留无关段落不变
```

错误做法：

```text
不读取文件，直接根据记忆生成整篇新内容
```

原因：仓库文件可能已经被前一次提交修改，新窗口里的记忆不一定是最新版本。

---

### 步骤 3：用 update_file 提交完整文件内容

使用 `GitHub.update_file`。

示例：

```json
{
  "repository_full_name": "fengkai11/Mathematical-principles-of-imitation-learning",
  "path": "成书级章节改造方法.md",
  "content": "修改后的完整文件内容",
  "message": "Update GitHub workflow notes",
  "sha": "fetch_file 返回的最新 sha",
  "branch": "master"
}
```

注意：

- `content` 必须是修改后的完整文件内容，不是 diff，也不是局部片段；
- `sha` 必须使用刚刚 `fetch_file` 返回的最新 sha；
- `path` 必须使用仓库真实路径，不要根据显示标题猜文件名。

---

### 步骤 4：提交后再抽查

提交成功后，再使用 `GitHub.fetch_file` 抽查修改位置。

推荐至少抽查：

```text
1. 新增段落是否存在；
2. 原有关键段落是否还在；
3. 公式、图片、链接路径是否正确；
4. 是否误删无关内容；
5. 是否还残留本次要删除的旧写法。
```

例如本书公式改造后，要抽查：

```text
是否还有 \tag{...}
是否已经改成“公式 (x.y)：公式名称 + 纯公式块”
公式索引是否同步
```

---

## 3. 处理 409 conflict

如果 `GitHub.update_file` 返回 409 conflict，通常说明 sha 过期。

处理方式：

```text
不要继续用旧 sha 重试；
重新 fetch_file 获取最新 content 和 sha；
把修改重新应用到最新 content；
再 update_file。
```

固定流程：

```text
fetch_file 最新文件
→ 获取新 sha
→ 在最新 content 基础上重新合并修改
→ update_file
→ fetch_file 抽查
```

---

## 4. 文件路径注意事项

仓库里的真实文件名不一定等于章节显示标题。

例如第12章显示标题是：

```text
第12章：Offline Imitation Learning：离线数据不是越多越好，是坑有没有录进去
```

但仓库真实路径是：

```text
模仿学习的数学原理_工程扩展版_第1-29章含附录/chapters/第12章_Offline_Imitation_Learning_离线数据不是越多越好_是坑有没有录进去.md
```

因此修改章节前，优先查看：

```text
模仿学习的数学原理_工程扩展版_第1-29章含附录/SUMMARY.md
```

以 SUMMARY 中的路径为准。

---

## 5. 长文件处理建议

如果目标文件很长，`fetch_file` 返回内容可能会在界面中截断。

此时应优先：

```text
1. 使用 start_line / end_line 抽查目标段落；
2. 避免在看不全全文时贸然 update_file；
3. 必要时先创建补充文档，记录操作流程或修改方案；
4. 等能完整拿到 content 后，再合入主文档。
```

不要为了完成提交而用不完整 content 覆盖长文件。

---

## 6. 本项目常用路径

章节目录：

```text
模仿学习的数学原理_工程扩展版_第1-29章含附录/chapters/
```

图片目录：

```text
模仿学习的数学原理_工程扩展版_第1-29章含附录/images/
```

目录索引：

```text
模仿学习的数学原理_工程扩展版_第1-29章含附录/SUMMARY.md
```

公式规范：

```text
公式格式说明.md
```

成书级改造方法：

```text
成书级章节改造方法.md
```

---

## 7. 最小安全检查清单

每次提交前后检查：

```text
是否 fetch_file 获取了最新 sha？
是否基于当前 content 修改？
update_file 的 sha 是否是最新 sha？
提交后是否 fetch_file 抽查？
路径是否来自 SUMMARY 或 fetch_file，而不是凭记忆猜？
是否误删了无关段落？
```
