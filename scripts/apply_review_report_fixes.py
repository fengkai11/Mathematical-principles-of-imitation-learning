#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply mechanical fixes from 审查报告.md.

Usage:
  python scripts/apply_review_report_fixes.py --scan
  python scripts/apply_review_report_fixes.py --write

This script is intentionally conservative and idempotent:
- ch2: move blockquote math blocks out of blockquotes by removing leading `> ` only inside `$$...$$` blocks.
- ch16: normalize norm notation from `\left\|...\right\|` to `\lVert...\rVert`.
- ch11/ch12: add missing 思考题 and 本章配图清单 only when absent.
- scan formula-index sections that may still contain Markdown tables.
- scan stale ch22/ch23 references.
- generate an image reference/name mapping table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "模仿学习的数学原理_工程扩展版_第1-29章含附录"
CHAPTERS = BOOK / "chapters"
IMAGES = BOOK / "images"

CH2 = CHAPTERS / "第02章_Behavior_Cloning_最朴素也最容易翻车的模仿学习.md"
CH11 = CHAPTERS / "第11章_GAIL_从判别器里偷一个奖励函数.md"
CH12 = CHAPTERS / "第12章_Offline_Imitation_Learning_离线数据不是越多越好_是坑有没有录进去.md"
CH16 = CHAPTERS / "第16章_BC_ACT_Diffusion_Policy_Flow_Matching_对比.md"

REPORT = ROOT / "审查报告修复执行.md"
IMAGE_MAP = ROOT / "图片命名映射表.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str, write: bool) -> None:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def count_occurrences(text: str, patterns: Iterable[str]) -> int:
    return sum(text.count(p) for p in patterns)


def move_blockquote_math_out(text: str) -> Tuple[str, int]:
    """Remove leading `> ` from lines inside blockquote math blocks.

    Example:
      > $$
      > x=y
      > $$
    becomes:
      $$
      x=y
      $$
    """
    lines = text.splitlines()
    out: List[str] = []
    in_bq_math = False
    changed = 0

    for line in lines:
        stripped = line[2:] if line.startswith("> ") else line[1:] if line == ">" else line

        if not in_bq_math and line.startswith("> ") and stripped.strip() == "$$":
            out.append(stripped)
            in_bq_math = True
            changed += 1
            continue

        if in_bq_math:
            if line.startswith("> "):
                out.append(line[2:])
                changed += 1
                if line[2:].strip() == "$$":
                    in_bq_math = False
                continue
            if line == ">":
                out.append("")
                changed += 1
                continue
            out.append(line)
            if line.strip() == "$$":
                in_bq_math = False
            continue

        out.append(line)

    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), changed


def normalize_norm_notation(text: str) -> Tuple[str, int]:
    before = count_occurrences(text, [r"\left\|", r"\right\|"])
    text = text.replace(r"\left\|", r"\lVert")
    text = text.replace(r"\right\|", r"\rVert")
    after = count_occurrences(text, [r"\left\|", r"\right\|"])
    return text, before - after


CH11_INSERT = """## 16. 思考题

1. 为什么说 GAIL 的核心不是“把 GAN 套到轨迹上”，而是 occupancy measure matching？
2. 请解释 $\rho_E(s,a)$ 与 $\rho_{\pi_\theta}(s,a)$ 的区别。它们为什么比单步动作误差更接近闭环行为？
3. 如果判别器很容易区分专家样本和策略样本，说明了什么？如果判别器完全分不清，又一定说明策略安全吗？
4. GAIL 中判别器输出为什么可以被构造成隐式 reward？这个 reward 为什么不一定可解释？
5. 对真实机械臂任务，GAIL 的 rollout 成本主要来自哪些方面？
6. 如果只能在仿真中 rollout，实机失败可能来自哪些 sim-to-real 差异？
7. 为什么说 GAIL 没有显式学习 reward，但仍然绕不开策略优化和交互成本？
8. 请比较 BC、IRL 和 GAIL：它们分别在匹配什么对象？
9. 如果一个任务无法频繁重置环境，你会如何调整 GAIL 或改用什么更稳妥的路线？
10. 第12章为什么自然转向 Offline Imitation Learning？

---

## 17. 本章配图清单

- **图11-1：BC 单步匹配与 GAIL occupancy matching 对比**：展示 BC 比较专家数据上的单步动作，GAIL 比较专家与策略的状态—动作访问分布。
- **图11-2：GAIL 判别器训练流程图**：展示专家样本、策略 rollout 样本进入判别器，判别器再向策略提供隐式 reward 信号。
- **图11-3：最优判别器与密度比关系图**：用两条分布曲线解释 $D^*(s,a)$ 如何反映专家分布和策略分布差异。
- **图11-4：GAIL 工程闭环风险图**：展示 rollout 成本、安全限制、仿真偏差和判别器震荡如何影响实机可用性。

---

"""

CH12_INSERT = """## 18. 思考题

1. 为什么 Offline IL 不是“普通 BC 换一个名字”？它额外强调了哪些风险？
2. 请解释 behavior policy $\beta(a\mid s)$ 为什么决定了离线数据的覆盖范围。
3. 数据量增加为什么不一定解决 support mismatch？请结合 $\mathcal{C}_\epsilon(\mathcal{D})$ 解释。
4. 什么是 OOD action？它为什么会通过环境转移继续制造 OOD state？
5. 只包含成功轨迹的数据集为什么可能让策略缺少恢复能力？
6. recovery data 和脏数据有什么区别？什么样的失败轨迹值得保留？
7. 如果 open-loop loss 很低但 closed-loop 失败，你会优先检查哪些数据覆盖问题？
8. 在机械臂插入、抓取或泊车任务中，请分别举一个 support mismatch 的例子。
9. 为什么更强的策略模型不能替代数据覆盖？
10. 第13章为什么从 Offline IL 自然过渡到 ACT 的 action chunk 建模？

---

## 19. 本章配图清单

- **图12-1：Offline IL 数据来源图**：展示 behavior policy 产生固定离线数据，训练策略只能在该数据覆盖区域内学习。
- **图12-2：近似覆盖区域 $\mathcal{C}_\epsilon(\mathcal{D})$ 示意图**：展示数据点附近可插值区域，以及部署策略可能访问的覆盖外区域。
- **图12-3：OOD action 到 OOD state 的风险链条**：展示动作出圈后通过状态转移造成下一状态继续出圈。
- **图12-4：成功数据与 recovery data 覆盖差异图**：展示只采成功轨迹和加入恢复轨迹后覆盖区域的变化。

---

"""


def insert_ch11_sections(text: str) -> Tuple[str, int]:
    if "## 16. 思考题" in text or "## 17. 本章配图清单" in text:
        return text, 0
    marker = "## 16. 建议阅读的附录条目"
    if marker not in text:
        return text, 0
    text = text.replace(marker, CH11_INSERT + "## 18. 建议阅读的附录条目", 1)
    text = text.replace("## 17. 本章小结：为什么下一章是 Offline IL", "## 19. 本章小结：为什么下一章是 Offline IL", 1)
    return text, 1


def insert_ch12_sections(text: str) -> Tuple[str, int]:
    if "## 18. 思考题" in text or "## 19. 本章配图清单" in text:
        return text, 0
    marker = "## 18. 建议阅读的附录条目"
    if marker not in text:
        return text, 0
    text = text.replace(marker, CH12_INSERT + "## 20. 建议阅读的附录条目", 1)
    text = text.replace("## 19. 本章小结：从离线数据覆盖走向动作块策略", "## 21. 本章小结：从离线数据覆盖走向动作块策略", 1)
    return text, 1


def markdown_files() -> List[Path]:
    return sorted([p for p in ROOT.rglob("*.md") if ".git" not in p.parts])


def scan_formula_index_tables() -> List[str]:
    rows: List[str] = []
    heading_re = re.compile(r"^##+\s+.*公式索引")
    for path in markdown_files():
        rel = path.relative_to(ROOT)
        lines = read_text(path).splitlines()
        for i, line in enumerate(lines):
            if heading_re.search(line):
                window = lines[i + 1 : i + 80]
                has_table = any(l.strip().startswith("|") for l in window)
                has_block_math_in_table = False
                for j, l in enumerate(window):
                    if l.strip().startswith("|") and ("$$" in l or "\\[" in l or "<div" in l):
                        has_block_math_in_table = True
                        break
                if has_table:
                    rows.append(f"| `{rel}` | {i+1} | 是 | {'是' if has_block_math_in_table else '否'} |")
    return rows


def scan_stale_refs() -> List[str]:
    patterns = ["第22章", "第23章", "ch22", "ch23", "图22-", "图23-"]
    rows: List[str] = []
    for path in markdown_files():
        rel = path.relative_to(ROOT)
        for idx, line in enumerate(read_text(path).splitlines(), start=1):
            hits = [p for p in patterns if p in line]
            if hits:
                snippet = line.strip().replace("|", "\\|")[:160]
                rows.append(f"| `{rel}` | {idx} | {', '.join(hits)} | {snippet} |")
    return rows


IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\((\.\./images/[^)]+)\)")
FIG_NO_RE = re.compile(r"图(\d+)-(\d+)")


def scan_image_mapping() -> List[str]:
    rows: List[str] = []
    sources = list(CHAPTERS.glob("*.md")) + list((BOOK / "导读").glob("*.md"))
    for path in sorted(sources):
        rel = path.relative_to(ROOT)
        for idx, line in enumerate(read_text(path).splitlines(), start=1):
            for alt, link in IMAGE_LINK_RE.findall(line):
                img_rel = link.replace("../images/", "")
                img_path = IMAGES / img_rel
                exists = "是" if img_path.exists() else "否"
                fig_match = FIG_NO_RE.search(alt)
                suggested = ""
                if fig_match:
                    c, n = fig_match.groups()
                    suffix = Path(img_rel).suffix or ".png"
                    title = re.sub(r"^图\d+-\d+\s*", "", alt).strip()
                    title = re.sub(r"[：:，,\s]+", "_", title).strip("_")
                    suggested = f"图{int(c)}-{int(n)}_{title}{suffix}" if title else f"图{int(c)}-{int(n)}{suffix}"
                mismatch = ""
                if suggested and Path(img_rel).name != suggested:
                    mismatch = "需核对"
                safe_alt = alt.replace("|", "\\|")
                rows.append(f"| `{rel}` | {idx} | {safe_alt} | `{img_rel}` | {exists} | `{suggested}` | {mismatch} |")
    return rows


def build_report(changes: List[str], formula_rows: List[str], stale_rows: List[str]) -> str:
    formula_body = "\n".join(formula_rows) if formula_rows else "| 无 | - | - | - |"
    stale_body = "\n".join(stale_rows) if stale_rows else "| 无 | - | - | - |"
    change_body = "\n".join(f"- {c}" for c in changes) if changes else "- scan 模式：未写入源文件。"
    return f"""# 审查报告修复执行记录

> 本文件由 `scripts/apply_review_report_fixes.py` 生成，用于记录审查报告后续修复状态。

## 1. 本次执行动作

{change_body}

## 2. 公式索引表格扫描

| 文件 | 行号 | 是否包含 Markdown 表格 | 表格内是否疑似包含块级公式 |
|---|---:|---|---|
{formula_body}

## 3. ch22 / ch23 旧引用扫描

| 文件 | 行号 | 命中项 | 内容摘录 |
|---|---:|---|---|
{stale_body}

## 4. 下一步建议

1. 若公式索引扫描仍有表格，优先改为“逐条公式索引 + 公式块 + 含义说明”。
2. 对 ch22/ch23 命中项逐条判断：正书固定为六篇二十一章时，应改为 ch20/ch21 或移入存档说明。
3. 使用 `图片命名映射表.md` 逐条确认图片重命名，不建议在未确认图片内容前盲目批量改名。
"""


def build_image_map(rows: List[str]) -> str:
    body = "\n".join(rows) if rows else "| 无 | - | - | - | - | - | - |"
    return f"""# 图片命名映射表

> 本表由 `scripts/apply_review_report_fixes.py` 生成，用于最后统一图片命名和引用。

| 引用文件 | 行号 | 图注 alt | 当前图片文件 | 文件是否存在 | 建议文件名 | 状态 |
|---|---:|---|---|---|---|---|
{body}

## 使用原则

1. 先确认图片内容确实对应图注，再改文件名和引用。
2. 正书建议统一采用 `图X-Y_简短中文说明.svg/png`。
3. 若同一张图被多处复用，优先保留内容语义最准确的命名，并在章节中保持图号一致。
4. 对旧章节号图片，例如 `图22-1_*` 被第20章引用，应优先核对是否来自章节重排遗留。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write changes to repository files")
    parser.add_argument("--scan", action="store_true", help="scan only")
    args = parser.parse_args()
    write = args.write

    changes: List[str] = []

    if CH2.exists():
        text = read_text(CH2)
        new_text, n = move_blockquote_math_out(text)
        if n:
            write_text(CH2, new_text, write)
            changes.append(f"ch2：移出 blockquote 内公式块，处理行数 {n}。")
        else:
            changes.append("ch2：未发现 blockquote 内公式块，跳过。")
    else:
        changes.append(f"ch2：文件不存在：{CH2}")

    if CH16.exists():
        text = read_text(CH16)
        new_text, n = normalize_norm_notation(text)
        if n:
            write_text(CH16, new_text, write)
            changes.append(f"ch16：范数写法统一，替换次数 {n}。")
        else:
            changes.append("ch16：未发现 \\left\\| / \\right\\|，跳过。")
    else:
        changes.append(f"ch16：文件不存在：{CH16}")

    if CH11.exists():
        text = read_text(CH11)
        new_text, n = insert_ch11_sections(text)
        if n:
            write_text(CH11, new_text, write)
            changes.append("ch11：补入思考题与本章配图清单，并顺延后续章节编号。")
        else:
            changes.append("ch11：思考题/配图清单已存在或未找到插入点，跳过。")
    else:
        changes.append(f"ch11：文件不存在：{CH11}")

    if CH12.exists():
        text = read_text(CH12)
        new_text, n = insert_ch12_sections(text)
        if n:
            write_text(CH12, new_text, write)
            changes.append("ch12：补入思考题与本章配图清单，并顺延后续章节编号。")
        else:
            changes.append("ch12：思考题/配图清单已存在或未找到插入点，跳过。")
    else:
        changes.append(f"ch12：文件不存在：{CH12}")

    formula_rows = scan_formula_index_tables()
    stale_rows = scan_stale_refs()
    image_rows = scan_image_mapping()

    write_text(REPORT, build_report(changes, formula_rows, stale_rows), write)
    write_text(IMAGE_MAP, build_image_map(image_rows), write)

    print("\n".join(changes))
    print(f"formula index table candidates: {len(formula_rows)}")
    print(f"stale ch22/ch23 refs: {len(stale_rows)}")
    print(f"image refs: {len(image_rows)}")
    if not write:
        print("scan only; rerun with --write to modify files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
