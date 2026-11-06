#!/usr/bin/env python3
"""Compare the broad-AI and LLM-oriented keyword candidate sets.

The comparison is descriptive.  Neither keyword set is treated as ground-truth
LLM membership.  Records selected only by the LLM-oriented rules are exported
for semantic-model and manual review.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


SCRIPT_VERSION = "keyword-set-comparison-v1"
SUMMARY_SCHEMA_VERSION = "keyword-set-comparison-summary-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 4)


def load_jsonl(
    path: Path,
    transform: Callable[[dict[str, Any]], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    records: dict[str, Any] = {}
    digest = hashlib.sha256()
    counts = Counter()
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                counts["empty_lines"] += 1
                continue
            counts["nonempty_lines"] += 1
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["malformed_lines"] += 1
                continue
            if not isinstance(value, dict):
                counts["non_object_lines"] += 1
                continue
            docid = value.get("docid")
            if not isinstance(docid, str) or not docid.strip():
                counts["missing_docids"] += 1
                continue
            docid = docid.strip()
            if docid in records:
                counts["duplicate_docids"] += 1
                continue
            records[docid] = transform(value)
            counts["valid_unique_docids"] += 1
    return records, {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        **dict(sorted(counts.items())),
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def qwen_stats(docids: set[str], decisions: dict[str, str]) -> dict[str, int]:
    counts = Counter(decisions[docid] for docid in docids if docid in decisions)
    return {"screened": sum(counts.values()), **dict(sorted(counts.items()))}


def broad_tag_name(tags: dict[str, Any]) -> str:
    strict = bool(tags.get("strict_ai_matched"))
    broad = bool(tags.get("broad_ai_matched"))
    return f"strict_ai={str(strict).lower()}, broad_ai={str(broad).lower()}"


def build_markdown(summary: dict[str, Any]) -> str:
    sets = summary["sets"]
    tier_rows = []
    for tier, values in summary["llm_oriented_tiers"].items():
        tier_rows.append(
            f"| `{tier}` | {values['total']:,} | {values['in_broad']:,} | "
            f"{values['llm_oriented_only']:,} | {values['broad_coverage_pct']:.2f}% |"
        )

    rule_rows = []
    for rule, count in list(summary["llm_oriented_only_rules"].items())[:20]:
        rule_rows.append(f"| `{rule}` | {count:,} |")

    qwen = summary["qwen_progress"]
    return f"""# 宽泛 AI 与 LLM 定向关键词候选集对比

生成时间：{summary['generated_at_utc']}  
对比脚本：`{summary['script_version']}`

## 1. 研究口径

- **宽泛 AI 集合 A**：旧关键词规则得到的候选池，包含独立 AI、机器学习、生成式图像/音频、LLM 等宽泛信号。
- **LLM 定向集合 B**：新规则从相同的 1,210,809 条英文文本中独立筛选，重点覆盖模型/平台身份、LLM 技术术语、AI 聊天与语言生成能力。
- 两者都是**候选检索结果**，不是已确认的 LLM 标签。`B − A` 只能解释为潜在漏检候选，必须经过相同语义模型和人工复核。

## 2. 集合对比结果

| 指标 | 数量/比例 |
|---|---:|
| 宽泛 AI 集合 A | {sets['broad_ai']:,} |
| LLM 定向集合 B | {sets['llm_oriented']:,} |
| 交集 A ∩ B | {sets['intersection']:,} |
| 仅宽泛集合 A − B | {sets['broad_only']:,} |
| 仅 LLM 定向集合 B − A | {sets['llm_oriented_only']:,} |
| 并集 A ∪ B | {sets['union']:,} |
| 宽泛集合对 B 的关键词覆盖率 | {sets['llm_oriented_covered_by_broad_pct']:.2f}% |
| B 中位于宽泛集合之外的比例 | {sets['llm_oriented_outside_broad_pct']:.2f}% |
| 宽泛集合中同时命中 B 的比例 | {sets['broad_matching_llm_oriented_pct']:.2f}% |
| Jaccard 相似度 | {sets['jaccard_pct']:.2f}% |

## 3. LLM 定向集合的证据层级

| 证据层级 | B 总数 | 位于 A 中 | B − A | A 对该层覆盖率 |
|---|---:|---:|---:|---:|
{chr(10).join(tier_rows)}

## 4. B − A 的主要命中规则

| 规则 | 涉及记录数 |
|---|---:|
{chr(10).join(rule_rows)}

`B − A` 已单独导出，可使用与原集合完全相同的 Qwen 提示词、模型版本和参数进行复核。只有复核为 `llm` 或经人工确认的记录，才能视为旧宽泛规则的实际漏检。

## 5. 当前 Qwen 复核进度

| 子集 | 已复核 | llm | uncertain | non_llm |
|---|---:|---:|---:|---:|
| A 全集 | {qwen['broad_all'].get('screened', 0):,} | {qwen['broad_all'].get('llm', 0):,} | {qwen['broad_all'].get('uncertain', 0):,} | {qwen['broad_all'].get('non_llm', 0):,} |
| A ∩ B | {qwen['intersection'].get('screened', 0):,} | {qwen['intersection'].get('llm', 0):,} | {qwen['intersection'].get('uncertain', 0):,} | {qwen['intersection'].get('non_llm', 0):,} |
| A − B | {qwen['broad_only'].get('screened', 0):,} | {qwen['broad_only'].get('llm', 0):,} | {qwen['broad_only'].get('uncertain', 0):,} | {qwen['broad_only'].get('non_llm', 0):,} |
| B − A | {qwen['llm_oriented_only'].get('screened', 0):,} | {qwen['llm_oriented_only'].get('llm', 0):,} | {qwen['llm_oriented_only'].get('uncertain', 0):,} | {qwen['llm_oriented_only'].get('non_llm', 0):,} |

当前 Qwen 结果尚未覆盖全部候选，因此上述分布只能作为运行进度，不能作为最终精确率比较。

## 6. 论文可报告与不可报告内容

可以报告：

1. 两种关键词策略在同一输入总体上的集合规模、交集、差集和规则层级分布。
2. 宽泛候选池对 LLM 定向关键词候选的**关键词覆盖率**。
3. 对 `B − A` 完成统一语义复核和人工抽查后，报告其确认 LLM 数量及旧规则的增量漏检。

暂时不能报告：

1. 不能把 {sets['llm_oriented']:,} 条 B 记录称为真实 LLM 应用。
2. 不能把 {sets['llm_oriented_covered_by_broad_pct']:.2f}% 称为 LLM 召回率；它只是相对于另一套关键词规则的覆盖率。
3. 未从 `A ∪ B` 之外抽样审计前，不能声称两套规则覆盖了全部 Google Play LLM 应用。

## 7. 推荐的下一步

1. 对全部 `B − A` 使用当前三分类提示词判断 `llm / uncertain / non_llm`。
2. 人工复核 `B − A` 中的 `llm`、`uncertain` 以及分层抽取的 `non_llm`。
3. 从 `A ∪ B` 之外抽样，估计两套关键词均未命中的残余漏检风险。
4. 完成复核前，论文统一使用“候选集”“关键词覆盖率”和“潜在漏检”，避免使用“完整 LLM 数据集”或“100% 召回”。
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    docs_dir = project_dir / "docs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--broad",
        type=Path,
        default=data_dir / "google_play_apps_ai_filtered.jsonl",
    )
    parser.add_argument(
        "--llm-oriented",
        type=Path,
        default=data_dir / "google_play_apps_llm_keyword_candidates.jsonl",
    )
    parser.add_argument(
        "--qwen-decisions",
        type=Path,
        default=data_dir / "google_play_apps_llm_screening.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=data_dir / "google_play_keyword_set_comparison.json",
    )
    parser.add_argument(
        "--llm-oriented-only-output",
        type=Path,
        default=data_dir / "google_play_apps_llm_keyword_only.jsonl",
    )
    parser.add_argument(
        "--llm-oriented-only-docids",
        type=Path,
        default=data_dir / "google_play_apps_llm_keyword_only_docids.txt",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=docs_dir / "3.1_LLM关键词候选集对比.md",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.broad, args.llm_oriented, args.qwen_decisions):
        if not path.is_file():
            raise ValueError(f"input file does not exist: {path}")
    inputs = {args.broad.resolve(), args.llm_oriented.resolve(), args.qwen_decisions.resolve()}
    outputs = {
        args.json_output.resolve(),
        args.llm_oriented_only_output.resolve(),
        args.llm_oriented_only_docids.resolve(),
        args.markdown_output.resolve(),
    }
    if inputs & outputs or len(outputs) != 4:
        raise ValueError("input and output paths must be distinct")
    if not args.force:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise ValueError(
                "refusing to replace existing output(s) without --force: "
                + ", ".join(existing)
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    broad, broad_inventory = load_jsonl(
        args.broad, lambda record: record.get("ai_analysis_tags", {})
    )
    llm_oriented, llm_inventory = load_jsonl(args.llm_oriented, lambda record: record)
    qwen, qwen_inventory = load_jsonl(
        args.qwen_decisions, lambda record: str(record.get("verdict", ""))
    )

    broad_ids = set(broad)
    llm_ids = set(llm_oriented)
    intersection = broad_ids & llm_ids
    broad_only = broad_ids - llm_ids
    llm_only = llm_ids - broad_ids
    union = broad_ids | llm_ids

    all_tiers = sorted(
        {
            record.get("llm_keyword_analysis", {}).get("candidate_tier")
            for record in llm_oriented.values()
        }
        - {None}
    )
    tier_stats: dict[str, dict[str, Any]] = {}
    for tier in all_tiers:
        ids = {
            docid
            for docid, record in llm_oriented.items()
            if record.get("llm_keyword_analysis", {}).get("candidate_tier") == tier
        }
        in_broad = ids & broad_ids
        tier_stats[tier] = {
            "total": len(ids),
            "in_broad": len(in_broad),
            "llm_oriented_only": len(ids - broad_ids),
            "broad_coverage_pct": pct(len(in_broad), len(ids)),
        }

    llm_only_rules = Counter()
    llm_only_categories = Counter()
    for docid in llm_only:
        analysis = llm_oriented[docid].get("llm_keyword_analysis", {})
        llm_only_rules.update(
            {
                match.get("rule_id")
                for match in analysis.get("matched_rules", [])
                if match.get("rule_id")
            }
        )
        llm_only_categories.update(set(analysis.get("matched_categories", [])))

    broad_tag_stats = {
        "intersection": dict(
            sorted(Counter(broad_tag_name(broad[docid]) for docid in intersection).items())
        ),
        "broad_only": dict(
            sorted(Counter(broad_tag_name(broad[docid]) for docid in broad_only).items())
        ),
    }
    qwen_progress = {
        "broad_all": qwen_stats(broad_ids, qwen),
        "intersection": qwen_stats(intersection, qwen),
        "broad_only": qwen_stats(broad_only, qwen),
        "llm_oriented_all": qwen_stats(llm_ids, qwen),
        "llm_oriented_only": qwen_stats(llm_only, qwen),
    }

    args.llm_oriented_only_output.parent.mkdir(parents=True, exist_ok=True)
    subset_temp = args.llm_oriented_only_output.with_name(
        f".{args.llm_oriented_only_output.name}.tmp"
    )
    docids_temp = args.llm_oriented_only_docids.with_name(
        f".{args.llm_oriented_only_docids.name}.tmp"
    )
    subset_digest = hashlib.sha256()
    with subset_temp.open("w", encoding="utf-8", newline="\n") as subset_handle, \
            docids_temp.open("w", encoding="utf-8", newline="\n") as docids_handle:
        for docid in sorted(llm_only):
            line = json.dumps(llm_oriented[docid], ensure_ascii=False) + "\n"
            subset_handle.write(line)
            subset_digest.update(line.encode("utf-8"))
            docids_handle.write(docid + "\n")
    os.replace(subset_temp, args.llm_oriented_only_output)
    os.replace(docids_temp, args.llm_oriented_only_docids)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "interpretation": (
            "Descriptive keyword-set comparison only; LLM-oriented-only records "
            "are potential misses until semantic and manual review."
        ),
        "inputs": {
            "broad_ai": broad_inventory,
            "llm_oriented": llm_inventory,
            "qwen_decisions": qwen_inventory,
        },
        "sets": {
            "broad_ai": len(broad_ids),
            "llm_oriented": len(llm_ids),
            "intersection": len(intersection),
            "broad_only": len(broad_only),
            "llm_oriented_only": len(llm_only),
            "union": len(union),
            "llm_oriented_covered_by_broad_pct": pct(len(intersection), len(llm_ids)),
            "llm_oriented_outside_broad_pct": pct(len(llm_only), len(llm_ids)),
            "broad_matching_llm_oriented_pct": pct(len(intersection), len(broad_ids)),
            "jaccard_pct": pct(len(intersection), len(union)),
        },
        "llm_oriented_tiers": tier_stats,
        "llm_oriented_only_categories": dict(llm_only_categories.most_common()),
        "llm_oriented_only_rules": dict(llm_only_rules.most_common()),
        "broad_rule_tags": broad_tag_stats,
        "qwen_progress": qwen_progress,
        "outputs": {
            "llm_oriented_only_jsonl": {
                "path": str(args.llm_oriented_only_output.resolve()),
                "records": len(llm_only),
                "sha256": subset_digest.hexdigest(),
            },
            "llm_oriented_only_docids": str(args.llm_oriented_only_docids.resolve()),
            "markdown_report": str(args.markdown_output.resolve()),
        },
    }
    atomic_write_json(args.json_output, summary)
    atomic_write_text(args.markdown_output, build_markdown(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 2
    sets = summary["sets"]
    print(
        f"A={sets['broad_ai']:,} B={sets['llm_oriented']:,} "
        f"intersection={sets['intersection']:,} "
        f"A-only={sets['broad_only']:,} B-only={sets['llm_oriented_only']:,}"
    )
    print(
        "Broad coverage of LLM-oriented candidates: "
        f"{sets['llm_oriented_covered_by_broad_pct']:.2f}%"
    )
    print(f"Summary: {args.json_output.resolve()}")
    print(f"Report: {args.markdown_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
