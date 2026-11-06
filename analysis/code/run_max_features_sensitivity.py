"""Controlled follow-up: 52/sqrt, 52/fixed-6 and 48/fixed-6 on identical folds.

Uses the preceding completed experiment's verified input snapshots. Does not
modify the production trainer/model or replace any previous experiment result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import run_feature_ablation as core

CONFIGS = {"S52_sqrt": (52, "sqrt"), "S52_fixed6": (52, 6), "S48_fixed6": (48, 6)}
PAIRS = [("S52_fixed6", "S52_sqrt"), ("S52_fixed6", "S48_fixed6"), ("S48_fixed6", "S52_sqrt")]


def model_for(max_features, n_jobs):
    return core.make_model(n_jobs).set_params(max_features=max_features)


def verify_file(source: Path, relative: str, expected: str) -> Path:
    path = source / relative
    if core.sha256(path) != expected:
        raise ValueError(f"Source experiment file has changed: {path}")
    return path


def run(args):
    source = args.source_run.resolve()
    source_manifest = source / "experiment_manifest.json"
    prior = json.loads(source_manifest.read_text(encoding="utf-8"))
    if prior["status"] != "completed":
        raise ValueError("Source experiment is not completed.")
    files = {}
    for name in ("features", "labels"):
        entry = prior["inputs"][name]
        files[f"{name}.csv"] = verify_file(source, entry["snapshot"], entry["sha256"])
    for name in ("oof_predictions.csv", "fold_assignments.csv"):
        files[name] = verify_file(source, name, prior["output_sha256"][name])
    files["source_experiment_manifest.json"] = source_manifest
    output = args.output_dir.resolve() if args.output_dir else core.ROOT / "ablation_results" / datetime.now().strftime("max_features6_%Y%m%d_%H%M%S_%f")
    core.create_run_dir(output)
    print(f"Output directory: {output}", flush=True)
    (output / "inputs").mkdir()
    file_records = {}
    for name, path in files.items():
        digest = core.sha256(path)
        shutil.copy2(path, output / "inputs" / name)
        if core.sha256(output / "inputs" / name) != digest:
            raise ValueError("Source changed during snapshotting.")
        file_records[name] = {"source": str(path), "sha256": digest}
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "follow-up sensitivity analysis motivated by preceding ablation results",
        "source_run": str(source), "inputs": file_records, "provenance": prior["provenance"],
        "configurations": {name: {"dimension": dim, "feature_ids": list(range(1, dim+1)),
                                 "model_parameters": model_for(mf, args.n_jobs).get_params()}
                           for name, (dim, mf) in CONFIGS.items()},
        "comparisons": PAIRS, "cross_validation": {"folds": 5, "random_state": 42, "shuffle": True},
        "bootstrap": {"repeats": 2000, "seed": 20261011, "paired": True, "stratified_by_true_class": True},
        "environment": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
        "source_sha256": {str(path): core.sha256(path) for path in (Path(__file__).resolve(), Path(core.__file__).resolve())},
    }
    core.write_json(output / "experiment_manifest.json", manifest)
    try:
        X, y, paths, names = core.load_dataset(output / "inputs/features.csv", output / "inputs/labels.csv")
        if np.bincount(y, minlength=5).tolist() != core.EXPECTED_COUNTS:
            raise ValueError("Expected the fixed 240-APK benchmark.")
        _, reference_rows = core.read_csv(output / "inputs/oof_predictions.csv")
        reference = core.keyed_rows(reference_rows, source)
        _, fold_rows = core.read_csv(output / "inputs/fold_assignments.csv")
        fold_lookup = core.keyed_rows(fold_rows, source)
        if set(reference) != set(paths) or set(fold_lookup) != set(paths):
            raise ValueError("Source sample sets differ.")
        splits = core.make_splits(y)
        folds = np.zeros(len(y), dtype=int)
        for fold, (_, test) in enumerate(splits, 1):
            folds[test] = fold
        for i, path in enumerate(paths):
            if (int(fold_lookup[path]["test_fold"]) != folds[i] or
                int(reference[path]["test_fold"]) != folds[i] or
                reference[path]["true_class"] != core.CLASSES[y[i]] or
                fold_lookup[path]["true_class"] != core.CLASSES[y[i]]):
                raise ValueError("Source labels/folds do not match this run.")
        manifest["class_counts"] = np.bincount(y).tolist()
        manifest["feature_names"] = names
        predictions, probabilities, metrics, timings, class_rows, matrices = {}, {}, {}, [], [], {}
        reference_checks = {}
        for key, (dimension, mf) in CONFIGS.items():
            print(f"Running {key}: dimension={dimension}, max_features={mf}", flush=True)
            pred, prob, timing = core.fit_oof(X, y, np.arange(dimension), splits, args.n_jobs,
                                             model_factory=lambda jobs, value=mf: model_for(value, jobs))
            expected_effective = int(np.sqrt(dimension)) if mf == "sqrt" else mf
            if any(row["effective_max_features"] != expected_effective for row in timing):
                raise ValueError("Unexpected effective max_features.")
            predictions[key], probabilities[key] = pred, prob
            metrics[key] = {**core.scores(y, pred), "errors": int(np.sum(pred != y))}
            print(f"  Macro-F1={metrics[key]['macro_f1']:.6f}; errors={metrics[key]['errors']}", flush=True)
            timings.extend({"configuration": key, **row} for row in timing)
            matrices[key] = confusion_matrix(y, pred, labels=range(5)).tolist()
            p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
            class_rows.extend({"configuration": key, "class": core.CLASSES[c], "precision": float(p[c]),
                               "recall": float(r[c]), "f1": float(f[c]), "support": int(n[c])} for c in range(5))
            old_key = {"S52_sqrt": "B_full", "S48_fixed6": "D_no_size_quality"}.get(key)
            if old_key:
                old_predictions = np.array([int(reference[path][old_key + "_prediction"][1:])-1 for path in paths])
                mismatches = int(np.sum(pred != old_predictions))
                reference_checks[key] = {"reference": old_key, "prediction_mismatches": mismatches}
                if mismatches:
                    raise ValueError(f"Control {key} differs from the source experiment.")
        boot = core.bootstrap_scores(y, predictions, 2000, 20261011)
        summary, pair_rows, cases, oof = [], [], [], []
        for key, (dimension, mf) in CONFIGS.items():
            row = {"configuration": key, "dimension": dimension, "max_features": mf,
                   "effective_max_features": int(np.sqrt(dimension)) if mf == "sqrt" else mf, **metrics[key]}
            for j, name in enumerate(core.METRICS):
                lo, hi = np.quantile(boot[key][:, j], [0.025, 0.975]).tolist()
                row[name + "_ci95_low"], row[name + "_ci95_high"] = lo, hi
            summary.append(row)
        for left, right in PAIRS:
            lp, rp = predictions[left], predictions[right]
            for j, name in enumerate(core.METRICS):
                lo, hi = np.quantile(boot[left][:, j] - boot[right][:, j], [0.025, 0.975]).tolist()
                pair_rows.append({"left": left, "right": right, "metric": name,
                                  "difference": metrics[left][name] - metrics[right][name],
                                  "ci95_low": lo, "ci95_high": hi,
                                  "left_only_correct": int(np.sum((lp == y) & (rp != y))),
                                  "right_only_correct": int(np.sum((lp != y) & (rp == y)))})
            for i in np.flatnonzero(lp != rp):
                cases.append({"left": left, "right": right, "apk_path": paths[i], "test_fold": int(folds[i]),
                              "true_class": core.CLASSES[y[i]], "left_prediction": core.CLASSES[lp[i]],
                              "right_prediction": core.CLASSES[rp[i]], "manual_explanation": ""})
        for i, path in enumerate(paths):
            row = {"apk_path": path, "test_fold": int(folds[i]), "true_class": core.CLASSES[y[i]]}
            for key in CONFIGS:
                row[key + "_prediction"] = core.CLASSES[predictions[key][i]]
                row.update({f"{key}_probability_C{c+1}": float(probabilities[key][i, c]) for c in range(5)})
            oof.append(row)
        for filename, rows in [("summary_metrics.csv", summary), ("paired_comparisons.csv", pair_rows),
                               ("per_class_metrics.csv", class_rows), ("fold_metrics.csv", timings),
                               ("oof_predictions.csv", oof)]:
            core.write_csv(output / filename, list(rows[0]), rows)
        core.write_csv(output / "prediction_changes.csv", ["left", "right", "apk_path", "test_fold", "true_class",
                       "left_prediction", "right_prediction", "manual_explanation"], cases)
        core.write_json(output / "confusion_matrices.json", matrices)
        core.write_json(output / "metrics.json", metrics)
        np.savez_compressed(output / "bootstrap_metrics.npz", **boot)
        manifest["reference_checks"] = reference_checks
        lines = ["# 固定 max_features=6 的补充对照", "",
                 "这是在首轮消融结果之后提出的敏感性分析，不替代原实验，也不自动更改正式模型。"
                 "使用原实验校验过的 240 个 APK 输入副本、完全相同的五折划分和随机种子 42。"
                 "三种配置均重新训练，每种五折，共 15 次拟合。", "",
                 "| 配置 | 维数 | max_features | 每次分裂候选特征数 | Macro-F1 | Accuracy | Balanced Accuracy | 错误数 |",
                 "|---|---:|---|---:|---:|---:|---:|---:|"]
        for row in summary:
            lines.append(f"| {row['configuration']} | {row['dimension']} | {row['max_features']} | {row['effective_max_features']} | "
                         f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['errors']} |")
        lines += ["", "## 配对差异", "", "差值为左侧减右侧。", "",
                  "| 对比 | Macro-F1 差值 | 条件 95% 区间 | 左对右错 | 左错右对 |", "|---|---:|---|---:|---:|"]
        for row in pair_rows:
            if row["metric"] == "macro_f1":
                lines.append(f"| {row['left']} − {row['right']} | {row['difference']:+.4f} | "
                             f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['left_only_correct']} | {row['right_only_correct']} |")
        lines += ["", "## 复现与解释范围", "",
                  "- S52_sqrt 与原完整方法 B、S48_fixed6 与原 48 维版本 D 的逐 APK 预测均完全一致。",
                  "- S52_fixed6 与 S52_sqrt 比较：只改变完整模型的 max_features 设置。",
                  "- S52_fixed6 与 S48_fixed6 比较：候选特征数均为 6，只删去 f49–f52，其他参数和划分相同。"
                  "特征子集改变仍会影响树结构及随机过程，不能将单次结果解释为普遍因果结论。",
                  "- 指标基于全部 OOF 预测。区间来自 2,000 次共享索引、按真值类别分层的配对 bootstrap。"
                  "以固定 OOF 预测为条件，未重跑训练/选择，未处理家族聚类和共享训练折依赖，未经多重比较校正。",
                  "- 这是针对已观察结果的补充分析，不能反复调节参数直到完整方法获胜。"
                  "不能将当前数据上的配置选择包装为独立测试结论。",
                  "- 继承原特征缺少提取来源清单的限制，不重新扫描或改变特征语义。", "",
                  "`prediction_changes.csv` 保留发生变化的样本；原因需核查 APK 证据，程序不推断原因。", ""]
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["source_inputs_unchanged"] = all(core.sha256(Path(record["source"])) == record["sha256"] for record in file_records.values())
        manifest["output_sha256"] = {str(path.relative_to(output)): core.sha256(path) for path in output.rglob("*")
                                     if path.is_file() and path.name != "experiment_manifest.json"}
        core.write_json(output / "experiment_manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        core.write_json(output / "experiment_manifest.json", manifest)
        raise
    print(f"Completed: {output / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=core.ROOT / "ablation_results/run_20261011_final")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.n_jobs == 0:
        parser.error("n-jobs cannot be zero")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
