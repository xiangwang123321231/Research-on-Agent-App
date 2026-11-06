"""59-dimensional, eight-configuration feature-block ablation study.

Run from any working directory:
    python code/run_feature_ablation.py --max-features 6

All variants use the adopted fixed max_features=6. No APK scanning, parameter
search or production-model replacement occurs. Results go to a NEW directory.
See feature_ablation_README.md for the protocol and interpretation boundaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from feature_vectorizer import FEATURE_SCHEMA_VERSION, FeatureVectorizer
from feature_contract import validate_records

ROOT = Path(__file__).resolve().parent.parent
LABEL_COLUMNS = ["is_direct_client", "is_server_proxy", "is_webview", "is_local", "is_non_ai"]
CLASSES = ["C1", "C2", "C3", "C4", "C5"]
SEED = 42
STUDY_VERSION = "feature-block-ablation-59d-v2026.10.28.1"
EXPECTED_FEATURE_SCHEMA = "features-59d-v2026.10.25.1"
FULL = "A_full"
# These are column blocks, not switches disabling extraction mechanisms.
REMOVED_GROUPS = {
    "B_no_service": tuple([*range(4, 10), *range(12, 20), 25]),
    "C_no_protocol": tuple([3, *range(20, 25), 58, 59]),
    "D_no_web": (10, 11, 38, 39, 40, 45),
    "E_no_device": tuple([*range(26, 38), 41, 46, *range(53, 58)]),
    "F_no_http_relations": (44, 47, 48),
    "G_no_density": (1, 2, 42, 43),
    "H_no_size_quality": (49, 50, 51, 52),
}
VARIANTS = {FULL: tuple(range(1, 60)), **{
    key: tuple(i for i in range(1, 60) if i not in removed)
    for key, removed in REMOVED_GROUPS.items()}}
TITLES = {
    FULL: "完整方法",
    "B_no_service": "移除服务身份与访问配置",
    "C_no_protocol": "移除语言生成与协议语义",
    "D_no_web": "移除网页交互",
    "E_no_device": "移除端侧推理",
    "F_no_http_relations": "移除 HTTP 局部方法关系",
    "G_no_density": "移除通用证据数量与来源",
    "H_no_size_quality": "移除 APK 规模与分析质量",
}
PLOT_TITLES = dict(zip(VARIANTS, ["Full", "No service/configuration", "No generation/protocol",
    "No web interaction", "No on-device", "No HTTP method relations", "No density/provenance",
    "No size/quality"]))
COMPARISONS = [(FULL, key) for key in REMOVED_GROUPS]
METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "mcc")


def validate_design(names):
    removed = [i for ids in REMOVED_GROUPS.values() for i in ids]
    if FEATURE_SCHEMA_VERSION != EXPECTED_FEATURE_SCHEMA or len(names) != 59:
        raise ValueError("This study is frozen to features-59d-v2026.10.25.1.")
    if sorted(removed) != list(range(1, 60)):
        raise ValueError("Feature blocks must partition all 59 columns exactly once.")
    if list(map(len, VARIANTS.values())) != [59, 44, 51, 53, 40, 56, 55, 55]:
        raise ValueError("Unexpected feature configuration dimensions.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Empty or duplicate column names: {path}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Empty table or malformed row width: {path}")
    return fields, rows


def keyed_rows(rows: list[dict], source: Path) -> dict[str, dict]:
    result = {}
    seen_sha = set()
    for row in rows:
        key = row.get("apk_path", "").strip()
        if not key or key in result:
            raise ValueError(f"Blank/duplicate apk_path in {source}: {key!r}")
        # A hash-named APK cannot appear twice under different directories.
        stem = key.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if re.fullmatch(r"[0-9a-fA-F]{64}", stem):
            identity = stem.lower()
            if identity in seen_sha:
                raise ValueError(f"Duplicate SHA-named APK in {source}: {stem}")
            seen_sha.add(identity)
        result[key] = row
    return result


def load_dataset(features: Path, labels: Path):
    names = FeatureVectorizer().feature_names
    validate_design(names)
    fields, rows = read_csv(features)
    if fields != ["apk_path", *names]:
        raise ValueError("Feature header/order must exactly match the frozen 59 named columns.")
    label_fields, label_rows = read_csv(labels)
    if not set(["apk_path", *LABEL_COLUMNS]).issubset(label_fields):
        raise ValueError("Missing required label columns.")
    keyed = keyed_rows(rows, features)
    labelled = keyed_rows(label_rows, labels)
    if set(keyed) != set(labelled):
        raise ValueError(f"Feature/label APK sets differ: {len(set(keyed)-set(labelled))} unlabelled, "
                         f"{len(set(labelled)-set(keyed))} without features. No silent dropping is allowed.")
    paths = list(keyed)  # Preserve historical CSV order for the seed-42 folds.
    values, targets = [], []
    for path in paths:
        raw = [keyed[path][name].strip() for name in names]
        if any(value == "" for value in raw):
            raise ValueError(f"Blank feature value for {path}; missing is not zero.")
        values.append([float(value) for value in raw])
        bits = [labelled[path][name].strip() for name in LABEL_COLUMNS]
        if any(bit not in ("0", "1") for bit in bits) or bits.count("1") != 1:
            raise ValueError(f"Labels must be binary and exactly one-hot: {path}")
        targets.append(bits.index("1"))
    X = np.asarray(values, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite feature values detected.")
    return X, np.asarray(targets, dtype=np.int32), paths, names


def make_model(n_jobs: int = -1) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2,
                               class_weight="balanced", max_features=6,
                               random_state=SEED, n_jobs=n_jobs)


def make_study_model(n_jobs: int = -1, max_features: int | str = 6) -> ExtraTreesClassifier:
    """The current study fixes feature sampling; no per-variant tuning."""
    if max_features != 6:
        raise ValueError("The 59d ablation protocol requires max_features=6.")
    return make_model(n_jobs).set_params(max_features=max_features)


def make_splits(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if np.bincount(y, minlength=5).min() < 5:
        raise ValueError("Every class needs at least five examples; no implicit CV fallback.")
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def fit_oof(X, y, columns, splits, n_jobs=-1, *, model_factory=None):
    """Use the frozen model by default; explicit factories are for separate studies."""
    predicted = np.full(len(y), -1, dtype=np.int32)
    probabilities = np.full((len(y), 5), np.nan)
    seen = np.zeros(len(y), dtype=int)
    timings = []
    for fold, (train, test) in enumerate(splits, 1):
        if np.intersect1d(train, test).size:
            raise ValueError("Train/test overlap.")
        started = time.perf_counter()
        model = make_model(n_jobs) if model_factory is None else model_factory(n_jobs)
        model.fit(X[train][:, columns], y[train])
        if not np.array_equal(model.classes_, np.arange(5)):
            raise ValueError("Training fold does not contain all five classes.")
        predicted[test] = model.predict(X[test][:, columns])
        probabilities[test] = model.predict_proba(X[test][:, columns])
        seen[test] += 1
        timings.append({"fold": fold, "train_n": len(train), "test_n": len(test),
                        "seconds": time.perf_counter() - started,
                        "effective_max_features": model.estimators_[0].max_features_})
    if not np.all(seen == 1) or not np.isfinite(probabilities).all():
        raise ValueError("Each APK must receive exactly one valid held-out prediction.")
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Prediction probabilities do not sum to one.")
    return predicted, probabilities, timings


def scores(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Metrics from a fixed five-class confusion matrix; also used by bootstrap."""
    cm = np.bincount(5 * y + predicted, minlength=25).reshape(5, 5).astype(float)
    tp = cm.diagonal()
    actual, assigned = cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2 * tp, actual + assigned, out=np.zeros(5), where=(actual + assigned) != 0)
    recall = np.divide(tp, actual, out=np.zeros(5), where=actual != 0)
    n, correct = cm.sum(), tp.sum()
    denom = np.sqrt((n*n - assigned @ assigned) * (n*n - actual @ actual))
    return {"macro_f1": float(f1.mean()), "accuracy": float(correct / n),
            "balanced_accuracy": float(recall.mean()),
            "mcc": float((correct*n - actual @ assigned) / denom) if denom else 0.0}


def bootstrap_scores(y, predictions, repeats, seed):
    """Same class-stratified resample for every variant: valid paired differences.

    Conditional on these OOF predictions; does not repeat fitting/model selection
    and does not account for family/developer clustering or training-set overlap.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in range(5)]
    output = {key: np.empty((repeats, len(METRICS))) for key in predictions}
    for b in range(repeats):
        idx = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        for key, pred in predictions.items():
            result = scores(y[idx], pred[idx])
            output[key][b] = [result[metric] for metric in METRICS]
    return output


def check_reference(path: Path, paths: list[str], y, predicted) -> dict:
    fields, rows = read_csv(path)
    if not {"apk_path", "true_class", "pred_class_extratrees"}.issubset(fields):
        raise ValueError("Reference predictions lack required columns.")
    reference = keyed_rows(rows, path)
    if set(reference) != set(paths):
        raise ValueError("Historical prediction APK set differs from the experiment.")
    historical = np.array([int(reference[p]["pred_class_extratrees"]) for p in paths])
    truth = np.array([int(reference[p]["true_class"]) for p in paths])
    if not np.array_equal(truth, y) or np.any((historical < 0) | (historical > 4)):
        raise ValueError("Historical labels differ or reference predictions are invalid.")
    mismatches = np.flatnonzero(historical != predicted)
    return {"path": str(path), "sha256": sha256(path),
            "status": "identical_predictions" if not len(mismatches) else "prediction_drift",
            "mismatch_n": len(mismatches), "mismatch_apks": [paths[i] for i in mismatches],
            "historical_metrics": scores(y, historical), "rerun_metrics": scores(y, predicted)}


def create_run_dir(output: Path) -> None:
    # Refuse ALL existing directories, including empty ones, to avoid overwrites.
    output.mkdir(parents=True, exist_ok=False)


def validate_signal_source(signals: Path, feature_manifest: dict, paths, names, X) -> dict:
    """Check the saved extraction facts and explicit 57->59 derivation, not APKs."""
    with signals.open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    raw = keyed_rows(records, signals)
    if set(raw) != set(paths):
        raise ValueError("Signal/feature APK sets differ; no implicit subset or dropping.")
    hashes = [str(r.get("sha256", "")).lower() for r in records]
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("Signals require valid, unique APK content SHA-256 values.")
    refresh = any(r.get("metadata", {}).get("feature_schema_version") != FEATURE_SCHEMA_VERSION
                  for r in records)
    contract = validate_records(records, allow_feature_refresh=refresh)
    if feature_manifest.get("contract") != contract or feature_manifest.get("feature_names") != names:
        raise ValueError("Feature manifest contract/names do not match current signal interpretation.")
    derivation = feature_manifest.get("derivation")
    if refresh or derivation:
        expected = {"operation": "revectorize", "source_sha256": sha256(signals),
                    "source_record_count": len(records),
                    "source_feature_schema_version": records[0]["metadata"]["feature_schema_version"],
                    "target_feature_schema_version": FEATURE_SCHEMA_VERSION}
        if not isinstance(derivation, dict) or any(derivation.get(k) != v for k, v in expected.items()):
            raise ValueError("Missing or inconsistent feature derivation; do not bypass source versions.")
    regenerated, generated_names = FeatureVectorizer().transform([raw[p] for p in paths])
    if generated_names != names or not np.array_equal(np.asarray(regenerated, dtype=np.float32), X):
        raise ValueError("Feature values do not reproduce from the supplied signal records.")
    return {"status": "revalidated_against_raw_signals", "contract": contract,
            "source_feature_schema_version": records[0]["metadata"]["feature_schema_version"],
            "recomputed_cells": int(X.size), "feature_refresh": refresh}


def plot_results(output: Path, y, predictions, metrics, boot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(VARIANTS)
    labels = [f"{key[0]}: {PLOT_TITLES[key]} ({len(VARIANTS[key])})" for key in keys]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for i, key in enumerate(keys):
        low, high = np.quantile(boot[key][:, 0], [0.025, 0.975])
        ax.hlines(i, low, high, color="#245b83", linewidth=2)
        ax.plot(metrics[key]["macro_f1"], i, "o", color="#245b83")
        ax.text(1.01, i, f"{metrics[key]['macro_f1']:.4f}", va="center", fontsize=10)
    ax.set(yticks=range(len(keys)), yticklabels=labels, xlim=(0, 1.12),
           xlabel="OOF Macro-F1 (conditional 95% bootstrap interval)",
           title="Feature-configuration comparison")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output / "macro_f1_comparison.png", dpi=200)
    fig.savefig(output / "macro_f1_comparison.svg")
    plt.close(fig)
    rows = (len(keys) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), layout="constrained", squeeze=False)
    for ax, key, label in zip(axes.flat, keys, labels):
        cm = confusion_matrix(y, predictions[key], labels=np.arange(5))
        normalized = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(5):
            for col in range(5):
                ax.text(col, row, f"{cm[row, col]}\n{normalized[row, col]:.0%}",
                        ha="center", va="center", fontsize=9,
                        color="white" if normalized[row, col] > 0.55 else "black")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted", ylabel="True", title=label)
        ax.set_title(label, fontsize=11)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle("OOF confusion matrices: counts and row-normalized percentages")
    fig.savefig(output / "confusion_matrices.png", dpi=200)
    fig.savefig(output / "confusion_matrices.svg")
    plt.close(fig)


def render_report(output, manifest, metrics, per_class, pairs):
    parameters = manifest["model_parameters"]
    sample_count = manifest["sample_count"]
    counts = manifest["class_counts"]
    fold_sizes = manifest["cross_validation"]["fold_sizes"]
    fold_description = "；".join(
        f"第 {row['fold']} 折 {row['train_n']} 个训练样本、{row['test_n']} 个测试样本"
        for row in fold_sizes
    )
    feature_sampling = (
        f"max_features={parameters['max_features']}；全部 {manifest['fit_budget']} 次折内拟合保持一致。"
        "特征子集改变仍会影响随机采样和树结构，因此固定候选数量不等于固定树结构。"
    )
    lines = ["# 59 维特征组消融实验", "",
             "## 1. 实验范围", "",
             f"本实验复用本轮固定的 {sample_count} 个 APK 特征及标签，比较 {len(VARIANTS)} 种特征配置。"
             "不重新扫描 APK，不修改信号定义，不重新选择分类器，也不替换正式模型。", "",
             "C1：第三方 LLM 服务直连；C2：服务端介导的 LLM 访问；"
             "C3：基于网页的 LLM 交互；C4：端侧 LLM 推理；C5：非目标候选应用。"
             "C3 不限于 WebView，也包括符合主要网页交互定义的 Custom Tabs、TWA 等形式。"
             "C5 表示不属于本文 LLM 研究范围，并非没有任何 AI 功能；处理失败或证据不足不能自动归入 C5。", "",
             f"类别数量：{manifest['class_counts']}。所有配置共用分层五折划分，划分与模型种子均为 42。"
             f"{fold_description}。指标基于全部 {sample_count} 个折外预测汇总，并非折指标简单平均。", "",
             "ExtraTrees：300 棵树、最大深度 8、叶节点最少 2 个样本、类别权重 balanced、"
             + feature_sampling, "",
             "## 2. 结果", "",
             "| 配置 | 维数 | Macro-F1 | 条件 95% 区间 | Accuracy | Balanced Accuracy | MCC | 错误数 |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for key, ids in VARIANTS.items():
        m = metrics[key]
        lo, hi = m["macro_f1_ci95"]
        lines.append(f"| {key[0]}：{TITLES[key]} | {len(ids)} | {m['macro_f1']:.4f} | "
                     f"[{lo:.4f}, {hi:.4f}] | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
                     f"{m['mcc']:.4f} | {m['errors']} |")
    lines += ["", "### 各类别 F1", "", "| 配置 | C1 | C2 | C3 | C4 | C5 |",
              "|---|---:|---:|---:|---:|---:|"]
    for key in VARIANTS:
        values = [row["f1"] for row in per_class if row["variant"] == key]
        lines.append(f"| {key} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "### 预先确定的配对比较", "",
              "差值方向为左侧配置减右侧配置；正值表示左侧 Macro-F1 更高。", "",
              "| 对比 | Macro-F1 差值 | 条件 95% 差值区间 | 左对右错 | 左错右对 |",
              "|---|---:|---|---:|---:|"]
    for row in pairs:
        if row["metric"] == "macro_f1":
            lines.append(f"| {row['left']} − {row['right']} | {row['difference']:+.4f} | "
                         f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
                         f"{row['left_only_correct']} | {row['right_only_correct']} |")
    lines += ["", "错误数相同并不保证 Macro-F1 相同：同一个误分类样本若被分配到不同的错误类别，"
              "会改变相应类别的 Precision/F1。双向正确数为零的比较也应查看逐样本分歧。", "",
              "## 3. 完整模型基线复现核对", "",
              f"核对状态：`{manifest['reference_check']['status']}`；"
              f"与指定 ExtraTrees 基线折外预测不同的 APK 数量：{manifest['reference_check'].get('mismatch_n', '未比较')}。", "",
              "## 4. 解释边界", "",
              "- 本实验删除特征列，不重新提取或过滤信号；其他列仍可包含同类信息。"
              "差值衡量专门特征组在其余列保留时的增量作用，不代表关闭了对应提取机制。",
              "- f22 由 f08 与 f20 推导；移除服务组后仍保留 f22 和 HTTP 关系等间接信息。"
              "网页、端侧及语义事实也可能继续影响 f01/f02/f42/f43。",
              "- F 仅删除 f44/f47/f48；网页关系 f45 和端侧关系 f46 分别归入 D、E。"
              "不能将 F 称为移除全部关系分析。",
              "- 七组大小不同且可能互补，不能直接按分数下降给证据重要性排序或将贡献相加。"
              "H 删除 f49–f52；性能变化本身不证明数据泄漏。",
              f"- 区间来自 {manifest['bootstrap']['repeats']} 次按真实类别分层的样本 bootstrap，"
              "所有配置共用每次抽样索引。区间以已得到的折外预测为条件，未重跑训练或模型选择，"
              "未处理开发者/代码家族聚类及训练折重叠依赖，也不代表独立测试不确定性。",
              "- 各比较区间未经多重比较校正，作为描述性敏感性结果，不自动宣称统计显著优越。"
              f"C3/C4 支持数为 {counts[2]}/{counts[3]}，须结合错误数解释。",
              "- 同一基准曾用于方法开发与模型选择，本实验仍为内部验证，不能替代独立样本验证。"
              "本轮固定参数沿用当前基线，不根据消融结果调整各组参数。",
              "- 沿用输入标签，不自动改标或删除待复核样本。标签争议应单独复核；"
              "与记录标签不一致不能一概称为已确认误判。",
              "- 不因本轮哪个版本得分更高而自动替换已经冻结的生产模型。", "",
              "## 5. 数据来源与产物", "",
              f"特征来源清单状态：`{manifest['provenance']['status']}`。"
              "主信号与特征映射经过逐值核对，源记录的旧特征版本仅在有明确重映射记录时接受。", "",
              "- `experiment_manifest.json`：运行状态、数据/脚本哈希、特征清单、模型参数、环境及抽样设置。",
              "- `inputs/`：本次实际使用的信号、特征、清单、标签与参考预测副本；`sources/` 保留代码快照。",
              "- `fold_assignments.csv`：每个 APK 唯一的测试折编号，其他四折为训练集。",
              "- `feature_configurations.csv`：全部 59 维在各配置中的保留情况。",
              "- `summary_metrics.csv`、`per_class_metrics.csv`：汇总及分类别指标。",
              "- `oof_predictions.csv`：八种配置逐 APK 的记录标签、预测和类别概率。",
              "- `paired_comparisons.csv`、`paired_case_review.csv`：配对差值及双向错误案例，"
              "案例原因尚需人工核查，脚本不推断原因。",
              "- `confusion_matrices.json`、`fold_metrics.csv`：混淆矩阵、逐折指标和训练耗时。",
              "- `macro_f1_comparison.*`、`confusion_matrices.*`：PNG 与 SVG 图。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args) -> Path:
    features, labels, reference = args.features.resolve(), args.labels.resolve(), args.reference_predictions.resolve()
    signals = args.signals.resolve()
    max_features = getattr(args, "max_features", 6)
    make_study_model(args.n_jobs, max_features)  # Validate before creating any output.
    if args.bootstrap < 100 or args.n_jobs == 0:
        raise ValueError("bootstrap must be >=100; n-jobs cannot be zero")
    source_manifest = features.with_name(features.name + ".manifest.json")
    input_paths = {"features": features, "labels": labels, "reference_predictions": reference,
                   "signals": signals, "feature_manifest": source_manifest}
    input_hashes = {key: sha256(path) for key, path in input_paths.items()}
    X, y, paths, names = load_dataset(features, labels)
    class_counts = np.bincount(y, minlength=5).tolist()
    splits = make_splits(y)
    # Validate reference now, before any costly fitting or output creation.
    check_reference(reference, paths, y, y)
    data = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
    provenance = validate_signal_source(signals, data, paths, names, X)
    provenance["sha256"] = input_hashes["feature_manifest"]
    output = args.output_dir.resolve() if args.output_dir else ROOT / "ablation_results" / datetime.now().strftime("59d_%Y%m%d_%H%M%S_%f")
    create_run_dir(output)
    print(f"Output directory: {output}", flush=True)
    inputs = output / "inputs"
    inputs.mkdir()
    input_records = {}
    snapshot_names = {"features": "features.csv", "labels": "labels.csv",
                      "reference_predictions": "reference_predictions.csv",
                      "signals": "apks.jsonl", "feature_manifest": "features.csv.manifest.json"}
    for key, path in input_paths.items():
        target = inputs / snapshot_names[key]
        shutil.copy2(path, target)
        input_records[key] = {"path": str(path), "sha256": input_hashes[key], "snapshot": str(target.relative_to(output))}
        if input_hashes[key] != sha256(target) or input_hashes[key] != sha256(path):
            raise ValueError("Input changed while snapshotting.")
    # Fit the immutable snapshots, not originals that may be edited during a run.
    X2, y2, paths2, names2 = load_dataset(inputs / "features.csv", inputs / "labels.csv")
    if not (np.array_equal(X, X2) and np.array_equal(y, y2) and paths == paths2 and names == names2):
        raise ValueError("Input changed between validation and snapshotting.")
    # Preserve the local Python dependency sources, including feature-admission helpers.
    source_files = sorted((ROOT / "code").glob("*.py"))
    sources = output / "sources"
    sources.mkdir()
    source_hashes = {}
    for path in source_files:
        before = sha256(path)
        shutil.copy2(path, sources / path.name)
        if before != sha256(sources / path.name) or before != sha256(path):
            raise ValueError("Source changed while snapshotting.")
        source_hashes[str(path)] = before
    model_factory = lambda jobs: make_study_model(jobs, max_features)
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "study_version": STUDY_VERSION,
        "study_type": "eight-configuration column-block ablation under a fixed shared classifier",
        "fit_budget": len(VARIANTS) * len(splits),
        "inputs": input_records, "provenance": provenance,
        "sample_count": len(y), "class_counts": class_counts,
        "label_mapping": dict(zip(CLASSES, LABEL_COLUMNS)),
        "variants": {key: {"title": TITLES[key], "feature_ids": list(ids),
                           "feature_names": [names[i-1] for i in ids],
                           "removed_feature_ids": list(REMOVED_GROUPS.get(key, ()))} for key, ids in VARIANTS.items()},
        "predefined_comparisons": [list(pair) for pair in COMPARISONS],
        "model_parameters": model_factory(args.n_jobs).get_params(),
        "cross_validation": {"folds": 5, "shuffle": True, "random_state": SEED, "grouped": False,
                             "fold_sizes": [{"fold": fold, "train_n": len(train), "test_n": len(test)}
                                            for fold, (train, test) in enumerate(splits, 1)]},
        "bootstrap": {"repeats": args.bootstrap, "seed": args.bootstrap_seed,
                      "method": "class-stratified paired percentile; fixed OOF predictions"},
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "source_sha256": source_hashes,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(output / "experiment_manifest.json", manifest)
    try:
        folds = np.zeros(len(y), dtype=int)
        for fold, (_, test) in enumerate(splits, 1):
            folds[test] = fold
        write_csv(output / "fold_assignments.csv", ["row_index", "apk_path", "true_class", "test_fold"],
                  [{"row_index": i, "apk_path": path, "true_class": CLASSES[y[i]], "test_fold": int(folds[i])} for i, path in enumerate(paths)])
        write_csv(output / "feature_configurations.csv", ["feature_id", "feature_name", *VARIANTS],
                  [{"feature_id": i, "feature_name": name, **{key: int(i in ids) for key, ids in VARIANTS.items()}}
                   for i, name in enumerate(names, 1)])
        predictions, probabilities, metrics, per_class, fold_rows, matrices = {}, {}, {}, [], [], {}
        for key in VARIANTS:
            print(f"Running {key}: {len(VARIANTS[key])} features, five folds", flush=True)
            cols = np.array(VARIANTS[key]) - 1
            pred, prob, timings = fit_oof(X, y, cols, splits, args.n_jobs,
                                        model_factory=model_factory)
            expected_candidates = 6
            if any(row["effective_max_features"] != expected_candidates for row in timings):
                raise ValueError(f"Unexpected candidate-feature count in {key}.")
            predictions[key], probabilities[key] = pred, prob
            if key == FULL:
                manifest["reference_check"] = check_reference(inputs / "reference_predictions.csv", paths, y, pred)
                write_json(output / "historical_reproduction.json", manifest["reference_check"])
                if manifest["reference_check"]["mismatch_n"] and not args.allow_reference_drift:
                    raise ValueError("Full-model OOF predictions differ from history. Inspect historical_reproduction.json; "
                                     "only explicitly use --allow-reference-drift in a NEW run after investigation.")
            metrics[key] = {**scores(y, pred), "errors": int(np.sum(pred != y))}
            print(f"  Macro-F1={metrics[key]['macro_f1']:.6f}; errors={metrics[key]['errors']}", flush=True)
            precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(5), zero_division=0)
            for c in range(5):
                per_class.append({"variant": key, "class": CLASSES[c], "precision": float(precision[c]),
                                  "recall": float(recall[c]), "f1": float(f1[c]), "support": int(support[c])})
            for timing, (_, test) in zip(timings, splits):
                fold_rows.append({"variant": key, **timing, **scores(y[test], pred[test])})
            cm = confusion_matrix(y, pred, labels=np.arange(5))
            matrices[key] = {"counts": cm.tolist(), "row_normalized": (cm/cm.sum(axis=1, keepdims=True)).tolist()}
        print(f"Paired stratified bootstrap: {args.bootstrap} resamples", flush=True)
        boot = bootstrap_scores(y, predictions, args.bootstrap, args.bootstrap_seed)
        summary, pairs, cases = [], [], []
        for key, ids in VARIANTS.items():
            row = {"variant": key, "feature_count": len(ids), **metrics[key]}
            for j, metric in enumerate(METRICS):
                low, high = np.quantile(boot[key][:, j], [0.025, 0.975]).tolist()
                metrics[key][metric + "_ci95"] = [low, high]
                row[metric + "_ci95_low"], row[metric + "_ci95_high"] = low, high
            summary.append(row)
        for left, right in COMPARISONS:
            lcorrect, rcorrect = predictions[left] == y, predictions[right] == y
            for j, metric in enumerate(METRICS):
                low, high = np.quantile(boot[left][:, j] - boot[right][:, j], [0.025, 0.975]).tolist()
                pairs.append({"left": left, "right": right, "metric": metric,
                              "difference": metrics[left][metric] - metrics[right][metric],
                              "ci95_low": low, "ci95_high": high,
                              "left_only_correct": int(np.sum(lcorrect & ~rcorrect)),
                              "right_only_correct": int(np.sum(~lcorrect & rcorrect)),
                              "prediction_disagreements": int(np.sum(predictions[left] != predictions[right])),
                              "both_wrong_changed": int(np.sum(~lcorrect & ~rcorrect & (predictions[left] != predictions[right])))})
            for i in np.flatnonzero(predictions[left] != predictions[right]):
                cases.append({"left": left, "right": right, "apk_path": paths[i], "test_fold": int(folds[i]),
                              "true_class": CLASSES[y[i]], "left_prediction": CLASSES[predictions[left][i]],
                              "right_prediction": CLASSES[predictions[right][i]],
                              "case_type": "left_only_correct" if lcorrect[i] else "right_only_correct" if rcorrect[i] else "both_wrong",
                              "manual_explanation": ""})
        oof_rows = []
        for i, path in enumerate(paths):
            row = {"row_index": i, "apk_path": path, "test_fold": int(folds[i]), "true_class": CLASSES[y[i]]}
            for key in VARIANTS:
                row[key + "_prediction"] = CLASSES[predictions[key][i]]
                for c in range(5):
                    row[f"{key}_probability_C{c+1}"] = float(probabilities[key][i, c])
            oof_rows.append(row)
        write_csv(output / "oof_predictions.csv", list(oof_rows[0]), oof_rows)
        write_csv(output / "summary_metrics.csv", list(summary[0]), summary)
        write_csv(output / "per_class_metrics.csv", list(per_class[0]), per_class)
        write_csv(output / "fold_metrics.csv", list(fold_rows[0]), fold_rows)
        write_csv(output / "paired_comparisons.csv", list(pairs[0]), pairs)
        write_csv(output / "paired_case_review.csv", ["left", "right", "apk_path", "test_fold", "true_class",
                  "left_prediction", "right_prediction", "case_type", "manual_explanation"], cases)
        write_json(output / "confusion_matrices.json", matrices)
        write_json(output / "metrics.json", metrics)
        np.savez_compressed(output / "bootstrap_metrics.npz", **boot)
        if not args.no_plots:
            plot_results(output, y, predictions, metrics, boot)
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["original_inputs_unchanged"] = all(sha256(Path(value["path"])) == value["sha256"] for value in input_records.values())
        render_report(output, manifest, metrics, per_class, pairs)
        manifest["output_sha256"] = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*")
                                     if path.is_file() and path.name != "experiment_manifest.json"}
        write_json(output / "experiment_manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "experiment_manifest.json", manifest)
        raise
    print(f"Completed. Report: {output / 'report.md'}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "output" / "features.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "output" / "labels.csv")
    parser.add_argument("--reference-predictions", type=Path, default=ROOT / "result" / "predictions.csv")
    parser.add_argument("--output-dir", type=Path, help="NEW directory; existing directories are always refused")
    parser.add_argument("--signals", type=Path, default=ROOT / "output" / "apks.jsonl",
                        help="Matching raw signal JSONL; snapshotted and revectorized for validation only")
    parser.add_argument("--allow-reference-drift", action="store_true", help="Explicitly continue despite changed historical OOF predictions")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-features", choices=["6"], default="6",
                        help="Fixed at 6 for all eight configurations")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.max_features = 6
    if args.bootstrap < 100 or args.n_jobs == 0:
        parser.error("bootstrap must be >=100; n-jobs cannot be zero")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
