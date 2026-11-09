"""Multi-model single-label multi-class classifier with LOOCV + K-fold.

Models: XGBoost / Logistic Regression / Random Forest / ExtraTrees / Linear SVM
Classes (C1-C5): Direct third-party LLM service access / Server-mediated LLM
access / Web-based LLM interaction / On-device LLM inference / Non-target
candidate applications.

C3 covers the main web-based LLM interaction, including WebView, Custom Tabs
and TWA. C5 means outside this study's LLM scope, not absence of all AI.
The legacy label-column identifiers and numeric class order remain unchanged.
Prediction normalizes display names from compatible saved models in memory;
it does not modify model files, probabilities or numeric predictions.

Usage::

    python trainer.py
    python trainer.py --features features.csv --labels labels.csv --cv 5
    python trainer.py --predict model_xgb.pkl --features new_features.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from feature_contract import read_contract, check_model_contract

warnings.filterwarnings("ignore")

LABEL_COLS = ["is_direct_client", "is_server_proxy", "is_webview", "is_local", "is_non_ai"]
CLASS_CN = [
    "第三方 LLM 服务直连",
    "服务端介导的 LLM 访问",
    "基于网页的 LLM 交互",
    "端侧 LLM 推理",
    "非目标候选应用",
]
# Historical display names are retained only to recognize older model metadata.
_LEGACY_CLASS_CN = ("客户端直连", "服务端中转", "WebView套壳", "本地部署", "非智能应用")
N_CLASSES = len(LABEL_COLS)  # 5

# n ≤ _LOOCV_MAX_N → LOOCV; otherwise → K-fold (auto or user-specified)
_LOOCV_MAX_N = 30

MODEL_META = [
    {"key": "xgb", "name": "XGBoost",      "long_name": "XGBoost",
     "make_clf": None},
    {"key": "lr",  "name": "LR",           "long_name": "Logistic Regression",
     "make_clf": None},
    {"key": "rf",  "name": "RF",           "long_name": "Random Forest",
     "make_clf": None},
    {"key": "extratrees", "name": "ExtraTrees", "long_name": "ExtraTrees",
     "make_clf": None},
    {"key": "linear_svm", "name": "Linear SVM", "long_name": "Linear SVM",
     "make_clf": None},
]


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_data(
    features_path: str, labels_path: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    label_map: dict[str, dict[str, int]] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            apk = row.get("apk_path", "").strip()
            if not apk:
                continue
            lbls = {col: int(row.get(col, "0")) for col in LABEL_COLS
                    if row.get(col, "").strip() in ("0", "1")}
            if lbls:
                label_map[apk] = lbls

    feature_names: list[str] = []
    apk_names: list[str] = []
    X_rows: list[list[float]] = []
    y_rows: list[list[int]] = []

    with open(features_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        feature_names = [k for k in (reader.fieldnames or []) if k != "apk_path"]
        for row in reader:
            apk = row.get("apk_path", "").strip()
            if not apk or apk not in label_map:
                continue
            apk_names.append(apk)
            X_rows.append([float(row.get(fn, 0) or 0) for fn in feature_names])
            y_rows.append([label_map[apk].get(col, 0) for col in LABEL_COLS])

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), apk_names, feature_names


# ═══════════════════════════════════════════════════════════════════
# Multi-label → single-label conversion
# ═══════════════════════════════════════════════════════════════════

def _to_single_label(y_ml: np.ndarray) -> np.ndarray:
    """Convert n×5 one-hot labels to n single-label integers (0-4).

    With 5 explicit columns and exactly one ``1`` per row, this is
    simply ``argmax``.  Warns if any row has !=1 positive labels.
    """
    row_sums = y_ml.sum(axis=1)
    bad = np.where(row_sums != 1)[0]
    if len(bad):
        print(
            f"  WARNING: {len(bad)} sample(s) have !=1 positive labels. "
            f"Affected rows: {bad.tolist()[:10]}"
            f"{'…' if len(bad) > 10 else ''}",
            file=sys.stderr,
        )
    return np.argmax(y_ml, axis=1).astype(np.int32)


# ═══════════════════════════════════════════════════════════════════
# Sample weight — only for XGBoost (other models use class_weight where needed)
# ═══════════════════════════════════════════════════════════════════

def _compute_sample_weight(y_single: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights for multi-class imbalance.

    XGBoost's ``scale_pos_weight`` only supports binary classification,
    so we pass per-sample weights explicitly.
    """
    counts = np.bincount(y_single, minlength=N_CLASSES).astype(np.float64)
    counts[counts == 0] = 1
    class_weights = y_single.shape[0] / (N_CLASSES * counts)
    return class_weights[y_single].astype(np.float32)


def _fit_model(clf, X, y_single, model_key: str) -> None:
    """Fit a classifier, passing ``sample_weight`` only to XGBoost.

    The other models use their own class balancing or calibration pipeline —
    applying additional sample weights would change their intended weighting.
    """
    if model_key == "xgb":
        clf.fit(X, y_single, sample_weight=_compute_sample_weight(y_single))
    else:
        clf.fit(X, y_single)


# ═══════════════════════════════════════════════════════════════════
# Model factories
# ═══════════════════════════════════════════════════════════════════

def _make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        objective="multi:softmax", num_class=N_CLASSES,
        n_estimators=250, max_depth=4, min_child_weight=2,
        subsample=0.85, colsample_bytree=0.85,
        learning_rate=0.05, eval_metric="mlogloss",
        reg_lambda=2.0,
        random_state=42, verbosity=0,
    )


def _make_lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        class_weight="balanced",
        max_iter=800, multi_class="multinomial",
        random_state=42,
    )


def _make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=2,
        class_weight="balanced",
        max_features="sqrt", random_state=42, n_jobs=-1,
    )


def _make_extratrees():
    from sklearn.ensemble import ExtraTreesClassifier
    # Researcher adopted leaf=1, max_features=6 on 2026-11-06 after the
    # fixed-60d comparison. Historical ablations retain their frozen leaf=2.
    return ExtraTreesClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=1,
        class_weight="balanced", max_features=6,
        random_state=42, n_jobs=-1,
    )


def _make_linear_svm():
    """Return a calibrated, standardized linear SVM.

    Calibration preserves the ``predict_proba`` interface required by the
    OOF metrics, CSV export, and prediction CLI.  A linear margin is a stable
    complement to LR and tree ensembles for this small, mixed-feature dataset.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    base = LinearSVC(
        C=0.03, class_weight="balanced", dual=False,
        max_iter=5000, random_state=42,
    )
    try:
        calibrated = CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv=3,
        )
    except TypeError:  # scikit-learn < 1.2 compatibility
        calibrated = CalibratedClassifierCV(
            base_estimator=base, method="sigmoid", cv=3,
        )
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", calibrated),
    ])


for m in MODEL_META:
    if m["key"] == "xgb":   m["make_clf"] = _make_xgb
    elif m["key"] == "lr":  m["make_clf"] = _make_lr
    elif m["key"] == "rf":  m["make_clf"] = _make_rf
    elif m["key"] == "extratrees": m["make_clf"] = _make_extratrees
    elif m["key"] == "linear_svm": m["make_clf"] = _make_linear_svm


# ═══════════════════════════════════════════════════════════════════
# CV runners
# ═══════════════════════════════════════════════════════════════════

def _run_loocv_single(
    X: np.ndarray, y_single: np.ndarray, model_key: str, make_clf,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    probas = np.zeros((n, N_CLASSES), dtype=np.float32)
    preds  = np.zeros(n, dtype=np.int32)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        clf = make_clf()
        _fit_model(clf, X[mask], y_single[mask], model_key)
        probas[i] = clf.predict_proba(X[i:i+1])[0]
        preds[i]  = int(clf.predict(X[i:i+1])[0])
    return probas, preds


def _run_kfold_single(
    X: np.ndarray, y_single: np.ndarray, model_key: str, make_clf, k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedKFold
    n = X.shape[0]
    probas = np.zeros((n, N_CLASSES), dtype=np.float32)
    preds  = np.zeros(n, dtype=np.int32)

    min_pos = min(np.bincount(y_single, minlength=N_CLASSES))
    if min_pos < k:
        print(
            f"  WARNING: smallest class has only {min_pos} samples "
            f"(need ≥{k} for {k}-fold CV), falling back to LOOCV.",
            file=sys.stderr,
        )
        return _run_loocv_single(X, y_single, model_key, make_clf)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y_single):
        clf = make_clf()
        _fit_model(clf, X[train_idx], y_single[train_idx], model_key)
        probas[test_idx] = clf.predict_proba(X[test_idx])
        preds[test_idx]  = clf.predict(X[test_idx])
    return probas, preds


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Return per-class precision / recall / F1."""
    result = []
    for ci in range(N_CLASSES):
        tp = int(((y_pred == ci) & (y_true == ci)).sum())
        fp = int(((y_pred == ci) & (y_true != ci)).sum())
        fn = int(((y_pred != ci) & (y_true == ci)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        result.append({"prec": prec, "rec": rec, "f1": f1,
                       "tp": tp, "fp": fp, "fn": fn,
                       "support": int((y_true == ci).sum())})
    return result


def _global_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probas: np.ndarray,
) -> dict[str, float]:
    """Compute paper-facing metrics from out-of-fold predictions."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_onehot = np.eye(N_CLASSES, dtype=np.int32)[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_pr_auc": float(average_precision_score(
            y_onehot, probas, average="macro")),
        "macro_roc_auc": float(roc_auc_score(
            y_onehot, probas, average="macro", multi_class="ovr")),
        "log_loss": float(log_loss(y_true, probas, labels=range(N_CLASSES))),
        "brier": float(np.mean(np.sum((probas - y_onehot) ** 2, axis=1))),
    }


def _bootstrap_metric_cis(
    y_true: np.ndarray, y_pred: np.ndarray,
    n_resamples: int = 2000, seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap 95% CIs while preserving class support."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
    )

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == ci) for ci in range(N_CLASSES)]
    values: dict[str, list[float]] = {
        "accuracy": [], "balanced_accuracy": [], "macro_f1": [], "mcc": [],
    }

    for _ in range(n_resamples):
        sampled = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True)
            for idx in class_indices if len(idx)
        ])
        yt = y_true[sampled]
        yp = y_pred[sampled]
        values["accuracy"].append(float(accuracy_score(yt, yp)))
        values["balanced_accuracy"].append(
            float(balanced_accuracy_score(yt, yp)))
        values["macro_f1"].append(float(f1_score(
            yt, yp, average="macro", zero_division=0)))
        values["mcc"].append(float(matthews_corrcoef(yt, yp)))

    return {
        key: (float(np.percentile(vals, 2.5)),
              float(np.percentile(vals, 97.5)))
        for key, vals in values.items()
    }


def _feature_importance_values(
    clf, X: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return model-appropriate, descriptive feature-effect magnitudes."""
    estimator = clf
    if hasattr(clf, "named_steps"):
        estimator = list(clf.named_steps.values())[-1]

    if hasattr(estimator, "feature_importances_"):
        return (np.asarray(estimator.feature_importances_, dtype=float),
                "Tree feature importance")
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        feature_scale = np.std(X, axis=0)
        values = np.mean(np.abs(coefficients), axis=0) * feature_scale
        return values, "Mean |coefficient| × feature SD"

    # CalibratedClassifierCV stores one fitted base estimator per calibration
    # fold.  Average their linear coefficients so the SVM remains explainable
    # in the same feature space as the other models.
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated:
        coefficients = []
        for calibrated_model in calibrated:
            base = getattr(calibrated_model, "estimator", None)
            if base is None:
                base = getattr(calibrated_model, "base_estimator", None)
            if base is not None and hasattr(base, "coef_"):
                coefficients.append(np.asarray(base.coef_, dtype=float))
        if coefficients:
            values = np.mean(np.abs(np.concatenate(coefficients, axis=0)), axis=0)
            return values, "Mean |standardized SVM coefficient|"
    raise TypeError(f"Unsupported classifier for feature importance: {type(clf)!r}")


# ═══════════════════════════════════════════════════════════════════
# Model persistence
# ═══════════════════════════════════════════════════════════════════

def save_models(all_results: dict, output_dir: Path,
                feature_names: list[str], feature_contract: dict | None = None) -> dict[str, Path]:
    """Persist final (full-data) models to disk.

    Returns ``{model_key: saved_path}``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for key, data in all_results.items():
        clf = data.get("final_clf")
        if clf is None:
            continue
        path = output_dir / f"model_{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": clf,
                "feature_names": feature_names,
                "class_names": CLASS_CN,
                "label_cols": LABEL_COLS,
                "feature_contract": feature_contract,
            }, f)
        saved[key] = path
    return saved


def load_model(model_path: str | Path) -> dict:
    """Load a saved model bundle (dict with model, feature_names, …)."""
    with open(Path(model_path), "rb") as f:
        return pickle.load(f)


def _prediction_class_names(bundle: dict) -> list[str]:
    """Return current display names for the unchanged C1-C5 label schema.

    Stable label columns identify this task even if a saved model uses old
    display text. Without that metadata, recognize only the exact legacy name
    sequence. Preserve other explicit mappings rather than relabeling them.
    This function never changes the bundle, class indices or probability order.
    """
    stored_names = bundle.get("class_names")
    if stored_names is None:
        return list(CLASS_CN)
    label_cols = bundle.get("label_cols")
    if label_cols is not None:
        if list(label_cols) == LABEL_COLS:
            return list(CLASS_CN)
        return list(stored_names)
    if tuple(stored_names) == _LEGACY_CLASS_CN:
        return list(CLASS_CN)
    return list(stored_names)


def predict(model_path: str | Path, X: np.ndarray) -> dict:
    """Run inference with a saved model.

    Returns
    -------
    dict with keys: pred_class (0-4, corresponding to C1-C5), pred_label,
    probas, class_names. Compatible saved models use the current display names.
    """
    bundle = load_model(model_path)
    clf = bundle["model"]
    probas = clf.predict_proba(X)
    preds = clf.predict(X).astype(int)
    class_names = _prediction_class_names(bundle)
    return {
        "pred_class": preds,
        "pred_label": [class_names[p] for p in preds],
        "probas": probas,
        "class_names": class_names,
    }


# ═══════════════════════════════════════════════════════════════════
# Main evaluation + report
# ═══════════════════════════════════════════════════════════════════

def evaluate_and_report(
    X: np.ndarray, y_ml: np.ndarray,
    apk_names: list[str], feature_names: list[str],
    output_dir: Path, cv: int = 0,
    save_model: bool = True,
    feature_contract: dict | None = None,
) -> None:

    _md: list[str] = []

    def _p(*args, **kwargs) -> None:
        buf = io.StringIO()
        print(*args, **kwargs, file=buf)
        _md.append(buf.getvalue())

    y_single = _to_single_label(y_ml)
    n = X.shape[0]
    method = f"{cv}-fold CV" if cv > 0 else "LOOCV"

    _p(f"\n# 单标签多分类评估报告\n")
    _p(f"**样本数**: {n} | **特征数**: {X.shape[1]} | **类别数**: {N_CLASSES} | **CV**: {method}\n")

    # ── class distribution ────────────────────────────────────────
    _p(f"\n## 类别分布\n")
    _p(f"| 类别 | 样本数 | 占比 |")
    _p(f"|------|------|------|")
    for ci, cn in enumerate(CLASS_CN):
        cnt = int((y_single == ci).sum())
        _p(f"| {cn} | {cnt} | {cnt / n * 100:.1f}% |")

    # ══════════════════════════════════════════════════════════
    # Train all models via CV
    # ══════════════════════════════════════════════════════════
    _p(f"\n---\n\n## 训练\n")

    all_results: dict[str, dict] = {}

    for mm in MODEL_META:
        _p(f"\n### {mm['long_name']}\n")
        key = mm["key"]
        if cv > 0:
            probas, preds = _run_kfold_single(X, y_single, key, mm["make_clf"], cv)
        else:
            probas, preds = _run_loocv_single(X, y_single, key, mm["make_clf"])

        # Train final model on full data (for saving + feature importance)
        final_clf = mm["make_clf"]()
        _fit_model(final_clf, X, y_single, key)

        # Pre-compute per-class metrics once, cache for reuse
        metrics = _per_class_metrics(y_single, preds)
        global_metrics = _global_metrics(y_single, preds, probas)
        metric_cis = _bootstrap_metric_cis(y_single, preds)

        all_results[key] = {
            "probas": probas, "preds": preds,
            "final_clf": final_clf, "metrics": metrics,
            "global_metrics": global_metrics, "metric_cis": metric_cis,
        }

    # ══════════════════════════════════════════════════════════
    # Per-Model Classification Reports (sklearn)
    # ══════════════════════════════════════════════════════════
    from sklearn.metrics import classification_report

    _p(f"\n---\n\n## Per-Model 分类报告\n")

    for mm in MODEL_META:
        key = mm["key"]
        preds = all_results[key]["preds"]
        _p(f"\n### {mm['long_name']}\n")
        cr = classification_report(
            y_single, preds, target_names=CLASS_CN,
            output_dict=True, zero_division=0,
        )
        # Per-class table
        _p(f"| 类别 | Precision | Recall | F1 | Support |")
        _p(f"|------|:--:|:--:|:--:|:--:|")
        for ci, cn in enumerate(CLASS_CN):
            m = cr.get(cn, {})
            _p(f"| {cn} | {m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | {m.get('f1-score', 0):.4f} | {int(m.get('support', 0))} |")
        # Averages
        ma = cr.get("macro avg", {})
        wa = cr.get("weighted avg", {})
        acc = cr.get("accuracy", 0)
        _p(f"| **Macro Avg** | {ma.get('precision', 0):.4f} | {ma.get('recall', 0):.4f} | {ma.get('f1-score', 0):.4f} | {int(ma.get('support', 0))} |")
        _p(f"| **Weighted Avg** | {wa.get('precision', 0):.4f} | {wa.get('recall', 0):.4f} | {wa.get('f1-score', 0):.4f} | {int(wa.get('support', 0))} |")
        _p(f"\nAccuracy: **{acc:.4f}**")

    # ══════════════════════════════════════════════════════════
    # Overall comparison
    # ══════════════════════════════════════════════════════════
    _p(f"\n---\n\n## 全局指标对比\n")
    _p("以下 95% CI 为保持各类别样本量不变的分层 bootstrap 区间（2000 次重采样）。\n")
    _p("| 模型 | Accuracy (95% CI) | Balanced Accuracy (95% CI) | Macro F1 (95% CI) | MCC (95% CI) | Weighted F1 |")
    _p("|------|:--:|:--:|:--:|:--:|:--:|")

    for mm in MODEL_META:
        key = mm["key"]
        gm = all_results[key]["global_metrics"]
        ci = all_results[key]["metric_cis"]

        def _with_ci(metric: str) -> str:
            low, high = ci[metric]
            return f"{gm[metric]:.4f} ({low:.4f}-{high:.4f})"

        _p(
            f"| {mm['long_name']} | {_with_ci('accuracy')} | "
            f"{_with_ci('balanced_accuracy')} | {_with_ci('macro_f1')} | "
            f"{_with_ci('mcc')} | {gm['weighted_f1']:.4f} |"
        )

    _p("\n### 概率质量指标\n")
    _p("所有概率指标均基于交叉验证的 out-of-fold 概率计算。类别不均衡时，Macro PR-AUC 比 ROC-AUC 更具判别力。\n")
    _p("| 模型 | Macro PR-AUC | Macro ROC-AUC (OvR) | Log Loss | Multiclass Brier |")
    _p("|------|:--:|:--:|:--:|:--:|")
    for mm in MODEL_META:
        gm = all_results[mm["key"]]["global_metrics"]
        _p(
            f"| {mm['long_name']} | {gm['macro_pr_auc']:.4f} | "
            f"{gm['macro_roc_auc']:.4f} | {gm['log_loss']:.4f} | "
            f"{gm['brier']:.4f} |"
        )

    # ══════════════════════════════════════════════════════════
    # Model ranking
    # ══════════════════════════════════════════════════════════
    _p("\n---\n\n## 模型排名\n")
    ranked_models = sorted(
        MODEL_META,
        key=lambda mm: all_results[mm["key"]]["global_metrics"]["macro_f1"],
        reverse=True,
    )
    _p("模型不预设主次，按 out-of-fold Macro F1 从高到低排序。\n")
    _p("| Rank | 模型 | Macro F1 | Balanced Accuracy | MCC |")
    _p("|:--:|------|:--:|:--:|:--:|")
    for rank, mm in enumerate(ranked_models, start=1):
        gm = all_results[mm["key"]]["global_metrics"]
        _p(f"| {rank} | {mm['long_name']} | {gm['macro_f1']:.4f} | "
           f"{gm['balanced_accuracy']:.4f} | {gm['mcc']:.4f} |")

    # ══════════════════════════════════════════════════════════
    # Confusion Matrices — all registered models
    # ══════════════════════════════════════════════════════════
    _p(f"\n---\n\n## 混淆矩阵 (5×5)\n")

    for mm in MODEL_META:
        key = mm["key"]
        preds = all_results[key]["preds"]
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
        for i in range(n):
            cm[y_single[i], preds[i]] += 1

        _p(f"\n### {mm['name']}\n")
        header_cells = " | ".join([f"Pred {cn}" for cn in CLASS_CN])
        _p(f"| True \\ Pred | {header_cells} |")
        _p("|" + "|".join(["---"] * (N_CLASSES + 1)) + "|")
        for ci, cn in enumerate(CLASS_CN):
            row_cells = " | ".join(str(cm[ci, cj]) for cj in range(N_CLASSES))
            _p(f"| True {cn} | {row_cells} |")

    # ══════════════════════════════════════════════════════════
    # Model-specific feature importance
    # ══════════════════════════════════════════════════════════
    _p("\n---\n\n## 特征重要性\n")
    _p("树模型使用模型内置特征重要性；逻辑回归使用各类别平均绝对系数乘以特征标准差；Linear SVM 使用校准折叠中线性系数的平均绝对值。不同方法的数值尺度不可直接横向比较。\n")
    for mm in MODEL_META:
        importance, importance_label = _feature_importance_values(
            all_results[mm["key"]]["final_clf"], X)
        top = np.argsort(importance)[::-1][:15]
        _p(f"\n### {mm['long_name']}\n")
        _p(f"度量：{importance_label}\n")
        _p("| Rank | Feature | Importance |")
        _p("|------|---------|------------|")
        for rank, idx in enumerate(top, start=1):
            _p(f"| {rank} | {feature_names[idx]} | {importance[idx]:.4f} |")

    # ── save predictions for all models ──────────────────────
    pred_path = output_dir / "predictions.csv"
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["apk_path", "true_class", "true_label"]
        for mm in MODEL_META:
            key = mm["key"]
            header.extend([f"pred_class_{key}", f"pred_label_{key}"])
            header.extend(
                f"proba_{key}_{ci}_{CLASS_CN[ci]}" for ci in range(N_CLASSES))
        writer.writerow(header)
        for i in range(n):
            row = [apk_names[i], str(y_single[i]), CLASS_CN[y_single[i]]]
            for mm in MODEL_META:
                result = all_results[mm["key"]]
                pred = int(result["preds"][i])
                row.extend([str(pred), CLASS_CN[pred]])
                row.extend(f"{result['probas'][i, ci]:.6f}"
                           for ci in range(N_CLASSES))
            writer.writerow(row)
    _p(f"\n**Predictions CSV**: `{pred_path}`\n")

    # ── error analysis for all models ─────────────────────────
    _p("\n---\n\n## 错误分析\n")
    for mm in MODEL_META:
        probas = all_results[mm["key"]]["probas"]
        preds = all_results[mm["key"]]["preds"]
        _p(f"\n### {mm['long_name']}\n")
        for ci, cn in enumerate(CLASS_CN):
            fp_i = np.where((preds == ci) & (y_single != ci))[0]
            fn_i = np.where((preds != ci) & (y_single == ci))[0]
            if len(fp_i) == 0 and len(fn_i) == 0:
                _p(f"\n**{cn}**: 无错误。")
                continue
            _p(f"\n**{cn}** — FP: {len(fp_i)}, FN: {len(fn_i)}")
            if len(fp_i):
                show = (fp_i if len(fp_i) <= 5 else
                        sorted(fp_i, key=lambda idx: -probas[idx, ci])[:5])
                _p("- False Positives (top confidence):")
                for idx in show:
                    _p(f"  - {Path(apk_names[idx]).name[:55]} "
                       f"(true={CLASS_CN[y_single[idx]]}, proba={probas[idx, ci]:.3f})")
            if len(fn_i):
                show = (fn_i if len(fn_i) <= 4 else
                        sorted(fn_i, key=lambda idx: probas[idx, ci])[:4])
                _p("- False Negatives (lowest confidence):")
                for idx in show:
                    _p(f"  - {Path(apk_names[idx]).name[:55]} "
                       f"(true={cn}, proba={probas[idx, ci]:.3f})")

    # ══════════════════════════════════════════════════════════
    # Model saving
    # ══════════════════════════════════════════════════════════
    if save_model:
        saved = save_models(all_results, output_dir, feature_names, feature_contract)
        _p(f"\n---\n\n## 模型保存\n")
        for key, path in saved.items():
            _p(f"- `{key}` → `{path}`")
        _p("\n所有模型使用相同的模型包格式，可分别通过 `load_model()` 加载。")

    # ── visualization ──────────────────────────────────────────
    if n >= 10:
        _plot_results(X, y_single, all_results, output_dir, feature_names)

    # ── write markdown ─────────────────────────────────────────
    md_path = output_dir / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(_md)
    print(f"\n  Report saved to: {md_path}")


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def _plot_results(
    X: np.ndarray, y_single: np.ndarray, all_results: dict[str, dict],
    output_dir: Path, feature_names: list[str],
) -> None:
    """Generate the compact, paper-facing figure set.

    The report contains the complete numeric tables.  Figures are limited to
    dataset balance, normalized confusion matrices, model comparison, macro
    ROC/PR curves, and the primary model's feature importance.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    model_count = len(MODEL_META)
    class_colors = ["#2c7bb6", "#74add1", "#fdae61", "#d7191c", "#7b6888"]
    model_colors = ["#2c7bb6", "#fdae61", "#5e9e6b", "#8c6bb1", "#d95f02"]
    cm_colors = ["Blues", "Oranges", "Greens", "Purples", "Reds"]

    # Remove only plots generated by older versions of this trainer.  Data,
    # model files, CSVs, and user-provided files are intentionally untouched.
    legacy_plot_names = [
        "cm_5x5.png", "per_class_f1.png",
        "xgb_per_class_metrics.png", "lr_per_class_metrics.png",
        "rf_per_class_metrics.png", "extratrees_per_class_metrics.png",
        "linear_svm_per_class_metrics.png",
        "pr_curves_xgb.png", "pr_curves_lr.png", "pr_curves_rf.png",
        "pr_curves_extratrees.png", "pr_curves_linear_svm.png",
        "roc_curves_xgb.png", "roc_curves_lr.png", "roc_curves_rf.png",
        "roc_curves_extratrees.png", "roc_curves_linear_svm.png",
        "feature_importance_xgb.png", "feature_importance_lr.png",
        "feature_importance_rf.png", "feature_importance_extratrees.png",
        "feature_importance_linear_svm.png",
    ]
    for filename in legacy_plot_names:
        old_plot = output_dir / filename
        if old_plot.exists():
            old_plot.unlink()

    # 1. Dataset class distribution.
    class_counts = np.bincount(y_single, minlength=N_CLASSES)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASS_CN, class_counts, color=class_colors)
    for bar, value in zip(bars, class_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(class_counts.max() * 0.015, 1),
            str(int(value)), ha="center", fontsize=10,
        )
    ax.set_ylabel("Number of APKs")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    # 2. Row-normalized confusion matrices.  Row normalization is preferable
    # for this imbalanced dataset because it makes minority-class recall clear.
    ncols = min(3, model_count)
    nrows = int(np.ceil(model_count / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 5.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for mi, mm in enumerate(MODEL_META):
        preds = all_results[mm["key"]]["preds"]
        cm = confusion_matrix(
            y_single, preds, labels=range(N_CLASSES), normalize="true",
        )
        ax = axes_flat[mi]
        im = ax.imshow(cm, cmap=cm_colors[mi], vmin=0, vmax=1)
        for i2 in range(N_CLASSES):
            for j2 in range(N_CLASSES):
                ax.text(
                    j2, i2, f"{cm[i2, j2]:.0%}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i2, j2] > 0.5 else "black",
                )
        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels([cn[:4] for cn in CLASS_CN], fontsize=8, rotation=30)
        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels([f"True {cn[:4]}" for cn in CLASS_CN], fontsize=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(mm["long_name"], fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[model_count:]:
        ax.axis("off")
    fig.suptitle("Row-Normalized Confusion Matrices", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "cm_normalized.png", dpi=180)
    plt.close(fig)

    # 3. Per-class F1 comparison across all models.
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(N_CLASSES)
    width = 0.78 / model_count
    for mi, mm in enumerate(MODEL_META):
        values = [m["f1"] for m in all_results[mm["key"]]["metrics"]]
        offset = (mi - (model_count - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width, label=mm["long_name"],
            color=model_colors[mi], edgecolor="white", linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.2f}", ha="center", fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_CN, fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_f1.png", dpi=180)
    plt.close(fig)

    # 4. Overall metrics with the bootstrap intervals from the report.
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "mcc"]
    metric_names = ["Accuracy", "Balanced Acc.", "Macro F1", "MCC"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    width = 0.78 / model_count
    for mi, mm in enumerate(MODEL_META):
        result = all_results[mm["key"]]
        values = np.array([
            result["global_metrics"][key] for key in metric_keys
        ])
        intervals = [result["metric_cis"][key] for key in metric_keys]
        yerr = np.array([
            [value - low for value, (low, _) in zip(values, intervals)],
            [high - value for value, (_, high) in zip(values, intervals)],
        ])
        offset = (mi - (model_count - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width, label=mm["long_name"],
            color=model_colors[mi], yerr=yerr, capsize=2,
            edgecolor="white", linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.04),
                f"{value:.2f}", ha="center", fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison with 95% Bootstrap CIs")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "global_metrics.png", dpi=180)
    plt.close(fig)

    y_onehot = np.eye(N_CLASSES, dtype=np.int32)[y_single]

    # 5. Macro ROC curves: one line per model, so the figure remains readable
    # after adding ExtraTrees and Linear SVM.
    roc_grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(8, 6))
    for mi, mm in enumerate(MODEL_META):
        probas = all_results[mm["key"]]["probas"]
        class_tprs = []
        for ci in range(N_CLASSES):
            fpr, tpr, _ = roc_curve(y_onehot[:, ci], probas[:, ci])
            order = np.argsort(fpr)
            class_tprs.append(np.interp(roc_grid, fpr[order], tpr[order]))
        macro_tpr = np.mean(class_tprs, axis=0)
        macro_auc = roc_auc_score(
            y_onehot, probas, average="macro", multi_class="ovr",
        )
        ax.plot(
            roc_grid, macro_tpr, color=model_colors[mi], linewidth=2,
            label=f"{mm['long_name']} (AUC={macro_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-Average ROC Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves_all_models.png", dpi=180)
    plt.close(fig)

    # 6. Macro PR curves, interpolated across the one-vs-rest class curves.
    recall_grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(8, 6))
    for mi, mm in enumerate(MODEL_META):
        probas = all_results[mm["key"]]["probas"]
        class_precisions = []
        for ci in range(N_CLASSES):
            precision, recall, _ = precision_recall_curve(
                y_onehot[:, ci], probas[:, ci],
            )
            order = np.argsort(recall)
            class_precisions.append(
                np.interp(recall_grid, recall[order], precision[order]),
            )
        macro_precision = np.mean(class_precisions, axis=0)
        macro_ap = average_precision_score(
            y_onehot, probas, average="macro",
        )
        ax.plot(
            recall_grid, macro_precision, color=model_colors[mi], linewidth=2,
            label=f"{mm['long_name']} (AP={macro_ap:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Macro-Average Precision-Recall Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curves_all_models.png", dpi=180)
    plt.close(fig)

    # 7. Feature importance for the best out-of-fold Macro-F1 model only.
    ranked = sorted(
        MODEL_META,
        key=lambda mm: all_results[mm["key"]]["global_metrics"]["macro_f1"],
        reverse=True,
    )
    primary = ranked[0]
    try:
        importance, importance_label = _feature_importance_values(
            all_results[primary["key"]]["final_clf"], X,
        )
        top = np.argsort(importance)[-15:]
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.barh(
            [feature_names[i] for i in top], importance[top],
            color=model_colors[MODEL_META.index(primary)],
        )
        ax.set_xlabel(importance_label)
        ax.set_title(f"{primary['long_name']} — Top 15 Feature Importance")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "feature_importance_primary.png", dpi=180)
        plt.close(fig)
    except (TypeError, ValueError):
        pass

    print(f"  Paper plots saved to: {output_dir}")

# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Single-label multi-class classifier")
    parser.add_argument("--features", default=r"D:\work\analysis\output\features.csv")
    parser.add_argument("--labels", default=r"D:\work\analysis\output\labels.csv")
    parser.add_argument("--output-dir", default=r"D:\work\analysis\result")
    parser.add_argument("--cv", type=int, default=0,
                        help="K-fold CV (0=auto: LOOCV if n≤%d, else 5-fold)" % _LOOCV_MAX_N)
    parser.add_argument("--no-save-model", action="store_true",
                        help="Skip saving trained models")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Explicitly reproduce historical features/models without provenance manifests")
    parser.add_argument("--predict", default=None, metavar="MODEL.pkl",
                        help="Prediction mode: load MODEL.pkl and predict on --features")
    parser.add_argument("--output", default=None, metavar="PREDICTIONS.csv",
                        help="Output CSV for --predict (default: model_dir/predictions.csv)")
    parser.add_argument("--threshold", type=float, default=None, metavar="T",
                        help="Minimum probability for C2 to be accepted (e.g. 0.6); "
                             "below this the 2nd-best class is used instead")
    args = parser.parse_args()

    # ── Prediction mode ────────────────────────────────────────────
    if args.predict is not None:
        return _predict_cli(args)

    # ── Training mode ──────────────────────────────────────────────
    fp = Path(args.features); lp = Path(args.labels); od = Path(args.output_dir)
    if not fp.exists(): print(f"Features not found: {fp}", file=sys.stderr); return 1
    if not lp.exists(): print(f"Labels not found: {lp}", file=sys.stderr); return 1
    feature_contract = read_contract(fp)
    if feature_contract is None and not args.allow_legacy:
        print("Features have no extraction contract; re-export current signals or use --allow-legacy for historical experiments.", file=sys.stderr)
        return 1
    od.mkdir(parents=True, exist_ok=True)

    X, y, apk_names, feature_names = load_data(str(fp), str(lp))
    if X.shape[0] == 0:
        print("No labelled samples found.", file=sys.stderr); return 1

    cv = args.cv
    if cv == 0:
        cv = 0 if X.shape[0] <= _LOOCV_MAX_N else 5

    evaluate_and_report(X, y, apk_names, feature_names, od, cv=cv,
                        save_model=not args.no_save_model, feature_contract=feature_contract)
    return 0


# ── Prediction CLI ──────────────────────────────────────────────────────

def _predict_cli(args) -> int:
    """Run prediction on a features CSV using a saved model."""
    model_path = Path(args.predict)
    fp = Path(args.features)

    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr); return 1
    if not fp.exists():
        print(f"Features not found: {fp}", file=sys.stderr); return 1

    # Load model bundle
    bundle = load_model(model_path)
    try:
        check_model_contract(bundle.get("feature_contract"), read_contract(fp), getattr(args, "allow_legacy", False))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    clf = bundle["model"]
    model_fnames: list[str] = bundle.get("feature_names", [])
    class_names = _prediction_class_names(bundle)

    # Read features CSV
    apk_names: list[str] = []
    X_rows: list[list[float]] = []
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_fnames = [k for k in (reader.fieldnames or []) if k != "apk_path"]
        for row in reader:
            apk = row.get("apk_path", "").strip()
            if not apk:
                continue
            apk_names.append(apk)
            X_rows.append([float(row.get(fn, 0) or 0) for fn in csv_fnames])

    if not X_rows:
        print("No features found.", file=sys.stderr); return 1

    X = np.array(X_rows, dtype=np.float32)

    # Align feature columns to model's training order if needed
    if model_fnames and csv_fnames != model_fnames:
        # Reorder / warn on mismatch
        if set(csv_fnames) != set(model_fnames):
            missing = set(model_fnames) - set(csv_fnames)
            extra = set(csv_fnames) - set(model_fnames)
            if missing:
                print(f"  WARNING: features missing from CSV: {sorted(missing)}",
                      file=sys.stderr)
            if extra:
                print(f"  WARNING: extra features in CSV: {sorted(extra)}",
                      file=sys.stderr)
        idx_map = [csv_fnames.index(fn) for fn in model_fnames if fn in csv_fnames]
        if len(idx_map) != len(model_fnames):
            print("  ERROR: feature set mismatch — cannot align.", file=sys.stderr)
            return 1
        X = X[:, idx_map]
        csv_fnames = [csv_fnames[i] for i in idx_map]
    else:
        model_fnames = csv_fnames

    # Predict
    probas = clf.predict_proba(X)
    preds = np.argmax(probas, axis=1).astype(int)

    # ── Optional C2 threshold (reduces C2 false positives) ────────
    # If C2 probability is below *threshold*, suppress it and pick
    # the best non-C2 class instead.  C2 is index 1.
    if args.threshold is not None:
        mask_c2_low = (preds == 1) & (probas[:, 1] < args.threshold)
        if mask_c2_low.any():
            suppressed = probas[mask_c2_low].copy()
            suppressed[:, 1] = -1.0  # exclude C2
            preds[mask_c2_low] = np.argmax(suppressed, axis=1).astype(int)
            n_changed = int(mask_c2_low.sum())
            print(f"  C2 threshold={args.threshold}: {n_changed} prediction(s) re-classified",
                  file=sys.stderr)

    # ── Output ─────────────────────────────────────────────────────
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (od / "predictions.csv")

    col_proba = [f"proba_{i}_{cn}" for i, cn in enumerate(class_names)]
    header = ["apk_path", "pred_class", "pred_label"] + col_proba

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, apk in enumerate(apk_names):
            n = apk.split("\\")[-1] if "\\" in apk else apk.split("/")[-1]
            w.writerow([
                apk,
                int(preds[i]),
                class_names[preds[i]],
            ] + [f"{p:.4f}" for p in probas[i]])

    # Summary
    counts = [int((preds == c).sum()) for c in range(N_CLASSES)]
    print(f"  Model: {model_path}")
    print(f"  Features: {fp}  ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"  Output: {out_path}")
    print(f"  Prediction distribution:")
    for c in range(N_CLASSES):
        print(f"    {class_names[c]}: {counts[c]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
