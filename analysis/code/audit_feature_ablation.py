"""Independently audit the saved 59d ablation study, including 40 fresh fits.

This verifier does not call the experiment's loading, fitting, metric, or
bootstrap helpers. It writes a separate JSON audit and never changes results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold

from feature_contract import validate_records
from feature_vectorizer_59d import FEATURE_SCHEMA_VERSION, FeatureVectorizer

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["is_direct_client", "is_server_proxy", "is_webview", "is_local", "is_non_ai"]
CLASSES = [f"C{i}" for i in range(1, 6)]
METRICS = ["macro_f1", "accuracy", "balanced_accuracy", "mcc"]
# Specify these independently rather than importing the experiment constants.
REMOVED = {
    "B_no_service": [4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 25],
    "C_no_protocol": [3, 20, 21, 22, 23, 24, 58, 59],
    "D_no_web": [10, 11, 38, 39, 40, 45],
    "E_no_device": [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 46, 53, 54, 55, 56, 57],
    "F_no_http_relations": [44, 47, 48],
    "G_no_density": [1, 2, 42, 43],
    "H_no_size_quality": [49, 50, 51, 52],
}
CONFIGS = {"A_full": list(range(1, 60)), **{
    key: [i for i in range(1, 60) if i not in removed] for key, removed in REMOVED.items()}}
PAIRS = [("A_full", key) for key in REMOVED]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path, *, allow_empty=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names, rows = reader.fieldnames, list(reader)
    if not names or len(names) != len(set(names)) or (not rows and not allow_empty):
        raise ValueError(f"Invalid table: {path}")
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Malformed row: {path}")
    return names, rows


def keyed(rows):
    result = {row["apk_path"]: row for row in rows}
    if len(result) != len(rows) or not all(result):
        raise ValueError("Blank or duplicate APK identities")
    return result


def metrics(y, p):
    return dict(zip(METRICS, [f1_score(y, p, labels=range(5), average="macro", zero_division=0),
                             accuracy_score(y, p), balanced_accuracy_score(y, p),
                             matthews_corrcoef(y, p)]))


def check_float(actual, expected, label):
    if not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"Numeric mismatch: {label}")


def bootstrap_from_confusions(y, predictions, repeats, seed):
    """Independent batch formulation; validate selected draws with sklearn."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in range(5)]
    indices = np.stack([np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                        for _ in range(repeats)])
    batches = np.repeat(np.arange(repeats), len(y))
    output = {}
    for key, p in predictions.items():
        cm = np.zeros((repeats, 5, 5), dtype=np.float64)
        np.add.at(cm, (batches, y[indices].ravel(), p[indices].ravel()), 1)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        actual, assigned = cm.sum(axis=2), cm.sum(axis=1)
        f1 = np.divide(2*tp, actual+assigned, out=np.zeros_like(tp), where=actual+assigned != 0)
        correct, n = tp.sum(axis=1), len(y)
        denom = np.sqrt((n*n-(assigned**2).sum(axis=1)) * (n*n-(actual**2).sum(axis=1)))
        mcc = np.divide(correct*n-(actual*assigned).sum(axis=1), denom,
                        out=np.zeros(repeats), where=denom != 0)
        output[key] = np.column_stack([f1.mean(axis=1), correct/n,
                                      (tp/actual).mean(axis=1), mcc])
        for b in np.linspace(0, repeats-1, 20, dtype=int):
            check_float(output[key][b], list(metrics(y[indices[b]], p[indices[b]]).values()),
                        f"sklearn bootstrap draw {key}/{b}")
    return output


def audit(run, destination, jobs):
    if not __debug__:
        raise RuntimeError("Do not run the audit with python -O; assertions are required.")
    if destination.exists():
        raise FileExistsError("Audit destination already exists; use a new path.")
    manifest = json.loads((run / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["study_version"] == "feature-block-ablation-59d-v2026.10.28.1"
    assert manifest["model_parameters"]["max_features"] == 6
    assert manifest["fit_budget"] == 40
    assert manifest["predefined_comparisons"] == [list(p) for p in PAIRS]
    assert sorted(i for ids in REMOVED.values() for i in ids) == list(range(1, 60))
    protected = [run / relative for relative in manifest["output_sha256"]]
    protected += [run / "experiment_manifest.json"]
    before = {str(p): digest(p) for p in protected}
    for relative, expected in manifest["output_sha256"].items():
        assert digest(run / relative) == expected, f"Artifact hash mismatch: {relative}"
    for source, expected in manifest["source_sha256"].items():
        name = source.replace("\\", "/").rsplit("/", 1)[-1]
        assert digest(run / "sources" / name) == expected, f"Source snapshot changed: {name}"
        assert digest(ROOT / "code" / name) == expected, f"Runtime source changed; restore pinned code: {name}"
    for item in manifest["inputs"].values():
        assert item["sha256"] == digest(run / item["snapshot"])
    names, feature_rows = table(run / "inputs" / "features.csv")
    names = names[1:]
    assert names == FeatureVectorizer().feature_names
    paths = list(keyed(feature_rows))
    _, label_rows = table(run / "inputs" / "labels.csv")
    labels = keyed(label_rows)
    assert set(paths) == set(labels)
    bits = np.array([[int(labels[p][col]) for col in LABELS] for p in paths])
    assert np.isin(bits, [0, 1]).all() and (bits.sum(axis=1) == 1).all()
    y = bits.argmax(axis=1)
    X64 = np.array([[float(row[name]) for name in names] for row in feature_rows])
    X = X64.astype(np.float32)
    assert X.shape == (manifest["sample_count"], 59) and np.isfinite(X).all()
    assert np.bincount(y, minlength=5).tolist() == manifest["class_counts"]

    with (run / "inputs" / "apks.jsonl").open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    raw = keyed(records)
    assert len(raw) == len(paths) and set(raw) == set(paths)
    hashes = [r["sha256"].lower() for r in records]
    assert len(set(hashes)) == len(hashes), "Repeated APK content SHA-256"
    assert all(len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in hashes)
    feature_manifest = json.loads((run / "inputs" / "features.csv.manifest.json").read_text(encoding="utf-8-sig"))
    source_versions = {r["metadata"]["feature_schema_version"] for r in records}
    target = feature_manifest["contract"]["feature_schema_version"]
    refresh = source_versions != {target}
    contract = validate_records(records, allow_feature_refresh=refresh,
                                feature_schema_version=FEATURE_SCHEMA_VERSION)
    if refresh or feature_manifest.get("derivation"):
        derivation = feature_manifest["derivation"]
        assert len(source_versions) == 1
        assert derivation["operation"] == "revectorize"
        assert derivation["source_sha256"] == digest(run / "inputs" / "apks.jsonl")
        assert derivation["source_record_count"] == len(records)
        assert derivation["source_feature_schema_version"] in source_versions
        assert derivation["target_feature_schema_version"] == target
    assert feature_manifest["feature_names"] == names
    assert contract == feature_manifest["contract"] == manifest["provenance"]["contract"]
    regenerated, regenerated_names = FeatureVectorizer().transform([raw[p] for p in paths])
    assert regenerated_names == names
    check_float(np.array(regenerated), X64, "raw signals to 59 features")
    assert np.array_equal(np.array(regenerated, dtype=np.float32), X)
    params = ExtraTreesClassifier(n_estimators=300, max_features=6, max_depth=8,
                                  min_samples_leaf=2, class_weight="balanced",
                                  random_state=42, n_jobs=jobs).get_params()
    for field, value in params.items():
        if field != "n_jobs":
            assert manifest["model_parameters"][field] == value

    _, saved_folds = table(run / "fold_assignments.csv")
    _, saved_oof = table(run / "oof_predictions.csv")
    assert [r["apk_path"] for r in saved_folds] == paths == [r["apk_path"] for r in saved_oof]
    folds = np.array([int(r["test_fold"]) for r in saved_folds])
    assert np.array_equal(folds, [int(r["test_fold"]) for r in saved_oof])
    assert [r["true_class"] for r in saved_folds] == [CLASSES[t] for t in y]
    assert [r["true_class"] for r in saved_oof] == [CLASSES[t] for t in y]
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    fold_sizes = []
    for f, (train, test) in enumerate(splits, 1):
        assert not set(train).intersection(test)
        assert np.array_equal(test, np.flatnonzero(folds == f))
        assert set(y[train]) == set(y[test]) == set(range(5))
        fold_sizes.append({"fold": f, "train_n": len(train), "test_n": len(test)})
    assert fold_sizes == manifest["cross_validation"]["fold_sizes"]
    _, summary_rows = table(run / "summary_metrics.csv")
    summary = {r["variant"]: r for r in summary_rows}
    saved_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    saved_cms = json.loads((run / "confusion_matrices.json").read_text(encoding="utf-8"))
    _, class_rows = table(run / "per_class_metrics.csv")
    _, fold_rows = table(run / "fold_metrics.csv")
    assert len(fold_rows) == 40 and len(class_rows) == 40
    assert set(summary) == set(CONFIGS) == set(saved_metrics) == set(saved_cms)
    predictions, summary_audit = {}, {}
    for key, feature_ids in CONFIGS.items():
        assert manifest["variants"][key]["feature_ids"] == feature_ids
        assert manifest["variants"][key]["removed_feature_ids"] == REMOVED.get(key, [])
        assert manifest["variants"][key]["feature_names"] == [names[i-1] for i in feature_ids]
        assert int(summary[key]["feature_count"]) == len(feature_ids)
        pred = np.array([CLASSES.index(r[key+"_prediction"]) for r in saved_oof])
        probabilities = np.array([[float(r[f"{key}_probability_C{c}"]) for c in range(1, 6)]
                                  for r in saved_oof])
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        check_float(probabilities.sum(axis=1), np.ones(len(y)), f"probability sums {key}")
        assert np.array_equal(probabilities.argmax(axis=1), pred)
        independent = metrics(y, pred)
        for metric, value in independent.items():
            check_float(value, float(summary[key][metric]), f"summary {key}/{metric}")
            check_float(value, saved_metrics[key][metric], f"metrics.json {key}/{metric}")
        errors = int((pred != y).sum())
        assert errors == int(summary[key]["errors"]) == saved_metrics[key]["errors"]
        cm = confusion_matrix(y, pred, labels=range(5))
        assert cm.tolist() == saved_cms[key]["counts"]
        check_float(cm/cm.sum(axis=1, keepdims=True), saved_cms[key]["row_normalized"], key)
        pcs = [r for r in class_rows if r["variant"] == key]
        assert [r["class"] for r in pcs] == CLASSES
        pr, re, f1, support = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
        for c, row in enumerate(pcs):
            check_float([pr[c], re[c], f1[c]], [float(row[x]) for x in ["precision", "recall", "f1"]], key)
            assert int(row["support"]) == int(support[c])
        max_prob_delta = 0.0
        seen = np.zeros(len(y), dtype=int)
        cols = np.array(feature_ids)-1
        for fold, (train, test) in enumerate(splits, 1):
            clf = ExtraTreesClassifier(n_estimators=300, max_features=6, max_depth=8,
                                       min_samples_leaf=2, class_weight="balanced",
                                       random_state=42, n_jobs=jobs)
            clf.fit(X[np.ix_(train, cols)], y[train])
            assert all(tree.max_features_ == 6 for tree in clf.estimators_)
            assert all(tree.tree_.n_node_samples[0] == len(train) for tree in clf.estimators_)
            assert np.array_equal(clf.predict(X[np.ix_(test, cols)]), pred[test]), f"Fresh-fit drift {key}/{fold}"
            fresh_proba = clf.predict_proba(X[np.ix_(test, cols)])
            check_float(fresh_proba, probabilities[test], f"fresh-fit probabilities {key}/{fold}")
            max_prob_delta = max(max_prob_delta, float(np.abs(fresh_proba-probabilities[test]).max()))
            seen[test] += 1
            row, = [r for r in fold_rows if r["variant"] == key and int(r["fold"]) == fold]
            assert int(row["effective_max_features"]) == 6
            assert int(row["train_n"]) == len(train) and int(row["test_n"]) == len(test)
            for metric, value in metrics(y[test], pred[test]).items():
                check_float(value, float(row[metric]), f"fold metric {key}/{fold}/{metric}")
        assert (seen == 1).all()
        predictions[key] = pred
        summary_audit[key] = {**independent, "errors": errors, "fresh_fit_prediction_mismatches": 0,
                              "fresh_fit_probability_max_absolute_difference": max_prob_delta}
        print(f"PASS {key}: five fresh fits; Macro-F1={independent['macro_f1']:.6f}; errors={errors}", flush=True)
    _, configuration_rows = table(run / "feature_configurations.csv")
    assert len(configuration_rows) == 59
    for i, row in enumerate(configuration_rows, 1):
        assert int(row["feature_id"]) == i and row["feature_name"] == names[i-1]
        assert all(int(row[key]) == int(i in ids) for key, ids in CONFIGS.items())
    _, references = table(run / "inputs" / "reference_predictions.csv")
    reference = keyed(references)
    assert set(reference) == set(paths)
    assert np.array_equal([int(reference[p]["true_class"]) for p in paths], y)
    ref_predictions = np.array([int(reference[p]["pred_class_extratrees"]) for p in paths])
    mismatch = int((ref_predictions != predictions["A_full"]).sum())
    assert mismatch == manifest["reference_check"]["mismatch_n"]
    assert not mismatch or manifest["arguments"]["allow_reference_drift"]

    boot = bootstrap_from_confusions(y, predictions, manifest["bootstrap"]["repeats"], manifest["bootstrap"]["seed"])
    with np.load(run / "bootstrap_metrics.npz") as saved_boot:
        assert set(saved_boot.files) == set(CONFIGS)
        for key in CONFIGS:
            check_float(saved_boot[key], boot[key], f"all bootstrap draws {key}")
            for j, metric in enumerate(METRICS):
                interval = np.quantile(boot[key][:, j], [0.025, 0.975])
                check_float(interval, saved_metrics[key][metric+"_ci95"], f"CI {key}/{metric}")
                check_float(interval, [float(summary[key][metric+"_ci95_"+bound]) for bound in ["low", "high"]], key)
    _, pairs = table(run / "paired_comparisons.csv")
    _, cases = table(run / "paired_case_review.csv", allow_empty=True)
    assert len(pairs) == len(PAIRS)*4
    expected_cases = []
    for left, right in PAIRS:
        lcorrect, rcorrect = predictions[left] == y, predictions[right] == y
        for j, metric in enumerate(METRICS):
            row, = [r for r in pairs if (r["left"], r["right"], r["metric"]) == (left, right, metric)]
            check_float(float(row["difference"]), summary_audit[left][metric]-summary_audit[right][metric], "paired difference")
            check_float([float(row["ci95_low"]), float(row["ci95_high"])],
                        np.quantile(boot[left][:, j]-boot[right][:, j], [0.025, 0.975]), "paired interval")
            assert int(row["left_only_correct"]) == int((lcorrect & ~rcorrect).sum())
            assert int(row["right_only_correct"]) == int((~lcorrect & rcorrect).sum())
            assert int(row["prediction_disagreements"]) == int((predictions[left] != predictions[right]).sum())
            assert int(row["both_wrong_changed"]) == int((~lcorrect & ~rcorrect & (predictions[left] != predictions[right])).sum())
        for i in np.flatnonzero(predictions[left] != predictions[right]):
            expected_cases.append((left, right, paths[i], str(folds[i]), CLASSES[y[i]],
                                   CLASSES[predictions[left][i]], CLASSES[predictions[right][i]],
                                   "left_only_correct" if lcorrect[i] else "right_only_correct" if rcorrect[i] else "both_wrong"))
    case_fields = ["left", "right", "apk_path", "test_fold", "true_class", "left_prediction", "right_prediction", "case_type"]
    assert [tuple(r[k] for k in case_fields) for r in cases] == expected_cases

    packages = defaultdict(list)
    for i, p in enumerate(paths):
        package = raw[p]["metadata"].get("manifest_package")
        if package:
            packages[package].append({"apk_path": p, "sha256": raw[p]["sha256"],
                                      "true_class": CLASSES[y[i]], "test_fold": int(folds[i])})
    repeated_packages = [{"package": pkg, "samples": rows,
                          "crosses_test_folds": len({r["test_fold"] for r in rows}) > 1}
                         for pkg, rows in packages.items() if len(rows) > 1]
    identical_vectors = {}
    for key, ids in CONFIGS.items():
        _, inverse, counts = np.unique(X[:, np.array(ids)-1], axis=0, return_inverse=True, return_counts=True)
        identical_vectors[key] = {"groups": int((counts > 1).sum()),
                                  "samples": int(counts[counts > 1].sum()),
                                  "cross_fold_groups": int(sum(len(set(folds[inverse == g])) > 1
                                                              for g in np.flatnonzero(counts > 1)))}
    after = {p: digest(p) for p in before}
    assert before == after, "Protected source/input/result files changed during audit"
    report = {"status": "passed", "audited_utc": datetime.now(timezone.utc).isoformat(),
              "experiment": str(run), "sample_count": len(y), "class_counts": manifest["class_counts"],
              "checks": {"recorded_output_hashes": len(manifest["output_sha256"]),
                         "raw_feature_cells_recomputed": int(X.size), "raw_content_hashes_unique": len(set(hashes)),
                         "fresh_model_fits": len(CONFIGS)*5, "fresh_prediction_mismatches": 0,
                         "sklearn_bootstrap_spot_checks": len(CONFIGS)*20,
                         "all_saved_bootstrap_values_verified": int(sum(b.size for b in boot.values())),
                         "protected_files_unchanged": len(before)},
              "fold_sizes": fold_sizes, "results": summary_audit,
              "repeated_manifest_packages": repeated_packages, "identical_feature_vectors": identical_vectors,
              "interpretation_limits": ["These are fixed-feature ablations on a development benchmark.",
                                        "Ordinary stratified folds do not isolate developer or code lineage groups.",
                                        "Shared package identity is a grouping warning, not proof of identical code.",
                                        "Feature removal does not disable upstream contextual/relation extraction.",
                                        "Bootstrap intervals are conditional on fixed OOF predictions, without multiplicity correction.",
                                        "Cases have verified predictions but do not yet have manual causal explanations.",
                                        "Recorded content hashes were checked; APK binaries were not rescanned."],
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "audit_source_sha256": digest(__file__), "verified_file_sha256": after}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"AUDIT PASSED: {destination}", flush=True)
    print(f"Repeated packages: {len(repeated_packages)}; crossing folds: "
          f"{sum(r['crosses_test_folds'] for r in repeated_packages)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.n_jobs == 0:
        parser.error("n-jobs cannot be zero")
    run = args.run_dir.resolve()
    destination = args.output or run / "audit.json"
    if destination.exists():
        parser.error("Use a new audit output path; existing audit files are preserved")
    audit(run, destination, args.n_jobs)
