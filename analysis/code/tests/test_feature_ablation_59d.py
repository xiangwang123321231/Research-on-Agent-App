"""59d study regressions; all fitting/output uses synthetic temporary inputs."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import run_feature_ablation as run
import audit_feature_ablation as verify
from feature_contract import validate_records


def fixture(directory, *, constant=True):
    contract = validate_records([])
    old = {**contract, "feature_schema_version": "features-57d-v2026.10.24.1"}
    # Constant vectors and five equally represented classes exercise empty
    # paired-case output without mock estimators or changing the frozen model.
    records = [{"apk_path": f"/synthetic/sample_{i}.apk", "sha256": f"{i+1:064x}",
                "signals": [], "metadata": {**old, "size_bytes": 1, "entries": 1,
                                                "jadx_status": "success"}} for i in range(25)]
    if not constant:
        for i, record in enumerate(records):
            record["metadata"]["size_bytes"] = 2 ** (i % 5 + 1)
            record["metadata"]["entries"] = 3 ** (i % 5 + 1)
    signals = directory / "apks.jsonl"
    signals.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    X, names = run.FeatureVectorizer().transform(records)
    features, labels, reference = [directory / n for n in ("features.csv", "labels.csv", "reference.csv")]
    run.write_csv(features, ["apk_path", *names],
                  [{"apk_path": r["apk_path"], **dict(zip(names, vector))} for r, vector in zip(records, X)])
    run.write_csv(labels, ["apk_path", *run.LABEL_COLUMNS],
                  [{"apk_path": r["apk_path"], **{c: int(j == i % 5) for j, c in enumerate(run.LABEL_COLUMNS)}}
                   for i, r in enumerate(records)])
    reference_predictions = np.zeros(25, dtype=int)
    if not constant:
        truth = np.arange(25) % 5
        matrix = np.array(X, dtype=np.float32)
        for train, test in StratifiedKFold(5, shuffle=True, random_state=42).split(matrix, truth):
            clf = ExtraTreesClassifier(n_estimators=300, max_features=6, max_depth=8,
                                       min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)
            clf.fit(matrix[train], truth[train])
            reference_predictions[test] = clf.predict(matrix[test])
    run.write_csv(reference, ["apk_path", "true_class", "pred_class_extratrees"],
                  [{"apk_path": r["apk_path"], "true_class": i % 5,
                    "pred_class_extratrees": int(reference_predictions[i])}
                   for i, r in enumerate(records)])
    manifest = {"contract": contract, "feature_names": names, "derivation": {
        "operation": "revectorize", "source_sha256": run.sha256(signals), "source_record_count": 25,
        "source_feature_schema_version": old["feature_schema_version"],
        "target_feature_schema_version": contract["feature_schema_version"]}}
    run.write_json(features.with_name(features.name + ".manifest.json"), manifest)
    args = argparse.Namespace(features=features, labels=labels, reference_predictions=reference,
        signals=signals, output_dir=directory / "synthetic_run", max_features=6, n_jobs=1,
        bootstrap=100, bootstrap_seed=20261011, no_plots=False, allow_reference_drift=False)
    return args, manifest


class Ablation59Tests(unittest.TestCase):
    def test_design_and_independent_audit_agree(self):
        run.validate_design(run.FeatureVectorizer().feature_names)
        self.assertEqual({k: list(v) for k, v in run.VARIANTS.items()}, verify.CONFIGS)
        self.assertEqual(run.COMPARISONS, verify.PAIRS)
        self.assertEqual([len(v) for v in run.VARIANTS.values()], [59, 44, 51, 53, 40, 56, 55, 55])
        for key, removed in run.REMOVED_GROUPS.items():
            self.assertEqual(set(run.VARIANTS[run.FULL]) - set(run.VARIANTS[key]), set(removed))

    def test_fixed_model_and_no_sqrt(self):
        p = run.make_study_model(1).get_params()
        self.assertEqual([p[k] for k in ("n_estimators", "max_features", "max_depth", "min_samples_leaf")],
                         [300, 6, 8, 2])
        with self.assertRaises(ValueError):
            run.make_study_model(1, "sqrt")

    def test_metrics_and_paired_bootstrap_independently(self):
        y = np.tile(np.arange(5), 5)
        p = y.copy()
        p[[1, 4, 12]] = [2, 0, 3]
        expected = [f1_score(y, p, average="macro"), accuracy_score(y, p),
                    balanced_accuracy_score(y, p), matthews_corrcoef(y, p)]
        np.testing.assert_allclose(list(run.scores(y, p).values()), expected, atol=1e-12)
        predictions = {"full": p, "other": (y + 1) % 5}
        actual = run.bootstrap_scores(y, predictions, 100, 42)
        independent = verify.bootstrap_from_confusions(y, predictions, 100, 42)
        for key in predictions:
            np.testing.assert_allclose(actual[key], independent[key], atol=1e-12)

    def test_old_or_reordered_header_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = fixture(Path(tmp))
            fields, rows = run.read_csv(args.features)
            for bad in (fields[:53], [fields[0], fields[2], fields[1], *fields[3:]]):
                run.write_csv(args.features, bad, [{k: r[k] for k in bad} for r in rows])
                with self.assertRaisesRegex(ValueError, "59 named columns"):
                    run.load_dataset(args.features, args.labels)

    def test_invalid_labels_and_set_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = fixture(Path(tmp))
            fields, rows = run.read_csv(args.labels)
            for bits in ([0, 0, 0, 0, 0], [1, 1, 0, 0, 0]):
                broken = copy.deepcopy(rows)
                broken[0].update(dict(zip(run.LABEL_COLUMNS, bits)))
                run.write_csv(args.labels, fields, broken)
                with self.assertRaisesRegex(ValueError, "one-hot"):
                    run.load_dataset(args.features, args.labels)
            run.write_csv(args.labels, fields, rows[:-1])
            with self.assertRaisesRegex(ValueError, "APK sets differ"):
                run.load_dataset(args.features, args.labels)

    def test_derivation_hash_and_feature_values_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest = fixture(Path(tmp))
            X, _, paths, names = run.load_dataset(args.features, args.labels)
            result = run.validate_signal_source(args.signals, manifest, paths, names, X)
            self.assertTrue(result["feature_refresh"])
            for bad in ({k: v for k, v in manifest.items() if k != "derivation"},
                        {**manifest, "derivation": {**manifest["derivation"], "source_sha256": "0" * 64}}):
                with self.assertRaisesRegex(ValueError, "derivation"):
                    run.validate_signal_source(args.signals, bad, paths, names, X)
            X[0, 58] = 1
            with self.assertRaisesRegex(ValueError, "values do not reproduce"):
                run.validate_signal_source(args.signals, manifest, paths, names, X)

    def test_mixed_source_versions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest = fixture(Path(tmp))
            X, _, paths, names = run.load_dataset(args.features, args.labels)
            records = [json.loads(line) for line in args.signals.read_text().splitlines()]
            records[0]["metadata"]["feature_schema_version"] = run.FEATURE_SCHEMA_VERSION
            args.signals.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed"):
                run.validate_signal_source(args.signals, manifest, paths, names, X)

    def test_duplicate_content_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest = fixture(Path(tmp))
            X, _, paths, names = run.load_dataset(args.features, args.labels)
            records = [json.loads(line) for line in args.signals.read_text().splitlines()]
            records[1]["sha256"] = records[0]["sha256"]
            args.signals.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique APK"):
                run.validate_signal_source(args.signals, manifest, paths, names, X)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                run.create_run_dir(Path(tmp))

    def test_end_to_end_snapshot_audit_and_all_eight_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                result = run.run(args)
            manifest = json.loads((result / "experiment_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["fit_budget"], 40)
            self.assertTrue(manifest["original_inputs_unchanged"])
            self.assertEqual(manifest["reference_check"]["mismatch_n"], 0)
            _, cases = verify.table(result / "paired_case_review.csv", allow_empty=True)
            self.assertEqual(cases, [])
            for name in ("macro_f1_comparison.svg", "confusion_matrices.svg"):
                svg = (result / name).read_text(encoding="utf-8")
                for key, title in run.PLOT_TITLES.items():
                    self.assertIn(f"{key[0]}: {title} ({len(run.VARIANTS[key])})", svg)
            # Production inputs may change after a run; audit must use snapshots.
            args.features.write_text("changed after experiment", encoding="utf-8")
            args.labels.write_text("changed after experiment", encoding="utf-8")
            args.signals.write_text("changed after experiment", encoding="utf-8")
            args.reference_predictions.write_text("changed after experiment", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                verify.audit(result, result / "audit.json", 1)
            audit = json.loads((result / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["checks"]["fresh_model_fits"], 40)
            # Tampering with a recorded result must stop the audit before fitting.
            (result / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "Artifact hash mismatch"):
                verify.audit(result, result / "audit_tampered.json", 1)

    def test_nonconstant_features_produce_auditable_paired_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = fixture(Path(tmp), constant=False)
            args.no_plots = True
            with contextlib.redirect_stdout(io.StringIO()):
                result = run.run(args)
                verify.audit(result, result / "audit.json", 1)
            _, cases = verify.table(result / "paired_case_review.csv")
            self.assertTrue(cases)
            self.assertEqual({r["right"] for r in cases}, {"H_no_size_quality"})
            self.assertEqual({r["case_type"] for r in cases}, {"left_only_correct"})


if __name__ == "__main__":
    unittest.main()
