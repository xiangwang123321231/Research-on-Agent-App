"""Display-only regressions; no training, real data export or model writes."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import trainer


EXPECTED_NAMES = [
    "第三方 LLM 服务直连", "服务端介导的 LLM 访问", "基于网页的 LLM 交互",
    "端侧 LLM 推理", "非目标候选应用",
]
EXPECTED_COLUMNS = [
    "is_direct_client", "is_server_proxy", "is_webview", "is_local", "is_non_ai",
]
OLD_NAMES = ["客户端直连", "服务端中转", "WebView套壳", "本地部署", "非智能应用"]


class FakeClassifier:
    def predict_proba(self, features):
        return np.eye(5)[np.arange(len(features)) % 5] * 0.8 + 0.04

    def predict(self, features):
        return np.argmax(self.predict_proba(features), axis=1)


class ClassifierDisplayNameTests(unittest.TestCase):
    def test_current_names_and_stable_columns(self):
        self.assertEqual(trainer.CLASS_CN, EXPECTED_NAMES)
        self.assertEqual(trainer.LABEL_COLS, EXPECTED_COLUMNS)
        self.assertEqual(trainer.N_CLASSES, 5)

    def test_old_model_metadata_is_normalized_without_mutation(self):
        bundle = {"class_names": OLD_NAMES.copy(), "label_cols": EXPECTED_COLUMNS.copy()}
        self.assertEqual(trainer._prediction_class_names(bundle), EXPECTED_NAMES)
        self.assertEqual(bundle["class_names"], OLD_NAMES)
        self.assertEqual(bundle["label_cols"], EXPECTED_COLUMNS)

    def test_legacy_names_without_label_columns(self):
        self.assertEqual(
            trainer._prediction_class_names({"class_names": OLD_NAMES}), EXPECTED_NAMES,
        )

    def test_missing_names_and_current_names(self):
        self.assertEqual(trainer._prediction_class_names({}), EXPECTED_NAMES)
        self.assertEqual(
            trainer._prediction_class_names({"class_names": EXPECTED_NAMES}), EXPECTED_NAMES,
        )

    def test_other_explicit_label_order_is_not_reinterpreted(self):
        self.assertEqual(
            trainer._prediction_class_names({
                "class_names": OLD_NAMES,
                "label_cols": list(reversed(EXPECTED_COLUMNS)),
            }), OLD_NAMES,
        )
        other_names = ["A", "B", "C", "D", "E"]
        self.assertEqual(
            trainer._prediction_class_names({"class_names": other_names}), other_names,
        )

    def test_api_changes_only_display_names(self):
        classifier = FakeClassifier()
        bundle = {
            "model": classifier, "class_names": OLD_NAMES.copy(),
            "label_cols": EXPECTED_COLUMNS.copy(),
        }
        features = np.zeros((5, 52), dtype=np.float32)
        with mock.patch.object(trainer, "load_model", return_value=bundle):
            result = trainer.predict("unused.pkl", features)
        np.testing.assert_array_equal(result["pred_class"], classifier.predict(features))
        np.testing.assert_array_equal(result["probas"], classifier.predict_proba(features))
        self.assertEqual(result["pred_label"], EXPECTED_NAMES)
        self.assertEqual(result["class_names"], EXPECTED_NAMES)
        self.assertEqual(bundle["class_names"], OLD_NAMES)

    def test_future_model_saves_use_current_names(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = trainer.save_models(
                {"fake": {"final_clf": FakeClassifier()}}, Path(directory), ["f01"],
            )
            bundle = trainer.load_model(saved["fake"])
        self.assertEqual(bundle["class_names"], EXPECTED_NAMES)
        self.assertEqual(bundle["label_cols"], EXPECTED_COLUMNS)

    def test_cli_uses_same_names_and_keeps_numeric_outputs(self):
        classifier = FakeClassifier()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_file = root / "features.csv"
            model_file = root / "model.pkl"
            output_file = root / "predictions.csv"
            feature_file.write_text(
                "apk_path,f01\n" + "".join(f"sample_{index}.apk,0\n" for index in range(5)),
                encoding="utf-8",
            )
            model_file.touch()
            bundle = {
                "model": classifier, "feature_names": ["f01"],
                "class_names": OLD_NAMES.copy(), "label_cols": EXPECTED_COLUMNS.copy(),
            }
            args = argparse.Namespace(
                predict=str(model_file), features=str(feature_file),
                output=str(output_file), output_dir=str(root),
                threshold=None, allow_legacy=False,
            )
            with mock.patch.object(trainer, "load_model", return_value=bundle), \
                    mock.patch.object(trainer, "read_contract", return_value={}), \
                    mock.patch.object(trainer, "check_model_contract"), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(trainer._predict_cli(args), 0)
            with output_file.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["pred_label"] for row in rows], EXPECTED_NAMES)
            self.assertEqual([int(row["pred_class"]) for row in rows], list(range(5)))
            for index, row in enumerate(rows):
                probabilities = [float(row[f"proba_{c}_{name}"]) for c, name in enumerate(EXPECTED_NAMES)]
                np.testing.assert_allclose(probabilities, classifier.predict_proba(np.zeros((5, 1)))[index])
            self.assertEqual(bundle["class_names"], OLD_NAMES)


if __name__ == "__main__":
    unittest.main()
