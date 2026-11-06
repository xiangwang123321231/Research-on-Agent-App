"""Tests use temporary dummy files; no research APK or dataset is modified."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "random_copy_apks.py"
SPEC = importlib.util.spec_from_file_location("random_copy_apks", SCRIPT)
SAMPLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLER)


class RandomCopyApksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "apks"
        self.source.mkdir()

    def populate(self, count):
        for number in range(count):
            (self.source / f"app_{number:03}.apk").write_bytes(
                f"dummy APK {number}".encode("utf-8")
            )

    def run_sample(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return SAMPLER.copy_sample(self.source, **kwargs)

    def test_default_copies_200_and_preserves_originals(self):
        self.populate(205)
        (self.source / "ignore.txt").write_text("not an APK", encoding="utf-8")
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        report = self.run_sample()
        output = self.root / "apks_random_200"
        self.assertEqual(len(list(output.glob("*.apk"))), 200)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["candidate_count"], 205)
        self.assertEqual(report["unique_sample_sha256_count"], 200)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        saved = json.loads((output / "sampling_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        for record in report["samples"]:
            self.assertEqual(
                (output / record["relative_path"]).read_bytes(),
                (self.source / record["relative_path"]).read_bytes(),
            )

    def test_reproducible_sorted_pool_and_seed(self):
        self.populate(15)
        first = self.run_sample(count=5, seed=7, output=self.root / "first")
        second = self.run_sample(count=5, seed=7, output=self.root / "second")
        self.assertEqual(first["samples"], second["samples"])
        self.assertEqual(first["candidate_relative_paths"], sorted(first["candidate_relative_paths"]))

    def test_too_few_files_does_not_create_output(self):
        self.populate(3)
        with self.assertRaisesRegex(ValueError, "only 3 APK"):
            self.run_sample()
        self.assertFalse((self.root / "apks_random_200").exists())

    def test_existing_output_is_untouched(self):
        self.populate(3)
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_sample(count=2, output=output)
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_rejects_output_inside_source(self):
        self.populate(3)
        with self.assertRaisesRegex(ValueError, "outside the source"):
            self.run_sample(count=2, output=self.source / "sample")
        self.assertFalse((self.source / "sample").exists())

    def test_recursive_and_uppercase_extension(self):
        self.populate(1)
        nested = self.source / "nested"
        nested.mkdir()
        (nested / "other.APK").write_bytes(b"nested dummy")
        with self.assertRaisesRegex(ValueError, "only 1 APK"):
            self.run_sample(count=2)
        report = self.run_sample(count=2, recursive=True)
        self.assertEqual(report["candidate_count"], 2)
        self.assertTrue((self.root / "apks_random_2" / "nested" / "other.APK").is_file())

    def test_copy_failure_retains_plan_and_originals(self):
        self.populate(3)
        with mock.patch.object(SAMPLER.shutil, "copyfileobj", side_effect=OSError("simulated")):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.run_sample(count=2)
        saved = json.loads(
            (self.root / "apks_random_2" / "sampling_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved["status"], "incomplete")
        self.assertEqual(saved["copied_count"], 0)
        self.assertEqual(len(saved["samples"]), 2)
        self.assertEqual(len(list(self.source.glob("*.apk"))), 3)

    def test_identical_binaries_reported_without_resampling(self):
        self.populate(2)
        (self.source / "app_001.apk").write_bytes((self.source / "app_000.apk").read_bytes())
        report = self.run_sample(count=2)
        self.assertEqual(report["copied_count"], 2)
        self.assertEqual(report["unique_sample_sha256_count"], 1)

    def test_invalid_count_or_missing_source(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.run_sample(count=0)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            SAMPLER.copy_sample(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
