"""Semantic boundaries, source isolation and migration for the 60d mapping."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from business_facts import scan_llm_business_literal
from feature_contract import check_model_contract, read_contract, validate_records
from feature_vectorizer import FEATURE_SCHEMA_VERSION, FeatureVectorizer
from feature_vectorizer_59d import FeatureVectorizer as HistoricalVectorizer
from signal_types import SignalAggregator
from export_features import main as export_main


def literal(text='ai-chat', source='assets/interface/chunks/main.js'):
    return SignalAggregator().merge(list(scan_llm_business_literal(text, source, 'zip')))


def frontend(entry='assets/interface/index.html', binding='framework_entry_configuration'):
    return {'frontend_analysis': {'status': 'partial', 'reason': 'unsupported_html_base',
                                 'entries': [{'asset': entry, 'binding': binding}]}}


def route(value='https://service.example/llm/answer', strength='strong'):
    return dict(signal_type='business_ai_endpoint', value=value, role='relation',
                scope='value', verification='observed', source_kind='binary_url',
                relation='custom_domain_with_llm_path', source_files=['classes.dex'],
                components=['route_strength='+strength, 'llm_path=/llm/answer',
                            'custom_domain=service.example'])


class SemanticFeatures60Tests(unittest.TestCase):
    def setUp(self):
        self.fv = FeatureVectorizer()

    def features(self, signals=(), metadata=None):
        return dict(zip(self.fv.feature_names,
                        self.fv.vectorize_one(list(signals), metadata or {})))

    def test_schema_removes_only_three_old_columns_and_adds_four(self):
        old = HistoricalVectorizer().feature_names
        deleted = {'f50_zip_entries_log', 'f51_jadx_source_file_count', 'f52_jadx_failed'}
        self.assertEqual([n.split('_', 1)[1] for n in self.fv.feature_names[:56]],
                         [n.split('_', 1)[1] for n in old if n not in deleted])
        self.assertEqual(self.fv.n_features, 60)
        self.assertEqual(len(set(self.fv.feature_names)), 60)
        self.assertEqual([n.split('_', 1)[0] for n in self.fv.feature_names],
                         [f'f{i:02d}' for i in range(1, 61)])
        self.assertEqual(FEATURE_SCHEMA_VERSION, 'features-60d-v2026.11.06.2')

    def test_audit_metadata_no_longer_changes_vector(self):
        a = {'size_bytes': 16, 'entries': 1, 'jadx_status': 'success',
             'layer_coverage': {'jadx_app_source_files': 3}}
        b = {**a, 'entries': 9000, 'jadx_status': 'failed',
             'layer_coverage': {'jadx_app_source_files': 10000}}
        self.assertEqual(self.features(metadata=a), self.features(metadata=b))
        self.assertNotEqual(self.features(metadata=a)['f49_apk_size_log'],
                            self.features(metadata={**a, 'size_bytes': 32})['f49_apk_size_log'])

    def test_web_provenance_without_navigation_or_full_chain(self):
        f = self.features(literal(), frontend())
        self.assertEqual(f['f57_web_frontend_llm_observation_present'], 1)
        self.assertEqual([f[n] for n in ('f38_web_container_present',
                         'f39_web_navigation_call_present', 'f40_web_llm_relation_present')], [0, 0, 0])
        self.assertEqual(f['f55_programmatic_llm_usage_present'], 0)

    def test_dynamic_asset_roots_and_inline_html(self):
        for root in ('assets/a_new_ui', 'assets/renamed/deeper'):
            for source in (root+'/chunks/panel.mjs', root+'/index.html',
                           'D:\\decoded\\'+root.replace('/', '\\')+'\\main.js'):
                with self.subTest(root=root, source=source):
                    f = self.features(literal(source=source), frontend(root+'/index.html', 'local_asset_argument'))
                    self.assertEqual(f['f57_web_frontend_llm_observation_present'], 1)

    def test_no_web_frontend_guess_from_js_or_metadata_only(self):
        for signals, meta in ((literal(), {}), ([], frontend()),
                              (literal(), frontend(binding='unresolved')),
                              (literal(source='assets/another/main.js'), frontend()),
                              (literal(), frontend('https://example.org/assets/interface/index.html')),
                              (literal('chat'), frontend())):
            self.assertEqual(self.features(signals, meta)['f57_web_frontend_llm_observation_present'], 0)

    def test_auxiliary_and_non_browser_material_not_web(self):
        for path in ('assets/interface/docs/help.js', 'assets/interface/vendor/engine.js',
                     'assets/interface/node_modules/sdk/main.js', 'assets/interface/tests/main.js',
                     'assets/interface/main.js.map', 'assets/interface/index.android.bundle',
                     'assets/interface/README.js', 'assets/interface/../outside.js',
                     'https://example.org/assets/interface/main.js'):
            with self.subTest(path=path):
                self.assertEqual(self.features(literal(source=path), frontend())[
                    'f57_web_frontend_llm_observation_present'], 0)
        for kind in ('hermes_string_literal', 'ram_string_literal'):
            s = literal()[0]
            s['evidence_instances'][0]['source_kind'] = kind
            self.assertEqual(self.features([s], frontend())['f57_web_frontend_llm_observation_present'], 0)

    def test_web_does_not_borrow_admission_from_other_occurrence(self):
        s = literal()[0]
        valid = copy.deepcopy(s['evidence_instances'][0])
        valid['source_file'] = 'sources/Panel.java'
        invalid = copy.deepcopy(s['evidence_instances'][0])
        invalid['verification'] = 'candidate'
        s['evidence_instances'] = [valid, invalid]
        s['source_files'] = [valid['source_file'], invalid['source_file']]
        self.assertEqual(self.features([s], frontend())['f57_web_frontend_llm_observation_present'], 0)

    def test_business_route_is_not_invocation_or_ownership(self):
        f = self.features([route()])
        self.assertEqual(f['f16_business_ai_endpoint_present'], 1)
        self.assertEqual(f['f58_business_llm_route_observation_present'], 1)
        self.assertEqual(f['f55_programmatic_llm_usage_present'], 0)
        relative = route('/llm/answer')
        self.assertEqual(self.features([relative])['f58_business_llm_route_observation_present'], 1)

    def test_host_only_and_noise_do_not_gain_route_bit(self):
        for s in (route('https://service.example'),
                  route('https://service.example/llm/answer', 'host_identity'),
                  route('https://translation.googleapis.com/v2/llm/answer')):
            if s['value'] == 'https://service.example':
                s['components'] = ['route_strength=strong', 'llm_path=']
            self.assertEqual(self.features([s])['f58_business_llm_route_observation_present'], 0)

    def test_root_slash_is_host_identity_not_a_business_route(self):
        for path in ('', '/', '/?mode=llm'):
            s = route('https://service.example'+path)
            s['components'] = ['route_strength=strong', 'llm_path='+path]
            self.assertEqual(self.features([s])['f58_business_llm_route_observation_present'], 0)

    def test_route_proof_source_and_admission_are_one_occurrence(self):
        s = route()
        left, right = copy.deepcopy(s), copy.deepcopy(s)
        left['components'] = ['route_strength=strong']
        right['components'] = ['llm_path=/llm/answer']
        s['evidence_instances'] = [left, right]
        self.assertEqual(self.features([s])['f58_business_llm_route_observation_present'], 0)
        right['components'] = list(route()['components'])
        right['verification'] = 'candidate'
        self.assertEqual(self.features([s])['f58_business_llm_route_observation_present'], 0)

    def test_protocol_property_and_generic_reference_stay_distinct(self):
        for text, expected in (('ai-chat', 0), ('/v1/chat/completions', 1), ('gpt-4_badge_visible', 1)):
            with self.subTest(text=text):
                f = self.features(literal(text, 'sources/Panel.java'))
                self.assertEqual(f['f56_llm_operation_reference_present'], 1)
                self.assertEqual(f['f59_llm_protocol_structure_reference_present'], expected)
                self.assertEqual(f['f55_programmatic_llm_usage_present'], 0)
                if text == 'gpt-4_badge_visible':
                    self.assertEqual(f['f23_model_name_present'], 0)

    def test_protocol_cannot_borrow_fact_from_invalid_instance(self):
        s = literal('/v1/chat/completions')[0]
        valid, invalid = copy.deepcopy(s['evidence_instances'][0]), copy.deepcopy(s['evidence_instances'][0])
        valid['components'] = ['business_fact=llm_related_operation']
        invalid['role'] = 'literal'
        s['evidence_instances'] = [valid, invalid]
        self.assertEqual(self.features([s])['f59_llm_protocol_structure_reference_present'], 0)

    def test_device_identity_unifies_dedicated_stacks_not_generic_keywords(self):
        def identity(st, value):
            return dict(signal_type=st, value=value, role='reference', scope='value',
                        verification='observed', source_files=['sources/Panel.java'])
        for s, expected in ((identity('native_llm_library', 'libllama.so'), 1),
                            (identity('known_llm_sdk_package', 'com.google.mlkit.genai.prompt'), 1),
                            (identity('llm_framework_keyword', 'gguf'), 0),
                            (identity('local_inference_runtime', 'tvm'), 0),
                            (identity('onnx_tflite_runtime', 'tflite'), 0)):
            f = self.features([s])
            self.assertEqual(f['f60_dedicated_device_stack_present'], expected)
            self.assertEqual(f['f54_on_device_generation_call_present'], 0)

    def test_duplicate_observations_are_binary_and_inputs_unchanged(self):
        signals = literal('/v1/chat/completions') + [route()]
        original = copy.deepcopy(signals)
        f = self.features(signals * 4, frontend())
        self.assertEqual([f[n] for n in self.fv.feature_names[-4:]], [1, 1, 1, 0])
        self.assertEqual(signals, original)

    def test_retained_56_columns_identical_to_historical_mapping(self):
        old = HistoricalVectorizer()
        signals = literal('gpt-4_badge_visible') + literal('/v1/chat/completions') + [route()]
        meta = {**frontend(), 'size_bytes': 256, 'entries': 48, 'jadx_status': 'failed',
                'callgraph': {'api_domain_near_http': 5, 'native_near_jni': 3}}
        deleted = {'f50_zip_entries_log', 'f51_jadx_source_file_count', 'f52_jadx_failed'}
        previous = [value for name, value in zip(old.feature_names,
                    old.vectorize_one(signals, meta)) if name not in deleted]
        self.assertEqual(self.fv.vectorize_one(signals, meta)[:56], previous)

    def test_previous_60d_contract_cannot_silently_cross_column_rename(self):
        current = validate_records([])
        previous = {**current, 'feature_schema_version': 'features-60d-v2026.11.06.1'}
        for allow_legacy in (False, True):
            with self.assertRaisesRegex(ValueError, 'retrain'):
                check_model_contract(previous, current, allow_legacy)

    def test_revectorize_old_signals_keeps_labels_and_rejects_old_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {**validate_records([]), 'feature_schema_version': 'features-59d-v2026.10.25.1'}
            source = root/'apks.jsonl'
            source.write_text(json.dumps({'apk_path': 'a.apk', 'metadata': metadata, 'signals': []})+'\n', encoding='utf-8')
            labels = root/'labels.csv'; labels.write_text('preserve', encoding='utf-8')
            with patch.object(sys, 'argv', ['export_features.py', '--jsonl', str(source), '--revectorize']), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(export_main(), 0)
            target = root/'features_revectorized.csv'
            self.assertEqual(len(target.read_text(encoding='utf-8').splitlines()[0].split(',')), 61)
            self.assertEqual(labels.read_text(encoding='utf-8'), 'preserve')
            with self.assertRaisesRegex(ValueError, 'retrain'):
                check_model_contract(metadata, read_contract(target))


if __name__ == '__main__':
    unittest.main()
