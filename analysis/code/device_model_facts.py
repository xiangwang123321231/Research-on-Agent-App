"""Structured generation configurations and generic runtime identities.

A configuration is not a weight file, a runtime is not proof of language
generation, and neither implies that inference executed on a device.
"""
from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

from signal_types import RawSignal, SignalRole, SignalScope, SignalType, SignalVerification


def generation_config_facts(text: str, source: str, layer: str):
    """Recognize complete config objects, never keyword proximity or examples."""
    path = source.replace('\\', '/')
    name = PurePosixPath(path).name.lower()
    if name not in {'mlc-chat-config.json', 'config.json'}:
        return []
    # Only shipped resource/configuration trees. Documentation and source-code
    # snippets with the same basename cannot establish model configuration.
    if not re.search(r'(?:^|/)(?:assets|res/raw)/', path, re.I):
        return []
    if re.search(r'(?:^|/)(?:docs?|examples?|tests?|samples?)/', path, re.I):
        return []
    try:
        config = json.loads(text)
    except (ValueError, RecursionError):
        return []
    if not isinstance(config, dict):
        return []

    def positive_int(value):
        return type(value) is int and value > 0

    architecture = config.get('model_type')
    if not isinstance(architecture, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', architecture):
        return []
    components = [f'model_architecture={architecture.lower()}',
                  'config_evidence=structured_generation_config',
                  'artifact_kind=configuration', 'execution=not_observed']
    if name == 'mlc-chat-config.json':
        model = config.get('model_config')
        if not isinstance(model, dict) or not all(positive_int(model.get(k)) for k in
                ('hidden_size', 'num_attention_heads', 'num_hidden_layers', 'vocab_size')):
            return []
        tokenizer = config.get('tokenizer_files')
        if not (isinstance(tokenizer, list) and tokenizer
                and all(isinstance(v, str) and v and len(v) <= 256 for v in tokenizer)):
            return []
        # Language decoder structure plus conversation/generation fields; a
        # CV model renamed to this filename does not qualify.
        if not (isinstance(config.get('conv_template'), (str, dict))
                and any(k in config for k in ('temperature', 'top_p', 'max_gen_len'))):
            return []
        components += ['config_format=mlc_chat', 'runtime=mlc_llm',
                       'generation_task=autoregressive_language',
                       f'vocab_size={model["vocab_size"]}']
    else:
        architectures = config.get('architectures')
        if not (isinstance(architectures, list) and architectures
                and all(isinstance(a, str) and re.fullmatch(r'[A-Za-z0-9_]{1,128}', a)
                        for a in architectures)
                and any(a.endswith('ForCausalLM') for a in architectures)
                and all(positive_int(config.get(k)) for k in
                        ('vocab_size', 'hidden_size', 'num_hidden_layers'))):
            return []
        components += ['config_format=causal_lm', 'generation_task=autoregressive_language',
                       f'vocab_size={config["vocab_size"]}']
    return [RawSignal(SignalType.LOCAL_GENERATION_CONFIG, path, layer, source,
                      role=SignalRole.CONFIG, scope=SignalScope.CONFIG,
                      verification=SignalVerification.VERIFIED,
                      source_kind='structured_generation_config', components=tuple(components))]


def generic_runtime_fact(name: str, layer: str = 'zip'):
    path = name.replace('\\', '/')
    if not re.fullmatch(r'lib/[^/]+/libtvm(?:4j)?_runtime(?:_packed)?\.so', path, re.I):
        return []
    return [RawSignal(SignalType.LOCAL_INFERENCE_RUNTIME, 'tvm', layer, name,
                      role=SignalRole.ARTIFACT, scope=SignalScope.VALUE,
                      source_kind='generic_native_runtime',
                      components=('runtime=tvm', 'runtime_specificity=general_ml',
                                  'language_generation=not_established'))]
