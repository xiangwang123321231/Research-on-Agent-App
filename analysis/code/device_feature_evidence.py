"""Per-occurrence checks for local model/config/interface feature semantics."""
from __future__ import annotations

from device_generation_smali import typed_generation_runtime
from feature_evidence import occurrences, sources


GENERATION_SDKS = frozenset({'com.google.ai.edge.litertlm',
                            'com.google.mlkit.genai.prompt', 'ai.mlc.mlcllm'})


def generation_configuration(signal):
    if signal.get('signal_type') != 'local_generation_config':
        return False
    return any(i.get('verification') == 'verified'
               and i.get('role') == 'config' and i.get('scope') == 'config'
               and i.get('source_kind') == 'structured_generation_config'
               and {'config_evidence=structured_generation_config',
                    'artifact_kind=configuration', 'generation_task=autoregressive_language'}
               <= set(i.get('components') or []) and sources(i)
               for i in occurrences(signal))


def model_binary_artifact(signal):
    if signal.get('signal_type') != 'model_file_extension':
        return False
    return any(i.get('verification') == 'verified' and i.get('role') == 'artifact'
               and 'artifact_check=header_valid' in (i.get('components') or [])
               and any(f'artifact_format={fmt}' in (i.get('components') or [])
                       for fmt in ('gguf', 'ggml', 'litertlm'))
               and sources(i) for i in occurrences(signal))


def generation_interface(signal, *, call=False):
    if signal.get('signal_type') == 'known_llm_sdk_package' and not call:
        return str(signal.get('value', '')).lower() in GENERATION_SDKS
    if signal.get('signal_type') == 'known_llm_sdk_package' and call:
        if signal.get('value') != 'com.google.mlkit.genai.prompt':
            return False
        for i in occurrences(signal):
            cs = set(i.get('components') or [])
            target = next((s.split('=', 1)[1] for s in cs if s.startswith('target=')), '')
            if (i.get('verification') == 'verified' and i.get('scope') == 'method'
                    and i.get('role') == 'call' and i.get('source_kind') == 'smali_on_device_generation_call'
                    and {'runtime=mlkit_genai_prompt', 'inference_location=same_device',
                         'binding=typed_sdk_invocation', 'evidence=static_invoke',
                         'operation=text_generation'} <= cs
                    and sources(i) and typed_generation_runtime(target) == 'mlkit_genai_prompt'):
                return True
        return False
    if signal.get('signal_type') != 'on_device_generation_usage':
        return False
    runtime = signal.get('value')
    for i in occurrences(signal):
        cs = set(i.get('components') or [])
        if not (i.get('verification') == 'verified' and i.get('scope') == 'method'
                and i.get('source_kind') == 'smali_device_generation_interface'
                and 'inference_location=same_device' in cs and sources(i)
                and f'runtime={runtime}' in cs):
            continue
        target = next((s.split('=', 1)[1] for s in cs if s.startswith('target=')), '')
        if runtime == 'mlc_llm':
            valid = 'runtime_function_family=mlc.llm_chat_create' in cs and (
                ('binding=typed_runtime_function_lookup' in cs and i.get('role') == 'reference')
                or ('binding=typed_runtime_function_invoke' in cs and i.get('role') == 'call'
                    and target == 'Lorg/apache/tvm/Function;->invoke()Lorg/apache/tvm/TVMValue;'))
        else:
            valid = (typed_generation_runtime(target) == runtime
                     and 'binding=typed_sdk_invocation' in cs and i.get('role') == 'call')
        if not valid:
            continue
        if not call:
            return True
        if (i.get('role') == 'call' and 'evidence=static_invoke' in cs
                and any(f'operation={op}' in cs for op in ('text_generation', 'prefill', 'decode'))):
            return True
    return False
