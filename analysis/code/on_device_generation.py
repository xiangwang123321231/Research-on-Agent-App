"""Typed on-device generation API calls; no category assignment or execution claim."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import re
from signal_types import RawSignal, SignalType, SignalRole, SignalScope, SignalRelation, SignalVerification
from webview_fields import METHOD, without_comments
from device_generation_smali import extract_device_calls, referenced_owners, OWNER

MLKIT_PROMPT_PACKAGE = "com.google.mlkit.genai.prompt"
_GENERATION_CALL = re.compile(
    r"(?m)^\s*invoke-(?:virtual|interface)(?:/range)?\s+\{[^}]*\},\s*"
    r"(Lcom/google/mlkit/genai/prompt/GenerativeModel;->"
    r"generateContent\(Lcom/google/mlkit/genai/prompt/GenerateContentRequest;"
    r"(?:Lkotlin/coroutines/Continuation;)?\)"
    r"(?:Ljava/lang/Object;|Lcom/google/mlkit/genai/prompt/GenerateContentResponse;|Lcom/google/android/gms/tasks/Task;))\s*$"
)


def extract_on_device_generation(root: Path, scope, diagnostics: dict | None = None):
    info = diagnostics if diagnostics is not None else {}
    info.update(status="complete", files_checked=0, bytes_checked=0, calls=0,
                mlkit_calls=0,
                referenced_classes_checked=0, interface_observations=0,
                generation_calls=0, unsupported_runtime_binding_methods=0,
                coverage='typed_sdk_calls_and_local_tvm_cfg_constructor_fields')
    if not root.is_dir() or not scope.usable:
        info.update(status="skipped", reason="missing_root_or_scope")
        return []
    signals = []
    sources = {}
    indexes = {}
    for dex in sorted(root.glob("smali*")):
        if dex.is_dir():
            for path in sorted(dex.rglob('*.smali')):
                owner = 'L' + path.relative_to(dex).with_suffix('').as_posix() + ';'
                indexes.setdefault(owner, []).append(path)

    def read(path):
        try:
            size = path.stat().st_size
            if info['files_checked'] >= 4096 or info['bytes_checked'] + size > 64 * 1024 * 1024:
                info.update(status='partial', reason='source_budget')
                return None
            text = path.read_text(encoding='utf-8')
            info['files_checked'] += 1
            info['bytes_checked'] += size
            return text
        except (OSError, UnicodeError):
            info.update(status='partial', reason='source_read_error')
            return None
    for dex in sorted(root.glob("smali*")):
        if not dex.is_dir():
            continue
        for path in sorted(dex.rglob("*.smali")):
            if not scope.matches_smali_path(path, root):
                continue
            text = read(path)
            if text is None:
                continue
            owner = re.search(r"(?m)^\s*\.class\b[^\n]*\s(L[^;\s]+;)\s*$", text)
            if not owner or owner[1].startswith(("Lcom/google/mlkit/", "Lcom/google/android/gms/", "Landroid/", "Landroidx/")):
                continue
            if not scope.matches_owner(owner[1]):
                continue
            sources[path] = text
            if "Lcom/google/mlkit/genai/prompt/GenerativeModel;->generateContent(" not in text:
                continue
            for method in METHOD.finditer(without_comments(text)):
                signature = method[1].split()[-1]
                # Abstract/native declarations have no invoke and cannot match.
                targets = sorted(set(_GENERATION_CALL.findall(method[2])))
                for target in targets:
                    signals.append(RawSignal(
                        SignalType.KNOWN_LLM_SDK_PACKAGE, MLKIT_PROMPT_PACKAGE, "apktool", str(path),
                        role=SignalRole.CALL, scope=SignalScope.METHOD,
                        relation=SignalRelation.RUNTIME_LLM_CONTEXT,
                        verification=SignalVerification.VERIFIED,
                        source_kind="smali_on_device_generation_call",
                        components=("runtime=mlkit_genai_prompt", "inference_location=same_device",
                                    "binding=typed_sdk_invocation", "operation=text_generation",
                                    "evidence=static_invoke", f"caller={owner[1]}->{signature}",
                                    f"target={target}"),
                    ))
                    info['mlkit_calls'] += 1
    # Audit exact directly invoked classes without globally broadening app
    # ownership. This admits renamed/moved adapters, not an entire namespace
    # or an unreferenced bundled SDK. Only one hop is followed.
    visited = set(sources)
    reference_sources = {}
    for path, text in sources.items():
        for owner in referenced_owners(text):
            reference_sources.setdefault(owner, set()).add(str(path))
    owners = set(reference_sources)
    related_sources = {}
    excluded = ('Landroid/', 'Landroidx/', 'Ljava/', 'Ljavax/', 'Lkotlin/', 'Lkotlinx/',
                'Lcom/google/', 'Lorg/apache/tvm/', 'Lorg/json/', 'Lokhttp3/', 'Lretrofit2/')
    for owner in sorted(owners):
        if owner.startswith(excluded):
            continue
        paths = indexes.get(owner, [])
        if len(paths) != 1 or paths[0] in visited:
            continue
        path = paths[0]
        visited.add(path)
        text = read(path)
        if text is not None and (declaration := OWNER.search(text)) and declaration[1] == owner:
            sources[path] = text
            related_sources[path] = sorted(reference_sources[owner])
            info['referenced_classes_checked'] += 1
    for path, text in sources.items():
        if 'Lorg/apache/tvm/' not in text and '/genai/' not in text and '/litertlm/' not in text:
            continue
        recovered = extract_device_calls(text, str(path), info)
        if path in related_sources:
            recovered = [replace(s, components=(*s.components,
                         'adapter_reference=direct_typed_class_invocation',
                         *(f'adapter_referenced_by={p}' for p in related_sources[path]))) for s in recovered]
        # ML Kit already has a typed call observation above. Preserve that
        # existing fact rather than counting the same invocation twice.
        existing_mlkit = {tuple(c for c in s.components if c.startswith(('caller=', 'target=')))
                          for s in signals if s.source_file == str(path)
                          and s.source_kind == 'smali_on_device_generation_call'}
        recovered = [s for s in recovered if not (
            s.value == 'mlkit_genai_prompt'
            and tuple(c for c in s.components if c.startswith(('caller=', 'target='))) in existing_mlkit)]
        signals.extend(recovered)
        info['interface_observations'] += len(recovered)
    info['generation_calls'] = sum(s.signal_type in (SignalType.ON_DEVICE_GENERATION_USAGE,
                                                     SignalType.KNOWN_LLM_SDK_PACKAGE)
                                   and s.role == SignalRole.CALL
                                   and any(c in s.components for c in
                                           ('operation=text_generation', 'operation=prefill', 'operation=decode'))
                                   for s in signals)
    info['calls'] = info['generation_calls']
    if info['unsupported_runtime_binding_methods'] and info['status'] == 'complete':
        info.update(status='partial', reason='unsupported_runtime_binding_control_flow')
    return signals

