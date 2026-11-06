"""Instruction-scoped independent facts; no reachability prerequisite."""
from dataclasses import replace
import re

from business_facts import scan_llm_literal, llm_serialized_name
from registry import is_reference_only_source
from signal_types import RawSignal, SignalType, SignalRole, SignalScope
from smali_literals import decode_smali_string

_METHOD = re.compile(r'(?m)^\s*\.method\s+([^\n]+)')
_CONST = re.compile(r'^\s*const-string(?:/jumbo)?\s+[vp]\d+,\s*("(?:\\.|[^"\\])*")\s*(?:#.*)?$')


def scan_smali_facts(text, source, diagnostics):
    """Only real const instructions, plus explicit serialization annotations.

    Input files are selected by the caller's application scope. All literal
    observations retain their file and method; annotations are not code calls.
    """
    if is_reference_only_source(source):
        return []
    facts, seen = [], set()
    owner = re.search(r'(?m)^\s*\.class\b[^\n]*\s(L[^;\s]+;)\s*$', text)
    method, annotation = '', 0
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith('.annotation'):
            annotation += 1
        if stripped.startswith('.end annotation'):
            annotation = max(0, annotation-1)
        m = _METHOD.match(line)
        if m:
            method = (owner[1] if owner else '') + '->' + m[1].rsplit(None, 1)[-1]
        if stripped == '.end method':
            method = ''
        if not method or annotation:
            continue
        m = _CONST.fullmatch(line)
        if not m:
            continue
        diagnostics['constants_read'] = diagnostics.get('constants_read', 0) + 1
        try:
            value = decode_smali_string(m[1])
        except ValueError:
            diagnostics['invalid_constants'] = diagnostics.get('invalid_constants', 0) + 1
            continue
        for fact in scan_llm_literal(value, source, 'apktool', 'smali_instruction_literal'):
            key = fact.signal_type, fact.value, method
            if key in seen:
                continue
            seen.add(key)
            facts.append(replace(fact, source_kind='smali_instruction_literal',
                                 components=(*fact.components, f'source_method={method}',
                                             f'source_line={line_no}', 'coverage=instruction_literal')))
    # Kotlin SerialName is also emitted on generated property-annotation
    # methods. Do not scan d1/d2 Metadata strings as serialized fields.
    for a in re.finditer(r'(?ms)^\s*\.annotation\s+runtime\s+L(?:kotlinx/serialization/SerialName|com/google/gson/annotations/SerializedName|com/fasterxml/jackson/annotation/JsonProperty);(.*?)^\s*\.end annotation', text):
        name = re.search(r'(?m)^\s*value\s*=\s*"([\w]+)"\s*$', a[1])
        if name and llm_serialized_name(name[1]):
            facts.append(RawSignal(SignalType.LLM_OPERATION_CONTEXT, name[1], 'apktool', source,
                                   role=SignalRole.REFERENCE, scope=SignalScope.VALUE,
                                   source_kind='serialized_field_reference',
                                   components=('business_fact=llm_serialized_field',)))
    diagnostics['signals'] = diagnostics.get('signals', 0) + len(facts)
    return facts
