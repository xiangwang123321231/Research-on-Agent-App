"""Format facts shared by extraction; no research conclusions or labels."""
from __future__ import annotations
import json
import struct
import re


def font_format(data: bytes) -> str | None:
    """Recognize font headers independently of resource names, with bounds checks."""
    if len(data) >= 12 and data[:4] in {b'\x00\x01\x00\x00', b'OTTO', b'true', b'typ1'}:
        count = int.from_bytes(data[4:6], 'big')
        if 1 <= count <= 256 and len(data) >= 12 + 16 * count:
            entries = [struct.unpack_from('>4sIII', data, 12 + 16 * i) for i in range(count)]
            if all(offset <= len(data) and size <= len(data) - offset
                   for _, _, offset, size in entries):
                return 'sfnt'
    if len(data) >= 44 and data[:4] in {b'wOFF', b'wOF2'}:
        size = int.from_bytes(data[8:12], 'big')
        count = int.from_bytes(data[12:14], 'big')
        if size == len(data) and 1 <= count <= 256:
            return 'woff'
    if len(data) >= 16 and data[:4] == b'ttcf':
        count = int.from_bytes(data[8:12], 'big')
        if 1 <= count <= 256 and len(data) >= 12 + 4 * count:
            if all(12 <= int.from_bytes(data[12+4*i:16+4*i], 'big') <= len(data)-12
                   for i in range(count)):
                return 'font_collection'
    return None


_LICENSE_PHRASES = (
    'redistribution and use in source and binary forms',
    'permission is hereby granted, free of charge',
    'this software is provided',
    'limitations under the license',
    'licensed under the apache license',
    'redistributions of source code must retain',
)


def is_license_document(text: str, source: str = '') -> bool:
    """Whole-document attribution, never a comment header on executable source."""
    lower_source = source.lower().split('?', 1)[0]
    if lower_source.endswith(('.java', '.kt', '.smali', '.js', '.mjs', '.cjs', '.jsx', '.tsx', '.json')):
        return False
    # Executable declarations veto whole-document suppression even after rename.
    if re.search(r'(?m)^\s*(?:package\s+[\w.]|import\s|(?:(?:export|public|private|protected|final|abstract|static)\s+)*(?:class|function|const|let|var)\s+\w|\.class\s)', text):
        return False
    if re.search(r'<script\b|<iframe\b|\b(?:fetch|loadUrl|generateContent)\s*\(', text, re.I):
        return False
    lowered = text.lower()
    return ('copyright' in lowered and
            sum(phrase in lowered for phrase in _LICENSE_PHRASES) >= 2)


def model_header_facts(data: bytes) -> dict[str, object]:
    """Inspect a bounded prefix. A format is not a language-model identity."""
    if data.startswith(b"GGUF"):
        if len(data) < 24:
            return {"format": "gguf", "status": "truncated"}
        version, tensors, entries = struct.unpack_from("<IQQ", data, 4)
        valid = version in {2, 3} and tensors < 10_000_000 and entries < 100_000
        return {"format": "gguf", "status": "header_valid" if valid else "invalid",
                "version": version, "tensor_count": tensors, "metadata_count": entries}
    if len(data) >= 8:
        length = struct.unpack_from("<Q", data)[0]
        if 2 <= length <= 1024 * 1024 and len(data) >= 8 + length:
            try:
                header = json.loads(data[8:8 + length].decode("utf-8"))
                if isinstance(header, dict) and any(
                    isinstance(v, dict) and {"dtype", "shape", "data_offsets"}.issubset(v)
                    for k, v in header.items() if k != "__metadata__"
                ):
                    return {"format": "safetensors", "status": "header_valid"}
            except (UnicodeError, ValueError, RecursionError):
                pass
    return {"format": "unknown", "status": "unrecognized"}
