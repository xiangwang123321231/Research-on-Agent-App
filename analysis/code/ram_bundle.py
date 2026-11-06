"""Validate Metro indexed RAM bundles; never treat their index as JS text.

Format: https://github.com/facebook/metro/blob/main/packages/metro/src/lib/RamBundleParser.js
Offsets are relative to the start of the startup code, after the module table.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct

MAGIC = 0xFB0BD1E5


@dataclass(frozen=True)
class RamModule:
    module_id: str
    byte_offset: int
    text: str


def is_ram_bundle(data: bytes) -> bool:
    return len(data) >= 4 and struct.unpack_from('<I', data)[0] == MAGIC


def ram_modules(data: bytes) -> list[RamModule]:
    if not is_ram_bundle(data) or len(data) < 12:
        raise ValueError('invalid_ram_header')
    _, count, startup_size = struct.unpack_from('<3I', data)
    base = 12 + count * 8
    if base > len(data) or startup_size > len(data)-base:
        raise ValueError('invalid_ram_table')
    spans = [('startup', base, startup_size)]
    for index in range(count):
        offset, size = struct.unpack_from('<2I', data, 12 + index * 8)
        if not size:
            continue
        if offset < startup_size or offset > len(data)-base or size > len(data)-base-offset:
            raise ValueError('invalid_ram_module_bounds')
        spans.append((str(index), base+offset, size))
    # Shared module groups may have identical spans; partial overlaps are invalid.
    unique = sorted({(offset, size) for _, offset, size in spans if size})
    for (offset, size), (next_offset, _) in zip(unique, unique[1:]):
        if offset+size > next_offset:
            raise ValueError('overlapping_ram_modules')
    modules = []
    for name, offset, size in spans:
        if not size:
            continue
        raw = data[offset:offset+size]
        if raw[-1:] != b'\0' or b'\0' in raw[:-1]:
            raise ValueError('invalid_ram_module_terminator')
        modules.append(RamModule(name, offset, raw[:-1].decode('utf-8')))
    return modules
