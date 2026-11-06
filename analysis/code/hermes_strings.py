"""Read HBC string values without treating compressed storage as source.

Layout reference: facebook/hermes include/hermes/BCGen/HBC/BytecodeFileFormat.h
and lib/BCGen/HBC/BytecodeDataProvider.cpp. This reader does not decode calls.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterator


HBC_MAGIC = 0x1F1903C103BC1FC6
HBC_DELTA_MAGIC = (~HBC_MAGIC) & 0xffffffffffffffff
# These released layouts have a 128-byte header and 16-byte function headers.
# Unknown versions must be explicit coverage gaps, never guessed text files.
SUPPORTED_HBC_VERSIONS = frozenset(range(84, 97))
MAX_HBC_BYTES = 64 * 1024 * 1024
MAX_HBC_STRINGS = 1_000_000
MAX_LITERAL_CHARS = 16 * 1024


class HermesFormatError(ValueError):
    pass


class UnsupportedHermesVersion(HermesFormatError):
    pass


class UnsupportedHermesForm(HermesFormatError):
    pass


def is_hermes_bytecode(data: bytes) -> bool:
    return len(data) >= 8 and struct.unpack_from('<Q', data)[0] in {HBC_MAGIC, HBC_DELTA_MAGIC}


@dataclass(frozen=True)
class HermesLiteral:
    index: int
    value: str


def hermes_string_literals(data: bytes) -> Iterator[HermesLiteral]:
    """Validate the complete table before yielding bounded, separate literals.

    Shared prefix/suffix storage is addressed by each entry's offset/length.
    Overflow lengths and UTF-16 lengths are interpreted independently. No
    table order, adjacency or storage overlap establishes a program relation.
    """
    if not is_hermes_bytecode(data) or not 148 <= len(data) <= MAX_HBC_BYTES:
        raise HermesFormatError('invalid HBC size/header')
    if struct.unpack_from('<Q', data)[0] == HBC_DELTA_MAGIC:
        raise UnsupportedHermesForm('delta HBC is not execution bytecode')
    version = struct.unpack_from('<I', data, 8)[0]
    if version not in SUPPORTED_HBC_VERSIONS:
        raise UnsupportedHermesVersion('unsupported HBC version')
    fields = struct.unpack_from('<19I', data, 32)
    file_size, _, functions, kinds, identifiers, count, overflow_count, storage_size = fields[:8]
    if file_size != len(data) or count > MAX_HBC_STRINGS:
        raise HermesFormatError('invalid HBC counts/size')
    end = file_size - 20
    table = 128 + functions * 16 + kinds * 4 + identifiers * 4
    overflow = table + count * 4
    storage = overflow + overflow_count * 8
    if storage > end or storage_size > end - storage:
        raise HermesFormatError('HBC tables outside file')
    if hashlib.sha1(memoryview(data)[:end]).digest() != data[end:]:
        raise HermesFormatError('HBC footer mismatch')

    def entry(index):
        packed = struct.unpack_from('<I', data, table + index * 4)[0]
        utf16 = bool(packed & 1)
        offset, length = (packed >> 1) & 0x7fffff, packed >> 24
        if length == 255:
            if offset >= overflow_count:
                raise HermesFormatError('invalid HBC overflow index')
            offset, length = struct.unpack_from('<II', data, overflow + offset * 8)
        size = length * (2 if utf16 else 1)
        if offset > storage_size or size > storage_size - offset:
            raise HermesFormatError('HBC string outside storage')
        return utf16, offset, length, size

    for index in range(count):
        entry(index)
    for index in range(count):
        utf16, offset, length, size = entry(index)
        if not length or length > MAX_LITERAL_CHARS:
            continue
        raw = data[storage + offset:storage + offset + size]
        try:
            value = raw.decode('utf-16-le' if utf16 else 'utf-8')
        except UnicodeDecodeError:
            # HBC also permits arbitrary non-UTF8 byte strings. They cannot
            # be interpreted as textual LLM observations without guessing.
            continue
        if '\x00' not in value:
            yield HermesLiteral(index, value)
