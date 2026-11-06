"""Decode quoted Smali string constants without evaluating source code.

Escape grammar: google/smali smaliLexer.jflex, STRING state.
Unlike JSON, Smali accepts an escaped apostrophe and rejects an escaped slash.
"""
from __future__ import annotations

_ESCAPES = {
    "b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r",
    "'": "'", '"': '"', "\\": "\\",
}
_HEX = frozenset("0123456789abcdefABCDEF")


def decode_smali_string(token: str) -> str:
    """Decode one complete double-quoted token, or raise a non-secret error.

    Valid UTF-16 pairs are combined. Unpaired surrogates remain unknown rather
    than producing an unencodable value or a replacement-character URL.
    """
    if len(token) < 2 or token[0] != '"' or token[-1] != '"':
        raise ValueError("invalid_smali_string_quotes")
    result = []
    index, end = 1, len(token) - 1
    while index < end:
        char = token[index]
        index += 1
        if char in {'"', "\r", "\n"}:
            raise ValueError("unescaped_smali_string_delimiter")
        if char != "\\":
            result.append(char)
            continue
        if index >= end:
            raise ValueError("incomplete_smali_escape")
        escape = token[index]
        index += 1
        if escape in _ESCAPES:
            result.append(_ESCAPES[escape])
        elif escape == "u":
            digits = token[index:index + 4]
            if index + 4 > end or len(digits) != 4 or not set(digits) <= _HEX:
                raise ValueError("invalid_smali_unicode_escape")
            result.append(chr(int(digits, 16)))
            index += 4
        else:
            raise ValueError("unsupported_smali_escape")
    value = "".join(result)
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        try:
            value = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
        except UnicodeError:
            raise ValueError("unpaired_utf16_surrogate") from None
    return value
