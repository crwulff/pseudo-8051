"""
colorize.py — IDA color-tag syntax colorizer for 8051 pseudocode lines.

Uses IDA's ida_lines SCOLOR_* constants embedded in strings.
Falls back to returning text unchanged when not running inside IDA.
"""

import re

def _to_char(val) -> str:
    """Convert an IDA SCOLOR constant to a single character.

    IDA 9 exposes SCOLOR_* as single-character strings; older versions
    expose them as integers.  Handle both.
    """
    return val if isinstance(val, str) else chr(val)


try:
    import ida_lines as _il

    _ON  = _to_char(_il.SCOLOR_ON)
    _OFF = _to_char(_il.SCOLOR_OFF)

    def _col(code, text: str) -> str:
        c = _to_char(code)
        return _ON + c + text + _OFF + c

    _C_KEYWORD  = _il.SCOLOR_KEYWORD
    _C_NUMBER   = _il.SCOLOR_NUMBER
    _C_COMMENT  = _il.SCOLOR_RPTCMT
    _C_FUNCNAME = _il.SCOLOR_CNAME
    _C_MEMREF   = _il.SCOLOR_DREF
    _C_TYPE     = getattr(_il, 'SCOLOR_TYPE', _il.SCOLOR_KEYWORD)

    _HAVE_IDA = True

except ImportError:
    def _col(code, text: str) -> str:  # type: ignore[misc]
        return text
    _HAVE_IDA = False
    _C_KEYWORD = _C_NUMBER = _C_COMMENT = _C_FUNCNAME = _C_MEMREF = _C_TYPE = 0


_KEYWORDS = frozenset({
    'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'default',
    'break', 'continue', 'return', 'void', 'goto',
})

_TYPES = frozenset({
    'uint8_t',  'int8_t',
    'uint16_t', 'int16_t',
    'uint32_t', 'int32_t',
    'char', 'int', 'unsigned', 'signed', 'bool', 'size_t',
})

# Identifiers that render as memory-region prefixes: XRAM[...], IRAM[...], CROM[...]
_MEMREFS = frozenset({'XRAM', 'IRAM', 'CROM'})

# Tokenizer — alternatives tried left-to-right, first match wins.
# 'call' matches an identifier immediately followed by '(' (lookahead).
_TOKEN_RE = re.compile(
    r'(?P<comment>//.*$)'
    r'|(?P<hexnum>0[xX][0-9a-fA-F]+)'
    r'|(?P<decnum>\b\d+\b)'
    r'|(?P<call>[A-Za-z_][A-Za-z0-9_]*)(?=\s*\()'
    r'|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)'
    r'|(?P<other>.)',
    re.MULTILINE,
)


def colorize(text: str) -> str:
    """Apply IDA color tags to a single pseudocode line.

    Returns text unchanged when not running inside IDA or if ida_lines
    is unavailable.
    """
    if not _HAVE_IDA:
        return text

    parts: list = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        val  = m.group()

        if kind == 'comment':
            parts.append(_col(_C_COMMENT, val))

        elif kind in ('hexnum', 'decnum'):
            parts.append(_col(_C_NUMBER, val))

        elif kind == 'call':
            # Identifier directly before '(' — keyword, type, or function name
            if val in _KEYWORDS:
                parts.append(_col(_C_KEYWORD, val))
            elif val in _TYPES:
                parts.append(_col(_C_TYPE, val))
            elif val in _MEMREFS:
                parts.append(_col(_C_MEMREF, val))
            else:
                parts.append(_col(_C_FUNCNAME, val))

        elif kind == 'ident':
            if val in _KEYWORDS:
                parts.append(_col(_C_KEYWORD, val))
            elif val in _TYPES:
                parts.append(_col(_C_TYPE, val))
            elif val in _MEMREFS:
                parts.append(_col(_C_MEMREF, val))
            else:
                parts.append(val)

        else:
            parts.append(val)

    return ''.join(parts)
