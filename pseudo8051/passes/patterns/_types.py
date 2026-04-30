"""
passes/patterns/_types.py — Type classes and type-introspection helpers.

VarInfo, TypeGroup, type-size/sign helpers, XRAM byte-field helpers,
legacy text-based substitution, and constant formatting.
"""

import re
import sys
from typing import Dict, FrozenSet, List, Optional, Tuple


# ── Type helpers ──────────────────────────────────────────────────────────────

_ARRAY_TYPE_RE = re.compile(r'^(\w+)\[(\d+)\]$')


def _parse_array_type(type_str: str) -> Optional[Tuple[str, int]]:
    """Parse an array type string: 'uint8_t[6]' → ('uint8_t', 6), or None."""
    m = _ARRAY_TYPE_RE.match(type_str.strip())
    if m:
        return m.group(1), int(m.group(2))
    return None


def _type_bytes(t: str) -> int:
    if t in ("bool", "uint8_t",  "int8_t",  "char"):  return 1
    if t in ("uint16_t", "int16_t"):                   return 2
    if t in ("uint32_t", "int32_t"):                   return 4
    arr = _parse_array_type(t)
    if arr is not None:
        elem_type, count = arr
        eb = _type_bytes(elem_type)
        return eb * count if eb > 0 else 0
    try:
        from pseudo8051.prototypes import struct_size as _ss
        n = _ss(t)
        if n > 0:
            return n
    except ImportError:
        pass
    # Fallback: ask IDA's type system (handles typedefs, enums, etc.)
    try:
        import ida_typeinf as _idt
        _tif = _idt.tinfo_t()
        if _tif.get_named_type(None, t):
            sz = _tif.get_size()
            _BADSIZE = getattr(_idt, 'BADSIZE', 0xFFFFFFFF)
            if sz and sz != _BADSIZE:
                return int(sz)
    except Exception:
        pass
    return 0


def _is_signed(t: str) -> bool:
    return t in ("int8_t", "int16_t", "int32_t")


# ── TypeGroup ────────────────────────────────────────────────────────────────

class TypeGroup:
    """A named, typed register group with ordered bytes and a live active subset.

    ``full_regs`` is the canonical ordered tuple of all registers belonging to
    this variable (high-byte first, e.g. ``('R6', 'R7')`` for int16_t).
    ``active_regs`` is the subset of ``full_regs`` that have not yet been
    killed (overwritten) at the current program point.

    TypeGroup is immutable: mutation returns a new instance.
    When ``active_regs`` becomes empty the group should be discarded.
    """

    __slots__ = ('name', 'type', 'full_regs', 'active_regs',
                 'xram_sym', 'is_param', 'xram_addr', 'is_byte_field')

    def __init__(self, name: str, type_str: str,
                 full_regs: Tuple[str, ...],
                 active_regs: Optional[FrozenSet[str]] = None,
                 xram_sym: str = "",
                 is_param: bool = False,
                 xram_addr: int = 0,
                 is_byte_field: bool = False):
        self.name         = name
        self.type         = type_str
        self.full_regs    = full_regs
        self.active_regs  = (active_regs if active_regs is not None
                             else frozenset(full_regs))
        self.xram_sym     = xram_sym
        self.is_param     = is_param
        self.xram_addr    = xram_addr
        self.is_byte_field = is_byte_field

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def pair_name(self) -> str:
        """Concatenated full register key, e.g. 'R6R7'."""
        return "".join(self.full_regs)

    @property
    def is_complete(self) -> bool:
        """True when all registers are still live (active_regs == full_regs)."""
        return self.active_regs == frozenset(self.full_regs)

    def byte_of(self, reg: str) -> int:
        """Byte index from MSB (0 = full_regs[0] = most-significant byte)."""
        return self.full_regs.index(reg)

    def covers(self, regs) -> bool:
        """True when every register in *regs* is in active_regs."""
        return all(r in self.active_regs for r in regs)

    # ── Mutation helpers (return new instances) ───────────────────────────────

    def killed(self, reg: str) -> Optional['TypeGroup']:
        """Return a new TypeGroup with *reg* removed from active_regs.

        Returns ``None`` when active_regs would become empty (group is dead).
        """
        new_active = self.active_regs - {reg}
        if not new_active:
            return None
        return TypeGroup(self.name, self.type, self.full_regs,
                         active_regs=new_active,
                         xram_sym=self.xram_sym,
                         is_param=self.is_param,
                         xram_addr=self.xram_addr,
                         is_byte_field=self.is_byte_field)

    def __repr__(self) -> str:
        return (f"TypeGroup({self.name!r}, {self.type!r}, "
                f"full={self.full_regs!r}, active={tuple(self.active_regs)!r})")


# ── VarInfo ───────────────────────────────────────────────────────────────────

class VarInfo:
    """One named variable occupying one or more consecutive registers or an XRAM address."""

    def __init__(self, name: str, type_str: str, regs: Tuple[str, ...],
                 xram_sym: str = "", is_byte_field: bool = False,
                 xram_addr: int = 0, is_param: bool = False,
                 array_size: int = 0, elem_type: str = "",
                 iram_addr: int = 0):
        self.name          = name
        self.type          = type_str
        self.regs          = regs              # high → low order, e.g. ('R6', 'R7'); () for XRAM/IRAM locals
        self.pair_name     = "".join(regs)     # e.g. 'R6R7'; '' for XRAM/IRAM locals
        self.xram_sym      = xram_sym          # XRAM base address symbol, e.g. 'EXT_DC8A'; '' for reg vars
        self.is_byte_field = is_byte_field     # True for per-byte entries of multi-byte locals
        self.xram_addr     = xram_addr         # raw integer XRAM address (0 for non-XRAM vars)
        self.iram_addr     = iram_addr         # raw integer IRAM address (0 for non-IRAM vars)
        self.is_param      = is_param          # True only for params from the current function's proto
        self.array_size    = array_size        # >0 for array locals (e.g. uint8_t[6] → 6)
        self.elem_type     = elem_type         # element type for arrays (e.g. 'uint8_t')

    @property
    def hi(self) -> Optional[str]:
        """Most-significant register, or None for single-byte vars."""
        return self.regs[0] if len(self.regs) >= 2 else None

    @property
    def lo(self) -> Optional[str]:
        """Least-significant register."""
        return self.regs[-1] if self.regs else None


# ── XRAM byte-field helpers ───────────────────────────────────────────────────

def _byte_names(var_name: str, n: int, type_str: str = "") -> List[str]:
    """Per-byte field names for a multi-byte XRAM local or register.

    When type_str is a known struct, uses field names recursively.
    Otherwise: ['var.hi', 'var.lo'] for 2-byte vars, ['var.b0'..'var.bn'] for larger.
    Names are ordered high-byte first (big-endian, matching 8051 XRAM layout).
    """
    if type_str:
        try:
            from pseudo8051.prototypes import get_struct
            sd = get_struct(type_str)
            if sd is not None:
                result: List[str] = []
                for field in sd.fields:
                    field_n = _type_bytes(field.type)
                    if field_n == 1:
                        result.append(f"{var_name}.{field.name}")
                    elif field_n > 1:
                        result.extend(_byte_names(f"{var_name}.{field.name}",
                                                  field_n, field.type))
                if len(result) == n:
                    return result
        except ImportError:
            pass
    if n == 2:
        return [f"{var_name}.hi", f"{var_name}.lo"]
    return [f"{var_name}.b{i}" for i in range(n)]


# ── Legacy text-based substitution (kept until Phase 8 cleanup) ──────────────

def _replace_xram_syms(text: str, reg_map: Dict[str, "VarInfo"]) -> str:
    """Replace XRAM[sym] and *sym references with declared XRAM local variable names."""
    sym_map: Dict[str, str] = {}
    for vinfo in reg_map.values():
        if not isinstance(vinfo, VarInfo) or not vinfo.xram_sym:
            continue
        if vinfo.is_byte_field:
            sym_map[vinfo.xram_sym] = vinfo.name
        elif vinfo.xram_sym not in sym_map:
            sym_map[vinfo.xram_sym] = vinfo.name

    if not sym_map:
        return text

    def _repl_xram(m: "re.Match") -> str:
        return sym_map.get(m.group(1).strip(), m.group(0))

    def _repl_deref(m: "re.Match") -> str:
        return sym_map.get(m.group(1), m.group(0))

    text = re.sub(r'XRAM\[([^\]]+)\]', _repl_xram, text)
    text = re.sub(r'\*([A-Za-z_]\w*)', _repl_deref, text)
    return text


def _replace_pairs(text: str, reg_map: Dict[str, VarInfo]) -> str:
    """Replace register-pair tokens (e.g. R6R7) with the variable name.

    Derives the text token from VarInfo.regs (concatenated) so no pair key
    needs to exist in reg_map.  Also handles single-register consolidated
    XRAM local tokens (len==2 key, regs=(r,)) via the legacy key path.
    """
    # Collect substitutions: (token_str, var_name) — deduped by VarInfo identity.
    seen_vi_ids: set = set()
    subs: list = []
    for key, vinfo in reg_map.items():
        if not isinstance(vinfo, VarInfo) or vinfo.xram_sym:
            continue
        if id(vinfo) not in seen_vi_ids and len(vinfo.regs) > 1:
            seen_vi_ids.add(id(vinfo))
            subs.append(("".join(vinfo.regs), vinfo.name))
        # Legacy: single-reg consolidated token stored under a len-2 key
        if (len(key) == 2 and len(vinfo.regs) == 1 and not vinfo.is_param
                and (key, vinfo.name) not in subs):
            subs.append((key, vinfo.name))
    # Longest token first to avoid partial matches
    for token, name in sorted(subs, key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r"\b" + re.escape(token) + r"\b", name, text)
    return text


_RE_SINGLE_REG = re.compile(r'^R[0-7]$')


def _param_byte_name(reg: str, vinfo: "VarInfo") -> str:
    """Return the substitution name for a single register that is part of a
    multi-byte parameter, appending the appropriate .hi/.lo/.bN suffix."""
    if len(vinfo.regs) <= 1:
        return vinfo.name
    try:
        idx = list(vinfo.regs).index(reg)
    except ValueError:
        return vinfo.name
    return _byte_names(vinfo.name, len(vinfo.regs), vinfo.type)[idx]


def _replace_single_regs(text: str, reg_map: Dict[str, VarInfo]) -> str:
    """Substitute single-register variable names in read (RHS) positions.
    For multi-byte variables, appends .hi/.lo/.bN to identify the byte accessed."""
    singles = [(k, _param_byte_name(k, v)) for k, v in reg_map.items()
               if _RE_SINGLE_REG.match(k)
               and isinstance(v, VarInfo)
               and not v.xram_sym]
    if not singles:
        return text

    eq_pos = text.find(" = ")
    if eq_pos > 0:
        rhs = text[eq_pos + 3:]
        for reg, name in singles:
            rhs = re.sub(r'\b' + re.escape(reg) + r'\b', name, rhs)
        return text[:eq_pos + 3] + rhs
    else:
        for reg, name in singles:
            text = re.sub(r'\b' + re.escape(reg) + r'\b', name, text)
        return text


# ── Constant formatting ───────────────────────────────────────────────────────

def _parse_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _const_str(value: int, type_str: str) -> str:
    _c = sys.modules.get("pseudo8051.constants")
    if _c is None or not getattr(_c, "USE_HEX", True):
        return str(value)
    size = _type_bytes(type_str)
    if size >= 4: return f"0x{value:08x}"
    if size == 2: return f"0x{value:04x}"
    return hex(value) if value > 9 else str(value)
