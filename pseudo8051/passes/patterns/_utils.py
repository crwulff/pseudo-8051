"""
passes/patterns/_utils.py — Shared types and helpers for pattern modules.

VarInfo, type introspection, register-pair text substitution,
and constant formatting are here so each pattern file can import
what it needs without pulling in the full typesimplify module.

Phase 6 adds Expr-tree walking functions alongside the legacy text functions.
"""

import re
import sys
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,  # noqa: F401
                               ExprStmt, ReturnStmt, IfGoto, IfNode)
from pseudo8051.ir.expr import (  # noqa: F401
    Expr, Reg, Regs, Name, XRAMRef, RegGroup, ArrayRef,
)


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
                 'xram_sym', 'is_param', 'xram_addr')

    def __init__(self, name: str, type_str: str,
                 full_regs: Tuple[str, ...],
                 active_regs: Optional[FrozenSet[str]] = None,
                 xram_sym: str = "",
                 is_param: bool = False,
                 xram_addr: int = 0):
        self.name        = name
        self.type        = type_str
        self.full_regs   = full_regs
        self.active_regs = (active_regs if active_regs is not None
                            else frozenset(full_regs))
        self.xram_sym    = xram_sym
        self.is_param    = is_param
        self.xram_addr   = xram_addr

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
                         xram_addr=self.xram_addr)

    def __repr__(self) -> str:
        return (f"TypeGroup({self.name!r}, {self.type!r}, "
                f"full={self.full_regs!r}, active={tuple(self.active_regs)!r})")


# ── VarInfo ───────────────────────────────────────────────────────────────────

class VarInfo:
    """One named variable occupying one or more consecutive registers or an XRAM address."""

    def __init__(self, name: str, type_str: str, regs: Tuple[str, ...],
                 xram_sym: str = "", is_byte_field: bool = False,
                 xram_addr: int = 0, is_param: bool = False,
                 array_size: int = 0, elem_type: str = ""):
        self.name          = name
        self.type          = type_str
        self.regs          = regs              # high → low order, e.g. ('R6', 'R7'); () for XRAM locals
        self.pair_name     = "".join(regs)     # e.g. 'R6R7'; '' for XRAM locals
        self.xram_sym      = xram_sym          # XRAM base address symbol, e.g. 'EXT_DC8A'; '' for reg vars
        self.is_byte_field = is_byte_field     # True for per-byte entries of multi-byte XRAM locals
        self.xram_addr     = xram_addr         # raw integer XRAM address (0 for register vars)
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

def _byte_names(var_name: str, n: int) -> List[str]:
    """Per-byte field names for a multi-byte XRAM local.

    Returns ['var.hi', 'var.lo'] for 2-byte vars, ['var.b0'...'var.bn'] for larger.
    Names are ordered high-byte first (big-endian, matching 8051 XRAM layout).
    """
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
    return _byte_names(vinfo.name, len(vinfo.regs))[idx]


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


# ── Expr tree-walk helpers (Phase 6) ─────────────────────────────────────────

def _walk_expr(expr: Expr, fn: Callable[[Expr], Expr]) -> Expr:
    """
    Post-order tree walk over an Expr.

    fn receives each node (leaves first, composites after their children are
    updated) and may return a replacement or the same node.

    Dispatch is via Expr.children() / Expr.rebuild(), so new Expr subclasses
    are automatically handled as long as they implement those two methods.
    """
    children = expr.children()
    new_children = [_walk_expr(c, fn) for c in children]
    if any(nc is not oc for nc, oc in zip(new_children, children)):
        expr = expr.rebuild(new_children)
    return fn(expr)


def _subst_pairs_in_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Attach alias to RegGroup/Reg nodes for multi-reg VarInfo entries.

    Previously replaced these nodes with Name(vinfo.name); now sets alias so
    register identity (regs/name) is preserved for conflict detection while
    render() still returns the human-readable variable name.
    """
    # Build single-reg lookup: short key → display name (skip XRAM locals, skip params)
    single_map: Dict[str, str] = {}
    for key, vinfo in reg_map.items():
        if not isinstance(vinfo, VarInfo) or vinfo.xram_sym or vinfo.is_param:
            continue
        if len(vinfo.regs) == 1 and len(key) <= 2:
            # Single-register consolidated entry (e.g. "R3" → "_src_type", "A" → "retval1")
            single_map[key] = vinfo.name

    # Build multi-reg lookup: frozenset of individual regs → display name
    # Also: concatenated-pair-name → display name (for legacy Reg("R6R7") nodes)
    seen_vi_ids: set = set()
    multi_entries: list = []   # [(frozenset_of_regs, name), ...]
    pair_name_map: Dict[str, str] = {}   # "R6R7" → "val" (legacy single-node pairs)
    for vinfo in reg_map.values():
        if not isinstance(vinfo, VarInfo) or vinfo.xram_sym or id(vinfo) in seen_vi_ids:
            continue
        if len(vinfo.regs) > 1:
            seen_vi_ids.add(id(vinfo))
            multi_entries.append((frozenset(vinfo.regs), vinfo.name))
            pair_name_map["".join(vinfo.regs)] = vinfo.name

    if not single_map and not multi_entries:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, Regs):
            # Only alias on the first pass — once an alias is set it must not be
            # overwritten by a later _subst_pairs_in_expr call that sees updated
            # reg_map entries (e.g. retval VarInfo added after the arg alias).
            if not e.alias:
                if e.is_single:
                    if e.name in single_map:
                        return RegGroup(e.names, e.brace, alias=single_map[e.name])
                    # Legacy: Reg("R6R7") — single node whose name is a pair concat
                    if e.name in pair_name_map:
                        return RegGroup(e.names, e.brace, alias=pair_name_map[e.name])
                else:
                    reg_set = frozenset(e.names)
                    for entry_regs, entry_name in multi_entries:
                        if reg_set == entry_regs:
                            return RegGroup(e.names, e.brace, alias=entry_name)
        elif isinstance(e, Name):
            # Name("R7") style register placeholders in call args
            if e.name in single_map:
                return Reg(e.name, alias=single_map[e.name])
        return e

    return _walk_expr(expr, _fn)


def _subst_single_regs_in_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Attach alias to Reg(rx) nodes for register-backed single-register entries.

    Covers is_param entries and any other register-backed VarInfo (e.g. struct
    return fields) that is_param doesn't mark.  For multi-byte variables, appends
    .hi/.lo/.bN to identify which byte of the wider variable is being accessed.
    The alias is only set when not already aliased (by _subst_pairs_in_expr).
    """
    singles = {k: _param_byte_name(k, v) for k, v in reg_map.items()
               if _RE_SINGLE_REG.match(k)
               and isinstance(v, VarInfo)
               and not v.xram_sym}
    if not singles:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, Regs) and e.is_single and not e.alias and e.name in singles:
            return Reg(e.name, alias=singles[e.name])
        if isinstance(e, Name) and e.name in singles:   # covers Name("R3") call args
            return Reg(e.name, alias=singles[e.name])
        return e

    return _walk_expr(expr, _fn)


def _subst_xram_in_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Replace XRAMRef(Name(sym)) → Name(local_var_name) or ArrayRef(Name(arr), idx)."""
    from pseudo8051.ir.expr import BinOp as _BinOp, Const as _Const

    sym_map: Dict[str, str] = {}
    # Array base lookups for dynamic indexed access: XRAM[base + idx] → arr[idx]
    arr_sym_map: Dict[str, "VarInfo"] = {}    # xram_sym  → array VarInfo
    arr_addr_map: Dict[int, "VarInfo"] = {}   # xram_addr → array VarInfo

    for vinfo in reg_map.values():
        if not isinstance(vinfo, VarInfo) or not vinfo.xram_sym:
            continue
        if vinfo.is_byte_field:
            sym_map[vinfo.xram_sym] = vinfo.name
        elif vinfo.array_size > 0:
            # Static access to base addr maps to element [0] (also covered by elem entries)
            if vinfo.xram_sym not in sym_map:
                sym_map[vinfo.xram_sym] = f"{vinfo.name}[0]"
            arr_sym_map[vinfo.xram_sym] = vinfo
            arr_addr_map[vinfo.xram_addr] = vinfo
        elif vinfo.xram_sym not in sym_map:
            sym_map[vinfo.xram_sym] = vinfo.name

    if not sym_map and not arr_addr_map:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, XRAMRef):
            inner = e.inner
            inner_text = inner.render()
            if inner_text in sym_map:
                return Name(sym_map[inner_text])
            # Dynamic indexed access: XRAM[base_sym + idx] or XRAM[base_const + idx]
            if isinstance(inner, _BinOp) and inner.op == '+':
                lhs = inner.lhs
                arr_vi = None
                if isinstance(lhs, Name) and lhs.name in arr_sym_map:
                    arr_vi = arr_sym_map[lhs.name]
                elif isinstance(lhs, _Const) and lhs.value in arr_addr_map:
                    arr_vi = arr_addr_map[lhs.value]
                if arr_vi is not None and _type_bytes(arr_vi.elem_type) == 1:
                    return ArrayRef(Name(arr_vi.name), inner.rhs)
        return e

    return _walk_expr(expr, _fn)


def _subst_all_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Apply all three substitutions to an Expr tree."""
    expr = _subst_xram_in_expr(expr, reg_map)
    expr = _subst_pairs_in_expr(expr, reg_map)
    expr = _subst_single_regs_in_expr(expr, reg_map)
    return expr


def _apply_expr_subst_to_node(node: HIRNode,
                               expr_fn: Callable[[Expr], Expr]) -> HIRNode:
    """Apply expr_fn to every read-position Expr in node.

    Handles Assign/CompoundAssign (rhs), ExprStmt (expr), ReturnStmt (value).
    LHS is never transformed. Returns node unchanged if nothing changed.
    """
    if isinstance(node, Assign):
        new_rhs = expr_fn(node.rhs)
        return Assign(node.ea, node.lhs, new_rhs) if new_rhs is not node.rhs else node
    if isinstance(node, CompoundAssign):
        new_rhs = expr_fn(node.rhs)
        return CompoundAssign(node.ea, node.lhs, node.op, new_rhs) if new_rhs is not node.rhs else node
    if isinstance(node, ExprStmt):
        new_expr = expr_fn(node.expr)
        return ExprStmt(node.ea, new_expr) if new_expr is not node.expr else node
    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = expr_fn(node.value)
        return ReturnStmt(node.ea, new_val) if new_val is not node.value else node
    return node


def _replace_pairs_in_node(node: HIRNode,
                            reg_map: Dict[str, "VarInfo"]) -> HIRNode:
    """Apply pair substitution to an Assign / ExprStmt / ReturnStmt rhs/expr."""
    return _apply_expr_subst_to_node(
        node,
        lambda e: _subst_pairs_in_expr(e, reg_map),
    )


def _replace_single_regs_in_node(node: HIRNode,
                                  reg_map: Dict[str, "VarInfo"]) -> HIRNode:
    """Apply single-reg param substitution to RHS/value/expr of a node."""
    return _apply_expr_subst_to_node(
        node,
        lambda e: _subst_single_regs_in_expr(e, reg_map),
    )


def _count_reg_uses_in_node(r: str, node: HIRNode) -> int:
    """Count read-position occurrences of Reg/Name(r) in node.

    Also counts multi-component Regs nodes that contain r as one of their
    individual register names, so that RegGroup(('R4','R5','R6','R7')) is
    counted as a use of 'R4', 'R5', 'R6', and 'R7'.
    """
    count = [0]

    def _fn(e: Expr) -> Expr:
        if e == Reg(r):
            count[0] += 1
        elif isinstance(e, Name) and e.name == r:
            count[0] += 1
        elif isinstance(e, Regs) and not e.is_single and r in e.names:
            count[0] += 1
        return e

    if isinstance(node, Assign):
        _walk_expr(node.rhs, _fn)
        # Also count in compound LHS (e.g. XRAMRef inner)
        if not isinstance(node.lhs, Regs):
            _walk_expr(node.lhs, _fn)
    elif isinstance(node, CompoundAssign):
        _walk_expr(node.rhs, _fn)
    elif isinstance(node, ExprStmt):
        _walk_expr(node.expr, _fn)
    elif isinstance(node, ReturnStmt) and node.value is not None:
        _walk_expr(node.value, _fn)
    elif isinstance(node, IfGoto):
        _walk_expr(node.cond, _fn)
    elif isinstance(node, IfNode):
        _walk_expr(node.condition, _fn)
    return count[0]


def _subst_reg_in_node(node: HIRNode, r: str,
                        replacement: Expr) -> Optional[HIRNode]:
    """
    Replace Reg/Name(r) → replacement in read positions of node.
    Returns updated node, or None if r does not appear.
    """
    def _fn(e: Expr) -> Expr:
        if e == Reg(r):
            return replacement
        if isinstance(e, Name) and e.name == r:
            return replacement
        return e

    def _out(new_node: HIRNode) -> HIRNode:
        new_node.ann = node.ann
        return new_node

    if isinstance(node, Assign):
        new_rhs = _walk_expr(node.rhs, _fn)
        new_lhs = node.lhs
        if not isinstance(node.lhs, Regs):
            new_lhs = _walk_expr(node.lhs, _fn)
        if new_rhs is node.rhs and new_lhs is node.lhs:
            return None
        if isinstance(node, TypedAssign):
            return _out(TypedAssign(node.ea, node.type_str, new_lhs, new_rhs))
        return _out(Assign(node.ea, new_lhs, new_rhs))

    if isinstance(node, CompoundAssign):
        new_rhs = _walk_expr(node.rhs, _fn)
        if new_rhs is node.rhs:
            return None
        return _out(CompoundAssign(node.ea, node.lhs, node.op, new_rhs))

    if isinstance(node, ExprStmt):
        new_expr = _walk_expr(node.expr, _fn)
        if new_expr is node.expr:
            return None
        return _out(ExprStmt(node.ea, new_expr))

    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _walk_expr(node.value, _fn)
        if new_val is node.value:
            return None
        return _out(ReturnStmt(node.ea, new_val))

    if isinstance(node, IfGoto):
        new_cond = _walk_expr(node.cond, _fn)
        if new_cond is node.cond:
            return None
        return _out(IfGoto(node.ea, new_cond, node.label))

    if isinstance(node, IfNode):
        new_cond = _walk_expr(node.condition, _fn)
        if new_cond is node.condition:
            return None
        new_node = IfNode(node.ea, new_cond, node.then_nodes, node.else_nodes)
        new_node.ann = node.ann
        return new_node

    return None


def _fold_into_node(node: HIRNode, name_expr: Expr,
                    replacement: Expr,
                    reg_map: Dict[str, "VarInfo"]) -> Optional[HIRNode]:
    """
    Try to substitute name_expr → replacement in the expression position of node.

    For Assign: substitutes in rhs.
    For ReturnStmt/ExprStmt: substitutes in value/expr.
    Returns None if name_expr does not appear in an expression position.
    """
    name_str = name_expr.render() if isinstance(name_expr, Expr) else str(name_expr)
    repl_str = replacement.render() if isinstance(replacement, Expr) else str(replacement)

    def _subst_fn(e: Expr) -> Expr:
        if e == name_expr:
            return replacement
        if isinstance(e, (Name, Regs)) and e.render() == name_str:
            return replacement
        return e

    def _finalize(new_node: HIRNode) -> HIRNode:
        result = _replace_pairs_in_node(new_node, reg_map)
        result.ann = node.ann
        return result

    if isinstance(node, Assign):
        new_rhs = _walk_expr(node.rhs, _subst_fn)
        if new_rhs is node.rhs:
            return None  # not found in rhs
        return _finalize(Assign(node.ea, node.lhs, new_rhs))

    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _walk_expr(node.value, _subst_fn)
        if new_val is node.value:
            return None
        return _finalize(ReturnStmt(node.ea, new_val))

    if isinstance(node, ExprStmt):
        new_expr = _walk_expr(node.expr, _subst_fn)
        if new_expr is node.expr:
            return None
        return _finalize(ExprStmt(node.ea, new_expr))

    return None
