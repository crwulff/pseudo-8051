"""
passes/patterns/_utils.py — Shared types and helpers for pattern modules.

VarInfo, type introspection, register-pair text substitution,
and constant formatting are here so each pattern file can import
what it needs without pulling in the full typesimplify module.

Phase 6 adds Expr-tree walking functions alongside the legacy text functions.
"""

import re
import sys
from typing import Callable, Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Statement, Assign, CompoundAssign, ExprStmt, ReturnStmt  # noqa: F401 (re-exported for patterns)
from pseudo8051.ir.expr import (  # noqa: F401
    Expr, Reg, Const, Name, XRAMRef, IRAMRef, CROMRef, RegGroup, BinOp, UnaryOp, Call, Cast,
)


# ── Type helpers ──────────────────────────────────────────────────────────────

def _type_bytes(t: str) -> int:
    if t in ("bool", "uint8_t",  "int8_t",  "char"):  return 1
    if t in ("uint16_t", "int16_t"):                   return 2
    if t in ("uint32_t", "int32_t"):                   return 4
    return 0


def _is_signed(t: str) -> bool:
    return t in ("int8_t", "int16_t", "int32_t")


# ── VarInfo ───────────────────────────────────────────────────────────────────

class VarInfo:
    """One named variable occupying one or more consecutive registers or an XRAM address."""

    def __init__(self, name: str, type_str: str, regs: Tuple[str, ...],
                 xram_sym: str = "", is_byte_field: bool = False,
                 xram_addr: int = 0, is_param: bool = False):
        self.name         = name
        self.type         = type_str
        self.regs         = regs           # high → low order, e.g. ('R6', 'R7'); () for XRAM locals
        self.pair_name    = "".join(regs)  # e.g. 'R6R7'; '' for XRAM locals
        self.xram_sym     = xram_sym       # XRAM base address symbol, e.g. 'EXT_DC8A'; '' for reg vars
        self.is_byte_field = is_byte_field # True for per-byte entries of multi-byte XRAM locals
        self.xram_addr    = xram_addr      # raw integer XRAM address (0 for register vars)
        self.is_param     = is_param       # True only for params from the current function's proto

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
    """Replace register-pair tokens (e.g. R6R7) with the variable name."""
    for key in sorted((k for k in reg_map
                       if len(k) > 2 and isinstance(reg_map[k], VarInfo)),
                      key=len, reverse=True):
        if reg_map[key].xram_sym:
            continue
        text = re.sub(r"\b" + re.escape(key) + r"\b", reg_map[key].name, text)
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
    """Substitute single-register parameter names in read (RHS) positions.
    For multi-byte parameters, appends .hi/.lo/.bN to identify the byte accessed."""
    singles = [(k, _param_byte_name(k, v)) for k, v in reg_map.items()
               if _RE_SINGLE_REG.match(k)
               and isinstance(v, VarInfo)
               and not v.xram_sym
               and v.is_param]
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


# ── Legacy statement folding helper ──────────────────────────────────────────

def _fold_into_stmt(next_text: str, name_to_fold: str,
                    replacement: str,
                    reg_map: Dict[str, "VarInfo"]) -> Optional[str]:
    """
    Try to substitute `name_to_fold` → `replacement` in `next_text`.
    Returns the folded text, or None when name_to_fold is not in an expr position.
    """
    if not next_text:
        return None
    pat = re.compile(r'\b' + re.escape(name_to_fold) + r'\b')
    if not pat.search(next_text):
        return None

    if next_text.startswith("return "):
        return _replace_pairs(pat.sub(replacement, next_text), reg_map)

    eq_pos = next_text.find(" = ")
    if eq_pos > 0:
        rhs = next_text[eq_pos + 3:]
        if pat.search(rhs):
            new_rhs = pat.sub(replacement, rhs)
            return _replace_pairs(next_text[:eq_pos + 3] + new_rhs, reg_map)
        return None

    return _replace_pairs(pat.sub(replacement, next_text), reg_map)


# ── Expr tree-walk helpers (Phase 6) ─────────────────────────────────────────

def _walk_expr(expr: Expr, fn: Callable[[Expr], Expr]) -> Expr:
    """
    Post-order tree walk over an Expr.

    fn receives each node (leaves first, composites after their children are
    updated) and may return a replacement or the same node.
    """
    if isinstance(expr, (Reg, Const, Name)):
        return fn(expr)

    if isinstance(expr, (XRAMRef, IRAMRef, CROMRef)):
        new_inner = _walk_expr(expr.inner, fn)
        if new_inner is not expr.inner:
            expr = type(expr)(new_inner)
        return fn(expr)

    if isinstance(expr, RegGroup):
        # RegGroup holds string names, not Expr children
        return fn(expr)

    if isinstance(expr, BinOp):
        new_lhs = _walk_expr(expr.lhs, fn)
        new_rhs = _walk_expr(expr.rhs, fn)
        if new_lhs is not expr.lhs or new_rhs is not expr.rhs:
            expr = BinOp(new_lhs, expr.op, new_rhs)
        return fn(expr)

    if isinstance(expr, UnaryOp):
        new_operand = _walk_expr(expr.operand, fn)
        if new_operand is not expr.operand:
            expr = UnaryOp(expr.op, new_operand, expr.post)
        return fn(expr)

    if isinstance(expr, Call):
        new_args = [_walk_expr(a, fn) for a in expr.args]
        if any(na is not oa for na, oa in zip(new_args, expr.args)):
            expr = Call(expr.func_name, new_args)
        return fn(expr)

    if isinstance(expr, Cast):
        new_inner = _walk_expr(expr.inner, fn)
        if new_inner is not expr.inner:
            expr = Cast(expr.type_str, new_inner)
        return fn(expr)

    return fn(expr)


def _subst_pairs_in_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Replace RegGroup(regs) → Name(vinfo.name) for multi-reg VarInfo entries."""
    # Build lookup: pair_name → VarInfo  (skip XRAM locals)
    pair_map: Dict[str, str] = {}
    for key, vinfo in reg_map.items():
        if (isinstance(vinfo, VarInfo) and len(key) > 2
                and not vinfo.xram_sym):
            pair_map[key] = vinfo.name

    if not pair_map:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, RegGroup):
            pair = e.render()  # e.g. "R6R7"
            if pair in pair_map:
                return Name(pair_map[pair])
        if isinstance(e, (Reg, Name)):
            n = e.render()
            if n in pair_map:
                return Name(pair_map[n])
        return e

    return _walk_expr(expr, _fn)


def _subst_single_regs_in_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Replace Reg(rx) → Name(param.name[.suffix]) for is_param entries.
    For multi-byte parameters, appends .hi/.lo/.bN to identify the byte accessed."""
    singles = {k: _param_byte_name(k, v) for k, v in reg_map.items()
               if _RE_SINGLE_REG.match(k)
               and isinstance(v, VarInfo)
               and not v.xram_sym
               and v.is_param}
    if not singles:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, Reg) and e.name in singles:
            return Name(singles[e.name])
        return e

    return _walk_expr(expr, _fn)


def _subst_xram_in_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Replace XRAMRef(Name(sym)) → Name(local_var_name)."""
    sym_map: Dict[str, str] = {}
    for vinfo in reg_map.values():
        if not isinstance(vinfo, VarInfo) or not vinfo.xram_sym:
            continue
        if vinfo.is_byte_field:
            sym_map[vinfo.xram_sym] = vinfo.name
        elif vinfo.xram_sym not in sym_map:
            sym_map[vinfo.xram_sym] = vinfo.name

    if not sym_map:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, XRAMRef):
            inner_text = e.inner.render()
            if inner_text in sym_map:
                return Name(sym_map[inner_text])
        return e

    return _walk_expr(expr, _fn)


def _subst_all_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Apply all three substitutions to an Expr tree."""
    expr = _subst_xram_in_expr(expr, reg_map)
    expr = _subst_pairs_in_expr(expr, reg_map)
    expr = _subst_single_regs_in_expr(expr, reg_map)
    return expr


def _replace_pairs_in_node(node: HIRNode,
                            reg_map: Dict[str, "VarInfo"]) -> HIRNode:
    """Apply pair substitution to an Assign / ExprStmt / ReturnStmt rhs/expr."""
    if isinstance(node, Assign):
        new_rhs = _subst_pairs_in_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return Assign(node.ea, node.lhs, new_rhs)
        return node
    if isinstance(node, CompoundAssign):
        new_rhs = _subst_pairs_in_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return CompoundAssign(node.ea, node.lhs, node.op, new_rhs)
        return node
    if isinstance(node, ExprStmt):
        new_expr = _subst_pairs_in_expr(node.expr, reg_map)
        if new_expr is not node.expr:
            return ExprStmt(node.ea, new_expr)
        return node
    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _subst_pairs_in_expr(node.value, reg_map)
        if new_val is not node.value:
            return ReturnStmt(node.ea, new_val)
        return node
    # Legacy Statement — fall back to text substitution
    if isinstance(node, Statement):
        new_text = _replace_pairs(node.text, reg_map)
        return Statement(node.ea, new_text) if new_text != node.text else node
    return node


def _replace_single_regs_in_node(node: HIRNode,
                                  reg_map: Dict[str, "VarInfo"]) -> HIRNode:
    """Apply single-reg param substitution to RHS/value/expr of a node."""
    if isinstance(node, Assign):
        new_rhs = _subst_single_regs_in_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return Assign(node.ea, node.lhs, new_rhs)
        return node
    if isinstance(node, CompoundAssign):
        new_rhs = _subst_single_regs_in_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return CompoundAssign(node.ea, node.lhs, node.op, new_rhs)
        return node
    if isinstance(node, ExprStmt):
        new_expr = _subst_single_regs_in_expr(node.expr, reg_map)
        if new_expr is not node.expr:
            return ExprStmt(node.ea, new_expr)
        return node
    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _subst_single_regs_in_expr(node.value, reg_map)
        if new_val is not node.value:
            return ReturnStmt(node.ea, new_val)
        return node
    if isinstance(node, Statement):
        new_text = _replace_single_regs(node.text, reg_map)
        return Statement(node.ea, new_text) if new_text != node.text else node
    return node


def _fold_into_node(node: HIRNode, name_expr: Expr,
                    replacement: Expr,
                    reg_map: Dict[str, "VarInfo"]) -> Optional[HIRNode]:
    """
    Try to substitute name_expr → replacement in the expression position of node.

    For Assign: substitutes in rhs.
    For ReturnStmt/ExprStmt: substitutes in value/expr.
    For legacy Statement: falls back to text-based _fold_into_stmt.
    Returns None if name_expr does not appear in an expression position.
    """
    name_str = name_expr.render() if isinstance(name_expr, Expr) else str(name_expr)
    repl_str = replacement.render() if isinstance(replacement, Expr) else str(replacement)

    def _subst_fn(e: Expr) -> Expr:
        if e == name_expr:
            return replacement
        if isinstance(e, (Name, Reg)) and e.render() == name_str:
            return replacement
        return e

    if isinstance(node, Assign):
        new_rhs = _walk_expr(node.rhs, _subst_fn)
        if new_rhs is node.rhs:
            return None  # not found in rhs
        new_node = Assign(node.ea, node.lhs, new_rhs)
        return _replace_pairs_in_node(new_node, reg_map)

    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _walk_expr(node.value, _subst_fn)
        if new_val is node.value:
            return None
        new_node = ReturnStmt(node.ea, new_val)
        return _replace_pairs_in_node(new_node, reg_map)

    if isinstance(node, ExprStmt):
        new_expr = _walk_expr(node.expr, _subst_fn)
        if new_expr is node.expr:
            return None
        new_node = ExprStmt(node.ea, new_expr)
        return _replace_pairs_in_node(new_node, reg_map)

    if isinstance(node, Statement):
        folded = _fold_into_stmt(node.text, name_str, repl_str, reg_map)
        return Statement(node.ea, folded) if folded is not None else None

    return None
