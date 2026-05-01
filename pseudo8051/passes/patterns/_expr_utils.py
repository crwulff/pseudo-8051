"""
passes/patterns/_expr_utils.py — Expression-tree walking and substitution helpers.

_walk_expr, _contains_a, single-node matchers, canonicalization, and all
_subst_*_in_expr functions.
"""

from typing import Callable, Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign
from pseudo8051.ir.expr import (
    Expr, Reg, Regs, Const, Name, XRAMRef, IRAMRef, RegGroup, ArrayRef, BinOp, UnaryOp,
)
from pseudo8051.passes.patterns._types import (
    VarInfo, _type_bytes, _RE_SINGLE_REG, _param_byte_name,
)


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


def _contains_a(expr: Expr) -> bool:
    """Return True if Reg("A") appears anywhere in the Expr tree."""
    found = [False]

    def _fn(e: Expr) -> Expr:
        if e == Reg("A"):
            found[0] = True
        return e

    _walk_expr(expr, _fn)
    return found[0]


# ── Common single-node matching helpers ──────────────────────────────────────

def _node_a_from_reg(node: HIRNode) -> Optional[str]:
    """If node is 'A = Rn;', return Rn; else None."""
    if isinstance(node, Assign):
        if node.lhs == Reg("A") and isinstance(node.rhs, Regs) and node.rhs.is_single:
            return node.rhs.name
    return None


def _node_assign_imm(node: HIRNode) -> Optional[Tuple[str, int]]:
    """If node assigns an immediate to a register, return (dst_name, int_value)."""
    if isinstance(node, Assign):
        if isinstance(node.lhs, Regs) and node.lhs.is_single and isinstance(node.rhs, Const):
            return (node.lhs.name, node.rhs.value)
    return None


def _node_assign_reg(node: HIRNode) -> Optional[Tuple[str, str]]:
    """If node is a simple reg=reg assignment, return (dst_name, src_name)."""
    if isinstance(node, Assign):
        if (isinstance(node.lhs, Regs) and node.lhs.is_single
                and isinstance(node.rhs, Regs) and node.rhs.is_single):
            return (node.lhs.name, node.rhs.name)
    return None


def _is_reg_free(expr: Expr) -> bool:
    """True if expr contains no Regs leaf — safe to forward-substitute at multiple sites."""
    found = [False]

    def _fn(e: Expr) -> Expr:
        if isinstance(e, Regs):
            found[0] = True
        return e

    _walk_expr(expr, _fn)
    return not found[0]


def _regs_in_expr(expr: Expr) -> set:
    """Return the set of register names (Regs.name) contained in expr."""
    found: set = set()

    def _fn(e: Expr) -> Expr:
        if isinstance(e, Regs):
            found.add(e.name)
        return e

    _walk_expr(expr, _fn)
    return found


def _fold_unary_const(e: Expr) -> Expr:
    """Fold UnaryOp(++/--, Const) → Const (alias always removed).

    Post-increment: returns the original value (side effect discarded).
    Pre-increment:  returns value ± 1.
    """
    if isinstance(e, UnaryOp) and e.op in ('++', '--') and isinstance(e.operand, Const):
        v = e.operand.value
        if e.post:
            return Const(v)                      # Const(v, alias)++ → Const(v)
        delta = +1 if e.op == '++' else -1
        return Const((v + delta) & 0xFFFF)       # ++Const(v) → Const(v+1)
    return e


def _canonicalize_expr(expr: Expr,
                       const_state: Dict[str, int],
                       groups: list,
                       expr_state: Dict[str, Expr]) -> Expr:
    """Canonicalize an expression by substituting known register values.

    Performs a bottom-up walk.  At each single-register Regs leaf:
    1. Const substitute:  const_state.get(r) → Const(v)
    2. TypeGroup substitute: first group where full_regs == [r] and name set → Name(g.name)
    3. expr_state substitute: expr_state.get(r) → that expr (then apply const+group only,
       no recursive expr_state lookup to prevent cycles)

    After leaf substitution, applies constant folding and identity elimination:
    - BinOp(Const(a), op, Const(b)) → Const(result)
    - Identity: x+0, x-0, x|0, x^0, x<<0, x>>0 → x; 0+x, 0|x → x; x*1, 1*x → x
    """
    _FOLD_OPS = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a // b if b != 0 else None,
        '&': lambda a, b: a & b,
        '|': lambda a, b: a | b,
        '^': lambda a, b: a ^ b,
        '>>': lambda a, b: a >> b,
        '<<': lambda a, b: a << b,
    }

    def _subst_leaf(e: Expr) -> Expr:
        """Substitute a single-register Regs leaf once."""
        if not (isinstance(e, Regs) and e.is_single):
            return e
        r = e.name
        # 1. Const
        if r in const_state:
            return Const(const_state[r])
        # 2. TypeGroup name
        for g in groups:
            if (len(g.full_regs) == 1
                    and g.full_regs[0] == r
                    and g.name):
                return Name(g.name)
        # 3. expr_state (apply one more level of const+group, no expr_state)
        if r in expr_state:
            sub = expr_state[r]
            def _const_group_only(e2: Expr) -> Expr:
                if not (isinstance(e2, Regs) and e2.is_single):
                    return e2
                r2 = e2.name
                if r2 in const_state:
                    return Const(const_state[r2])
                for g in groups:
                    if (len(g.full_regs) == 1
                            and g.full_regs[0] == r2
                            and g.name):
                        return Name(g.name)
                return e2
            return _walk_expr(sub, _const_group_only)
        return e

    def _mul_base_coeff(e: Expr):
        """Return (base, coeff) if e is base*Const(k) or Const(k)*base, else (e, 1)."""
        if isinstance(e, BinOp) and e.op == '*':
            if isinstance(e.rhs, Const):
                return e.lhs, e.rhs.value
            if isinstance(e.lhs, Const):
                return e.rhs, e.lhs.value
        return e, 1

    def _fold(e: Expr) -> Expr:
        """Constant fold, identity-eliminate, and algebraically normalize."""
        # Strip trivial Paren(Const) wrappers — allows identity rules to fire
        # through parentheses (e.g. (0 << 8) | x  →  x after inner fold).
        from pseudo8051.ir.expr import Paren as _Paren
        if isinstance(e, _Paren) and isinstance(e.inner, Const):
            return e.inner
        if not isinstance(e, BinOp):
            return e
        lhs, rhs = e.lhs, e.rhs
        # Unwrap Paren(Const) on operands for the same reason.
        if isinstance(lhs, _Paren) and isinstance(lhs.inner, Const):
            lhs = lhs.inner
        if isinstance(rhs, _Paren) and isinstance(rhs.inner, Const):
            rhs = rhs.inner
        lc = isinstance(lhs, Const)
        rc = isinstance(rhs, Const)
        op = e.op
        if lc and rc:
            fn = _FOLD_OPS.get(op)
            if fn is not None:
                result = fn(lhs.value, rhs.value)
                if result is not None:
                    return Const(result & 0xFFFFFFFF)
        # Identity eliminations
        if rc and rhs.value == 0 and op in ('+', '-', '|', '^', '<<', '>>'):
            return lhs
        if lc and lhs.value == 0 and op in ('+', '|'):
            return rhs
        if rc and rhs.value == 1 and op == '*':
            return lhs
        if lc and lhs.value == 1 and op == '*':
            return rhs
        # Coefficient merging: e*k1 + e*k2 → e*(k1+k2)
        # Handles: e+e → e*2, e*k+e → e*(k+1), e*k1+e*k2 → e*(k1+k2)
        if op == '+':
            lbase, lcoeff = _mul_base_coeff(lhs)
            rbase, rcoeff = _mul_base_coeff(rhs)
            if lbase == rbase and not isinstance(lbase, Const):
                total = lcoeff + rcoeff
                return BinOp(lbase, '*', Const(total))
        # Division cancellation: (e * k1) / k2 → e * (k1//k2) when k2 divides k1
        if op == '/' and rc and rhs.value != 0:
            lbase, lcoeff = _mul_base_coeff(lhs)
            if not isinstance(lbase, Const) and lcoeff % rhs.value == 0:
                new_coeff = lcoeff // rhs.value
                if new_coeff == 1:
                    return lbase
                return BinOp(lbase, '*', Const(new_coeff))
        # Additive chain collapse: (base +/- Const(a)) +/- Const(b) → base +/- Const_combined
        # Applied with 8-bit wrapping since all tracked registers are 8-bit.
        if (op in ('+', '-') and rc
                and isinstance(lhs, BinOp) and lhs.op in ('+', '-')
                and isinstance(lhs.rhs, Const)):
            s1 = +1 if lhs.op == '+' else -1
            s2 = +1 if op == '+' else -1
            combined = (s1 * lhs.rhs.value + s2 * rhs.value) & 0xFF
            if combined == 0:
                return lhs.lhs
            if combined <= 0x7F:
                return BinOp(lhs.lhs, '+', Const(combined))
            return BinOp(lhs.lhs, '-', Const(0x100 - combined))
        # 16-bit byte-field reconstruction: (x.hi << 8) | x.lo → Name("x")
        # Handles both (Paren(x.hi << 8) | x.lo) and (x.hi << 8) | x.lo).
        if op == '|' and isinstance(rhs, Name) and rhs.name.endswith('.lo'):
            inner_lhs = lhs.inner if isinstance(lhs, _Paren) else lhs
            if (isinstance(inner_lhs, BinOp) and inner_lhs.op == '<<'
                    and isinstance(inner_lhs.rhs, Const) and inner_lhs.rhs.value == 8
                    and isinstance(inner_lhs.lhs, Name)
                    and inner_lhs.lhs.name.endswith('.hi')
                    and inner_lhs.lhs.name[:-3] == rhs.name[:-3]):
                return Name(rhs.name[:-3])
        return e

    def _fn(e: Expr) -> Expr:
        e = _fold(_subst_leaf(e))
        return _fold_unary_const(e)

    return _walk_expr(expr, _fn)


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
    # Address-keyed fallback: maps raw int XRAM address → variable name.
    # Used when a Const(addr) was produced by constant folding (no alias) but
    # resolve_ext_addr(addr) returned an IDA symbol name, so sym_map key doesn't
    # match the rendered hex string.
    addr_map: Dict[int, str] = {}
    # Array base lookups for dynamic indexed access: XRAM[base + idx] → arr[idx]
    arr_sym_map: Dict[str, "VarInfo"] = {}    # xram_sym  → array VarInfo
    arr_addr_map: Dict[int, "VarInfo"] = {}   # xram_addr → array VarInfo

    for vinfo in reg_map.values():
        if not isinstance(vinfo, VarInfo) or not vinfo.xram_sym:
            continue
        if vinfo.is_byte_field:
            sym_map[vinfo.xram_sym] = vinfo.name
            if vinfo.xram_addr > 0:
                addr_map[vinfo.xram_addr] = vinfo.name
        elif vinfo.array_size > 0:
            # Static access to base addr maps to element [0] (also covered by elem entries)
            if vinfo.xram_sym not in sym_map:
                sym_map[vinfo.xram_sym] = f"{vinfo.name}[0]"
            arr_sym_map[vinfo.xram_sym] = vinfo
            arr_addr_map[vinfo.xram_addr] = vinfo
        elif vinfo.xram_sym not in sym_map:
            sym_map[vinfo.xram_sym] = vinfo.name
            if vinfo.xram_addr > 0:
                addr_map[vinfo.xram_addr] = vinfo.name

    if not sym_map and not arr_addr_map:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, XRAMRef):
            inner = e.inner
            inner_text = inner.render()
            if inner_text in sym_map:
                return Name(sym_map[inner_text])
            # Fallback: Const with no alias — match by raw address value.
            # Handles the case where resolve_ext_addr returned an IDA symbol name
            # but the Const was produced by _fold_unary_const (no alias set).
            if isinstance(inner, _Const) and inner.value in addr_map:
                return Name(addr_map[inner.value])
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


def _subst_iram_in_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Replace IRAMRef(Const(addr)) → Name(local_var_name)."""
    addr_map: Dict[int, str] = {}
    for vinfo in reg_map.values():
        if not isinstance(vinfo, VarInfo) or not vinfo.iram_addr:
            continue
        addr_map[vinfo.iram_addr] = vinfo.name

    if not addr_map:
        return expr

    def _fn(e: Expr) -> Expr:
        if isinstance(e, IRAMRef) and isinstance(e.inner, Const):
            name = addr_map.get(e.inner.value)
            if name is not None:
                return Name(name)
        return e

    return _walk_expr(expr, _fn)


def _subst_all_expr(expr: Expr, reg_map: Dict[str, "VarInfo"]) -> Expr:
    """Apply all substitutions to an Expr tree."""
    expr = _subst_xram_in_expr(expr, reg_map)
    expr = _subst_iram_in_expr(expr, reg_map)
    expr = _subst_pairs_in_expr(expr, reg_map)
    expr = _subst_single_regs_in_expr(expr, reg_map)
    return expr
