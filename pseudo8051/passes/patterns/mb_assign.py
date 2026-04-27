"""
passes/patterns/mb_assign.py — collapse_mb_assigns().

Collapses consecutive byte-field Assign nodes produced after per-byte
XRAM-write pattern processing:

    Assign(ea, Name("var0.hi"), Name("count.hi"))
    DPTR++  (ExprStmt — skipped)
    Assign(ea, Name("var0.lo"), Name("count.lo"))

into a single assignment:

    Assign(ea, Name("var0"), Name("count"))

Also handles constant sources:

    Assign(ea, Name("var0.hi"), Name("0x12"))  or Const(0x12)
    DPTR++
    Assign(ea, Name("var0.lo"), Name("0x34"))  or Const(0x34)
  →
    Assign(ea, Name("var0"), Const(0x1234))

This must run as a SECOND PASS on the already-simplified HIR output, because
the byte-field Assign nodes are only visible after the first simplification
round (XRAMLocalWritePattern + AccumRelayPattern have already transformed the
raw XRAM-write nodes into named Assign nodes).
"""

import re
from typing import List, Optional, Tuple

from pseudo8051.ir.hir   import HIRNode, Assign, ExprStmt
from pseudo8051.ir.expr  import Reg, Regs, UnaryOp, Name, Const, BinOp, Cast
from pseudo8051.constants import dbg

# Match a bare integer constant (used to parse Name("0x12") from XRAMLocalWritePattern)
_RE_CONST_VAL = re.compile(r'^(0x[0-9a-fA-F]+|\d+)$')


def _parse_int(s: str) -> Optional[int]:
    m = _RE_CONST_VAL.match(s.strip())
    if not m:
        return None
    t = m.group(1)
    return int(t, 16) if t.lower().startswith("0x") else int(t)


def _split_field(name: str) -> Optional[Tuple[str, str]]:
    """Split 'parent.suffix' into (parent, suffix), or return None."""
    if "." in name:
        parent, suffix = name.split(".", 1)
        return (parent, suffix)
    return None


def _is_dptr_inc(node: HIRNode) -> bool:
    """True if node is a DPTR++ increment."""
    if isinstance(node, ExprStmt) and isinstance(node.expr, UnaryOp):
        return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
    return False


def _valid_suffix_sequence(suffixes: List[str]) -> bool:
    """True if suffixes form a valid ordered byte-field sequence."""
    if suffixes == ["hi", "lo"] or suffixes == ["lo", "hi"]:
        return True
    return suffixes == [f"b{i}" for i in range(len(suffixes))]


def _rhs_as_const(rhs) -> Optional[int]:
    """Return integer value if rhs is a Const or a Name holding a numeric literal."""
    if isinstance(rhs, Const):
        return rhs.value
    if isinstance(rhs, Name):
        return _parse_int(rhs.name)
    return None


def _try_collapse_rmw_inc(nodes: List[HIRNode], i: int):
    """
    Recognize the accumulator read-modify-write increment pattern for XRAM locals:

      A = <var>;       (load XRAM local into A)
      <var> = ++A;     (increment A and store back)

    Collapses to a post-increment statement:
      <var>++;
    """
    n0 = nodes[i]
    if not (isinstance(n0, Assign)
            and isinstance(n0.lhs, Regs) and n0.lhs.is_single
            and n0.lhs.names == ('A',)
            and isinstance(n0.rhs, Name)):
        return None, i

    var_name = n0.rhs.name
    if i + 1 >= len(nodes):
        return None, i

    n1 = nodes[i + 1]
    if not (isinstance(n1, Assign)
            and isinstance(n1.lhs, Name)
            and n1.lhs.name == var_name):
        return None, i

    rhs = n1.rhs
    is_inc_a = (
        (isinstance(rhs, Name) and rhs.name == '++A')
        or (isinstance(rhs, UnaryOp) and rhs.op == '++'
            and not getattr(rhs, 'post', True)
            and isinstance(rhs.operand, Regs)
            and rhs.operand.names == ('A',))
    )
    if not is_inc_a:
        return None, i

    dbg("typesimp", f"  mb_assign: {var_name}++ (rmw-inc)")
    return ExprStmt(n0.ea, UnaryOp('++', Name(var_name), post=True)), i + 2


def _try_collapse(nodes: List[HIRNode], i: int):
    """
    Attempt to collapse a byte-field assignment sequence starting at nodes[i].

    Returns (Assign, new_i) on success, or (None, i) if no collapse.
    """
    n0 = nodes[i]
    if not isinstance(n0, Assign) or not isinstance(n0.lhs, Name):
        return None, i

    field0 = _split_field(n0.lhs.name)
    if field0 is None or field0[1] not in ("hi", "lo", "b0"):
        return None, i

    lhs_parent, first_suffix = field0

    # Scan forward collecting (suffix, rhs_expr) pairs, skipping DPTR++ nodes
    pairs: List[Tuple[str, object]] = [(first_suffix, n0.rhs)]
    j = i + 1
    while j < len(nodes):
        nd = nodes[j]
        if _is_dptr_inc(nd):
            j += 1
            continue
        if isinstance(nd, Assign) and isinstance(nd.lhs, Name):
            f = _split_field(nd.lhs.name)
            if f is not None and f[0] == lhs_parent:
                pairs.append((f[1], nd.rhs))
                j += 1
                continue
        break

    if len(pairs) < 2:
        return None, i

    suffixes = [p[0] for p in pairs]
    if not _valid_suffix_sequence(suffixes):
        return None, i

    rhs_exprs = [p[1] for p in pairs]

    # Try: all RHS are Name("rhs_parent.same_suffix") — field copy
    rhs_parent = None
    valid_field = True
    for suf, rhs in zip(suffixes, rhs_exprs):
        if isinstance(rhs, Name):
            f = _split_field(rhs.name)
            if f is not None and f[1] == suf:
                if rhs_parent is None:
                    rhs_parent = f[0]
                elif f[0] != rhs_parent:
                    valid_field = False
                    break
            else:
                valid_field = False
                break
        else:
            valid_field = False
            break

    if valid_field and rhs_parent is not None:
        dbg("typesimp", f"  mb_assign: {lhs_parent} = {rhs_parent}")
        return Assign(n0.ea, Name(lhs_parent), Name(rhs_parent)), j

    # Try: all RHS are integer constants (Const or Name with numeric literal)
    const_vals = [_rhs_as_const(rhs) for rhs in rhs_exprs]
    if all(v is not None for v in const_vals):
        value = 0
        for cv in const_vals:
            value = (value << 8) | (cv & 0xFF)
        dbg("typesimp", f"  mb_assign: {lhs_parent} = {hex(value)} (const)")
        return Assign(n0.ea, Name(lhs_parent), Const(value)), j

    # Try: 16-bit RMW add-with-carry from 8051 MUL result.
    # Pattern (lo first or hi first):
    #   var.lo = var.lo + Cast('uint8_t', full_expr)
    #   var.hi = var.hi + mul_hi_expr [+ C]
    # where mul_hi_expr is typically B (MUL high byte) and full_expr is the
    # complete 16-bit MUL expression (e.g. _var2 * 3).
    # → var = var + full_expr  (CompoundAssign var += full_expr)
    #
    # Only handles the 2-byte case.
    if set(suffixes) == {"hi", "lo"} and len(rhs_exprs) == 2:
        if suffixes == ["lo", "hi"]:
            lo_rhs, hi_rhs = rhs_exprs
        else:
            hi_rhs, lo_rhs = rhs_exprs
        full_expr = _extract_mul_addc_pair(lhs_parent, hi_rhs, lo_rhs)
        if full_expr is not None:
            from pseudo8051.ir.hir import CompoundAssign
            dbg("typesimp",
                f"  mb_assign: {lhs_parent} += {full_expr.render()} (mul-addc)")
            node = CompoundAssign(n0.ea, Name(lhs_parent), "+=", full_expr)
            return node, j

    return None, i


def _extract_mul_addc_pair(var_name: str, hi_rhs, lo_rhs) -> Optional['Expr']:
    """
    Detect the 8051 16-bit add-with-carry RMW pattern from a MUL result:

      var.hi = var.hi + <hi_expr> [+ C]
      var.lo = var.lo + Cast('uint8_t', full_expr)

    Returns full_expr if the pattern matches, else None.

    Rules:
      - lo_rhs must be BinOp('+', Name(var.lo)_or_alias, Cast('uint8_t', full_expr))
        or BinOp('+', Cast('uint8_t', full_expr), Name(var.lo)_or_alias)
      - hi_rhs must be BinOp('+', Name(var.hi)_or_alias, hi_addend)
        where hi_addend contains Reg('B') (the MUL high-byte register) with
        optional Reg('C') carry
      - The cast-stripped full_expr in lo_rhs matches the value that B would
        hold (same expression tree, or B is the only non-C non-var addend in hi)
    """
    # Unwrap lo: var.lo + Cast('uint8_t', full_expr)
    full_expr = _unwrap_rmw_add(f"{var_name}.lo", lo_rhs, _is_uint8_cast)
    if full_expr is None:
        return None

    # Unwrap hi: var.hi + <hi_addend [+ C]>
    hi_addend = _unwrap_rmw_add(f"{var_name}.hi", hi_rhs, None)
    if hi_addend is None:
        return None

    # hi_addend must be B, or B+C, or C+B (C is the carry from the lo add)
    if not _is_b_plus_carry(hi_addend):
        return None

    return full_expr


def _name_matches(expr, field_name: str) -> bool:
    """True if expr is Name(field_name) or Regs with alias==field_name."""
    if isinstance(expr, Name) and expr.name == field_name:
        return True
    if isinstance(expr, Regs) and expr.alias == field_name:
        return True
    return False


def _unwrap_rmw_add(field_name: str, rhs, inner_pred) -> Optional['Expr']:
    """
    Match  rhs == field_name + inner  (or inner + field_name).
    If inner_pred is given, apply it to inner and return its result.
    If inner_pred is None, return inner directly.
    Returns None on mismatch.
    """
    if not isinstance(rhs, BinOp) or rhs.op != '+':
        return None
    if _name_matches(rhs.lhs, field_name):
        inner = rhs.rhs
    elif _name_matches(rhs.rhs, field_name):
        inner = rhs.lhs
    else:
        return None
    if inner_pred is not None:
        return inner_pred(inner)
    return inner


def _is_uint8_cast(expr) -> Optional['Expr']:
    """If expr is Cast('uint8_t', inner), return inner; else None."""
    if isinstance(expr, Cast) and expr.type_str == 'uint8_t':
        return expr.inner
    return None


def _is_b_plus_carry(expr) -> bool:
    """True if expr is B, B+C, or C+B (8051 ADDC high-byte pattern)."""
    if isinstance(expr, Regs) and expr.is_single and expr.name == 'B':
        return True
    if isinstance(expr, BinOp) and expr.op == '+':
        lhs_b = isinstance(expr.lhs, Regs) and expr.lhs.is_single and expr.lhs.name == 'B'
        rhs_b = isinstance(expr.rhs, Regs) and expr.rhs.is_single and expr.rhs.name == 'B'
        lhs_c = isinstance(expr.lhs, Regs) and expr.lhs.is_single and expr.lhs.name == 'C'
        rhs_c = isinstance(expr.rhs, Regs) and expr.rhs.is_single and expr.rhs.name == 'C'
        return (lhs_b and rhs_c) or (lhs_c and rhs_b)
    return False


def collapse_mb_assigns(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Second-pass collapse of byte-field assignment sequences.

    Recurses into all structured node bodies (IfNode, WhileNode, ForNode,
    DoWhileNode, SwitchNode) via map_bodies so nested byte-field sequences
    inside any control structure are also collapsed.
    """
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]

        # Recurse into structured nodes via map_bodies (returns new node for
        # structured types, self for leaf types).
        mapped = node.map_bodies(collapse_mb_assigns)
        if mapped is not node:
            out.append(mapped)
            i += 1
            continue

        collapsed, new_i = _try_collapse_rmw_inc(nodes, i)
        if collapsed is None:
            collapsed, new_i = _try_collapse(nodes, i)
        if collapsed is not None:
            out.append(collapsed)
            i = new_i
        else:
            out.append(node)
            i += 1

    return out
