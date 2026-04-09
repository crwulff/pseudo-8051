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

from pseudo8051.ir.hir   import HIRNode, Assign, ExprStmt, IfNode, WhileNode, ForNode, DoWhileNode
from pseudo8051.ir.expr  import Reg, Regs, UnaryOp, Name, Const
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
    if suffixes == ["hi", "lo"]:
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
    if field0 is None or field0[1] not in ("hi", "b0"):
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

    return None, i


def collapse_mb_assigns(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Second-pass collapse of byte-field assignment sequences.

    Recurses into IfNode / WhileNode / ForNode bodies so nested byte-field
    sequences inside control structures are also collapsed.
    """
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]

        # Recurse into structured nodes
        if isinstance(node, IfNode):
            node.then_nodes = collapse_mb_assigns(node.then_nodes)
            node.else_nodes = collapse_mb_assigns(node.else_nodes)
            out.append(node)
            i += 1
            continue
        if isinstance(node, WhileNode):
            node.body_nodes = collapse_mb_assigns(node.body_nodes)
            out.append(node)
            i += 1
            continue
        if isinstance(node, ForNode):
            node.body_nodes = collapse_mb_assigns(node.body_nodes)
            out.append(node)
            i += 1
            continue
        if isinstance(node, DoWhileNode):
            node.body_nodes = collapse_mb_assigns(node.body_nodes)
            out.append(node)
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
