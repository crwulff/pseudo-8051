"""
passes/patterns/mb_assign.py — collapse_mb_assigns().

Collapses consecutive byte-field Statement nodes produced after per-byte
XRAM-write pattern processing:

    var0.hi = count.hi;     (Statement)
    DPTR++;                 (ExprStmt or Statement — skipped)
    var0.lo = count.lo;     (Statement)

into a single statement:

    var0 = count;

Also handles constant sources:

    var0.hi = 0x12;
    DPTR++;
    var0.lo = 0x34;
  →
    var0 = 0x1234;

This must run as a SECOND PASS on the already-simplified HIR output, because
the byte-field Statement nodes are only visible after the first simplification
round (XRAMLocalWritePattern + AccumRelayPattern have already transformed the
raw Assign nodes into named Statement nodes).
"""

import re
from typing import List, Optional

from pseudo8051.ir.hir   import HIRNode, Statement, ExprStmt, IfNode, WhileNode, ForNode, DoWhileNode
from pseudo8051.ir.expr  import Reg, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns._utils import _const_str

# Match "lhs_parent.suffix = rhs_expr;"
_RE_MB_ASSIGN  = re.compile(r'^(\w+)\.(hi|lo|b\d+)\s*=\s*(.+);$')

# Match "rhs_parent.rhs_suffix"
_RE_FIELD_RHS  = re.compile(r'^(\w+)\.(hi|lo|b\d+)$')

# Match a bare integer constant
_RE_CONST_VAL  = re.compile(r'^(0x[0-9a-fA-F]+|\d+)$')


def _parse_int(s: str) -> Optional[int]:
    m = _RE_CONST_VAL.match(s.strip())
    if not m:
        return None
    t = m.group(1)
    return int(t, 16) if t.lower().startswith("0x") else int(t)


def _is_dptr_inc(node: HIRNode) -> bool:
    """True if node is a DPTR++ increment."""
    if isinstance(node, ExprStmt) and isinstance(node.expr, UnaryOp):
        return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
    if isinstance(node, Statement) and node.text == "DPTR++;":
        return True
    return False


def _valid_suffix_sequence(suffixes: List[str]) -> bool:
    """True if suffixes form a valid ordered byte-field sequence."""
    if suffixes == ["hi", "lo"]:
        return True
    return suffixes == [f"b{i}" for i in range(len(suffixes))]


def _try_collapse(nodes: List[HIRNode], i: int):
    """
    Attempt to collapse a byte-field assignment sequence starting at nodes[i].

    Returns (Statement, new_i) on success, or (None, i) if no collapse.
    """
    n0 = nodes[i]
    if not isinstance(n0, Statement):
        return None, i

    m0 = _RE_MB_ASSIGN.match(n0.text)
    # Only start a sequence at "hi" or "b0"
    if not m0 or m0.group(2) not in ("hi", "b0"):
        return None, i

    lhs_parent, first_suffix, rhs_str_0 = m0.group(1), m0.group(2), m0.group(3)

    # Scan forward collecting (suffix, rhs_expr) pairs, skipping DPTR++ nodes
    pairs = [(first_suffix, rhs_str_0)]
    j = i + 1
    while j < len(nodes):
        nd = nodes[j]
        if _is_dptr_inc(nd):
            j += 1
            continue
        if isinstance(nd, Statement):
            mx = _RE_MB_ASSIGN.match(nd.text)
            if mx and mx.group(1) == lhs_parent:
                pairs.append((mx.group(2), mx.group(3)))
                j += 1
                continue
        break

    if len(pairs) < 2:
        return None, i

    suffixes = [p[0] for p in pairs]
    if not _valid_suffix_sequence(suffixes):
        return None, i

    rhs_vals = [p[1] for p in pairs]

    # Try: all RHS are rhs_parent.same_suffix
    rhs_parent = None
    valid_field = True
    for suf, rhs_str in zip(suffixes, rhs_vals):
        mr = _RE_FIELD_RHS.match(rhs_str)
        if not mr:
            valid_field = False
            break
        parent, rsuf = mr.group(1), mr.group(2)
        if rhs_parent is None:
            rhs_parent = parent
        elif parent != rhs_parent:
            valid_field = False
            break
        if rsuf != suf:
            valid_field = False
            break

    if valid_field and rhs_parent is not None:
        dbg("typesimp", f"  mb_assign: {lhs_parent} = {rhs_parent}")
        return Statement(n0.ea, f"{lhs_parent} = {rhs_parent};"), j

    # Try: all RHS are integer constants
    const_vals = [_parse_int(rv) for rv in rhs_vals]
    if all(v is not None for v in const_vals):
        value = 0
        for cv in const_vals:
            value = (value << 8) | (cv & 0xFF)
        n_bytes = len(const_vals)
        type_str = {1: "uint8_t", 2: "uint16_t", 4: "uint32_t"}.get(n_bytes, "uint16_t")
        val_str = _const_str(value, type_str)
        dbg("typesimp", f"  mb_assign: {lhs_parent} = {val_str} (const)")
        return Statement(n0.ea, f"{lhs_parent} = {val_str};"), j

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

        collapsed, new_i = _try_collapse(nodes, i)
        if collapsed is not None:
            out.append(collapsed)
            i = new_i
        else:
            out.append(node)
            i += 1

    return out
