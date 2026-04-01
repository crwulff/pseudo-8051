"""
passes/patterns/reg_inc.py — RegPostIncPattern and RegPreIncPattern.

Post-increment/decrement:  Any node using register Rn exactly once,
followed by ExprStmt(Rn++/--), is folded to embed Rn++ (or Rn--) inside
the expression.  Example:

    A = IRAM[R1];   R1++;   →   A = IRAM[R1++];
    IRAM[R0] = A;   R0++;   →   IRAM[R0++] = A;

Pre-increment/decrement:  ExprStmt(Rn++/--) followed by any node using
Rn exactly once is folded to embed ++Rn (or --Rn) inside the expression.
Example:

    R1++;   A = IRAM[R1];   →   A = IRAM[++R1];

Both patterns:
  - work with any HIR node type (Assign, ExprStmt, ReturnStmt, IfGoto)
  - include DPTR (previously excluded; DPTR++ folding prevents incorrect pruning)
  - handle both ++ and --
  - guard against chains: if the candidate node is itself an inc/dec node,
    the pattern does not fire (avoids composing pre/post patterns)
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, ExprStmt
from pseudo8051.ir.expr import Reg, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _count_reg_uses_in_node, _subst_reg_in_node,
)


def _match_inc_node(node: HIRNode):
    """If node is ExprStmt(Rn++/--) with post=True, return (rn, op). Else None."""
    if not isinstance(node, ExprStmt):
        return None
    e = node.expr
    if not (isinstance(e, UnaryOp)
            and e.op in ("++", "--")
            and isinstance(e.operand, Reg)
            and e.post):
        return None
    return (e.operand.name, e.op)


def _embed_op(node: HIRNode, rn: str, op: str, post: bool) -> Optional[HIRNode]:
    """Substitute Reg(rn) → UnaryOp(op, Reg(rn), post) in node, iff exactly 1 read use."""
    if _count_reg_uses_in_node(rn, node) != 1:
        return None
    replacement = UnaryOp(op, Reg(rn), post=post)
    return _subst_reg_in_node(node, rn, replacement)


class RegPostIncPattern(Pattern):
    """
    Collapse 'node-using-Rn-once; Rn++/--;' into 'node-with-Rn++/-- embedded;'.

    Guards:
    - n0 must not itself be an inc/dec node (ExprStmt(Rm++/--)
    - The ExprStmt must use post=True
    - Rn must appear exactly once in read positions of n0
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:

        n0 = nodes[i]
        # Don't consume another inc node (avoid conflicting with RegPreIncPattern)
        if _match_inc_node(n0) is not None:
            return None
        if i + 1 >= len(nodes):
            return None
        inc = _match_inc_node(nodes[i + 1])
        if inc is None:
            return None
        rn, op = inc

        new_node = _embed_op(n0, rn, op, post=True)
        if new_node is None:
            return None
        dbg("typesimp", f"  [{hex(n0.ea)}] reg_post_inc: embedded {rn}{op} into node")
        return ([new_node], i + 2)


class RegPreIncPattern(Pattern):
    """
    Collapse 'Rn++/--; node-using-Rn-once;' into 'node-with-++Rn/--Rn embedded;'.

    Guards:
    - n1 must not itself be an inc/dec node
    - The ExprStmt must use post=True
    - Rn must appear exactly once in read positions of n1
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:

        inc = _match_inc_node(nodes[i])
        if inc is None:
            return None
        rn, op = inc
        if i + 1 >= len(nodes):
            return None
        n1 = nodes[i + 1]
        # Don't consume another inc node
        if _match_inc_node(n1) is not None:
            return None

        new_node = _embed_op(n1, rn, op, post=False)
        if new_node is None:
            return None
        dbg("typesimp", f"  [{hex(n1.ea)}] reg_pre_inc: embedded {op}{rn} into node")
        return ([new_node], i + 2)
