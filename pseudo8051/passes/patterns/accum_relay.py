"""
passes/patterns/accum_relay.py — AccumRelayPattern.

Collapses the 8051 idiom of routing a value through the accumulator:

    A = <expr>;
    <target> = A;

into a single statement:

    <target> = <expr>;

Handles both expression-tree nodes (Assign) and legacy Statement nodes.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign
from pseudo8051.ir.expr import Reg
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import InlineTransform, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo,
    _subst_all_expr,
)


class AccumRelayPattern(InlineTransform):
    """Collapse 'A = expr; target = A;' into 'target = expr;'."""

    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Match]:
        if i + 1 >= len(nodes):
            return None

        n0, n1 = nodes[i], nodes[i + 1]

        if isinstance(n0, Assign) and isinstance(n1, Assign):
            if (n0.lhs == Reg("A")
                    and n1.rhs == Reg("A")
                    and n1.lhs != Reg("A")
                    and n0.rhs != Reg("A")):
                new_rhs = _subst_all_expr(n0.rhs, reg_map)
                new_lhs = n1.lhs
                dbg("typesimp", f"  [{hex(n0.ea)}] accum_relay (expr): {n0.lhs.render()} = {n0.rhs.render()} + {n1.lhs.render()} = {n1.rhs.render()}")
                from pseudo8051.ir.hir import NodeAnnotation as _NA
                new_node = Assign(n0.ea, new_lhs, new_rhs)
                new_node.ann = _NA.merge(n0, n1)
                return (new_node, i + 2)

        return None
