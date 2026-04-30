"""
passes/typesimplify/_propagate.py — Forward single-use value propagation pass.

Exports:
  _propagate_values   master pass (calls sub-passes A0/A/A1/B/C in a fixed-point loop)

Sub-passes (also exported for testing):
  _fold_compound_assigns      A0: fold Assign(r, e) + CompoundAssign(r, op=, rhs) → Assign
  _propagate_register_copies  A:  substitute single-use Assign(Reg, expr) into its use
  _subst_from_reg_exprs       A1: substitute reg_exprs annotations into node expressions
  _inline_retvals             B:  inline TypedAssign retval = call() into its single use
  _inline_group_setups        C:  fold single-use multi-reg setup into call args
"""

from typing import Dict, List

from pseudo8051.ir.hir import HIRNode
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.constants import dbg
from pseudo8051.passes.typesimplify._propagate_utils import _dbg_node  # noqa: F401
from pseudo8051.passes.typesimplify._propagate_regcopy import (  # noqa: F401
    _fold_compound_assigns,
    _propagate_register_copies,
)
from pseudo8051.passes.typesimplify._propagate_inline import (  # noqa: F401
    _subst_from_reg_exprs,
    _inline_retvals,
    _inline_group_setups,
)


def _propagate_values(nodes: List[HIRNode],
                      reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Forward single-use propagation pass.

    Recurses into structured nodes first, then applies five sub-passes in a
    fixed-point loop until no changes occur:
      A0: fold Assign + CompoundAssign into a single Assign
      A:  substitute single-use (or reg-free multi-use) register copies
      A1: substitute reg_exprs annotation values directly into nodes
      B:  inline single-use retval TypedAssign into its target
      C:  fold single-use multi-register group setup into call arguments
    """
    work = [node.map_bodies(lambda ns: _propagate_values(ns, reg_map))
            for node in nodes]

    dbg("propagate", f"_propagate_values flat({len(work)}): "
        + ", ".join(_dbg_node(n) for n in work))

    changed = True
    while changed:
        changed = False
        work, c0  = _fold_compound_assigns(work)
        work, cA  = _propagate_register_copies(work, reg_map)
        work, cA1 = _subst_from_reg_exprs(work)
        work, cB  = _inline_retvals(work, reg_map)
        work, cC  = _inline_group_setups(work, reg_map)
        changed = c0 or cA or cA1 or cB or cC

    return work
