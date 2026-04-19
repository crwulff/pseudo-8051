"""
passes/typesimplify/_post.py — Post-simplify passes facade.

Re-exports everything from the individual sub-modules so that existing
`from pseudo8051.passes.typesimplify._post import ...` statements continue
to work without change.
"""

from pseudo8051.passes.typesimplify._dptr import (           # noqa: F401
    _is_dptr_inc_node,
    _collapse_dpl_dph,
    _collapse_dpl_dph_arithmetic,
    _dptr_live_after,
    _prune_orphaned_dptr_inc,
)
from pseudo8051.passes.typesimplify._xram_loads import (     # noqa: F401
    _consolidate_xram_local_loads,
    _subst_reg_in_scope,
)
from pseudo8051.passes.typesimplify._setup_fold import (     # noqa: F401
    _is_call_setup_assign,
    _collect_hir_name_refs,
    _fold_and_prune_setups,
    _fold_call_arg_pairs,
)
from pseudo8051.passes.typesimplify._return_fold import (    # noqa: F401
    _fold_return_chains,
)
from pseudo8051.passes.typesimplify._propagate import (      # noqa: F401
    _propagate_values,
    _fold_compound_assigns,
    _propagate_register_copies,
    _inline_retvals,
)
from pseudo8051.passes.typesimplify._carry import (          # noqa: F401
    _simplify_carry_comparison,
    _simplify_cjne_jnc,
    _simplify_orl_zero_check,
    _simplify_arithmetic,
    _simplify_acc_bit_test,
)
from pseudo8051.passes.typesimplify._xram_call_args import ( # noqa: F401
    _fold_xram_call_args,
)


def recurse_bodies(nodes, fn):
    """Recurse fn into structured node bodies; pass other nodes through."""
    return [node.map_bodies(fn) for node in nodes]


def _subst_xram_in_hir(nodes, reg_map):
    """
    Walk the HIR and apply _subst_xram_in_expr to every expression.

    Called after _collapse_dpl_dph_arithmetic to convert newly-created
    XRAM[base + idx] nodes into array subscript expressions (e.g. foo[R6R7]).
    Also applied to LHS write positions (XRAMRef indirect writes) so that
    indexed writes like XRAM[base + idx] = ... become arr[idx] = ... .
    """
    from pseudo8051.passes.patterns._utils import _subst_xram_in_expr, _apply_expr_subst_to_node
    from pseudo8051.ir.hir import Assign
    from pseudo8051.ir.expr import XRAMRef

    def _subst_fn(e):
        return _subst_xram_in_expr(e, reg_map)

    def _visit(ns):
        return _subst_xram_in_hir(ns, reg_map)

    result = []
    for node in nodes:
        patched = _apply_expr_subst_to_node(node, _subst_fn)
        # Also transform XRAMRef LHS (write destination) so indexed XRAM writes
        # like XRAM[base + idx] = expr become arr[idx] = expr.
        if isinstance(patched, Assign) and isinstance(patched.lhs, XRAMRef):
            new_lhs = _subst_fn(patched.lhs)
            if new_lhs is not patched.lhs:
                patched = patched.copy_meta_to(Assign(patched.ea, new_lhs, patched.rhs))
        result.append(patched.map_bodies(_visit))
    return result
