"""
passes/typesimplify/_post.py — Post-simplify passes facade.

Re-exports everything from the individual sub-modules so that existing
`from pseudo8051.passes.typesimplify._post import ...` statements continue
to work without change.
"""

from pseudo8051.passes.typesimplify._dptr import (           # noqa: F401
    _is_dptr_inc_node,
    _collapse_dpl_dph,
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
    _expand_pair_refs,
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
)
from pseudo8051.passes.typesimplify._xram_call_args import ( # noqa: F401
    _fold_xram_call_args,
)


def recurse_bodies(nodes, fn):
    """Recurse fn into structured node bodies; pass other nodes through."""
    return [node.map_bodies(fn) for node in nodes]
