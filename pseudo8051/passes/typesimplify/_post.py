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


def _fold_inline_trampolines(hir):
    """
    Detect inlined cross-page trampoline patterns that survived as intra-function
    chunks (IDA owns them as part of the calling function, so resolve_callee is
    never invoked during lifting).

    Pattern:
        Assign(Regs(DPTR), Const(offset))
        GotoStatement(label_Y)              ← label_Y matches *_call_page_N

    Replace both nodes with ExprStmt(Call(real_target)) and remove the now-dead
    page-switch helper body that follows.
    Recurses into structured bodies first.
    """
    import re
    from pseudo8051.ir.hir import Assign, GotoStatement, Label, ExprStmt, ReturnStmt
    from pseudo8051.ir.expr import Regs, Const, Call
    from pseudo8051.constants import dbg

    # Recurse into structured bodies first
    hir = [node.map_bodies(_fold_inline_trampolines) for node in hir]

    def _dptr_offset(node):
        """Return the DPTR constant offset if node is Assign(DPTR..., Const), else None."""
        if not isinstance(node, Assign):
            return None
        if not isinstance(node.rhs, Const):
            return None
        lhs = node.lhs
        if isinstance(lhs, Regs) and set(lhs.names) <= {'DPTR', 'DPL', 'DPH'}:
            return node.rhs.value & 0xFFFF
        return None

    def _collect_goto_labels(nodes):
        """Collect all GotoStatement target labels in a flat node list."""
        targets = set()
        for nd in nodes:
            if isinstance(nd, GotoStatement):
                targets.add(nd.label)
        return targets

    result = list(hir)
    i = 0
    while i < len(result):
        offset = _dptr_offset(result[i])
        if offset is None:
            i += 1
            continue

        # Find next non-Label node
        j = i + 1
        while j < len(result) and isinstance(result[j], Label):
            j += 1
        if j >= len(result) or not isinstance(result[j], GotoStatement):
            i += 1
            continue

        target_label = result[j].label
        # Detect page-switch helper by name pattern: *_call_page_N
        m = re.match(r'.*_call_page_(\d+)$', target_label)
        if not m:
            i += 1
            continue
        page_num = int(m.group(1))

        try:
            import ida_name, ida_funcs
            from pseudo8051.trampolines import _page_segment_base
            real_ea = _page_segment_base(page_num) | offset
            real_name = (ida_funcs.get_func_name(real_ea)
                         or ida_name.get_name(real_ea)
                         or hex(real_ea))
        except Exception:
            i += 1
            continue

        dbg("trampolines",
            f"  fold-inline-trampoline @{hex(result[i].ea)}: "
            f"DPTR={hex(offset)} → page {page_num} → {real_name}")

        # Build call args from the real function's prototype (same as LcallHandler).
        try:
            from pseudo8051.prototypes import get_proto, param_regs
            from pseudo8051.ir.expr import RegGroup, Name as _Name
            _proto = get_proto(real_name)
            if _proto:
                _p_regs = param_regs(_proto)
                _args = []
                for _p, _regs in zip(_proto.params, _p_regs):
                    if _regs:
                        _args.append(RegGroup(_regs) if len(_regs) > 1 else _Name("".join(_regs)))
                    else:
                        _args.append(_Name(_p.name))
            else:
                _args = []
        except Exception:
            _args = []

        dptr_node  = result[i]
        goto_node  = result[j]
        new_node = ExprStmt(result[i].ea, Call(real_name, _args))
        new_node.source_nodes = [dptr_node, goto_node]
        # Replace result[i] (DPTR assign) with new_node; remove result[j] (goto)
        result = result[:i] + [new_node] + result[i + 1:j] + result[j + 1:]
        # result[i] = new call node; dead page-switch body starts at i+1

        # Remove the now-dead page-switch helper body.
        # Collect goto targets from the live portion (everything up to and
        # including the inserted call) so we know which labels are still needed.
        live_refs = _collect_goto_labels(result[:i + 1])

        k = i + 1
        while k < len(result):
            nd = result[k]
            # Stop at a Label that is referenced from outside the dead zone
            if isinstance(nd, Label) and nd.name in live_refs:
                break
            was_return = isinstance(nd, ReturnStmt)
            k += 1
            if was_return:
                break  # page-switch chain ends at its RET

        if k > i + 1:
            result = result[:i + 1] + result[k:]

        i += 1

    return result


def _rename_byte_field_lhs(hir, byte_field_by_reg: dict):
    """
    Rename raw register LHS to their byte-field Name equivalents.

    After TypeAwareSimplifier substitutes RHS reads (e.g. R1 → src.lo) but leaves
    LHS registers unchanged, this produces mismatched nodes like:
        R1 = src.lo + expr;

    This pass renames the LHS when the register backs a byte-field variable:
        src.lo = src.lo + expr;

    This allows collapse_mb_assigns to subsequently collapse hi/lo field pairs.
    Only runs when byte_field_by_reg is non-empty (i.e. the function has 16-bit
    parameters whose byte-field VarInfo entries were recorded before _simplify).
    """
    if not byte_field_by_reg:
        return hir

    from pseudo8051.ir.hir import Assign
    from pseudo8051.ir.expr import Regs as RegExpr, Name as NameExpr

    def _rename(nodes):
        result = []
        for node in nodes:
            mapped = node.map_bodies(_rename)
            if mapped is not node:
                result.append(mapped)
                continue
            if (isinstance(node, Assign)
                    and isinstance(node.lhs, RegExpr)
                    and node.lhs.is_single):
                vi = byte_field_by_reg.get(node.lhs.name)
                if vi is not None and vi.name:
                    new_node = Assign(node.ea, NameExpr(vi.name), node.rhs)
                    node.copy_meta_to(new_node)
                    new_node.source_nodes = [node]
                    result.append(new_node)
                    continue
            result.append(mapped)
        return result

    return _rename(hir)


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
    from pseudo8051.passes.patterns._node_utils import _fold_exprs_in_node
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
                prev = patched
                patched = patched.copy_meta_to(Assign(patched.ea, new_lhs, patched.rhs))
                patched.source_nodes = [prev]
        # Fold after substitution so that byte-field reconstructions like
        # (x.hi << 8) | x.lo collapse to Name("x").
        if patched is not node:
            patched = _fold_exprs_in_node(patched)
        result.append(patched.map_bodies(_visit))
    return result


def _subst_iram_in_hir(nodes, reg_map):
    """Walk the HIR and apply _subst_iram_in_expr to every expression.

    Handles both RHS reads and IRAMRef LHS write positions so that
    IRAM[addr] = expr becomes local_name = expr as well.
    """
    from pseudo8051.passes.patterns._utils import _subst_iram_in_expr, _apply_expr_subst_to_node
    from pseudo8051.ir.hir import Assign
    from pseudo8051.ir.expr import IRAMRef

    def _subst_fn(e):
        return _subst_iram_in_expr(e, reg_map)

    def _visit(ns):
        return _subst_iram_in_hir(ns, reg_map)

    result = []
    for node in nodes:
        patched = _apply_expr_subst_to_node(node, _subst_fn)
        if isinstance(patched, Assign) and isinstance(patched.lhs, IRAMRef):
            new_lhs = _subst_fn(patched.lhs)
            if new_lhs is not patched.lhs:
                prev = patched
                patched = patched.copy_meta_to(Assign(patched.ea, new_lhs, patched.rhs))
                patched.source_nodes = [prev]
        result.append(patched.map_bodies(_visit))
    return result
