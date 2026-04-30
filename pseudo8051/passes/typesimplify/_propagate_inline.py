"""
passes/typesimplify/_propagate_inline.py — Sub-passes A1, B, and C.

A1: _subst_from_reg_exprs   — substitute reg_exprs annotation values into nodes
B:  _inline_retvals         — inline single-use retval TypedAssign into its use
C:  _inline_group_setups    — fold single-use multi-register setup into call args
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,
                                ExprStmt, ReturnStmt, IfGoto, IfNode,
                                NodeAnnotation)
from pseudo8051.ir.expr import (Expr, Call, Const, Regs as RegExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import (
    VarInfo, _count_reg_uses_in_node, _subst_reg_in_node,
    _walk_expr, _canonicalize_expr, _fold_exprs_in_node, _is_reg_free,
)
from pseudo8051.constants import dbg
from pseudo8051.passes.typesimplify._propagate_utils import (
    _expr_name_refs,
    _count_group_uses_in_node,
    _collect_mid_writes,
    _as_retval_stmt,
    _count_name_uses_in_nodes,
    _dbg_node,
)


# ── Sub-pass C: multi-register group setup inlining ──────────────────────────

def _subst_group_in_call_node(node: HIRNode, regs_tuple: tuple,
                               replacement: Expr) -> Optional[HIRNode]:
    """Replace Regs(names==regs_tuple) in call args of node with replacement."""
    def _patch(call: Call) -> Optional[Call]:
        new_args = []
        found = False
        for a in call.args:
            if isinstance(a, RegExpr) and not a.is_single and a.names == regs_tuple:
                new_args.append(replacement)
                found = True
            else:
                new_args.append(a)
        return Call(call.func_name, new_args) if found else None

    result: Optional[HIRNode] = None
    if isinstance(node, Assign) and isinstance(node.rhs, Call):
        new_call = _patch(node.rhs)
        if new_call is not None:
            result = node._rebuild(node.lhs, new_call)
    elif (isinstance(node, Assign)
          and isinstance(node.rhs, RegExpr)
          and not node.rhs.is_single
          and node.rhs.names == regs_tuple):
        result = node._rebuild(node.lhs, replacement)
    elif isinstance(node, ExprStmt) and isinstance(node.expr, Call):
        new_call = _patch(node.expr)
        if new_call is not None:
            result = ExprStmt(node.ea, new_call)
    elif isinstance(node, IfNode):
        def _patch_in_expr(e: Expr) -> Expr:
            if isinstance(e, Call):
                new_call = _patch(e)
                if new_call is not None:
                    return new_call
            return e
        new_cond = _walk_expr(node.condition, _patch_in_expr)
        if new_cond is not node.condition:
            result = IfNode(node.ea, new_cond, node.then_nodes, node.else_nodes)
    if result is not None:
        node.copy_meta_to(result)
        result.source_nodes = [node]
    return result


def _inline_group_setups(live: List[HIRNode],
                          reg_map: Dict = {}) -> Tuple[List[HIRNode], bool]:
    """
    C: Fold single-use multi-register setup assignments into call arguments.
    """
    changed = False
    live = list(live)
    i = 0
    while i < len(live):
        node = live[i]
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr)
                and not node.lhs.is_single):
            i += 1
            continue

        regs_tuple = node.lhs.names
        regs_set = set(regs_tuple)
        rhs = node.rhs
        if node.ann is not None:
            rhs = _canonicalize_expr(rhs,
                                     node.ann.reg_consts or {},
                                     node.ann.reg_groups or [],
                                     node.ann.reg_exprs or {})

        use_idx = None
        conflict = False
        for j in range(i + 1, len(live)):
            nd = live[j]
            uses = _count_group_uses_in_node(regs_tuple, nd)
            writes = nd.possibly_killed() & regs_set
            if uses > 0:
                if use_idx is not None:
                    conflict = True
                    break
                use_idx = j
            if writes:
                break

        if conflict or use_idx is None:
            i += 1
            continue

        if use_idx > i + 1:
            rhs_refs = _expr_name_refs(rhs)
            if rhs_refs:
                mid_writes = _collect_mid_writes(live[i + 1:use_idx], reg_map)
                if rhs_refs & mid_writes:
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] prop-group: blocked — "
                        f"intermediate writes {rhs_refs & mid_writes}")
                    i += 1
                    continue

        _GP_REGS = {'R0','R1','R2','R3','R4','R5','R6','R7','A','B'}
        use_ann = getattr(live[use_idx], 'ann', None)
        if use_ann is not None and use_ann.reg_exprs:
            _hidden_kill = False
            for _r in _expr_name_refs(rhs):
                if _r not in _GP_REGS:
                    continue
                _use_val = use_ann.reg_exprs.get(_r)
                if _use_val is None or not isinstance(_use_val, Const):
                    continue
                _def_val = node.ann.reg_exprs.get(_r) if node.ann is not None else None
                if _def_val is None or _def_val != _use_val:
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] prop-group: blocked — "
                        f"hidden kill {_r}: def={_def_val!r} use={_use_val.render()!r}")
                    _hidden_kill = True
                    break
            if _hidden_kill:
                i += 1
                continue

        group_node = live[i]
        new_use = _subst_group_in_call_node(live[use_idx], regs_tuple, rhs)
        if new_use is None:
            i += 1
            continue

        new_use.source_nodes = [group_node] + list(new_use.source_nodes)
        live[use_idx] = new_use
        live[i] = None
        dbg("typesimp",
            f"  [{hex(node.ea)}] prop-group: folded {''.join(regs_tuple)} = "
            f"{rhs.render()!r} into call")
        changed = True
        i += 1

    return [n for n in live if n is not None], changed


# ── Sub-pass A1: reg_exprs annotation substitution ───────────────────────────

def _expr_safe_to_subst(expr: Expr, ann) -> bool:
    """True if expr is safe to substitute at ann's node (reg-free or no TypeGroup regs)."""
    if _is_reg_free(expr):
        return True
    if ann is None:
        return False
    typegroup_regs: set = set()
    for g in (ann.reg_groups or []):
        typegroup_regs.update(g.active_regs)
    found = [False]
    def _fn(e: Expr) -> Expr:
        if isinstance(e, RegExpr) and e.is_single and e.name in typegroup_regs:
            found[0] = True
        return e
    _walk_expr(expr, _fn)
    return not found[0]


def _subst_from_reg_exprs(live: List[HIRNode]) -> Tuple[List[HIRNode], bool]:
    """Sub-pass A1: substitute reg_exprs annotations directly into node expressions."""
    result = list(live)
    any_changed = False
    for i, node in enumerate(result):
        if node.ann is None or not node.ann.reg_exprs:
            continue
        current = node
        for r, expr in current.ann.reg_exprs.items():
            if not _expr_safe_to_subst(expr, current.ann):
                continue
            if r in _expr_name_refs(expr):
                continue
            if _count_reg_uses_in_node(r, current) == 0:
                continue
            new_node = _subst_reg_in_node(current, r, expr)
            if new_node is None:
                continue
            new_node = _fold_exprs_in_node(new_node)
            current = new_node
            any_changed = True
        if (current.ann is not None
                and "DPTR" not in current.ann.reg_exprs
                and "DPH" in current.ann.reg_exprs
                and "DPL" in current.ann.reg_exprs
                and _count_reg_uses_in_node("DPTR", current) > 0):
            dph = current.ann.reg_exprs["DPH"]
            dpl = current.ann.reg_exprs["DPL"]
            if (_expr_safe_to_subst(dph, current.ann)
                    and _expr_safe_to_subst(dpl, current.ann)
                    and "DPTR" not in _expr_name_refs(dph)
                    and "DPTR" not in _expr_name_refs(dpl)):
                from pseudo8051.ir.expr import BinOp as _BinOp, Const as _Const, Paren as _Paren
                dptr_expr = _BinOp(_Paren(_BinOp(dph, "<<", _Const(8))), "|", dpl)
                new_node = _subst_reg_in_node(current, "DPTR", dptr_expr)
                if new_node is not None:
                    current = _fold_exprs_in_node(new_node)
                    any_changed = True
        if current is not node:
            result[i] = current
    return result, any_changed


# ── Sub-pass B: retval inlining ───────────────────────────────────────────────

def _inline_retvals(live: List[HIRNode],
                    reg_map: Dict = {}) -> Tuple[List[HIRNode], bool]:
    """
    B: For each retval TypedAssign with exactly one downstream use of the
    retval name, inline the call expression into the target and remove the TypedAssign.
    """
    changed = False
    live = list(live)
    i = 0
    while i < len(live):
        rv = _as_retval_stmt(live[i])
        if rv is None:
            i += 1
            continue
        retval_name, call_expr = rv
        remaining = live[i + 1:]
        total_uses = _count_name_uses_in_nodes(retval_name, remaining)

        if total_uses == 1:
            for j, tgt in enumerate(remaining):
                if _count_reg_uses_in_node(retval_name, tgt) == 1:
                    abs_j = i + 1 + j
                    dbg("propagate",
                        f"  inline-retval [{hex(live[i].ea)}]: {retval_name} "
                        f"j={j} abs_j={abs_j} "
                        f"intermediates={[_dbg_node(live[k]) for k in range(i+1, abs_j)]}")
                    if j > 0:
                        call_reads = _expr_name_refs(call_expr)
                        mid_writes = _collect_mid_writes(live[i + 1:abs_j], reg_map)
                        dbg("propagate",
                            f"    call_reads={sorted(call_reads)} "
                            f"mid_writes={sorted(mid_writes)} "
                            f"conflict={sorted(call_reads & mid_writes)}")
                        if call_reads & mid_writes:
                            dbg("typesimp",
                                f"  [{hex(live[i].ea)}] inline-retval: blocked — "
                                f"intermediate writes {call_reads & mid_writes}")
                            break
                    src_node = live[i]
                    if (isinstance(tgt, Assign)
                            and isinstance(tgt.rhs, (NameExpr, RegExpr))
                            and (not isinstance(tgt.rhs, RegExpr) or tgt.rhs.is_single)
                            and tgt.rhs.name == retval_name):
                        new_node = tgt._rebuild(tgt.lhs, call_expr)
                        new_node.ann = NodeAnnotation.merge(src_node, tgt)
                        new_node.source_nodes = [src_node, tgt]
                        live[abs_j] = new_node
                    else:
                        new_node = _subst_reg_in_node(tgt, retval_name, call_expr)
                        if new_node is not None:
                            new_node.ann = NodeAnnotation.merge(src_node, tgt)
                            live[abs_j] = new_node
                    src_ea = hex(src_node.ea)
                    live[i] = None
                    dbg("typesimp",
                        f"  [{src_ea}] prop-values: inlined {retval_name} into node {abs_j}")
                    changed = True
                    break

        i += 1

    return [n for n in live if n is not None], changed
