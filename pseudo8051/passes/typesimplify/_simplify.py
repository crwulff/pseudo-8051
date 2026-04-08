"""
passes/typesimplify/_simplify.py — Boolean helpers, default transform, core simplifier.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir    import (HIRNode, Assign, CompoundAssign,
                                   ExprStmt, ReturnStmt, IfGoto, IfNode, WhileNode, ForNode,
                                   DoWhileNode, SwitchNode)
from pseudo8051.passes.patterns         import _PATTERNS
from pseudo8051.passes.patterns._utils  import (
    VarInfo, _replace_pairs, _replace_xram_syms, _replace_single_regs,
    _subst_all_expr,
    _walk_expr,
)
from pseudo8051.ir.expr import (Expr, UnaryOp, BinOp)

# ── Boolean condition simplification ─────────────────────────────────────────

_NEGATE_OP = {
    "==": "!=", "!=": "==",
    "<":  ">=", ">":  "<=",
    "<=": ">",  ">=": "<",
}

_RE_NOT_CMP = re.compile(r'^!\((.+?)\s+(!=|==|>=|<=|>|<)\s+(.+)\)$')


def _simplify_bool_expr(expr: Expr) -> Expr:
    """Push `!` inward through comparisons; eliminate double negation.

    !(lhs op rhs)  →  lhs ~op rhs   (e.g. !(A != 0) → A == 0)
    !!x            →  x
    """
    def _fn(e: Expr) -> Expr:
        if isinstance(e, UnaryOp) and e.op == "!":
            inner = e.operand
            if isinstance(inner, BinOp) and inner.op in _NEGATE_OP:
                return BinOp(inner.lhs, _NEGATE_OP[inner.op], inner.rhs)
            if isinstance(inner, UnaryOp) and inner.op == "!":
                return inner.operand
        return e
    return _walk_expr(expr, _fn)


def _simplify_bool_str(cond: str) -> str:
    """String-condition version of boolean simplification."""
    m = _RE_NOT_CMP.match(cond)
    if m:
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        if op in _NEGATE_OP:
            return f"{lhs} {_NEGATE_OP[op]} {rhs}"
    # !(!(expr)) → expr
    if cond.startswith("!(!(") and cond.endswith("))"):
        return cond[2:-1]
    return cond


# ── Default node transformation ───────────────────────────────────────────────


def _effective_map(node: HIRNode, base_eff: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """Build per-node effective reg map from base_eff merged with node annotations.

    Adds entries from node.ann.reg_names, node.ann.call_arg_ann, and
    node.ann.callee_args via setdefault, so annotations never override entries
    already present in base_eff (which has already been pruned by _kill_written).
    """
    ann = node.ann
    if ann is None:
        return base_eff

    eff = dict(base_eff)
    for r, vi in ann.reg_names.items():
        if not vi.is_retval:
            eff.setdefault(r, vi)

    # call_arg_ann is back-propagated from the NEXT use of each register.
    # For registers being WRITTEN by this node, call_arg names the destination —
    # it takes precedence over stale pre-write reg_names entries so that patterns
    # (e.g. ConstGroupPattern) use the correct name for the write target.
    written = node.written_regs
    for r, vi in ann.call_arg_names.items():
        if r in written:
            # Evict stale VarInfo entries (for all of old_vi's regs)
            # so they don't appear as competing ConstGroupPattern candidates.
            old_vi = eff.get(r)
            if old_vi is not None and old_vi is not vi and getattr(old_vi, 'regs', None):
                for old_r in old_vi.regs:
                    if eff.get(old_r) is old_vi:
                        del eff[old_r]
            # Install the call_arg VarInfo for all its regs.
            for new_r in vi.regs:
                eff[new_r] = vi
        else:
            eff.setdefault(r, vi)

    _callee_dict = ann.callee_args_dict
    if _callee_dict:
        for r, vi in _callee_dict.items():
            eff.setdefault(r, vi)

    # Unconditionally evict stale retval struct-field entries for written registers.
    # RegCopyGroupPattern copies field regs to new regs but leaves source entries
    # in reg_map; without this, ConstGroupPattern would use the stale field name
    # as the write destination for subsequent const-loads into the same registers.
    for r in written:
        old_vi = eff.get(r)
        if (old_vi is not None and isinstance(old_vi, VarInfo)
                and old_vi.is_retval_field and old_vi.regs and not old_vi.xram_sym):
            for old_r in old_vi.regs:
                if eff.get(old_r) is old_vi:
                    del eff[old_r]

    return eff


def _kill_written(reg_map: Dict[str, VarInfo],
                  written_regs: frozenset,
                  pre_snap: Optional[Dict[str, VarInfo]] = None) -> Dict[str, VarInfo]:
    """Return reg_map with ALL register-backed entries whose registers overlap
    *written_regs* removed.

    If *pre_snap* is provided, only entries present in pre_snap are eligible
    for removal — entries that were absent from pre_snap (newly added by the
    current pattern match) are immune so they survive the node that created them.

    XRAM-backed entries (vinfo.xram_sym != '') are never removed.
    """
    if not written_regs:
        return reg_map
    result = {}
    for k, v in reg_map.items():
        if isinstance(v, VarInfo) and v.regs and not v.xram_sym:
            if any(r in written_regs for r in v.regs):
                if pre_snap is None or k in pre_snap:
                    continue   # kill this entry
        result[k] = v
    return result


def _subst_text(text: str, reg_map: Dict[str, VarInfo]) -> str:
    """Apply XRAM-symbol, register-pair, and single-reg-param substitutions to text."""
    text = _replace_xram_syms(text, reg_map)
    text = _replace_pairs(text, reg_map)
    text = _replace_single_regs(text, reg_map)
    return text


def _subst_expr(expr, reg_map: Dict[str, VarInfo]):
    """Apply all substitutions to an Expr node."""
    return _subst_all_expr(expr, reg_map)


def _transform_default(node: HIRNode,
                       reg_map: Dict[str, VarInfo],
                       simplify_fn=None) -> Optional[HIRNode]:
    """
    Fallback for nodes not consumed by any pattern.

    • Drops ``DPTR = sym;`` lines when sym is a declared XRAM local.
    • Applies substitutions to Statement text or Expr nodes.
    • Recurses into children of structured nodes.

    Returns None to signal that the node should be dropped.

    simplify_fn is called for nested node lists (IfNode/WhileNode/ForNode bodies)
    so that the caller's flow-sensitive kill state propagates inward.
    """
    if simplify_fn is None:
        simplify_fn = _simplify

    def _out(new_node: HIRNode) -> HIRNode:
        """Copy annotation from input node to any newly created output node."""
        new_node.ann = node.ann
        return new_node

    if isinstance(node, Assign):
        from pseudo8051.ir.hir import TypedAssign
        from pseudo8051.ir.expr import Reg, Regs as RegExpr, Name as NameExpr
        if node.lhs == Reg("DPTR"):
            sym = node.rhs.render()
            if any(v.xram_sym == sym for v in reg_map.values()
                   if isinstance(v, VarInfo) and v.xram_sym):
                return None
        # Substitute inside indirect LHS references (IRAMRef/XRAMRef pointer
        # registers should be renamed just like any other read position).
        # Simple Reg LHS are write destinations — do not fully substitute.
        # RegGroup LHS: apply pair alias only (adds the variable name as alias;
        # the kill mechanism ensures stale struct-member aliases are absent from eff).
        new_lhs = node.lhs
        if not isinstance(node.lhs, RegExpr):
            new_lhs = _subst_expr(node.lhs, reg_map)
        new_rhs = _subst_expr(node.rhs, reg_map)
        if new_lhs is not node.lhs or new_rhs is not node.rhs:
            if isinstance(node, TypedAssign):
                return _out(TypedAssign(node.ea, node.type_str, new_lhs, new_rhs))
            return _out(Assign(node.ea, new_lhs, new_rhs))
        return node

    if isinstance(node, CompoundAssign):
        new_rhs = _subst_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return _out(CompoundAssign(node.ea, node.lhs, node.op, new_rhs))
        return node

    if isinstance(node, ExprStmt):
        new_expr = _subst_expr(node.expr, reg_map)
        if new_expr is not node.expr:
            return _out(ExprStmt(node.ea, new_expr))
        return node

    if isinstance(node, ReturnStmt):
        if node.value is not None:
            new_val = _subst_expr(node.value, reg_map)
            if new_val is not node.value:
                return _out(ReturnStmt(node.ea, new_val))
        return node

    if isinstance(node, IfGoto):
        new_cond = _simplify_bool_expr(_subst_expr(node.cond, reg_map))
        if new_cond is not node.cond:
            return _out(IfGoto(node.ea, new_cond, node.label))
        return node

    if isinstance(node, IfNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return _out(IfNode(
            ea         = node.ea,
            condition  = new_cond,
            then_nodes = simplify_fn(node.then_nodes, reg_map),
            else_nodes = simplify_fn(node.else_nodes, reg_map),
        ))
    if isinstance(node, WhileNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return _out(WhileNode(
            ea         = node.ea,
            condition  = new_cond,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
        ))
    if isinstance(node, ForNode):
        init = node.init
        cond = node.condition
        update = node.update
        if isinstance(init, str):
            new_init = _subst_text(init, reg_map)
        elif isinstance(init, Assign):
            new_rhs = _subst_expr(init.rhs, reg_map)
            new_init = Assign(init.ea, init.lhs, new_rhs) if new_rhs is not init.rhs else init
        else:
            new_init = init
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        if isinstance(update, str):
            new_update = _subst_text(update, reg_map)
        else:
            new_update = _subst_expr(update, reg_map)
        return _out(ForNode(
            ea         = node.ea,
            init       = new_init,
            condition  = new_cond,
            update     = new_update,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
        ))
    if isinstance(node, DoWhileNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return _out(DoWhileNode(
            ea         = node.ea,
            condition  = new_cond,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
        ))
    if isinstance(node, SwitchNode):
        new_subject = _subst_expr(node.subject, reg_map)
        new_cases = [
            (vals, simplify_fn(body, reg_map) if isinstance(body, list) else body)
            for vals, body in node.cases
        ]
        new_default_body = (
            simplify_fn(node.default_body, reg_map)
            if isinstance(node.default_body, list) else node.default_body
        )
        return _out(SwitchNode(node.ea, new_subject, new_cases,
                               node.default_label, new_default_body))
    return node


# ── Core simplifier walk ──────────────────────────────────────────────────────

def _simplify_once(nodes: List[HIRNode], reg_map: Dict[str, VarInfo],
                   simplify_fn=None) -> List[HIRNode]:
    """Apply one round of pattern matching + default transforms to each node.

    Applies unified kill tracking: every register write removes all register-backed
    entries for those registers from the working copy of reg_map so that stale
    mappings never reach subsequent nodes.
    """
    if simplify_fn is None:
        simplify_fn = _simplify
    out: List[HIRNode] = []
    cur_map = reg_map   # updated after each node via _kill_written
    for node in nodes:
        eff = _effective_map(node, cur_map)
        for pat in _PATTERNS:
            result = pat.match([node], 0, eff, simplify_fn)
            if result is not None:
                out.extend(result[0])
                break
        else:
            transformed = _transform_default(node, eff, simplify_fn)
            if transformed is not None:
                out.append(transformed)
        written = node.written_regs | node.definitely_killed()
        cur_map = _kill_written(cur_map, written)
    return out


def _simplify(nodes: List[HIRNode], reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Walk nodes, trying each registered Pattern in turn.  Falls back to
    _transform_default for nodes not consumed by any pattern.

    Unified kill tracking: every register write removes ALL register-backed
    VarInfo entries for those registers from reg_map.  Entries newly added by
    a pattern match are immune for the creating node via pre-snapshot comparison.
    Structured-node definite kills are applied after each such node.
    """
    out: List[HIRNode] = []
    i = 0

    def _sub_simplify(ns: List[HIRNode], rm: Dict[str, VarInfo]) -> List[HIRNode]:
        """Recursive simplify; receives the already-killed reg_map."""
        return _simplify(ns, rm)

    while i < len(nodes):
        eff = _effective_map(nodes[i], reg_map)
        # Snapshot eff BEFORE the pattern runs so we can distinguish annotation-derived
        # entries (added by _effective_map from reg_groups/call_arg_ann/callee_args) from
        # entries mutated by the pattern itself.  Annotation-derived entries that the
        # pattern doesn't touch should be killable, not immune like true pattern mutations.
        pre_pattern_eff = dict(eff)

        for pat in _PATTERNS:
            result = pat.match(nodes, i, eff, _sub_simplify)
            if result is not None:
                replacement, new_i = result
                # Snapshot reg_map BEFORE propagating pattern mutations so that
                # _kill_written can distinguish pre-existing entries (eligible for
                # killing) from entries newly added by the pattern (immune).
                pre_snap = dict(reg_map)
                # Propagate new/changed keys from eff back to reg_map.
                if eff is not reg_map:
                    for k, v in eff.items():
                        if reg_map.get(k) is not v:
                            reg_map[k] = v
                # Remove from pre_snap any keys whose values were changed by the
                # pattern — changed entries are treated as newly-added and immune.
                for k in list(pre_snap.keys()):
                    if reg_map.get(k) is not pre_snap[k]:
                        del pre_snap[k]
                # Annotation-derived entries were in eff (from _effective_map) but not
                # in reg_map, so they're absent from pre_snap and would be treated as
                # immune.  If the pattern didn't mutate them (value unchanged from
                # pre_pattern_eff), mark them as eligible for killing by adding them to
                # pre_snap — they're stale annotation data, not new pattern outputs.
                for k, v_before in pre_pattern_eff.items():
                    if k not in pre_snap and eff.get(k) is v_before:
                        pre_snap[k] = reg_map.get(k)
                # Kill pre-existing entries for all registers written by consumed nodes.
                written: frozenset = frozenset()
                for j in range(i, new_i):
                    written |= nodes[j].written_regs | nodes[j].definitely_killed()
                reg_map = _kill_written(reg_map, written, pre_snap)
                i = new_i
                out.extend(_simplify_once(replacement, reg_map, _sub_simplify))
                break
        else:
            # Transform BEFORE killing so the node's own RHS sees current mapping.
            transformed = _transform_default(nodes[i], eff, _sub_simplify)
            if transformed is not None:
                out.append(transformed)
            # Kill AFTER transform — includes structured-node definite kills.
            written = nodes[i].written_regs | nodes[i].definitely_killed()
            reg_map = _kill_written(reg_map, written)
            i += 1
    return out
