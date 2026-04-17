"""
passes/typesimplify/_simplify.py — Boolean helpers, default transform, core simplifier.
"""

import re
from typing import Dict, FrozenSet, List, Optional, Tuple

from pseudo8051.ir.hir    import (HIRNode, Assign, CompoundAssign,
                                   ExprStmt, ReturnStmt, IfGoto, IfNode, WhileNode, ForNode,
                                   DoWhileNode, SwitchNode)
from pseudo8051.passes.patterns         import _PATTERNS
from pseudo8051.passes.patterns._utils  import (
    TypeGroup, VarInfo, _replace_pairs, _replace_xram_syms, _replace_single_regs,
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
    """Push `!` inward through comparisons and normalise arithmetic comparisons to == K form.

    !(lhs op rhs)  →  lhs ~op rhs   (e.g. !(A != 0) → A == 0)
    !!x            →  x
    ~x == 0        →  x == 0xFF     (CPL then JZ pattern)
    (x ^ K) == 0  →  x == K         (XRL then JZ pattern)
    (x + K) == 0  →  x == (256-K)&0xFF  (ADD then JZ pattern)
    (x - K) == 0  →  x == K
    Mirror forms for != 0.
    """
    from pseudo8051.ir.expr import Const as _Const

    def _fn(e: Expr) -> Expr:
        if isinstance(e, UnaryOp) and e.op == "!":
            inner = e.operand
            if isinstance(inner, BinOp) and inner.op in _NEGATE_OP:
                return BinOp(inner.lhs, _NEGATE_OP[inner.op], inner.rhs)
            if isinstance(inner, UnaryOp) and inner.op == "!":
                return inner.operand
        # Arithmetic normalisations: (expr op 0) → (base_expr == K)
        if (isinstance(e, BinOp)
                and e.op in ("==", "!=")
                and isinstance(e.rhs, _Const)
                and e.rhs.value == 0):
            lhs = e.lhs
            op  = e.op
            # ~x == 0  →  x == 0xFF
            if isinstance(lhs, UnaryOp) and lhs.op == "~":
                return BinOp(lhs.operand, op, _Const(0xFF))
            # (x ^ K) == 0  →  x == K
            if isinstance(lhs, BinOp) and lhs.op == "^" and isinstance(lhs.rhs, _Const):
                return BinOp(lhs.lhs, op, lhs.rhs)
            # (x + K) == 0  →  x == (256-K)&0xFF
            if isinstance(lhs, BinOp) and lhs.op == "+" and isinstance(lhs.rhs, _Const):
                return BinOp(lhs.lhs, op, _Const((256 - lhs.rhs.value) & 0xFF))
            # (x - K) == 0  →  x == K
            if isinstance(lhs, BinOp) and lhs.op == "-" and isinstance(lhs.rhs, _Const):
                return BinOp(lhs.lhs, op, lhs.rhs)
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


# ── TypeGroup working-state helpers ──────────────────────────────────────────


def _tg_to_varinfo(tg: TypeGroup) -> VarInfo:
    """Convert a TypeGroup to a VarInfo for pattern compat (uses full_regs)."""
    return VarInfo(tg.name, tg.type, tg.full_regs,
                   xram_sym=tg.xram_sym, is_param=tg.is_param)


def _varinfo_to_groups(reg_map: Dict) -> List[TypeGroup]:
    """Convert non-XRAM VarInfo entries in reg_map to TypeGroups.

    active_regs is inferred from which individual-register keys in reg_map
    point to each VarInfo (pair-name keys like 'R6R7' are ignored).
    """
    seen: Dict[int, Tuple[VarInfo, set]] = {}
    for k, v in reg_map.items():
        if not isinstance(v, VarInfo) or v.xram_sym or k == "__n__":
            continue
        if not v.regs or k not in v.regs:
            continue
        vi_id = id(v)
        if vi_id not in seen:
            seen[vi_id] = (v, set())
        seen[vi_id][1].add(k)
    return [TypeGroup(vi.name, vi.type, vi.regs,
                      active_regs=frozenset(active),
                      is_param=vi.is_param)
            for vi, active in seen.values()]


def _kill_groups_written(groups: List[TypeGroup],
                          killed_regs: FrozenSet[str],
                          written: FrozenSet[str]) -> Tuple[List[TypeGroup], FrozenSet[str]]:
    """Narrow active_regs of each group for written registers; discard empty groups.

    Returns updated (groups, killed_regs).  XRAM-backed groups are never touched.
    """
    if not written:
        return groups, killed_regs
    result: List[TypeGroup] = []
    new_killed = set(killed_regs)
    for g in groups:
        if g.xram_sym:
            result.append(g)
            continue
        ng: Optional[TypeGroup] = g
        for r in written:
            if ng is not None and r in ng.active_regs:
                ng = ng.killed(r)
                new_killed.add(r)
        if ng is not None:
            result.append(ng)
    return result, frozenset(new_killed)


def _absorb_eff_mutations(eff: Dict, pre_eff: Dict,
                           extra_groups: List[TypeGroup],
                           killed_regs: FrozenSet[str]
                           ) -> Tuple[List[TypeGroup], FrozenSet[str]]:
    """Detect new/changed/deleted VarInfo entries in eff after a pattern ran.

    New or changed entries → TypeGroup added to extra_groups (evicts overlapping).
    Deleted entries (present in pre_eff but removed from eff by the pattern) → those
    registers are killed from extra_groups so they cannot reappear as stale candidates.
    Newly installed registers are removed from killed_regs.
    """
    result = list(extra_groups)
    new_killed = set(killed_regs)

    # Deletions: regs that patterns explicitly removed from eff.
    for k, v_before in pre_eff.items():
        if k in eff:
            continue
        if not isinstance(v_before, VarInfo) or v_before.xram_sym or k == "__n__":
            continue
        if not v_before.regs or k not in v_before.regs:
            continue
        result2: List[TypeGroup] = []
        for g in result:
            ng = g.killed(k) if k in g.active_regs else g
            if ng is not None:
                result2.append(ng)
        result = result2
        new_killed.add(k)

    # Additions/changes: new or mutated VarInfo entries.
    seen_vi_ids: set = set()
    for k, v in eff.items():
        if not isinstance(v, VarInfo) or v.xram_sym or k == "__n__":
            continue
        if not v.regs or k not in v.regs:
            continue
        if pre_eff.get(k) is v:
            continue   # unchanged
        if id(v) in seen_vi_ids:
            continue
        seen_vi_ids.add(id(v))
        active = frozenset(r for r in v.regs if eff.get(r) is v)
        if not active:
            continue
        tg = TypeGroup(v.name, v.type, v.regs, active_regs=active,
                       xram_sym=v.xram_sym, is_param=v.is_param)
        result = [g for g in result if not (g.active_regs & tg.active_regs)]
        result.append(tg)
        new_killed -= active
    return result, frozenset(new_killed)


def _build_node_eff(node: HIRNode,
                    extra_groups: List[TypeGroup],
                    killed_regs: FrozenSet[str],
                    xram_map: Dict,
                    counter) -> Dict:
    """Build per-node effective map from extra_groups + annotation + xram_map.

    Priority (highest → lowest):
      1. xram_map / counter  — always present, never killed
      2. extra_groups        — pattern-accumulated, kill-tracked (force-install)
      3. call_arg_ann written regs — evict stale + force-install
      4. reg_groups (annotation snapshot) — setdefault, skip killed_regs
      5. call_arg_ann non-written regs — setdefault
      6. callee_args         — setdefault (lowest)
    """
    eff: Dict = dict(xram_map)
    if counter is not None:
        eff["__n__"] = counter

    # extra_groups — force (pattern-created state is authoritative)
    seen_tg_ids: set = set()
    for tg in extra_groups:
        if tg.xram_sym or id(tg) in seen_tg_ids:
            continue
        seen_tg_ids.add(id(tg))
        vi = _tg_to_varinfo(tg)
        for r in tg.active_regs:
            eff[r] = vi
        if len(tg.full_regs) > 1 and tg.is_complete:
            eff[tg.pair_name] = vi

    ann = node.ann
    if ann is None:
        return eff

    written = node.written_regs

    # reg_groups (annotation snapshot) — setdefault, honouring killed_regs
    for tg in ann.reg_groups:
        vi = _tg_to_varinfo(tg)
        for r in tg.active_regs:
            if r not in killed_regs:
                eff.setdefault(r, vi)
        if (len(tg.full_regs) > 1 and tg.is_complete
                and not any(r in killed_regs for r in tg.full_regs)):
            eff.setdefault(tg.pair_name, vi)

    # call_arg_ann — written regs: evict stale entries then force-install;
    #                non-written regs: setdefault
    for tg in ann.call_arg_ann:
        vi = _tg_to_varinfo(tg)
        written_in_tg = tg.active_regs & written
        if written_in_tg:
            for r in written_in_tg:
                old_vi = eff.get(r)
                if old_vi is not None and old_vi is not vi and getattr(old_vi, 'regs', None):
                    for old_r in old_vi.regs:
                        if eff.get(old_r) is old_vi:
                            del eff[old_r]
            for r in tg.active_regs:
                eff[r] = vi
            if len(tg.full_regs) > 1:
                eff[tg.pair_name] = vi
        else:
            for r in tg.active_regs:
                eff.setdefault(r, vi)

    # callee_args — setdefault (lowest priority)
    if ann.callee_args is not None:
        for tg in ann.callee_args:
            vi = _tg_to_varinfo(tg)
            for r in tg.active_regs:
                eff.setdefault(r, vi)
            if len(tg.full_regs) > 1:
                eff.setdefault(tg.pair_name, vi)

    return eff


def _rebuild_reg_map(extra_groups: List[TypeGroup], xram_map: Dict, counter) -> Dict:
    """Build a reg_map dict from extra_groups + xram_map for passing to _simplify."""
    result = dict(xram_map)
    if counter is not None:
        result["__n__"] = counter
    for tg in extra_groups:
        if tg.xram_sym:
            continue
        vi = _tg_to_varinfo(tg)
        for r in tg.active_regs:
            result[r] = vi
        if len(tg.full_regs) > 1 and tg.is_complete:
            result[tg.pair_name] = vi
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
        # Convert  x = ++x  /  x = --x  →  ExprStmt(++x) / ExprStmt(--x).
        # Pre-increment already updates x in place, so the enclosing Assign is
        # redundant; a bare increment/decrement statement is cleaner.
        from pseudo8051.ir.expr import UnaryOp as _UnaryOp
        if (isinstance(node.rhs, _UnaryOp)
                and node.rhs.op in ('++', '--')
                and not node.rhs.post
                and node.lhs == node.rhs.operand):
            return _out(ExprStmt(node.ea, node.rhs))
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

    Uses TypeGroup-based kill tracking: every register write narrows the
    active_regs of affected TypeGroups (removing them when empty) so stale
    mappings never reach subsequent nodes.
    """
    if simplify_fn is None:
        simplify_fn = _simplify

    xram_map = {k: v for k, v in reg_map.items()
                if isinstance(v, VarInfo) and v.xram_sym}
    counter = reg_map.get("__n__")
    extra_groups: List[TypeGroup] = _varinfo_to_groups(reg_map)
    killed_regs: FrozenSet[str] = frozenset()

    out: List[HIRNode] = []
    for node in nodes:
        # Inject user annotations into extra_groups so they propagate forward
        # regardless of killed_regs (user annotations override prior writes).
        node_ann = node.ann
        if node_ann is not None and node_ann.user_anns:
            for tg in node_ann.user_anns:
                extra_groups = [g for g in extra_groups
                                if not (g.active_regs & tg.active_regs)]
                extra_groups.append(tg)
                killed_regs = killed_regs - tg.active_regs
        eff = _build_node_eff(node, extra_groups, killed_regs, xram_map, counter)
        pre_eff = dict(eff)
        written = node.written_regs | node.definitely_killed()
        for pat in _PATTERNS:
            result = pat.match([node], 0, eff, simplify_fn)
            if result is not None:
                # Kill old entries BEFORE absorbing pattern mutations so that
                # newly-added TypeGroups are immune to the kill.
                extra_groups, killed_regs = _kill_groups_written(extra_groups, killed_regs, written)
                extra_groups, killed_regs = _absorb_eff_mutations(
                    eff, pre_eff, extra_groups, killed_regs)
                out.extend(result[0])
                break
        else:
            transformed = _transform_default(node, eff, simplify_fn)
            if transformed is not None:
                out.append(transformed)
            extra_groups, killed_regs = _kill_groups_written(extra_groups, killed_regs, written)
    return out


def _simplify(nodes: List[HIRNode], reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """Walk nodes, trying each registered Pattern in turn.

    Falls back to _transform_default for nodes not consumed by any pattern.
    Uses TypeGroup-based working state: extra_groups tracks pattern-created
    TypeGroups with kill-tracking via active_regs narrowing.  killed_regs
    prevents stale annotation reg_groups from re-appearing after a kill.
    At exit, reg_map is updated in-place with the final extra_groups state
    so downstream passes (e.g. _simplify_once in _pass.py) see retval names.
    """
    xram_map = {k: v for k, v in reg_map.items()
                if isinstance(v, VarInfo) and v.xram_sym}
    counter = reg_map.get("__n__")
    extra_groups: List[TypeGroup] = _varinfo_to_groups(reg_map)
    killed_regs: FrozenSet[str] = frozenset()

    out: List[HIRNode] = []
    i = 0

    def _sub_simplify(ns: List[HIRNode], rm: Dict) -> List[HIRNode]:
        return _simplify(ns, rm)

    while i < len(nodes):
        # Inject user annotations into extra_groups so they propagate forward
        # regardless of killed_regs (user annotations override prior writes).
        node_ann = nodes[i].ann
        if node_ann is not None and node_ann.user_anns:
            for tg in node_ann.user_anns:
                extra_groups = [g for g in extra_groups
                                if not (g.active_regs & tg.active_regs)]
                extra_groups.append(tg)
                killed_regs = killed_regs - tg.active_regs
        eff = _build_node_eff(nodes[i], extra_groups, killed_regs, xram_map, counter)
        pre_eff = dict(eff)

        for pat in _PATTERNS:
            result = pat.match(nodes, i, eff, _sub_simplify)
            if result is not None:
                replacement, new_i = result
                # Kill old entries BEFORE absorbing pattern mutations so that
                # newly-added TypeGroups (e.g. retval1) are immune to the kill.
                written: FrozenSet[str] = frozenset()
                for j in range(i, new_i):
                    written |= nodes[j].written_regs | nodes[j].definitely_killed()
                extra_groups, killed_regs = _kill_groups_written(
                    extra_groups, killed_regs, written)
                extra_groups, killed_regs = _absorb_eff_mutations(
                    eff, pre_eff, extra_groups, killed_regs)
                new_reg_map = _rebuild_reg_map(extra_groups, xram_map, counter)
                i = new_i
                out.extend(_simplify_once(replacement, new_reg_map, _sub_simplify))
                break
        else:
            transformed = _transform_default(nodes[i], eff, _sub_simplify)
            if transformed is not None:
                out.append(transformed)
            written = nodes[i].written_regs | nodes[i].definitely_killed()
            extra_groups, killed_regs = _kill_groups_written(
                extra_groups, killed_regs, written)
            i += 1

    # Update reg_map in-place with final extra_groups state so that downstream
    # passes (_simplify_once, _fold_call_arg_pairs, etc.) see retval/param names.
    for k in [k for k, v in reg_map.items()
              if isinstance(v, VarInfo) and not v.xram_sym]:
        del reg_map[k]
    for tg in extra_groups:
        if tg.xram_sym:
            continue
        vi = _tg_to_varinfo(tg)
        for r in tg.active_regs:
            reg_map[r] = vi
        if len(tg.full_regs) > 1 and tg.is_complete:
            reg_map[tg.pair_name] = vi

    return out
