"""
passes/typesimplify/_simplify.py — Boolean helpers, default transform, core simplifier.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir    import (HIRNode, Statement, Assign, CompoundAssign,
                                   ExprStmt, ReturnStmt, IfGoto, IfNode, WhileNode, ForNode,
                                   DoWhileNode, SwitchNode)
from pseudo8051.passes.patterns         import _PATTERNS
from pseudo8051.passes.patterns._utils  import (
    VarInfo, _replace_pairs, _replace_xram_syms, _replace_single_regs,
    _subst_all_expr,
    _walk_expr,
)
from pseudo8051.ir.expr import (Expr, UnaryOp, BinOp,
                                 Reg as RegExpr, RegGroup as RegGroupExpr)

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

_RE_DPTR_SETUP = re.compile(r'^DPTR = (.+?);')


def _get_written_regs(node: HIRNode) -> frozenset:
    """Return the set of register names written as the primary LHS of this node."""
    if isinstance(node, (Assign, CompoundAssign)):
        lhs = node.lhs
        if isinstance(lhs, RegExpr):
            return frozenset({lhs.name})
        if isinstance(lhs, RegGroupExpr):
            return frozenset(lhs.regs)
    return frozenset()


def _effective_map(node: HIRNode, base_eff: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """Build per-node effective reg map from base_eff merged with node annotations.

    For is_param entries: suppresses any whose registers are absent from
    node.ann.reg_names (the annotation forward-pass already tracked kills across
    all control-flow paths, including kills inside IfNode/WhileNode branches that
    the local `killed` set never sees).

    Then adds entries from node.ann.reg_names and node.ann.call_arg_ann via
    setdefault, so annotations never override entries that survived suppression.
    """
    ann = node.ann
    if ann is None:
        return base_eff

    eff: Dict[str, VarInfo] = {}
    for k, v in base_eff.items():
        if isinstance(v, VarInfo) and v.is_param and v.regs:
            # Keep this param only if at least one of its registers is still
            # live according to the annotation (not killed by any path).
            if any(r in ann.reg_names for r in v.regs):
                eff[k] = v
            # else: all paths through preceding control flow wrote to these
            # regs — the param name no longer applies here.
        else:
            eff[k] = v

    for r, vi in ann.reg_names.items():
        eff.setdefault(r, vi)
    for r, vi in ann.call_arg_ann.items():
        eff.setdefault(r, vi)
    return eff


def _kill_params(reg_map: Dict[str, VarInfo], killed: set) -> Dict[str, VarInfo]:
    """Return reg_map with entries whose param registers overlap *killed* removed."""
    if not killed:
        return reg_map
    result = {}
    for k, v in reg_map.items():
        if isinstance(v, VarInfo) and v.is_param and v.regs:
            if any(r in killed for r in v.regs):
                continue
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
    if isinstance(node, Statement):
        m = _RE_DPTR_SETUP.match(node.text)
        if m:
            sym = m.group(1).strip()
            if any(v.xram_sym == sym for v in reg_map.values()
                   if isinstance(v, VarInfo) and v.xram_sym):
                return None
        new_text = _subst_text(node.text, reg_map)
        return Statement(node.ea, new_text) if new_text != node.text else node

    if isinstance(node, Assign):
        from pseudo8051.ir.expr import Reg as RegExpr, Name as NameExpr
        if isinstance(node.lhs, RegExpr) and node.lhs.name == "DPTR":
            sym = node.rhs.render()
            if any(v.xram_sym == sym for v in reg_map.values()
                   if isinstance(v, VarInfo) and v.xram_sym):
                return None
        new_rhs = _subst_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return Assign(node.ea, node.lhs, new_rhs)
        return node

    if isinstance(node, CompoundAssign):
        new_rhs = _subst_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return CompoundAssign(node.ea, node.lhs, node.op, new_rhs)
        return node

    if isinstance(node, ExprStmt):
        new_expr = _subst_expr(node.expr, reg_map)
        if new_expr is not node.expr:
            return ExprStmt(node.ea, new_expr)
        return node

    if isinstance(node, ReturnStmt):
        if node.value is not None:
            new_val = _subst_expr(node.value, reg_map)
            if new_val is not node.value:
                return ReturnStmt(node.ea, new_val)
        return node

    if isinstance(node, IfGoto):
        new_cond = _simplify_bool_expr(_subst_expr(node.cond, reg_map))
        if new_cond is not node.cond:
            return IfGoto(node.ea, new_cond, node.label)
        return node

    if isinstance(node, IfNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return IfNode(
            ea         = node.ea,
            condition  = new_cond,
            then_nodes = simplify_fn(node.then_nodes, reg_map),
            else_nodes = simplify_fn(node.else_nodes, reg_map),
        )
    if isinstance(node, WhileNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return WhileNode(
            ea         = node.ea,
            condition  = new_cond,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
        )
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
        return ForNode(
            ea         = node.ea,
            init       = new_init,
            condition  = new_cond,
            update     = new_update,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
        )
    if isinstance(node, DoWhileNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return DoWhileNode(
            ea         = node.ea,
            condition  = new_cond,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
        )
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
        return SwitchNode(node.ea, new_subject, new_cases,
                          node.default_label, new_default_body)
    return node


# ── Core simplifier walk ──────────────────────────────────────────────────────

def _simplify_once(nodes: List[HIRNode], reg_map: Dict[str, VarInfo],
                   simplify_fn=None) -> List[HIRNode]:
    """Apply one round of pattern matching + default transforms to each node.

    Tracks registers written by previous nodes so that non-param callee-return
    placeholder mappings (e.g. R7 → "retval") are suppressed once the register
    has been overwritten.
    """
    if simplify_fn is None:
        simplify_fn = _simplify
    out: List[HIRNode] = []
    written: set = set()   # registers overwritten so far in this pass
    for node in nodes:
        # Suppress entries for registers already written in this pass.
        if written:
            base_eff = {k: v for k, v in reg_map.items()
                        if not (isinstance(v, VarInfo) and not v.xram_sym and v.regs
                                and any(r in written for r in v.regs))}
        else:
            base_eff = reg_map
        eff = _effective_map(node, base_eff)
        for pat in _PATTERNS:
            result = pat.match([node], 0, eff, simplify_fn)
            if result is not None:
                out.extend(result[0])
                break
        else:
            transformed = _transform_default(node, eff, simplify_fn)
            if transformed is not None:
                out.append(transformed)
        written.update(_get_written_regs(node))
        if isinstance(node, IfNode):
            written.update(_if_definite_kills(node))
    return out


def _if_definite_kills(node: IfNode) -> frozenset:
    """Registers definitely written in ALL execution paths through an IfNode.

    Both arms always execute exactly one, so the intersection of their writes
    is guaranteed to be killed at the merge point.  Used by _simplify_once to
    prevent stale param mappings from leaking into post-branch nodes that lack
    annotation (because they were created by patterns earlier in the pipeline).
    """
    return _definite_kills_list(node.then_nodes) & _definite_kills_list(node.else_nodes)


def _definite_kills_list(nodes: List[HIRNode]) -> frozenset:
    result: set = set()
    for n in nodes:
        result |= _get_written_regs(n)
        if isinstance(n, IfNode):
            result |= _if_definite_kills(n)
    return frozenset(result)


def _simplify(nodes: List[HIRNode], reg_map: Dict[str, VarInfo],
              _killed: Optional[set] = None) -> List[HIRNode]:
    """
    Walk nodes, trying each registered Pattern in turn.  Falls back to
    _transform_default for nodes not consumed by any pattern.

    Flow-sensitive kill tracking: when a node assigns to a register that
    carries an is_param mapping, that mapping is suppressed for all
    subsequent nodes (including nested IfNode/WhileNode/ForNode bodies).
    """
    out: List[HIRNode] = []
    i = 0
    killed: set = set() if _killed is None else set(_killed)

    def _sub_simplify(ns: List[HIRNode], rm: Dict[str, VarInfo]) -> List[HIRNode]:
        """Recursive simplify that carries the current kill-set inward."""
        return _simplify(ns, rm, killed)

    def _update_killed(node: HIRNode) -> None:
        for r in _get_written_regs(node):
            v = reg_map.get(r)
            if isinstance(v, VarInfo) and v.is_param:
                killed.add(r)

    while i < len(nodes):
        base_eff = _kill_params(reg_map, killed)
        eff = _effective_map(nodes[i], base_eff)

        for pat in _PATTERNS:
            result = pat.match(nodes, i, eff, _sub_simplify)
            if result is not None:
                replacement, new_i = result
                # Propagate new/changed keys from eff back to the original reg_map
                # (patterns mutate their reg_map arg, which may be a _kill_params copy).
                # Skip keys that _kill_params intentionally removed (killed params).
                if eff is not reg_map:
                    for k, v in eff.items():
                        if reg_map.get(k) is not v:
                            orig_v = reg_map.get(k)
                            if not (isinstance(orig_v, VarInfo) and orig_v.is_param
                                    and orig_v.regs
                                    and any(r in killed for r in orig_v.regs)):
                                reg_map[k] = v
                # Gather kills from the consumed range, then recompute eff
                for j in range(i, new_i):
                    _update_killed(nodes[j])
                base_eff = _kill_params(reg_map, killed)
                eff = _effective_map(nodes[i], base_eff)
                i = new_i
                out.extend(_simplify_once(replacement, eff, _sub_simplify))
                break
        else:
            # Transform BEFORE killing so the node's own RHS uses its old mapping
            transformed = _transform_default(nodes[i], eff, _sub_simplify)
            if transformed is not None:
                out.append(transformed)
            # Kill AFTER transform
            _update_killed(nodes[i])
            i += 1
    return out
