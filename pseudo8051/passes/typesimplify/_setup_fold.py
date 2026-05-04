"""
passes/typesimplify/_setup_fold.py — Register-setup folding, pruning, and call-arg pairing.

Exports:
  _is_call_setup_assign    predicate: Assign(Reg/RegGroup, Name/Const)
  _fold_and_prune_setups   fold Reg=Const into calls; prune dead setup nodes
  _fold_call_arg_pairs     combine consecutive byte-reg assigns into RegGroup assign
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import (HIRNode, Assign, CompoundAssign, ExprStmt,
                                ReturnStmt, IfGoto, IfNode, WhileNode, ForNode,
                                DoWhileNode, SwitchNode, TypedAssign,
                                GotoStatement, Label, BreakStmt, ContinueStmt)
from pseudo8051.ir.expr import (Expr, Const, Call, BinOp, Paren, XRAMRef, Cast,
                                 Reg as RegExpr, Regs as RegsExpr,
                                 RegGroup as RegGroupExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import TypeGroup, VarInfo, _count_reg_uses_in_node, _subst_reg_in_node, _const_str, _regs_in_expr, _is_reg_free
from pseudo8051.passes.typesimplify._dptr import _is_dptr_inc_node
from pseudo8051.constants import dbg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_const(expr: Expr, node: HIRNode) -> Expr:
    """If expr is a single-register Regs and its value is known in node.ann,
    return the resolved expression; otherwise return expr unchanged.

    Checks reg_consts first (integer constant), then reg_exprs (symbolic
    expression such as XRAM[EXT_DC41]).  reg_exprs results are only used when
    they are reg-free (safe to duplicate at multiple sites).
    """
    if isinstance(expr, RegsExpr) and expr.is_single:
        ann = getattr(node, 'ann', None)
        if ann is not None:
            val = ann.reg_consts.get(expr.name)
            if val is not None:
                return Const(val)
            sym = ann.reg_exprs.get(expr.name)
            if sym is not None and _is_reg_free(sym):
                return sym
    return expr


def _resolve_const_with_source(expr: Expr, node: HIRNode):
    """Like _resolve_const but also returns the defining HIR node when the value
    came from reg_exprs annotation (so callers can forward source provenance).

    Returns (resolved_expr, defining_node_or_None).
    """
    if isinstance(expr, RegsExpr) and expr.is_single:
        ann = getattr(node, 'ann', None)
        if ann is not None:
            val = ann.reg_consts.get(expr.name)
            if val is not None:
                return Const(val), None
            sym = ann.reg_exprs.get(expr.name)
            if sym is not None and _is_reg_free(sym):
                src = ann.reg_expr_sources.get(expr.name)
                return sym, src
    return expr, None


# ── Predicates ────────────────────────────────────────────────────────────────

def _references_xram_const(node: HIRNode, val: int) -> bool:
    """True if node (or a direct source_node) has XRAMRef(Const(val)) as lhs.

    Used to link pruned DPTR=Const(k) nodes back to XRAM writes whose address
    was folded from DPTR's constant value at lift time (so the XRAM node uses
    XRAMRef(Const(k)) directly rather than XRAMRef(Reg('DPTR'))).
    """
    def _check(n: HIRNode) -> bool:
        return (isinstance(n, Assign)
                and isinstance(n.lhs, XRAMRef)
                and isinstance(n.lhs.inner, Const)
                and n.lhs.inner.value == val)
    if _check(node):
        return True
    for sn in getattr(node, 'source_nodes', []):
        if _check(sn):
            return True
    return False


def _is_call_setup_assign(node: HIRNode) -> bool:
    """True for Assign(Reg/RegGroup, Name/Const) — a consolidated register-setup node."""
    return (isinstance(node, Assign)
            and isinstance(node.lhs, RegsExpr)
            and isinstance(node.rhs, (NameExpr, Const)))


# ── Reference collection ──────────────────────────────────────────────────────

def _collect_hir_name_refs(nodes: List[HIRNode]) -> frozenset:
    """Collect all Reg/Name/RegGroup name strings from read positions in nodes."""
    result: set = set()
    for n in nodes:
        result |= n.name_refs()
    return frozenset(result)


def _collect_unresolved_reg_refs(nodes: List[HIRNode]) -> frozenset:
    """Like _collect_hir_name_refs but excludes register names that appear only
    inside aliased Regs groups.

    An aliased Regs (e.g. Regs(('R2','R1'), alias='src')) means the register
    values were already resolved to a named variable; the individual register
    names are not truly 'needed' from upstream setup assigns — the alias is the
    real reference.  This allows _fold_and_prune_setups to prune setup assigns
    whose registers only appear as components of fully-aliased multi-byte args.
    """
    from pseudo8051.ir.hir._base import _refs_from_expr

    def _unresolved_refs_expr(expr: Expr) -> frozenset:
        """Walk expr collecting only non-aliased Regs and Name references."""
        if isinstance(expr, RegsExpr):
            if expr.alias:
                # Aliased: treat as a Name reference only (not raw registers)
                return frozenset({expr.alias})
            return expr.reg_set()
        if isinstance(expr, NameExpr):
            return frozenset({expr.name})
        children = expr.children()
        if not children:
            return frozenset()
        return frozenset().union(*(_unresolved_refs_expr(c) for c in children))

    result: set = set()
    for n in nodes:
        # Reuse the HIR node's traversal but with our expr walker
        # We replicate the read-position logic by checking node types
        from pseudo8051.ir.hir import (Assign, TypedAssign, CompoundAssign,
                                        ExprStmt, ReturnStmt, IfGoto, IfNode, SwitchNode)
        if isinstance(n, (Assign, TypedAssign)):
            result |= _unresolved_refs_expr(n.rhs)
            lhs = n.lhs
            if not isinstance(lhs, RegsExpr):
                result |= _unresolved_refs_expr(lhs)
        elif isinstance(n, CompoundAssign):
            result |= _unresolved_refs_expr(n.lhs)  # CompoundAssign reads its LHS
            result |= _unresolved_refs_expr(n.rhs)
        elif isinstance(n, ExprStmt):
            result |= _unresolved_refs_expr(n.expr)
        elif isinstance(n, ReturnStmt) and n.value is not None:
            result |= _unresolved_refs_expr(n.value)
        elif isinstance(n, IfGoto):
            result |= _unresolved_refs_expr(n.cond)
        elif isinstance(n, IfNode):
            result |= _unresolved_refs_expr(n.condition)
            result |= _collect_unresolved_reg_refs(n.then_nodes)
            result |= _collect_unresolved_reg_refs(n.else_nodes)
        elif isinstance(n, SwitchNode):
            result |= _unresolved_refs_expr(n.subject)
            for _, body in n.cases:
                if isinstance(body, list):
                    result |= _collect_unresolved_reg_refs(body)
            if isinstance(n.default_body, list):
                result |= _collect_unresolved_reg_refs(n.default_body)
        else:
            result |= n.name_refs()
    return frozenset(result)


# ── Setup-fold helpers ────────────────────────────────────────────────────────

def _subst_reg_in_call_node(node: HIRNode, reg: str, replacement: Expr) -> HIRNode:
    """Replace Name(reg) with replacement in the call args of node."""
    def _patch(call: Call) -> Call:
        new_args = [replacement if (isinstance(a, NameExpr) and a.name == reg) else a
                    for a in call.args]
        if any(na is not oa for na, oa in zip(new_args, call.args)):
            return Call(call.func_name, new_args)
        return call

    if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
        new_call = _patch(node.expr)
        if new_call is not node.expr:
            result = node.copy_meta_to(ExprStmt(node.ea, new_call))
            result.source_nodes = [node]
            return result
        return node
    if isinstance(node, Assign) and isinstance(node.rhs, Call):
        new_call = _patch(node.rhs)
        if new_call is not node.rhs:
            result = node.copy_meta_to(Assign(node.ea, node.lhs, new_call))
            result.source_nodes = [node]
            return result
        return node
    return node


def _first_kill_before_read(reg: str, nodes: List[HIRNode]) -> bool:
    """
    Return True if `reg` is written before being read in the flat prefix of `nodes`.

    Conservative: returns False for any structured node (IfNode, etc.) because
    a branch might read the register.
    """
    for node in nodes:
        if isinstance(node, (IfNode, WhileNode, ForNode, DoWhileNode, SwitchNode)):
            return False
        # GotoStatement/BreakStmt/ContinueStmt/Label indicate control-flow
        # boundaries: a goto/break/continue jumps over subsequent flat code (so
        # a kill after one may not be on every path), and a label is a merge
        # point where different predecessors may carry different values.
        # Return False (conservative) in all cases.
        if isinstance(node, (GotoStatement, Label, BreakStmt, ContinueStmt)):
            return False
        if (isinstance(node, CompoundAssign)
                and node.lhs == RegExpr(reg)):
            return False  # CompoundAssign reads its LHS
        reads = _count_reg_uses_in_node(reg, node)
        writes = reg in node.written_regs
        if reads > 0:
            return False
        if writes:
            return True
    return False


# ── Main pass ─────────────────────────────────────────────────────────────────

def _fold_and_prune_setups(nodes: List[HIRNode],
                            reg_map: Dict[str, VarInfo],
                            _outer_refs: frozenset = frozenset()) -> List[HIRNode]:
    """
    Post-simplify cleanup of register-setup lines before calls.

    1. Fold Assign(Reg, Const) into the next call node's args.
    2. Remove Assign(Reg/RegGroup, Name/Const) setup nodes whose LHS registers
       are not referenced in any subsequent node (including caller context via
       _outer_refs).
    3. Remove DPTR++ nodes whose DPTR value is not referenced afterwards.
    Recurses into IfNode / WhileNode / ForNode / SwitchNode bodies.
    """
    result: List[HIRNode] = []
    for k, node in enumerate(nodes):
        succ_refs = frozenset(_collect_hir_name_refs(nodes[k + 1:])) | _outer_refs
        if isinstance(node, SwitchNode):
            # A register written in one case body and read in another must not be
            # pruned.  Collect only registers that are written in at least one case
            # and read in at least one *other* case (or the default body), then
            # intersect with all case-body reads so only genuinely cross-case live
            # registers are added to outer_refs.
            all_bodies = [body for _, body in node.cases if isinstance(body, list)]
            if isinstance(node.default_body, list):
                all_bodies.append(node.default_body)
            body_writes = [frozenset().union(*(bn.written_regs for bn in b)) for b in all_bodies]
            body_reads  = [_collect_unresolved_reg_refs(b) for b in all_bodies]
            cross_case: set = set()
            for idx, (writes, reads) in enumerate(zip(body_writes, body_reads)):
                # Any register written in this body that is read in any other body
                other_reads = frozenset().union(*(body_reads[j] for j in range(len(all_bodies)) if j != idx))
                cross_case |= writes & other_reads
            extended = succ_refs | cross_case
            result.append(node.map_bodies(
                lambda ns, refs=extended: _fold_and_prune_setups(ns, reg_map, refs)))
        else:
            result.append(node.map_bodies(
                lambda ns, refs=succ_refs: _fold_and_prune_setups(ns, reg_map, refs)))

    work: List[HIRNode] = result

    # Phase 1: fold setup assigns into the next call's args.
    # Handles:
    #   Assign(single Reg, Const)       → substitute Reg name in call
    #   TypedAssign(Name(n), Const/Name) → substitute variable name in call
    for i in range(len(work)):
        node = work[i]
        is_reg_const = (isinstance(node, Assign)
                        and isinstance(node.lhs, RegsExpr) and node.lhs.is_single
                        and isinstance(node.rhs, Const))
        is_typed_const = (isinstance(node, TypedAssign)
                          and isinstance(node.lhs, NameExpr)
                          and isinstance(node.rhs, (Const, NameExpr)))
        if not (is_reg_const or is_typed_const):
            continue
        reg = node.lhs.name if is_reg_const else node.lhs.name
        val = node.rhs
        for j in range(i + 1, len(work)):
            nj = work[j]
            if _is_call_setup_assign(nj) or _is_dptr_inc_node(nj):
                continue
            new_nj = _subst_reg_in_call_node(nj, reg, val)
            if new_nj is not nj:
                # new_nj.source_nodes was set to [nj] by _subst_reg_in_call_node;
                # prepend the consumed setup node so the full chain is: new_nj →
                # [setup_node, nj] → nj's own sources.
                new_nj.source_nodes = [node] + list(new_nj.source_nodes)
                work[j] = new_nj
                work[i] = None
                dbg("typesimp", f"  [{hex(node.ea)}] fold-const: {reg}={val.render()} into call")
            break
    work = [n for n in work if n is not None]

    # Phase 2: remove dead setup-assign and DPTR++ nodes.
    out: List[HIRNode] = []
    for i, node in enumerate(work):
        if _is_call_setup_assign(node):
            lhs_regs = node.written_regs
            # Use unresolved-ref collection: aliased Regs (e.g. Regs(R2R1, alias='src'))
            # count as 'src' not as 'R2'/'R1', so setup assigns for those regs can be pruned.
            all_downstream = _collect_unresolved_reg_refs(work[i + 1:]) | _outer_refs
            # Don't prune setups whose next non-setup, non-label node is a goto:
            # GotoStatement.name_refs() returns frozenset() so the target's register
            # usage is invisible here, but the registers may be needed at the target.
            next_action = None
            for j in range(i + 1, len(work)):
                nj = work[j]
                if isinstance(nj, Label):
                    continue
                if _is_call_setup_assign(nj) or _is_dptr_inc_node(nj):
                    continue
                next_action = nj
                break
            if isinstance(next_action, GotoStatement):
                out.append(node)
                continue
            if lhs_regs.isdisjoint(all_downstream):
                dbg("typesimp",
                    f"  [{hex(node.ea)}] prune-setup: {node.lhs.render()} = {node.rhs.render()}")
                # Before dropping: if the RHS is a Name that IS used downstream,
                # link this node as a provenance source of the first user.
                # This preserves the contribution of instructions like "mov A, R7"
                # that were pruned only because A was renamed to R7's alias.
                if isinstance(node.rhs, NameExpr):
                    rhs_name = node.rhs.name
                    for j in range(i + 1, len(work)):
                        if rhs_name in _collect_hir_name_refs([work[j]]):
                            work[j].source_nodes = [node] + list(
                                work[j].source_nodes)
                            break
                        # Multi-reg setup (e.g. R6R7 = xarg2): _subst_from_reg_exprs
                        # may have already replaced the lhs registers with XRAM refs
                        # before this prune runs, leaving register names only in the
                        # immediate source_nodes of the downstream node.
                        if not node.lhs.is_single:
                            src_refs = frozenset().union(
                                *(sn.name_refs() for sn in work[j].source_nodes))
                            if not lhs_regs.isdisjoint(src_refs):
                                work[j].source_nodes = [node] + list(
                                    work[j].source_nodes)
                                break
                            # Stop searching if a downstream node kills any lhs reg
                            if not lhs_regs.isdisjoint(work[j].written_regs):
                                break
                elif isinstance(node.rhs, Const) and isinstance(node.lhs, RegsExpr):
                    if node.lhs.name == 'DPTR':
                        # DPTR = Const(k): the XRAM handler may have already folded
                        # DPTR's value into XRAMRef(Const(k)) at lift time, so the
                        # XRAM node no longer references 'DPTR' by name.  Find the
                        # first downstream node whose lhs (or a direct source lhs)
                        # is XRAMRef(Const(k)) and attach this node as provenance.
                        dptr_val = node.rhs.value
                        for j in range(i + 1, len(work)):
                            wj = work[j]
                            if _references_xram_const(wj, dptr_val):
                                if node not in wj.source_nodes:
                                    wj.source_nodes = [node] + list(
                                        wj.source_nodes)
                                break
                    else:
                        # Reg = Const: the register's constant value may have been
                        # tracked in a downstream call's reg_exprs annotation and
                        # folded in by _subst_from_reg_exprs (register renamed to
                        # a param alias, so not visible by name in the call args).
                        # Link to the first downstream call node as provenance.
                        lhs_name = node.lhs.name
                        for j in range(i + 1, len(work)):
                            wj = work[j]
                            if ((isinstance(wj, ExprStmt) and isinstance(wj.expr, Call))
                                    or (isinstance(wj, Assign) and isinstance(wj.rhs, Call))):
                                if node not in wj.source_nodes:
                                    wj.source_nodes = [node] + list(wj.source_nodes)
                                break
                            # Stop if the register is written before we reach a call
                            if lhs_name in wj.written_regs:
                                break
                continue
            live_lhs = lhs_regs & all_downstream
            if live_lhs and all(_first_kill_before_read(r, work[i + 1:])
                                 for r in live_lhs):
                dbg("typesimp",
                    f"  [{hex(node.ea)}] prune-setup-killed: "
                    f"{node.lhs.render()} = {node.rhs.render()}")
                continue
        elif _is_dptr_inc_node(node):
            if "DPTR" not in _collect_hir_name_refs(work[i + 1:]):
                dbg("typesimp", f"  [{hex(node.ea)}] prune-dptr++")
                continue
        out.append(node)
    return out


# ── Call-arg pair folding ─────────────────────────────────────────────────────

def _find_pair_groups(reg_map: Dict[str, VarInfo]) -> Dict[tuple, VarInfo]:
    """Return mapping: regs_tuple → VarInfo for multi-byte param pairs."""
    groups: Dict[tuple, VarInfo] = {}
    for r, v in reg_map.items():
        if isinstance(v, VarInfo) and len(v.regs) > 1 and r in v.regs:
            key = tuple(v.regs)
            if key not in groups:
                groups[key] = v
    return groups


def _reads_any_reg(node: HIRNode, regs: set) -> bool:
    """True if node reads any register in regs."""
    for r in regs:
        if _count_reg_uses_in_node(r, node) > 0:
            return True
    return False


def _writes_any_reg(node: HIRNode, regs: set) -> bool:
    """True if node writes any register in regs."""
    return bool(node.written_regs & regs)


def _inline_group_into_node(node: HIRNode, regs_key: tuple,
                             replacement: Expr) -> Optional[HIRNode]:
    """Replace Regs(names==regs_key) in call args of node with replacement.

    Returns a new node on success, None if the group is not found.
    """
    def _patch_call(call: Call) -> Optional[Call]:
        new_args = []
        found = False
        for a in call.args:
            if isinstance(a, RegsExpr) and not a.is_single and a.names == regs_key:
                new_args.append(replacement)
                found = True
            else:
                new_args.append(a)
        return Call(call.func_name, new_args) if found else None

    result = None
    if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
        new_call = _patch_call(node.expr)
        if new_call is not None:
            result = ExprStmt(node.ea, new_call)
    elif isinstance(node, TypedAssign) and isinstance(node.rhs, Call):
        new_call = _patch_call(node.rhs)
        if new_call is not None:
            result = TypedAssign(node.ea, node.type_str, node.lhs, new_call)
    elif isinstance(node, Assign) and isinstance(node.rhs, Call):
        new_call = _patch_call(node.rhs)
        if new_call is not None:
            result = Assign(node.ea, node.lhs, new_call)
    if result is not None:
        node.copy_meta_to(result)
        result.source_nodes = [node]
    return result


def _harvest_call_arg_pairs(nodes: List[HIRNode],
                             pair_groups: dict, reg_to_pair: dict) -> None:
    """Recursively collect multi-byte call_arg_ann TypeGroups from all nodes."""
    for node in nodes:
        ann = getattr(node, "ann", None)
        if ann is not None:
            for g in ann.call_arg_ann:
                if len(g.full_regs) > 1:
                    key = g.full_regs
                    if key not in pair_groups:
                        pair_groups[key] = g
                    for pr in g.full_regs:
                        reg_to_pair.setdefault(pr, key)
        # Recurse into structured bodies
        for _, body in node.child_body_groups():
            _harvest_call_arg_pairs(body, pair_groups, reg_to_pair)
        if isinstance(node, SwitchNode):
            for _, body in node.cases:
                if isinstance(body, list):
                    _harvest_call_arg_pairs(body, pair_groups, reg_to_pair)
            if isinstance(node.default_body, list):
                _harvest_call_arg_pairs(node.default_body, pair_groups, reg_to_pair)


def _fold_call_arg_pairs(nodes: List[HIRNode],
                          reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Fold consecutive assignments to all regs of a multi-byte call-arg pair into
    a single RegGroup assignment with a bit-shifted combination expression.

    Example: R3 = lo_expr; [safe nodes]; R2 = hi_expr;
         →   R2R3 = (hi_expr << 8) | lo_expr;

    Recurses into IfNode/WhileNode/ForNode/SwitchNode bodies.
    """
    pair_groups = _find_pair_groups(reg_map)

    reg_to_pair: Dict[str, tuple] = {}
    for regs_key in pair_groups:
        for r in regs_key:
            reg_to_pair[r] = regs_key

    # Harvest pair entries from call_arg_ann annotations, including nested bodies.
    _harvest_call_arg_pairs(nodes, pair_groups, reg_to_pair)

    if not reg_to_pair:
        return nodes

    nodes = [node.map_bodies(lambda ns: _fold_call_arg_pairs(ns, reg_map))
             for node in nodes]

    out: List[HIRNode] = []
    consumed: set = set()

    for i, node in enumerate(nodes):
        if i in consumed:
            continue

        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegsExpr) and node.lhs.is_single
                and node.lhs.name in reg_to_pair):
            out.append(node)
            continue

        r0 = node.lhs.name
        regs_key = reg_to_pair[r0]
        vinfo = pair_groups[regs_key]
        all_regs = set(regs_key)
        remaining_regs = all_regs - {r0}

        r0_expr, r0_src = _resolve_const_with_source(node.rhs, node)
        byte_assigns: Dict[str, tuple] = {r0: (i, r0_expr)}
        # Defining HIR nodes for registers whose values came from reg_exprs annotation.
        expr_sources: Dict[str, HIRNode] = {}
        if r0_src is not None:
            expr_sources[r0] = r0_src
        interleaved: List[int] = []
        conflict = False

        for k in range(i + 1, len(nodes)):
            if k in consumed:
                continue
            nd = nodes[k]
            reads_pair = _reads_any_reg(nd, all_regs)
            writes_pair = _writes_any_reg(nd, all_regs)

            if (isinstance(nd, Assign) and isinstance(nd.lhs, RegsExpr) and nd.lhs.is_single
                    and nd.lhs.name in remaining_regs and not reads_pair):
                rk_expr, rk_src = _resolve_const_with_source(nd.rhs, nd)
                byte_assigns[nd.lhs.name] = (k, rk_expr)
                if rk_src is not None:
                    expr_sources[nd.lhs.name] = rk_src
                remaining_regs -= {nd.lhs.name}
                if not remaining_regs:
                    break
            elif reads_pair or writes_pair:
                conflict = True
                break
            else:
                interleaved.append(k)

        if conflict or remaining_regs:
            out.append(node)
            continue

        # Prefer naming from consumed nodes' call_arg_ann over global pair_groups,
        # since pair_groups uses setdefault and a different callee's annotation may
        # have won for the same register tuple.
        naming_vinfo = vinfo
        naming_from_call_arg = False
        for r_check, (idx_check, _) in byte_assigns.items():
            nd_check = nodes[idx_check]
            ann_check = getattr(nd_check, "ann", None)
            if ann_check is not None:
                ca_g = ann_check.call_arg_for(r_check)
                if ca_g is not None and ca_g.full_regs == regs_key:
                    naming_vinfo = ca_g
                    naming_from_call_arg = True
                    break

        # Build combined expression: (byte0 << (n-1)*8) | byte1 | ... | byte_{n-1}
        n_bytes = len(regs_key)
        ordered_exprs = [byte_assigns[r][1] for r in regs_key]

        # When naming came only from pair_groups (register-backed VarInfo), verify
        # it against the actual hi-byte source.  If the hi byte is an XRAM load,
        # look up its parent VarInfo; a different (or unknown) variable means the
        # register pair is carrying unrelated data here, so suppress the name.
        if not naming_from_call_arg and not naming_vinfo.xram_sym and ordered_exprs:
            hi_expr = ordered_exprs[0]
            if isinstance(hi_expr, XRAMRef) and isinstance(hi_expr.inner, NameExpr):
                hi_sym = hi_expr.inner.name
                xram_parent = reg_map.get(hi_sym)
                if isinstance(xram_parent, VarInfo) and not xram_parent.is_byte_field:
                    naming_vinfo = xram_parent
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] fold-call-arg-pair: override naming "
                        f"{vinfo.name!r} → {xram_parent.name!r} from {hi_sym}")
                else:
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] fold-call-arg-pair: suppress naming "
                        f"{vinfo.name!r} (hi-byte XRAM {hi_sym!r} unknown)")
                    naming_vinfo = VarInfo("", naming_vinfo.type, ())

        dbg("typesimp",
            f"  [{hex(node.ea)}] fold-call-arg-pair exprs: "
            + ", ".join(f"{type(e).__name__}({e.render()!r})" for e in ordered_exprs))
        if all(isinstance(e, Const) for e in ordered_exprs):
            # All bytes are known constants — fold to a single integer Const.
            int_val = 0
            for b, e in enumerate(ordered_exprs):
                int_val = (int_val << 8) | (e.value & 0xFF)
            alias = _const_str(int_val, naming_vinfo.type) if naming_vinfo.type else None
            combined: Expr = Const(int_val, alias=alias)
            dbg("typesimp",
                f"  [{hex(node.ea)}] fold-call-arg-pair → const {combined.render()!r}")
        elif (n_bytes == 2
              and isinstance(ordered_exprs[0], Const)
              and ordered_exprs[0].value == 0):
            # Hi byte is zero, lo byte is non-constant: zero-extension cast.
            # (0 << 8) | lo_expr  →  (type)lo_expr
            lo_expr = ordered_exprs[1]
            combined: Expr = Cast(naming_vinfo.type, lo_expr) if naming_vinfo.type else lo_expr
            dbg("typesimp",
                f"  [{hex(node.ea)}] fold-call-arg-pair → zero-ext cast {combined.render()!r}")
        else:
            def _paren_if_binop(e: Expr) -> Expr:
                return Paren(e) if isinstance(e, BinOp) else e

            hi = _paren_if_binop(ordered_exprs[0])
            combined: Expr = Paren(BinOp(hi, "<<", Const((n_bytes - 1) * 8)))
            for b in range(1, n_bytes):
                shift = (n_bytes - 1 - b) * 8
                lo = _paren_if_binop(ordered_exprs[b])
                term: Expr = Paren(BinOp(lo, "<<", Const(shift))) if shift > 0 else lo
                combined = BinOp(combined, "|", term)
            dbg("typesimp",
                f"  [{hex(node.ea)}] fold-call-arg-pair → binop (not all Const)")

        if naming_vinfo.type and naming_vinfo.name:
            result_node = TypedAssign(node.ea, naming_vinfo.type,
                                      RegGroupExpr(regs_key, alias=naming_vinfo.name), combined)
        else:
            result_node = Assign(node.ea, RegGroupExpr(regs_key), combined)
        base_sources = [nodes[idx] for _, (idx, _) in byte_assigns.items()]
        # For registers whose values came from reg_exprs annotation (relay through
        # another register, e.g. R4 = A where A = XRAM[...]), include the defining
        # HIR node recorded in the annotation so the XRAM-load chain is preserved.
        # Place these before base_sources so the source order follows data flow:
        # XRAM load → register assignment → combined expression.
        seen_src_ids = {id(n) for n in base_sources}
        extra_sources = [src for r, src in expr_sources.items()
                         if id(src) not in seen_src_ids]
        result_node.source_nodes = extra_sources + base_sources
        dbg("typesimp",
            f"  [{hex(node.ea)}] fold-call-arg-pair: {''.join(regs_key)} = {combined.render()}")

        # Try to inline combined directly into the downstream call node.
        # This avoids emitting a TypedAssign(RegGroup, Const) that
        # _fold_and_prune_setups would prune (because R2/R1 aren't visible
        # in downstream refs when the call arg uses the alias 'src') before
        # _inline_group_setups gets a chance to fold it in.
        inlined = False
        max_consumed_idx = max(idx for _, (idx, _) in byte_assigns.items())
        source_regs = _regs_in_expr(combined)  # registers whose values combined depends on
        for k in range(max_consumed_idx + 1, len(nodes)):
            if k in consumed:
                continue
            nd = nodes[k]
            # If a source register is written before we reach the call, the
            # combined expression would be stale after propagation re-reads it
            # (e.g. R7 reused for sram_sel after being used as osdAddr.lo).
            # Stop inlining in that case and emit the TypedAssign instead.
            if source_regs and _writes_any_reg(nd, source_regs):
                break
            patched = _inline_group_into_node(nd, regs_key, combined)
            if patched is not None:
                # Merge byte-assign source nodes with any provenance already
                # attached to nd (e.g. AnnotationPass-linked defining nodes for
                # other call parameters like sram_sel=R7).
                seen_ids = {id(sn) for sn in result_node.source_nodes}
                extra = [sn for sn in nd.source_nodes if id(sn) not in seen_ids]
                patched.source_nodes = list(result_node.source_nodes) + extra
                nodes[k] = patched
                inlined = True
                dbg("typesimp",
                    f"  [{hex(node.ea)}] fold-call-arg-pair: inlined "
                    f"{''.join(regs_key)}={combined.render()} directly into downstream call")
                break
            # Can't patch — stop if this node reads or writes any group register
            if _reads_any_reg(nd, all_regs) or _writes_any_reg(nd, all_regs):
                break

        if not inlined:
            out.append(result_node)

        for r, (idx, _) in byte_assigns.items():
            consumed.add(idx)

    return out
