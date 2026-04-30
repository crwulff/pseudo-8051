"""
passes/typesimplify/_propagate.py — Forward single-use value propagation pass.

Exports:
  _propagate_values   master pass (calls three sub-passes in a fixed-point loop)

Sub-passes (also exported for testing):
  _fold_compound_assigns      A0: fold Assign(r, e) + CompoundAssign(r, op=, rhs) → Assign
  _propagate_register_copies  A:  substitute single-use Assign(Reg, expr) into its use
  _inline_retvals             B:  inline TypedAssign retval = call() into its single use
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,
                                ExprStmt, ReturnStmt, IfGoto, IfNode,
                                WhileNode, ForNode, DoWhileNode, NodeAnnotation,
                                Label, GotoStatement, BreakStmt, ContinueStmt,
                                SwitchNode)
from pseudo8051.ir.expr import (Expr, BinOp, Call, Const, XRAMRef, UnaryOp,
                                 Regs as RegExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import (
    VarInfo, _count_reg_uses_in_node, _subst_reg_in_node, _walk_expr,
    _canonicalize_expr, _fold_exprs_in_node, _is_reg_free,
)
from pseudo8051.constants import dbg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_xram_const_addr(node: HIRNode, val: int) -> bool:
    """True if node or a direct source_node has XRAMRef(Const(val)) as lhs.

    Used to link DPTR=Const(k) nodes to downstream XRAM writes whose address
    was folded from DPTR's constant value at lift time (XRAMRef(Const(k))
    instead of XRAMRef(Reg('DPTR'))).
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


def _name_possibly_written_in(name: str, node: HIRNode) -> bool:
    """Return True if any node inside node's bodies may assign to Name(name).

    Used to prevent propagating a Name-lhs value past a structured node
    (IfNode, WhileNode, SwitchNode, etc.) whose branches may redefine that name.
    """
    def _check_seq(nodes) -> bool:
        for n in nodes:
            lhs = getattr(n, 'lhs', None)
            if isinstance(lhs, NameExpr) and lhs.name == name:
                return True
            for _extra, body in n.child_body_groups():
                if _check_seq(body):
                    return True
            # SwitchNode.child_body_groups() returns [] (viewer handles it
            # specially), so check its case bodies explicitly.
            if isinstance(n, SwitchNode):
                for _vals, body in n.cases:
                    if isinstance(body, list) and _check_seq(body):
                        return True
                if isinstance(n.default_body, list) and _check_seq(n.default_body):
                    return True
        return False

    for _extra, body in node.child_body_groups():
        if _check_seq(body):
            return True
    # Also check if node itself is a SwitchNode (called directly on it)
    if isinstance(node, SwitchNode):
        for _vals, body in node.cases:
            if isinstance(body, list) and _check_seq(body):
                return True
        if isinstance(node.default_body, list) and _check_seq(node.default_body):
            return True
    return False


def _expr_name_refs(expr: Expr) -> frozenset:
    """Collect register/name identities referenced in an expression tree.

    For RegGroup nodes, emits each individual register name so that conflict
    detection operates in register-name space regardless of whether the node
    carries an alias.
    """
    refs: set = set()

    def _collect(e: Expr) -> Expr:
        if isinstance(e, RegExpr):
            refs.update(e.regs)
        elif isinstance(e, NameExpr):
            refs.add(e.name)
        return e

    _walk_expr(expr, _collect)
    return frozenset(refs)


def _count_group_uses_in_node(regs_tuple: tuple, node: HIRNode) -> int:
    """Count occurrences of Regs(regs_tuple) appearing as a full-group expression.

    Unlike _count_reg_uses_in_node, this only counts the multi-register expression
    used as a unit — individual uses of constituent registers don't count.
    E.g. for regs_tuple=('B','A'): 'DPTR = BA' counts as 1; 'R2 = R2 + B' counts 0.
    """
    count = [0]

    def _fn(e: Expr) -> Expr:
        if isinstance(e, RegExpr) and not e.is_single and e.names == regs_tuple:
            count[0] += 1
        return e

    if isinstance(node, Assign):
        _walk_expr(node.rhs, _fn)
        if not isinstance(node.lhs, (RegExpr, NameExpr)):
            _walk_expr(node.lhs, _fn)
    elif isinstance(node, (CompoundAssign, ExprStmt)):
        _walk_expr(getattr(node, 'rhs', None) or node.expr, _fn)
    elif isinstance(node, ReturnStmt) and node.value is not None:
        _walk_expr(node.value, _fn)
    elif isinstance(node, (IfGoto, IfNode)):
        _walk_expr(getattr(node, 'cond', None) or node.condition, _fn)
    return count[0]


def _xram_pre_incr_delta(node: HIRNode, r: str) -> Optional[int]:
    """If node is Assign(XRAMRef(++/-- r or r++/r--), ...), return +1 or -1.

    Detects whether substituting `r` into `node` will consume a side-effecting
    pre- or post-increment/decrement in the XRAMRef LHS (e.g. XRAM[++DPTR] or
    XRAM[DPTR++]).  After the node executes, `r` has changed by delta regardless
    of pre/post form:
      - XRAM[++DPTR]: address is old+1, DPTR becomes old+1  → delta +1
      - XRAM[DPTR++]: address is old,   DPTR becomes old+1  → delta +1
    The caller injects a synthetic Assign(r, old_val + delta) to keep the chain
    propagating to subsequent XRAM[++r] nodes.
    """
    if not isinstance(node, Assign):
        return None
    lhs = node.lhs
    if not isinstance(lhs, XRAMRef):
        return None
    inner = lhs.inner
    if not (isinstance(inner, UnaryOp)
            and isinstance(inner.operand, RegExpr)
            and inner.operand.is_single
            and inner.operand.name == r):
        return None
    return +1 if inner.op == '++' else -1


def _expr_stmt_incr_delta(node: HIRNode, r: str) -> Optional[int]:
    """If node is ExprStmt(r++ / r-- / ++r / --r), return +1 or -1.

    After substituting r=Const(K) into ExprStmt(r++) the statement becomes a
    no-op ExprStmt(Const(K)), but the register r has been incremented.  The
    caller should inject a synthetic Assign(r, K+delta) to keep any downstream
    XRAM[++r] chain propagating correctly.
    """
    if not isinstance(node, ExprStmt):
        return None
    expr = node.expr
    if not (isinstance(expr, UnaryOp)
            and isinstance(expr.operand, RegExpr)
            and expr.operand.is_single
            and expr.operand.name == r):
        return None
    return +1 if expr.op == '++' else -1


def _collect_mid_writes(nodes: List[HIRNode], reg_map: Dict) -> frozenset:
    """
    Collect all names/regs written by a sequence of nodes, with three expansions:

    1. Register-to-variable: for each register key in written_regs, look up the
       corresponding variable name in reg_map and add it.  This catches
       Assign(RegGroup(R4R5R6R7), ...) clobbering a variable named 'divisor'.

    2. Name-lhs: TypedAssign / Assign with a Name lhs don't appear in written_regs,
       so we add lhs.name directly.  This catches TypedAssign(Name('divisor'), ...).

    3. Variable-to-register (reverse): for each Name lhs, scan reg_map for VarInfo
       entries with that name and add their backing register keys.  This catches
       TypedAssign(Name('divisor'), ...) clobbering the R0R1R2R3 that backs divisor.
    """
    result: set = set()
    for node in nodes:
        # Use possibly_killed() so that structured nodes (IfNode, WhileNode, etc.)
        # conservatively contribute all registers they might write in any branch/body.
        # For leaf nodes possibly_killed() == written_regs, so there is no behaviour change.
        node_writes = node.possibly_killed()
        result.update(node_writes)
        # Expand register writes → variable names
        if reg_map:
            for reg in node_writes:
                info = reg_map.get(reg)
                if isinstance(info, VarInfo) and info.name:
                    result.add(info.name)
        # Name-lhs writes (TypedAssign / Assign with Name LHS)
        if isinstance(node, (Assign, TypedAssign)):
            lhs = getattr(node, 'lhs', None)
            if isinstance(lhs, NameExpr):
                result.add(lhs.name)
                # Reverse lookup: find backing register keys for this variable name
                if reg_map:
                    for reg_key, info in reg_map.items():
                        if isinstance(info, VarInfo) and info.name == lhs.name:
                            result.add(reg_key)
    # DPTR is composed of DPH+DPL.  A write to either component changes the
    # effective DPTR address, so treat DPH/DPL writes as writing DPTR (and vice
    # versa) so that expressions referencing DPTR are blocked from propagating
    # past DPH/DPL writes and vice versa.
    if 'DPH' in result or 'DPL' in result:
        result.add('DPTR')
    if 'DPTR' in result:
        result.add('DPH')
        result.add('DPL')
    return frozenset(result)


def _as_retval_stmt(node: HIRNode) -> Optional[Tuple[str, Call]]:
    """Return (retval_name, call_expr) if node is a TypedAssign retval node; else None."""
    if isinstance(node, TypedAssign) and isinstance(node.rhs, Call):
        return (node.lhs.render(), node.rhs)
    return None


def _count_name_uses_in_nodes(name: str, nodes: List[HIRNode]) -> int:
    """Count total occurrences of Name/Reg(name) in read positions across nodes."""
    return sum(_count_reg_uses_in_node(name, n) for n in nodes)




def _collect_hir_name_refs(nodes: List[HIRNode]) -> frozenset:
    result: set = set()
    for n in nodes:
        result |= n.name_refs()
    return frozenset(result)


def _dbg_node(n) -> str:
    try:
        return f"{type(n).__name__}:{n.render(0)[0][1][:50]!r}"
    except Exception:
        return type(n).__name__


# ── Sub-pass A0: compound-assign expansion ────────────────────────────────────

_COMPOUND_OPS = {"+=": "+", "-=": "-", "&=": "&", "|=": "|", "^=": "^"}


def _fold_compound_assigns(live: List[HIRNode]) -> Tuple[List[HIRNode], bool]:
    """
    A0: fold  Assign(Reg(r), expr) + CompoundAssign(Reg(r), op=, rhs)
    into a single Assign(Reg(r), expr op rhs).

    Needed because _count_reg_uses_in_node only counts RHS uses in a
    CompoundAssign, so the preceding Assign would appear unused without
    this expansion step.
    """
    changed = False
    live = list(live)
    i = 0
    while i < len(live):
        node = live[i]
        if (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr) and node.lhs.is_single
                and i + 1 < len(live)):
            nxt = live[i + 1]
            if (isinstance(nxt, CompoundAssign)
                    and nxt.lhs == node.lhs
                    and nxt.op in _COMPOUND_OPS):
                op_str = _COMPOUND_OPS[nxt.op]
                folded = Assign(node.ea, nxt.lhs, BinOp(node.rhs, op_str, nxt.rhs))
                folded.ann = NodeAnnotation.merge(node, nxt)
                folded.source_nodes = [node, nxt]
                live[i + 1] = folded
                live[i] = None
                dbg("typesimp",
                    f"  [{hex(node.ea)}] fold-compound: "
                    f"{node.lhs.render()} = {node.rhs.render()} "
                    f"{nxt.op} {nxt.rhs.render()}")
                changed = True
        i += 1
    return [n for n in live if n is not None], changed


# ── Sub-pass A: register copy propagation ─────────────────────────────────────

def _propagate_register_copies(live: List[HIRNode],
                                reg_map: Dict = {}) -> Tuple[List[HIRNode], bool]:
    """
    A: For each Assign(Reg(r), expr) or TypedAssign(Name(n), expr) at index i with
    exactly one downstream use before r/n is written again, substitute the replacement
    into that use and remove the assignment.

    Multi-use propagation is limited to reg-free replacements (no Reg leaves) to
    avoid duplicating side-effecting expressions.  Single-use propagation allows
    any expression; the intermediate guard checks for register clobbers.

    Name-lhs propagation folds typed variable assignments (e.g. ``arg1 = xarg1``)
    into their single call-argument use (→ ``cmp32(xarg1, 0)``).
    """
    changed = False
    live = list(live)
    i = 0
    while i < len(live):
        node = live[i]
        is_reg_lhs  = (isinstance(node, Assign)
                       and isinstance(node.lhs, RegExpr)
                       and node.lhs.is_single)
        is_name_lhs = (isinstance(node, (Assign, TypedAssign))
                       and isinstance(node.lhs, NameExpr))
        if not (is_reg_lhs or is_name_lhs):
            i += 1
            continue
        r = node.lhs.name
        replacement = node.rhs
        if is_reg_lhs and node.ann is not None:
            replacement = _canonicalize_expr(
                replacement,
                node.ann.reg_consts,
                node.ann.reg_groups,
                node.ann.reg_exprs)

        total_uses = 0
        use_idx = None
        kill_idx = None
        for j in range(i + 1, len(live)):
            # Label nodes are control-flow merge points: different predecessors
            # may carry different register values, so we cannot propagate past them.
            # GotoStatement/BreakStmt/ContinueStmt are unconditional jumps: fall-
            # through code after them is unreachable from this path.
            if isinstance(live[j], (GotoStatement, BreakStmt, ContinueStmt)):
                break   # unreachable fall-through; hard stop
            if isinstance(live[j], Label):
                # Label = CFG merge point.  Continue only if ALL predecessor paths
                # agree on the same expression for r (checked via ann.reg_exprs).
                if (is_reg_lhs
                        and live[j].ann is not None
                        and live[j].ann.reg_exprs.get(r) == replacement):
                    continue
                break
            uses_here = _count_reg_uses_in_node(r, live[j])
            total_uses += uses_here
            if uses_here > 0 and use_idx is None:
                use_idx = j
            if is_reg_lhs:
                # Use possibly_killed() so that an IfNode whose body may write r
                # is treated conservatively as a kill barrier.
                if r in live[j].possibly_killed():
                    kill_idx = j
                    break
                # 8051: Any A += / A -= instruction (ADD, ADDC, SUBB) always writes C
                # as a side-effect carry/borrow output, regardless of whether C also
                # appears in the rhs as the borrow-in.  SUBB in particular reads C
                # (borrow-in) AND writes C (borrow-out); the borrow-out kills the
                # previous C value even though uses_here > 0.
                if (r == 'C'
                        and isinstance(live[j], CompoundAssign)
                        and isinstance(live[j].lhs, RegExpr)
                        and live[j].lhs.is_single
                        and live[j].lhs.name == 'A'
                        and live[j].op in ('+=', '-=')):
                    kill_idx = j
                    break
                # AccumFoldPattern may have already consumed the ADD instruction, folding
                # it into Assign(DPL/DPH/A, BinOp(x, +/-, y)).  The original ADD still set C
                # as a hardware side effect, so treat these folded arithmetic assigns as C-kills.
                if (r == 'C'
                        and uses_here == 0
                        and isinstance(live[j], Assign)
                        and isinstance(live[j].lhs, RegExpr)
                        and live[j].lhs.is_single
                        and live[j].lhs.name in ('A', 'DPL', 'DPH')
                        and isinstance(live[j].rhs, BinOp)
                        and live[j].rhs.op in ('+', '-')):
                    kill_idx = j
                    break
            else:
                # Name-lhs kill: another assign to the same Name (re-definition),
                # either at this level or inside a structured node's bodies.
                lhs_j = getattr(live[j], 'lhs', None)
                if isinstance(lhs_j, NameExpr) and lhs_j.name == r:
                    kill_idx = j
                    break
                # Structured nodes (IfNode, WhileNode, etc.) may write the Name
                # in a branch body — treat as a kill barrier.
                if _name_possibly_written_in(r, live[j]):
                    kill_idx = j
                    break

        dbg("propagate", f"  sub-A: {r}={replacement.render()!r} "
            f"total_uses={total_uses} kill_idx={kill_idx} "
            f"reg_free={_is_reg_free(replacement)}")
        if total_uses > 0 or kill_idx is not None:
            _scan_end = (kill_idx + 1) if kill_idx is not None else min(len(live), i + 10)
            for _j in range(i + 1, _scan_end):
                _u = _count_reg_uses_in_node(r, live[_j])
                _wr = (r in live[_j].written_regs) if is_reg_lhs else False
                dbg("propagate", f"    scan[{_j}]={_dbg_node(live[_j])} uses={_u} kill={_wr}")

        if total_uses == 1 and use_idx is not None:
            # Guard: if the use-site's annotation explicitly records a DIFFERENT
            # value for r than replacement, a kill (e.g. movx A, @DPTR loading a
            # fresh XRAM value) happened between our def and the use but was folded
            # away from the HIR.  Block the substitution to avoid replacing r with
            # a stale constant.  (A use-site annotation saying r=Const(0) when
            # replacement is also Const(0) is consistent and allows propagation.)
            _use_ann = getattr(live[use_idx], 'ann', None)
            if is_reg_lhs:
                if _use_ann is not None:
                    _ann_val = _use_ann.reg_exprs.get(r)
                    if _ann_val is not None and _ann_val != replacement:
                        dbg("propagate",
                            f"  [{hex(node.ea)}] prop-blocked: {r}={replacement.render()!r} "
                            f"— use-site annotation says {r}={_ann_val.render()!r}")
                        i += 1
                        continue
            elif is_name_lhs and _use_ann is not None and reg_map:
                # For Name-lhs: guard against propagating XRAM-backed RMW values
                # (e.g. osdAddr.hi = osdAddr.hi + R6 + C) into a use site where
                # the annotation records the original XRAM value for the backing
                # register — meaning a reload happened between the write and the
                # use but was folded away from the HIR.
                # XRAM byte fields are keyed as '_byte_<sym>' not by name, so
                # search reg_map by VarInfo.name when direct lookup misses.
                _vinfo = reg_map.get(r)
                if _vinfo is None:
                    for _vi in reg_map.values():
                        if isinstance(_vi, VarInfo) and _vi.name == r:
                            _vinfo = _vi
                            break
                if isinstance(_vinfo, VarInfo):
                    if _vinfo.regs:
                        # Register-backed: compare use-site annotation reg value
                        # against def-site annotation reg value.
                        _backing_reg = _vinfo.regs[0]
                        _ann_val = _use_ann.reg_exprs.get(_backing_reg)
                        if _ann_val is not None:
                            _def_ann = getattr(node, 'ann', None)
                            _def_val = _def_ann.reg_exprs.get(_backing_reg) if _def_ann else None
                            if _def_val != _ann_val:
                                dbg("propagate",
                                    f"  [{hex(node.ea)}] prop-blocked: {r}={replacement.render()!r} "
                                    f"— backing reg {_backing_reg} use-site={_ann_val.render()!r}"
                                    f" def-site={_def_val!r}")
                                i += 1
                                continue
                    elif _vinfo.xram_sym:
                        # XRAM-backed name: if the use site's annotation shows any
                        # register holding the original XRAM value (XRAMRef with the
                        # same sym), a reload happened after the write — block
                        # propagation so the call keeps the reloaded value.
                        _sym = _vinfo.xram_sym
                        _blocked = any(
                            isinstance(_rv, XRAMRef)
                            and isinstance(_rv.inner, Const)
                            and _rv.inner.alias == _sym
                            for _rv in _use_ann.reg_exprs.values()
                        )
                        if _blocked:
                            dbg("propagate",
                                f"  [{hex(node.ea)}] prop-blocked: {r}={replacement.render()!r} "
                                f"— use-site has register holding {_sym}")
                            i += 1
                            continue

            # Guard: don't propagate past nodes that write to names (or their backing
            # registers) used in replacement.
            repl_refs = _expr_name_refs(replacement)
            if repl_refs and use_idx > i + 1:
                mid_writes = _collect_mid_writes(live[i + 1:use_idx], reg_map)
                if repl_refs & mid_writes:
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] prop-values: blocked {r} — "
                        f"intermediate writes {repl_refs & mid_writes}")
                    i += 1
                    continue

            # If replacement is a Const and the use is XRAM[pre ++/-- r] or a
            # standalone r++/r-- ExprStmt, detect now (before substitution
            # overwrites the node) so we can inject a synthetic Assign(r, K+δ)
            # to keep the chain propagating.
            old_use_node = live[use_idx]  # save before potential overwrite
            pre_incr_delta: Optional[int] = None
            pre_incr_is_xram = False
            if is_reg_lhs and isinstance(replacement, Const):
                pre_incr_delta = _xram_pre_incr_delta(old_use_node, r)
                if pre_incr_delta is not None:
                    pre_incr_is_xram = True
                else:
                    pre_incr_delta = _expr_stmt_incr_delta(old_use_node, r)

            new_node = _subst_reg_in_node(old_use_node, r, replacement)
            if new_node is not None:
                # Record the eliminated source node: the use site now represents
                # the combined effect of both instructions.
                define_node = live[i]
                new_node.source_nodes = [define_node] + list(old_use_node.source_nodes or [old_use_node])
                live[use_idx] = new_node
                live[i] = None
                dbg("typesimp", f"  [{hex(node.ea)}] prop-values: folded {r} into node {use_idx}")
                changed = True

                # Pre-link: the defining node is now gone from the live list.
                # Any downstream node that reads r only via its reg_exprs annotation
                # (not as a direct HIR name reference) will have the value substituted
                # later by _subst_from_reg_exprs without a source link.  Add the
                # defining node to those nodes' source_nodes now so the provenance
                # is preserved.  Only do this for register-lhs defines (Name-lhs
                # defines stay in the HIR and don't need pre-linking).
                if is_reg_lhs:
                    for _k in range(use_idx + 1, len(live)):
                        _nk = live[_k]
                        if _nk is None:
                            continue
                        _nk_ann = getattr(_nk, 'ann', None)
                        if (_nk_ann is not None
                                and r in _nk_ann.reg_exprs
                                and _count_reg_uses_in_node(r, _nk) > 0
                                and define_node not in _nk.source_nodes):
                            _nk.source_nodes = [define_node] + list(_nk.source_nodes)
                        # Stop if r is re-defined
                        if r in (_nk.written_regs if _nk is not None else frozenset()):
                            break

                # If a pre ++/-- r in XRAMRef was folded away, inject a synthetic
                # Assign(r, K+δ) so subsequent XRAM[++r] nodes continue propagating.
                # Only inject when there is actually another use of r beyond use_idx
                # (avoids leaving a dangling assignment at the end of a case body).
                if pre_incr_delta is not None:
                    new_r_val = Const((replacement.value + pre_incr_delta) & 0xFFFF)
                    has_more = any(
                        _count_reg_uses_in_node(r, live[k]) > 0
                        for k in range(use_idx + 1, len(live))
                        if not isinstance(live[k], (GotoStatement, BreakStmt, ContinueStmt))
                    )
                    if has_more:
                        # Use the EA and source of the inc/dec instruction itself,
                        # not the substituted XRAM node. For XRAM[++r], the inc
                        # instruction is source_nodes[0] of the original use node.
                        # Also include the defining node (new_node.source_nodes[0] ==
                        # original live[i]) so the full derivation is traceable.
                        defn_node = new_node.source_nodes[0]  # = original live[i]
                        if pre_incr_is_xram and old_use_node.source_nodes:
                            inc_node = old_use_node.source_nodes[0]
                            synthetic_ea = inc_node.ea
                            synthetic_sn: list = [defn_node, inc_node]
                        else:
                            synthetic_ea = old_use_node.ea
                            synthetic_sn = [defn_node, old_use_node]
                        synthetic = Assign(synthetic_ea, RegExpr((r,)), new_r_val)
                        synthetic.source_nodes = synthetic_sn
                        live.insert(use_idx + 1, synthetic)
                        dbg("typesimp",
                            f"  [{hex(node.ea)}] prop-values: injected "
                            f"{r}={hex(new_r_val.value)} after XRAM[++{r}] fold")
        elif total_uses > 1 and is_reg_lhs and _is_reg_free(replacement) and use_idx is not None:
            # Multi-use propagation only for Reg-lhs: Name-lhs replacements may be
            # Call exprs or other side-effecting expressions that cannot be duplicated.
            # All-or-nothing within any hidden-kill boundary: if a use site's
            # annotation records a DIFFERENT value for r than replacement, a kill
            # happened between our def and that site but was folded away from the HIR
            # (e.g. movx A,@DPTR that loaded a new value but was removed because its
            # only HIR use was already propagated).  Treat that site as a hard boundary
            # and only substitute use sites before it.
            end = kill_idx if kill_idx is not None else len(live)
            hidden_kill = end
            for _hk in range(i + 1, end):
                if _count_reg_uses_in_node(r, live[_hk]) > 0:
                    _hk_ann = getattr(live[_hk], 'ann', None)
                    if _hk_ann is not None:
                        _hk_val = _hk_ann.reg_exprs.get(r)
                        if _hk_val is not None and _hk_val != replacement:
                            hidden_kill = _hk
                            dbg("propagate",
                                f"  [{hex(node.ea)}] prop-multi: hidden kill of {r}"
                                f" at {_hk} (annotation says {r}={_hk_val.render()!r})")
                            break
            tentative: dict = {}
            all_ok = True
            for j in range(i + 1, hidden_kill):
                if _count_reg_uses_in_node(r, live[j]) > 0:
                    new_node = _subst_reg_in_node(live[j], r, replacement)
                    if new_node is None:
                        all_ok = False
                        break
                    tentative[j] = new_node
            if all_ok and tentative:
                src_node = live[i]
                for j, new_node in tentative.items():
                    # Record the defining node as a source of every use site.
                    new_node.source_nodes = [src_node] + list(live[j].source_nodes or [live[j]])
                    live[j] = new_node
                # Use the full end (not hidden_kill) for the remaining check so we
                # don't remove live[i] if r is still needed beyond the boundary.
                remaining = _collect_hir_name_refs(live[i + 1:end])
                if r not in remaining:
                    live[i] = None
                dbg("typesimp",
                    f"  [{hex(node.ea)}] prop-values-multi: {r} = {replacement.render()!r}"
                    f" into {len(tentative)} site(s)")
                changed = True
        elif (total_uses == 0 and is_reg_lhs
                and isinstance(replacement, Const)
                and r == 'DPTR'):
            # DPTR = Const(k) with no downstream name-uses: the XRAM handler may
            # have already folded DPTR's address into XRAMRef(Const(k)) at lift
            # time, so the XRAM write node doesn't reference 'DPTR' by name.
            # Link this node as a provenance source of the first downstream
            # XRAMRef(Const(k)) node so the original mov DPTR instruction is
            # traceable in the detail viewer.
            dptr_val = replacement.value
            end = (kill_idx + 1) if kill_idx is not None else len(live)
            for j in range(i + 1, end):
                lj = live[j]
                if lj is not None and _has_xram_const_addr(lj, dptr_val):
                    if node not in lj.source_nodes:
                        lj.source_nodes = [node] + list(lj.source_nodes)
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] prop-dptr-const: linked {hex(dptr_val)}"
                        f" provenance into node {j}")
                    break

        i += 1

    return [n for n in live if n is not None], changed


# ── Sub-pass C: multi-register group setup inlining ──────────────────────────

def _subst_group_in_call_node(node: HIRNode, regs_tuple: tuple,
                               replacement: Expr) -> Optional[HIRNode]:
    """Replace Regs(names==regs_tuple) in call args of node with replacement.

    Returns a new node on success, None if the group is not found in any call arg.
    """
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
    if isinstance(node, TypedAssign) and isinstance(node.rhs, Call):
        new_call = _patch(node.rhs)
        if new_call is not None:
            result = TypedAssign(node.ea, node.type_str, node.lhs, new_call)
    elif isinstance(node, Assign) and isinstance(node.rhs, Call):
        new_call = _patch(node.rhs)
        if new_call is not None:
            result = Assign(node.ea, node.lhs, new_call)
    elif (isinstance(node, Assign)
          and isinstance(node.rhs, RegExpr)
          and not node.rhs.is_single
          and node.rhs.names == regs_tuple):
        # Direct group-to-register copy: DPTR = BA → DPTR = replacement
        result = Assign(node.ea, node.lhs, replacement)
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

    Transforms:
      Assign(Regs(regs_tuple, alias=name), rhs)   [rhs is Const or Name]
      ...  [no intervening use of any reg in the group]
      call(..., Regs(regs_tuple, ...), ...)
    →
      call(..., rhs, ...)

    Handles both plain Assign and TypedAssign (typed multi-byte setups).
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
        # Canonicalize rhs using the source node's annotation so that constant
        # registers are folded in before substitution.  E.g. {B,A} = _var2 * B
        # with ann.reg_consts B=3 → rhs becomes _var2 * 3.
        if node.ann is not None:
            rhs = _canonicalize_expr(rhs,
                                     node.ann.reg_consts or {},
                                     node.ann.reg_groups or [],
                                     node.ann.reg_exprs or {})

        # Find first downstream use of the group as a unit (all registers together
        # as a Regs(regs_tuple) expression), not individual register uses.
        # Individual uses of group registers (e.g. B after {B,A} = MUL) don't
        # prevent folding the group into its single group-use site.
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
                break  # killed (or used-then-killed at call site — stop either way)

        if conflict or use_idx is None:
            i += 1
            continue

        # Guard: don't inline past intermediate writes of names referenced in rhs.
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

        # Hidden-kill guard: if rhs contains any Regs(r) and the call site's
        # annotation records r=Const (a pruned-away constant store), it means
        # a write to r was folded out of the HIR between the def and the call.
        # Inlining would embed the stale value of r into the call argument.
        _GP_REGS = {'R0','R1','R2','R3','R4','R5','R6','R7','A','B'}
        use_ann = getattr(live[use_idx], 'ann', None)
        if use_ann is not None and use_ann.reg_exprs:
            _hidden_kill = False
            for _r in _expr_name_refs(rhs):
                if _r not in _GP_REGS:
                    continue
                _use_val = use_ann.reg_exprs.get(_r)
                if not isinstance(_use_val, Const):
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
    """True if expr is safe to substitute at ann's node.

    An expression is safe when it is reg-free (no Regs leaves at all), OR when
    every Regs leaf it contains is NOT an active TypeGroup member at this node.
    TypeGroup members have already been renamed to Name nodes by _simplify, so
    re-introducing their raw Reg forms would be incorrect.  Non-TypeGroup
    registers were not renamed and remain valid in the HIR.
    """
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
    """Sub-pass A1: substitute reg_exprs annotations directly into node expressions.

    For each node whose annotation records a safe expression for register r,
    and where the node reads r, substitute r → expr then fold algebraically.

    "Safe" means reg-free OR containing only non-TypeGroup registers — the latter
    were not renamed by _simplify and are still valid in the HIR at this point.

    This handles accumulated CompoundAssign values that _propagate_register_copies
    cannot reach (e.g. A = arg1*3 tracked via A+=R0; A+=R0 → switch(A/3) → switch(arg1)).

    Returns (updated_list, changed).
    """
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
                continue  # circular: expr references r itself (e.g. R4 → rol9(R4, C))
            if _count_reg_uses_in_node(r, current) == 0:
                continue
            new_node = _subst_reg_in_node(current, r, expr)
            if new_node is None:
                continue
            new_node = _fold_exprs_in_node(new_node)
            current = new_node
            any_changed = True
        # Synthesize DPTR from DPH+DPL when DPTR is used but absent from reg_exprs.
        # Handles call sites where DPH/DPL were set piecemeal (annotation kills DPTR
        # TypeGroup but still tracks DPH/DPL individually via expr_state).
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
    retval name, inline the call expression into the target and remove the
    TypedAssign.
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
                    # Guard: don't inline past nodes that write to any name/reg
                    # (or their backing registers) referenced in the call expression.
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
                        new_node = Assign(tgt.ea, tgt.lhs, call_expr)
                        new_node.ann = NodeAnnotation.merge(src_node, tgt)
                        new_node.source_nodes = [src_node, tgt]
                        live[abs_j] = new_node
                    else:
                        new_node = _subst_reg_in_node(tgt, retval_name, call_expr)
                        if new_node is not None:
                            # _subst_reg_in_node already copies tgt.ann; merge in src callee_args
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


# ── Master propagation pass ───────────────────────────────────────────────────

def _propagate_values(nodes: List[HIRNode],
                       reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Forward single-use propagation pass.

    Recurses into structured nodes first, then applies three sub-passes in a
    fixed-point loop until no changes occur:
      A0: fold Assign + CompoundAssign into a single Assign
      A:  substitute single-use (or reg-free multi-use) register copies
      B:  inline single-use retval TypedAssign into its target
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
