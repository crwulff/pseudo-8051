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
from pseudo8051.ir.expr import (Expr, Const, Call, BinOp, Paren,
                                 Reg as RegExpr, Regs as RegsExpr,
                                 RegGroup as RegGroupExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import TypeGroup, VarInfo, _count_reg_uses_in_node, _subst_reg_in_node, _const_str
from pseudo8051.passes.typesimplify._dptr import _is_dptr_inc_node
from pseudo8051.constants import dbg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_const(expr: Expr, node: HIRNode) -> Expr:
    """If expr is a single-register Regs and its value is known in node.ann.reg_consts,
    return Const(val); otherwise return expr unchanged."""
    if isinstance(expr, RegsExpr) and expr.is_single:
        ann = getattr(node, 'ann', None)
        if ann is not None:
            val = ann.reg_consts.get(expr.name)
            if val is not None:
                return Const(val)
    return expr


# ── Predicates ────────────────────────────────────────────────────────────────

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
            new_node = ExprStmt(node.ea, new_call)
            new_node.ann = node.ann
            new_node.src_eas = node.src_eas
            return new_node
        return node
    if isinstance(node, Assign) and isinstance(node.rhs, Call):
        new_call = _patch(node.rhs)
        if new_call is not node.rhs:
            new_node = Assign(node.ea, node.lhs, new_call)
            new_node.ann = node.ann
            new_node.src_eas = node.src_eas
            return new_node
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
            # Be conservative for switch case bodies: any register written in any
            # case body is a potential output that might be read after the switch.
            # Add these to outer_refs so they're not pruned as dead.
            case_writes: set = set()
            for _, body in node.cases:
                if isinstance(body, list):
                    for bn in body:
                        case_writes |= bn.written_regs
            if isinstance(node.default_body, list):
                for bn in node.default_body:
                    case_writes |= bn.written_regs
            extended = succ_refs | frozenset(case_writes)
            result.append(node.map_bodies(
                lambda ns, refs=extended: _fold_and_prune_setups(ns, reg_map, refs)))
        else:
            result.append(node.map_bodies(
                lambda ns, refs=succ_refs: _fold_and_prune_setups(ns, reg_map, refs)))

    work: List[HIRNode] = result

    # Phase 1: fold Assign(Reg, Const) into the next call's args.
    for i in range(len(work)):
        node = work[i]
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegsExpr) and node.lhs.is_single
                and isinstance(node.rhs, Const)):
            continue
        reg = node.lhs.name
        val = node.rhs
        for j in range(i + 1, len(work)):
            nj = work[j]
            if _is_call_setup_assign(nj) or _is_dptr_inc_node(nj):
                continue
            new_nj = _subst_reg_in_call_node(nj, reg, val)
            if new_nj is not nj:
                new_nj.src_eas = new_nj.src_eas | node.src_eas
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
            all_downstream = _collect_hir_name_refs(work[i + 1:]) | _outer_refs
            if lhs_regs.isdisjoint(all_downstream):
                dbg("typesimp",
                    f"  [{hex(node.ea)}] prune-setup: {node.lhs.render()} = {node.rhs.render()}")
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

    # Harvest pair entries from per-node call_arg_ann annotations.
    for node in nodes:
        ann = getattr(node, "ann", None)
        if ann is None:
            continue
        for g in ann.call_arg_ann:
            if len(g.full_regs) > 1:
                key = g.full_regs
                if key not in pair_groups:
                    pair_groups[key] = g
                for pr in g.full_regs:
                    reg_to_pair.setdefault(pr, key)

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

        byte_assigns: Dict[str, tuple] = {r0: (i, _resolve_const(node.rhs, node))}
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
                byte_assigns[nd.lhs.name] = (k, _resolve_const(nd.rhs, nd))
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
        for r_check, (idx_check, _) in byte_assigns.items():
            nd_check = nodes[idx_check]
            ann_check = getattr(nd_check, "ann", None)
            if ann_check is not None:
                ca_g = ann_check.call_arg_for(r_check)
                if ca_g is not None and ca_g.full_regs == regs_key:
                    naming_vinfo = ca_g
                    break

        # Build combined expression: (byte0 << (n-1)*8) | byte1 | ... | byte_{n-1}
        n_bytes = len(regs_key)
        ordered_exprs = [byte_assigns[r][1] for r in regs_key]

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

        all_src_eas = frozenset().union(*(nodes[idx].src_eas for _, (idx, _) in byte_assigns.items()))
        if naming_vinfo.type and naming_vinfo.name:
            result_node = TypedAssign(node.ea, naming_vinfo.type,
                                      RegGroupExpr(regs_key, alias=naming_vinfo.name), combined)
        else:
            result_node = Assign(node.ea, RegGroupExpr(regs_key), combined)
        result_node.src_eas = all_src_eas
        out.append(result_node)
        dbg("typesimp",
            f"  [{hex(node.ea)}] fold-call-arg-pair: {''.join(regs_key)} = {combined.render()}")

        for r, (idx, _) in byte_assigns.items():
            consumed.add(idx)

    return out
