"""
passes/typesimplify/_post.py — Post-simplify passes and shared traversal helper.
"""

import re
from typing import Callable, Dict, List, Optional

from pseudo8051.ir.hir    import (HIRNode, Assign, TypedAssign, CompoundAssign,
                                   ExprStmt, ReturnStmt, IfGoto, IfNode, WhileNode, ForNode,
                                   DoWhileNode, SwitchNode)
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns._utils  import (
    VarInfo, _type_bytes, _byte_names, _walk_expr,
    _count_reg_uses_in_node, _subst_reg_in_node,
)
from pseudo8051.ir.expr import (Expr, UnaryOp, BinOp, Const, Call, Paren,
                                 Reg as RegExpr, RegGroup as RegGroupExpr, Name as NameExpr,
                                 XRAMRef, IRAMRef)

# ── Shared traversal helper ───────────────────────────────────────────────────

def recurse_bodies(nodes: List[HIRNode], fn: Callable) -> List[HIRNode]:
    """Recurse fn into structured node bodies; pass other nodes through."""
    return [node.map_bodies(fn) for node in nodes]


# ── Shared predicate ──────────────────────────────────────────────────────────

def _is_dptr_inc_node(node: HIRNode) -> bool:
    """True for ExprStmt(DPTR++) — a data-pointer advance node."""
    return (isinstance(node, ExprStmt)
            and isinstance(node.expr, UnaryOp)
            and node.expr.op == "++"
            and node.expr.operand == RegExpr("DPTR"))


_RE_BYTE_FIELD = re.compile(r'^(.+)\.(?:hi|lo|b\d+)$')


# ── XRAM local load consolidation ────────────────────────────────────────────

def _consolidate_xram_local_loads(nodes: List[HIRNode],
                                   reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Consolidate post-relay XRAM local byte loads into register-pair assignments.

    Two patterns handled:
    1. Rhi=Name("X.hi"); [DPTR++;...]; Rlo=Name("X.lo"); → RegGroup(Rhi,Rlo)=Name("X")
       Mutates reg_map["RhiRlo"] = VarInfo("X", type, (Rhi, Rlo)) (non-param, no xram_sym).
    2. Rn=Name("local") where "local" is a 1-byte XRAM local (no byte-field suffix)
       Kept as-is; mutates reg_map[Rn] = VarInfo("local", type, (Rn,), is_param=True).
    """
    def _parent_name(s):
        m = _RE_BYTE_FIELD.match(s)
        return m.group(1) if m else None

    def _find_parent_vinfo(parent_nm):
        for v in reg_map.values():
            if (isinstance(v, VarInfo) and v.name == parent_nm
                    and v.xram_sym and not v.is_byte_field):
                return v
        return None

    def _as_byte_assign(n):
        """Return (reg_name, name_str) if n is 'Reg(r) = Name(s)'; else None."""
        if (isinstance(n, Assign)
                and isinstance(n.lhs, RegExpr)
                and isinstance(n.rhs, NameExpr)):
            return (n.lhs.name, n.rhs.name)
        return None

    nodes = list(nodes)
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        ba = _as_byte_assign(node)
        if ba is not None:
            reg0, bname0 = ba
            parent_nm = _parent_name(bname0)
            if parent_nm is not None:
                parent_vinfo = _find_parent_vinfo(parent_nm)
                if parent_vinfo is not None:
                    n_bytes = _type_bytes(parent_vinfo.type)
                    expected_bnames = _byte_names(parent_nm, n_bytes)
                    if bname0 == expected_bnames[0] and n_bytes >= 2:
                        regs = [reg0]
                        j = i + 1
                        for k in range(1, n_bytes):
                            while j < len(nodes) and _is_dptr_inc_node(nodes[j]):
                                j += 1
                            if j >= len(nodes):
                                break
                            next_ba = _as_byte_assign(nodes[j])
                            if next_ba is None or next_ba[1] != expected_bnames[k]:
                                break
                            regs.append(next_ba[0])
                            j += 1
                        if len(regs) == n_bytes:
                            pair_key = "".join(regs)
                            new_vinfo = VarInfo(parent_nm, parent_vinfo.type, tuple(regs))
                            reg_map[pair_key] = new_vinfo
                            for r in regs:
                                reg_map[r] = new_vinfo
                            # If lo byte landed in DPL and the next node is DPH = Rhi,
                            # collapse to DPTR = var instead of RegGroup = var.
                            if (regs[-1] == "DPL" and j < len(nodes)
                                    and _as_dph_assign(nodes[j]) == regs[0]):
                                out.append(Assign(node.ea, RegExpr("DPTR"),
                                                  NameExpr(parent_nm)))
                                j += 1
                                dbg("typesimp",
                                    f"  [{hex(node.ea)}] xram-pair-consolidate: DPTR = {parent_nm}"
                                    f" (via {pair_key} + DPH)")
                            else:
                                out.append(Assign(node.ea,
                                                  RegGroupExpr(tuple(regs)),
                                                  NameExpr(parent_nm)))
                                dbg("typesimp",
                                    f"  [{hex(node.ea)}] xram-pair-consolidate: {pair_key} = {parent_nm}")
                            i = j
                            continue
            else:
                # No byte suffix — check if it's a 1-byte XRAM local
                for v in reg_map.values():
                    if (isinstance(v, VarInfo) and v.name == bname0
                            and v.xram_sym and not v.is_byte_field
                            and _type_bytes(v.type) == 1):
                        new_vinfo = VarInfo(bname0, v.type, (reg0,), is_param=False)
                        reg_map[reg0] = new_vinfo
                        dbg("typesimp", f"  [{hex(node.ea)}] xram-single-load: {reg0} = {bname0}")
                        # Immediately substitute Reg(reg0) → Name(bname0) in remaining
                        # nodes of this scope. Without this, a later reg_map[reg0]
                        # mutation from a different scope (e.g. the merge block) would
                        # corrupt _simplify_once's substitution for this scope.
                        nodes[i + 1:] = _subst_reg_in_scope(
                            nodes[i + 1:], reg0, NameExpr(bname0))
                        break
        # Recurse into structured nodes with a scope-local copy of reg_map so
        # that arm-internal mutations (xram-single-load, pair keys) don't leak
        # into the outer scope and cause incorrect substitutions at merge points.
        if isinstance(node, IfNode):
            out.append(IfNode(node.ea, node.condition,
                _consolidate_xram_local_loads(node.then_nodes, dict(reg_map)),
                _consolidate_xram_local_loads(node.else_nodes, dict(reg_map))))
        elif isinstance(node, WhileNode):
            out.append(WhileNode(node.ea, node.condition,
                _consolidate_xram_local_loads(node.body_nodes, dict(reg_map))))
        elif isinstance(node, ForNode):
            out.append(ForNode(node.ea, node.init, node.condition, node.update,
                _consolidate_xram_local_loads(node.body_nodes, dict(reg_map))))
        elif isinstance(node, DoWhileNode):
            out.append(DoWhileNode(node.ea, node.condition,
                _consolidate_xram_local_loads(node.body_nodes, dict(reg_map))))
        elif isinstance(node, SwitchNode):
            new_cases = [
                (vals, _consolidate_xram_local_loads(body, dict(reg_map))
                       if isinstance(body, list) else body)
                for vals, body in node.cases
            ]
            new_default = (
                _consolidate_xram_local_loads(node.default_body, dict(reg_map))
                if isinstance(node.default_body, list) else node.default_body
            )
            out.append(SwitchNode(node.ea, node.subject, new_cases,
                                  node.default_label, new_default))
        else:
            out.append(node)
        i += 1
    return out


# ── DPL/DPH → DPTR collapsing ────────────────────────────────────────────────

def _as_dph_assign(n) -> Optional[str]:
    """Return Rhi name if n is Assign(Reg('DPH'), Reg(Rhi)); else None."""
    if (isinstance(n, Assign)
            and isinstance(n.lhs, RegExpr) and n.lhs.name == "DPH"
            and isinstance(n.rhs, RegExpr)):
        return n.rhs.name
    return None


def _as_dpl_assign(n) -> Optional[str]:
    """Return Rlo name if n is Assign(Reg('DPL'), Reg(Rlo)); else None."""
    if (isinstance(n, Assign)
            and isinstance(n.lhs, RegExpr) and n.lhs.name == "DPL"
            and isinstance(n.rhs, RegExpr)):
        return n.rhs.name
    return None


def _collapse_dpl_dph(nodes: List[HIRNode],
                       reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Collapse paired DPH/DPL byte assignments into a single DPTR assignment.

    DPH = Rhi; [skippable...] DPL = Rlo;  →  DPTR = RhiRlo;  (or DPTR = var;)
    DPL = Rlo; [skippable...] DPH = Rhi;  →  same

    Skippable = _is_call_setup_assign or _is_dptr_inc_node.
    """
    # Recurse into structured nodes first.
    recursed = recurse_bodies(nodes, lambda ns: _collapse_dpl_dph(ns, reg_map))

    out: List[HIRNode] = []
    dead: set = set()   # indices consumed as the partner half of a pair

    for i, node in enumerate(recursed):
        if i in dead:
            continue

        rhi = _as_dph_assign(node)   # DPH = Rhi case
        rlo = _as_dpl_assign(node)   # DPL = Rlo case
        if rhi is None and rlo is None:
            out.append(node)
            continue

        # Search forward for the partner, skipping setup/dptr++ nodes.
        partner_idx = None
        partner_val = None
        for k in range(i + 1, len(recursed)):
            if k in dead:
                continue
            if rhi is not None:
                rlo2 = _as_dpl_assign(recursed[k])
                if rlo2 is not None:
                    partner_idx, partner_val = k, rlo2
                    break
            else:
                rhi2 = _as_dph_assign(recursed[k])
                if rhi2 is not None:
                    partner_idx, partner_val = k, rhi2
                    break
            if not (_is_call_setup_assign(recursed[k]) or _is_dptr_inc_node(recursed[k])):
                break   # non-skippable node blocks the search

        if partner_idx is None:
            out.append(node)
            continue

        # Build the collapsed DPTR assignment.
        if rhi is not None:
            reg_hi, reg_lo = rhi, partner_val
        else:
            reg_hi, reg_lo = partner_val, rlo

        pair_key = reg_hi + reg_lo
        vinfo = reg_map.get(pair_key)
        rhs: Expr = (NameExpr(vinfo.name)
                     if isinstance(vinfo, VarInfo)
                     else RegGroupExpr((reg_hi, reg_lo)))
        out.append(Assign(node.ea, RegExpr("DPTR"), rhs))
        dead.add(partner_idx)
        dbg("typesimp", f"  [{hex(node.ea)}] dpl-dph-collapse: DPTR = {pair_key}")

    return out


# ── Post-simplify call-arg fold and dead-setup pruning ────────────────────────


def _collect_hir_name_refs(nodes: List[HIRNode]) -> set:
    """Collect all Reg/Name .name strings from expression positions in nodes."""
    refs: set = set()

    def _add(expr: Expr) -> None:
        def _fn(e: Expr) -> Expr:
            if isinstance(e, (RegExpr, NameExpr)):
                refs.add(e.name)
            elif isinstance(e, RegGroupExpr):   # RegGroup in read position → each component reg
                for r in e.regs:
                    refs.add(r)
                refs.add("".join(e.regs))       # also the pair name, e.g. "R2R3"
            return e
        _walk_expr(expr, _fn)

    def _visit(node: HIRNode) -> None:
        if isinstance(node, Assign):
            _add(node.rhs)
            if not isinstance(node.lhs, (RegExpr, RegGroupExpr)):
                _add(node.lhs)
        elif isinstance(node, CompoundAssign):
            _add(node.rhs)
            if isinstance(node.lhs, (RegExpr, NameExpr)):   # compound assign READS its LHS
                refs.add(node.lhs.name)
        elif isinstance(node, ExprStmt):
            _add(node.expr)
        elif isinstance(node, ReturnStmt) and node.value is not None:
            _add(node.value)
        elif isinstance(node, IfGoto):
            _add(node.cond)
        elif isinstance(node, IfNode):
            for sub in list(node.then_nodes) + list(node.else_nodes):
                _visit(sub)
        elif isinstance(node, (WhileNode, ForNode, DoWhileNode)):
            for sub in node.body_nodes:
                _visit(sub)

    for node in nodes:
        _visit(node)
    return refs


def _expand_pair_refs(refs, reg_map: Dict[str, VarInfo]) -> frozenset:
    """
    Expand pair register names (e.g. 'R2R3') to their individual components
    ('R2', 'R3') using the reg_map.

    When a call uses Reg('R2R3') as an argument, it implicitly reads both R2
    and R3.  Without this expansion, an Assign(R3, ...) that feeds into R2R3
    would look dead because lhs_regs={'R3'} is disjoint from {'R2R3'}.
    """
    extra: set = set()
    for name in refs:
        vinfo = reg_map.get(name)
        if isinstance(vinfo, VarInfo) and len(vinfo.regs) > 1:
            extra.update(vinfo.regs)
    return frozenset(refs) | frozenset(extra) if extra else frozenset(refs)


def _is_call_setup_assign(node: HIRNode) -> bool:
    """True for Assign(Reg/RegGroup, Name/Const) — a consolidated register-setup node."""
    return (isinstance(node, Assign)
            and isinstance(node.lhs, (RegExpr, RegGroupExpr))
            and isinstance(node.rhs, (NameExpr, Const)))


def _lhs_reg_names(node: HIRNode) -> frozenset:
    """Return register name strings written by a setup-assign node."""
    if not isinstance(node, Assign):
        return frozenset()
    lhs = node.lhs
    if isinstance(lhs, RegExpr):
        return frozenset({lhs.name})
    if isinstance(lhs, RegGroupExpr):
        names = set(lhs.regs)
        names.add("".join(lhs.regs))
        return frozenset(names)
    return frozenset()


def _subst_reg_in_call_node(node: HIRNode, reg: str, replacement: Expr) -> HIRNode:
    """Replace Name(reg) with replacement in the call args of node."""
    repl_str = replacement.render()

    def _patch(call: Call) -> Call:
        new_args = [replacement if (isinstance(a, NameExpr) and a.name == reg) else a
                    for a in call.args]
        if any(na is not oa for na, oa in zip(new_args, call.args)):
            return Call(call.func_name, new_args)
        return call

    if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
        new_call = _patch(node.expr)
        return ExprStmt(node.ea, new_call) if new_call is not node.expr else node
    if isinstance(node, Assign) and isinstance(node.rhs, Call):
        new_call = _patch(node.rhs)
        return Assign(node.ea, node.lhs, new_call) if new_call is not node.rhs else node
    return node


def _first_kill_before_read(reg: str, nodes: List[HIRNode]) -> bool:
    """
    Return True if `reg` is written before being read in the flat prefix of `nodes`.

    Scans nodes in order; stops at the first node that reads OR writes `reg`.
    Returns True only if the first relevant node is a pure WRITE (not a read).
    For structured nodes (IfNode etc.) where either branch might read, returns False
    (conservative: assume the reg might be read).
    """
    for node in nodes:
        if isinstance(node, (IfNode, WhileNode, ForNode, DoWhileNode, SwitchNode)):
            return False  # conservative: might read in a branch
        # CompoundAssign LHS is both read and written — _count_reg_uses_in_node
        # only counts the RHS, so check the LHS explicitly here.
        if (isinstance(node, CompoundAssign)
                and isinstance(node.lhs, RegExpr)
                and node.lhs.name == reg):
            return False  # CompoundAssign reads its LHS operand
        reads = _count_reg_uses_in_node(reg, node)
        writes = reg in node.written_regs
        if reads > 0:
            return False  # reg is read first
        if writes:
            return True   # reg is written before being read
    return False


def _fold_and_prune_setups(nodes: List[HIRNode],
                            reg_map: Dict[str, VarInfo],
                            _outer_refs: frozenset = frozenset()) -> List[HIRNode]:
    """
    Post-simplify cleanup of register-setup lines before calls.

    1. Fold Assign(Reg, Const) into the next call node's args.
    2. Remove Assign(Reg/RegGroup, Name/Const) setup nodes whose LHS registers
       are not referenced in any subsequent node (including caller context via
       _outer_refs, so branches don't prune registers read after the enclosing
       IfNode/WhileNode/ForNode).
    3. Remove DPTR++ nodes whose DPTR value is not referenced afterwards.
    Recurses into IfNode / WhileNode / ForNode / SwitchNode bodies.
    """
    # Recurse into structured bodies first, passing the correct successor context.
    result: List[HIRNode] = []
    for k, node in enumerate(nodes):
        # refs visible after this node (siblings + caller context), with pair names expanded
        succ_refs = _expand_pair_refs(
            frozenset(_collect_hir_name_refs(nodes[k + 1:])) | _outer_refs, reg_map)
        if isinstance(node, IfNode):
            result.append(IfNode(node.ea, node.condition,
                _fold_and_prune_setups(node.then_nodes, reg_map, succ_refs),
                _fold_and_prune_setups(node.else_nodes, reg_map, succ_refs)))
        elif isinstance(node, WhileNode):
            result.append(WhileNode(node.ea, node.condition,
                _fold_and_prune_setups(node.body_nodes, reg_map, succ_refs)))
        elif isinstance(node, ForNode):
            result.append(ForNode(node.ea, node.init, node.condition, node.update,
                _fold_and_prune_setups(node.body_nodes, reg_map, succ_refs)))
        elif isinstance(node, DoWhileNode):
            result.append(DoWhileNode(node.ea, node.condition,
                _fold_and_prune_setups(node.body_nodes, reg_map, succ_refs)))
        elif isinstance(node, SwitchNode):
            new_cases = [(vals, _fold_and_prune_setups(body, reg_map, succ_refs)
                          if isinstance(body, list) else body)
                         for vals, body in node.cases]
            new_default = (_fold_and_prune_setups(node.default_body, reg_map, succ_refs)
                           if isinstance(node.default_body, list) else node.default_body)
            result.append(SwitchNode(node.ea, node.subject, new_cases,
                                     node.default_label, new_default))
        else:
            result.append(node)

    work: List[HIRNode] = result

    # Phase 1: fold Assign(Reg, Const) into the next call's args.
    for i in range(len(work)):
        node = work[i]
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr)
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
                work[j] = new_nj
                work[i] = None
                dbg("typesimp", f"  [{hex(node.ea)}] fold-const: {reg}={val.render()} into call")
            break
    work = [n for n in work if n is not None]

    # Phase 2: remove dead setup-assign and DPTR++ nodes.
    out: List[HIRNode] = []
    for i, node in enumerate(work):
        if _is_call_setup_assign(node):
            lhs_regs = _lhs_reg_names(node)
            all_downstream = _expand_pair_refs(
                _collect_hir_name_refs(work[i + 1:]) | _outer_refs, reg_map)
            if lhs_regs.isdisjoint(all_downstream):
                dbg("typesimp",
                    f"  [{hex(node.ea)}] prune-setup: {node.lhs.render()} = {node.rhs.render()}")
                continue
            # Also prune if every LHS reg that appears downstream is killed before first read
            live_lhs = lhs_regs & all_downstream
            if live_lhs and all(_first_kill_before_read(r, work[i + 1:])
                                 for r in live_lhs):
                dbg("typesimp",
                    f"  [{hex(node.ea)}] prune-setup-killed: "
                    f"{node.lhs.render()} = {node.rhs.render()}")
                continue
        elif _is_dptr_inc_node(node):
            # DPTR++ increments a pointer for immediate local use; its value never
            # flows across a control-flow merge, so we check only intra-scope refs.
            if "DPTR" not in _collect_hir_name_refs(work[i + 1:]):
                dbg("typesimp", f"  [{hex(node.ea)}] prune-dptr++")
                continue
        out.append(node)
    return out


# ── Return-chain folding ──────────────────────────────────────────────────────

def _try_combine_byte_fields(ret_regs: tuple,
                              exprs_by_reg: Dict[str, "Expr"]) -> Optional["Expr"]:
    """Try to combine hi/lo Name byte-fields into a single parent Name.

    E.g. ret_regs=('R2','R1'), exprs={'R2': Name('dest.hi'), 'R1': Name('dest.lo')}
    → Name('dest').  Returns None if the pattern does not match.
    """
    suffixes = ("hi", "lo") if len(ret_regs) == 2 else tuple(f"b{k}" for k in range(len(ret_regs)))
    parent: Optional[str] = None
    for r, suf in zip(ret_regs, suffixes):
        expr = exprs_by_reg.get(r)
        if not isinstance(expr, NameExpr):
            return None
        parts = expr.name.rsplit(".", 1)
        if len(parts) != 2 or parts[1] != suf:
            return None
        if parent is None:
            parent = parts[0]
        elif parts[0] != parent:
            return None
    return NameExpr(parent) if parent else None


def _fold_return_chains(hir: List[HIRNode], ret_regs: tuple,
                         reg_map: Optional[Dict] = None) -> List[HIRNode]:
    """Fold return-register assignments into ReturnStmt.

    Single-register return (len(ret_regs) == 1):
        Assign(ret_reg, expr); ReturnStmt(ret_reg | None) → ReturnStmt(expr)

    Multi-register return (len(ret_regs) > 1):
        Assign(RegGroup(ret_regs), expr); ReturnStmt(None) → ReturnStmt(expr)
        ReturnStmt(None): strip trailing ret-reg self-assigns; produce
            ReturnStmt(Name(pair_name)) from reg_map, or ReturnStmt(RegGroup(ret_regs)).
        If ALL ret_regs assigned consecutively just before ReturnStmt:
            try to collapse byte-field pair (Name("x.hi")/Name("x.lo") → Name("x")),
            else fall back to reg_map pair name.
    """
    ret_reg_set = set(ret_regs)
    pair_name   = "".join(ret_regs) if len(ret_regs) > 1 else None

    def _canon_return_expr() -> Optional["Expr"]:
        """Name (from reg_map) or RegGroup for the multi-reg return value."""
        if pair_name and reg_map:
            vinfo = reg_map.get(pair_name)
            if isinstance(vinfo, VarInfo) and vinfo.name:
                return NameExpr(vinfo.name)
        if pair_name:
            return RegGroupExpr(ret_regs)
        return None

    def _fold(nodes: List[HIRNode]) -> List[HIRNode]:
        out: List[HIRNode] = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            # ── Multi-reg: RegGroup assignment immediately before ReturnStmt ─────
            if (pair_name is not None
                    and isinstance(node, Assign)
                    and isinstance(node.lhs, RegGroupExpr)
                    and frozenset(node.lhs.regs) == frozenset(ret_regs)
                    and i + 1 < len(nodes)
                    and isinstance(nodes[i + 1], ReturnStmt)
                    and nodes[i + 1].value is None):
                out.append(ReturnStmt(node.ea, node.rhs))
                dbg("typesimp", f"  [{hex(node.ea)}] fold-return (RegGroup): {node.rhs.render()}")
                i += 2
                continue

            # ── Single-reg: only for single-register returns ──────────────────
            if (pair_name is None
                    and isinstance(node, Assign)
                    and isinstance(node.lhs, RegExpr)
                    and node.lhs.name in ret_reg_set
                    and i + 1 < len(nodes)
                    and isinstance(nodes[i + 1], ReturnStmt)
                    and (nodes[i + 1].value is None
                         or nodes[i + 1].value == RegExpr(node.lhs.name))):
                out.append(ReturnStmt(node.ea, node.rhs))
                dbg("typesimp", f"  [{hex(node.ea)}] fold-return (single): {node.rhs.render()}")
                i += 2
                continue

            # ── Multi-reg ReturnStmt(None): produce canonical pair expression ──
            if (pair_name is not None
                    and isinstance(node, ReturnStmt)
                    and node.value is None):
                # Collect consecutive ret_reg assignments at end of out
                collected: Dict[str, tuple] = {}   # reg → (out_index, expr)
                k = len(out) - 1
                while k >= 0:
                    prev = out[k]
                    if (isinstance(prev, Assign)
                            and isinstance(prev.lhs, RegExpr)
                            and prev.lhs.name in ret_reg_set
                            and prev.lhs.name not in collected):
                        collected[prev.lhs.name] = (k, prev.rhs)
                        k -= 1
                    else:
                        break

                if len(collected) == len(ret_regs):
                    # All ret_regs assigned consecutively — fold them all
                    exprs_by_reg = {r: collected[r][1] for r in ret_regs}
                    combined = (_try_combine_byte_fields(ret_regs, exprs_by_reg)
                                or _canon_return_expr()
                                or RegGroupExpr(ret_regs))
                    start_idx = min(v[0] for v in collected.values())
                    del out[start_idx:]
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] fold-return (multi-all): {combined.render()}")
                else:
                    # Partial or no individual assignments: strip trivial self-assigns,
                    # then use the canonical pair expression from reg_map.
                    while (out
                           and isinstance(out[-1], Assign)
                           and isinstance(out[-1].lhs, RegExpr)
                           and out[-1].lhs.name in ret_reg_set
                           and isinstance(out[-1].rhs, RegExpr)
                           and out[-1].rhs.name == out[-1].lhs.name):
                        out.pop()
                    combined = _canon_return_expr()
                    if combined is not None:
                        dbg("typesimp",
                            f"  [{hex(node.ea)}] fold-return (multi-canon): {combined.render()}")

                if combined is not None:
                    out.append(ReturnStmt(node.ea, combined))
                else:
                    out.append(node)
                i += 1
                continue

            # ── Recurse into structured bodies ────────────────────────────────
            if isinstance(node, IfNode):
                node = IfNode(node.ea, node.condition,
                              _fold(node.then_nodes), _fold(node.else_nodes))
            elif isinstance(node, WhileNode):
                node = WhileNode(node.ea, node.condition, _fold(node.body_nodes))
            elif isinstance(node, ForNode):
                node = ForNode(node.ea, node.init, node.condition,
                               node.update, _fold(node.body_nodes))
            elif isinstance(node, DoWhileNode):
                node = DoWhileNode(node.ea, node.condition, _fold(node.body_nodes))
            elif isinstance(node, SwitchNode):
                new_cases = [
                    (vals, _fold(body) if isinstance(body, list) else body)
                    for vals, body in node.cases
                ]
                new_default = _fold(node.default_body) \
                    if isinstance(node.default_body, list) else node.default_body
                node = SwitchNode(node.ea, node.subject, new_cases,
                                  node.default_label, new_default)
            out.append(node)
            i += 1
        return out

    return _fold(hir)


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

    # Build lookup: each individual reg → its regs tuple
    reg_to_pair: Dict[str, tuple] = {}
    for regs_key in pair_groups:
        for r in regs_key:
            reg_to_pair[r] = regs_key

    # Also harvest pair entries from per-node call_arg_ann annotations.
    # call_arg_ann entries may not have reached reg_map due to killed-param
    # blocking in _simplify, but are still present on the assignment nodes.
    for node in nodes:
        ann = getattr(node, "ann", None)
        if ann is None:
            continue
        for r, v in ann.call_arg_ann.items():
            if isinstance(v, VarInfo) and len(v.regs) > 1 and r in v.regs:
                key = tuple(v.regs)
                if key not in pair_groups:
                    pair_groups[key] = v
                for pr in key:
                    reg_to_pair.setdefault(pr, key)

    if not reg_to_pair:
        return nodes

    # Recurse into structured bodies first.
    nodes = recurse_bodies(nodes, lambda ns: _fold_call_arg_pairs(ns, reg_map))

    out: List[HIRNode] = []
    consumed: set = set()  # indices already folded

    for i, node in enumerate(nodes):
        if i in consumed:
            continue

        # Is this a plain Assign(Reg(r), expr) where r is part of a pair?
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr)
                and node.lhs.name in reg_to_pair):
            out.append(node)
            continue

        r0 = node.lhs.name
        regs_key = reg_to_pair[r0]
        vinfo = pair_groups[regs_key]
        all_regs = set(regs_key)
        remaining_regs = all_regs - {r0}

        # Collect byte assignments: reg → (index, expr)
        byte_assigns: Dict[str, tuple] = {r0: (i, node.rhs)}
        interleaved: List[int] = []
        conflict = False

        for k in range(i + 1, len(nodes)):
            if k in consumed:
                continue
            nd = nodes[k]
            # If this node reads or writes any pair reg, check carefully
            reads_pair = _reads_any_reg(nd, all_regs)
            writes_pair = _writes_any_reg(nd, all_regs)

            if isinstance(nd, Assign) and isinstance(nd.lhs, RegExpr) \
                    and nd.lhs.name in remaining_regs and not reads_pair:
                # Found another pair member assignment
                byte_assigns[nd.lhs.name] = (k, nd.rhs)
                remaining_regs -= {nd.lhs.name}
                if not remaining_regs:
                    break
            elif reads_pair or writes_pair:
                # Conflict: pair reg read or written by non-pair-member node
                conflict = True
                break
            else:
                # Safe interleaved node
                interleaved.append(k)

        if conflict or remaining_regs:
            out.append(node)
            continue

        # All pair members collected — build combined expression.
        # regs_key[0] is hi byte, regs_key[-1] is lo byte.
        n_bytes = len(regs_key)
        # Sort by position in regs_key (hi first)
        ordered_exprs = [byte_assigns[r][1] for r in regs_key]

        def _paren_if_binop(e: Expr) -> Expr:
            """Wrap in Paren for clarity when the expr itself is a BinOp."""
            return Paren(e) if isinstance(e, BinOp) else e

        # Build: (byte0 << (n-1)*8) | byte1 | ... | byte_{n-1}
        # Use Paren wrappers so that `<<` and `|` operands are always explicit.
        hi = _paren_if_binop(ordered_exprs[0])
        combined: Expr = Paren(BinOp(hi, "<<", Const((n_bytes - 1) * 8)))
        for b in range(1, n_bytes):
            shift = (n_bytes - 1 - b) * 8
            lo = _paren_if_binop(ordered_exprs[b])
            if shift > 0:
                term: Expr = Paren(BinOp(lo, "<<", Const(shift)))
            else:
                term = lo
            combined = BinOp(combined, "|", term)

        pair_group = RegGroupExpr(regs_key)
        out.append(Assign(node.ea, pair_group, combined))
        dbg("typesimp",
            f"  [{hex(node.ea)}] fold-call-arg-pair: {''.join(regs_key)} = {combined.render()}")

        # Mark all consumed indices
        for r, (idx, _) in byte_assigns.items():
            consumed.add(idx)

    return out


# ── Forward single-use propagation ────────────────────────────────────────────





def _subst_reg_in_scope(nodes: List[HIRNode], reg: str,
                         replacement: Expr) -> List[HIRNode]:
    """
    Replace Reg/Name(reg) → replacement in read positions of nodes,
    recursing into IfNode/WhileNode/ForNode/SwitchNode bodies.
    Stops at the first node that writes reg (i.e. redefines it), leaving
    that node and all subsequent nodes unchanged.
    Used by _consolidate_xram_local_loads to apply single-load substitutions
    immediately within the current scope, before a later reg_map mutation
    for the same register can corrupt _simplify_once's substitution.
    """
    result = []
    for i, node in enumerate(nodes):
        patched = _subst_reg_in_node(node, reg, replacement)
        if patched is not None:
            result.append(patched)
        elif isinstance(node, IfNode):
            result.append(IfNode(node.ea, node.condition,
                _subst_reg_in_scope(node.then_nodes, reg, replacement),
                _subst_reg_in_scope(node.else_nodes, reg, replacement)))
        elif isinstance(node, WhileNode):
            result.append(WhileNode(node.ea, node.condition,
                _subst_reg_in_scope(node.body_nodes, reg, replacement)))
        elif isinstance(node, ForNode):
            result.append(ForNode(node.ea, node.init, node.condition, node.update,
                _subst_reg_in_scope(node.body_nodes, reg, replacement)))
        elif isinstance(node, SwitchNode):
            new_cases = [(vals, _subst_reg_in_scope(body, reg, replacement)
                          if isinstance(body, list) else body)
                         for vals, body in node.cases]
            new_default = (_subst_reg_in_scope(node.default_body, reg, replacement)
                           if isinstance(node.default_body, list) else node.default_body)
            result.append(SwitchNode(node.ea, node.subject, new_cases,
                                     node.default_label, new_default))
        else:
            result.append(node)
        # If this node writes reg, subsequent nodes see the new value — stop here.
        if reg in node.written_regs:
            result.extend(nodes[i + 1:])
            return result
    return result


def _as_retval_stmt(node: HIRNode) -> Optional[tuple]:
    """Return (retval_name, call_expr) if node is a TypedAssign retval node; else None."""
    if isinstance(node, TypedAssign) and isinstance(node.rhs, Call):
        return (node.lhs.name, node.rhs)
    return None


def _count_name_uses_in_nodes(name: str, nodes: List[HIRNode]) -> int:
    """Count total occurrences of Name/Reg(name) in read positions across nodes."""
    total = 0
    for node in nodes:
        total += _count_reg_uses_in_node(name, node)
    return total


def _is_reg_free(expr: Expr) -> bool:
    """True if expr contains no Reg() leaf — safe to forward-substitute."""
    found_reg = [False]

    def _fn(e: Expr) -> Expr:
        if isinstance(e, RegExpr):
            found_reg[0] = True
        return e

    _walk_expr(expr, _fn)
    return not found_reg[0]


def _propagate_values(nodes: List[HIRNode],
                       reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Forward single-use propagation pass.

    Sub-pass A: For each Assign(Reg(r), Name/Const(n)) at index i with exactly
    one downstream use before r is written again, substitute n into that use
    and remove the assignment.

    Sub-pass B: For each retval Statement with exactly one downstream use of
    retval_name, inline the call expression into the target and remove the
    Statement.

    Recurses into IfNode/WhileNode/ForNode bodies first.
    """
    # Recurse into structured nodes first.
    work = recurse_bodies(nodes, lambda ns: _propagate_values(ns, reg_map))

    # Iterate until stable (multiple passes may be needed).
    changed = True
    while changed:
        changed = False

        # Sub-pass A: register copy propagation.
        live = list(work)
        i = 0
        while i < len(live):
            node = live[i]
            if not (isinstance(node, Assign)
                    and isinstance(node.lhs, RegExpr)
                    and (isinstance(node.rhs, (NameExpr, Const))
                         or _is_reg_free(node.rhs))):
                i += 1
                continue
            r = node.lhs.name
            replacement = node.rhs

            # Scan forward counting uses, stopping when r is written.
            total_uses = 0
            use_idx = None
            for j in range(i + 1, len(live)):
                written = live[j].written_regs
                uses_here = _count_reg_uses_in_node(r, live[j])
                total_uses += uses_here
                if uses_here > 0 and use_idx is None:
                    use_idx = j
                if r in written:
                    break

            if total_uses == 1 and use_idx is not None:
                new_node = _subst_reg_in_node(live[use_idx], r, replacement)
                if new_node is not None:
                    live[use_idx] = new_node
                    live[i] = None
                    dbg("typesimp", f"  [{hex(node.ea)}] prop-values: folded {r} into node {use_idx}")
                    changed = True

            i += 1

        live = [n for n in live if n is not None]

        # Sub-pass B: retval statement inlining.
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
                # Find the target node.
                for j, tgt in enumerate(remaining):
                    if _count_reg_uses_in_node(retval_name, tgt) == 1:
                        abs_j = i + 1 + j
                        if (isinstance(tgt, Assign)
                                and isinstance(tgt.rhs, (NameExpr, RegExpr))
                                and tgt.rhs.name == retval_name):
                            live[abs_j] = Assign(tgt.ea, tgt.lhs, call_expr)
                        else:
                            new_node = _subst_reg_in_node(tgt, retval_name, call_expr)
                            if new_node is not None:
                                live[abs_j] = new_node
                        src_ea = hex(live[i].ea)
                        live[i] = None
                        dbg("typesimp",
                            f"  [{src_ea}] prop-values: inlined {retval_name} into node {abs_j}")
                        changed = True
                        break

            i += 1

        live = [n for n in live if n is not None]
        work = live

    return work


# ── 16-bit SUBB comparison collapsing ────────────────────────────────────────

def _match_subb16(nodes: List[HIRNode], k: int):
    """Try to match CLR C + SUBB lo + MOV A,Rhi + SUBB hi starting at index k.
    Returns (Rlo_sub, Rhi_min, Rhi_sub) or None."""
    if k + 3 >= len(nodes):
        return None
    n0, n1, n2, n3 = nodes[k], nodes[k+1], nodes[k+2], nodes[k+3]
    # CLR C: Assign(Reg("C"), Const(0))
    if not (isinstance(n0, Assign) and isinstance(n0.lhs, RegExpr) and n0.lhs.name == "C"
            and isinstance(n0.rhs, Const) and n0.rhs.value == 0):
        return None
    # SUBB lo: CompoundAssign(Reg("A"), "-=", BinOp(Reg(Rlo_sub), "+", Reg("C")))
    if not (isinstance(n1, CompoundAssign) and isinstance(n1.lhs, RegExpr) and n1.lhs.name == "A"
            and n1.op == "-=" and isinstance(n1.rhs, BinOp) and n1.rhs.op == "+"
            and isinstance(n1.rhs.lhs, RegExpr) and isinstance(n1.rhs.rhs, RegExpr)
            and n1.rhs.rhs.name == "C"):
        return None
    Rlo_sub = n1.rhs.lhs.name
    # MOV A, Rhi_min: Assign(Reg("A"), Reg(Rhi_min))
    if not (isinstance(n2, Assign) and isinstance(n2.lhs, RegExpr) and n2.lhs.name == "A"
            and isinstance(n2.rhs, RegExpr)):
        return None
    Rhi_min = n2.rhs.name
    # SUBB hi: CompoundAssign(Reg("A"), "-=", BinOp(Reg(Rhi_sub), "+", Reg("C")))
    if not (isinstance(n3, CompoundAssign) and isinstance(n3.lhs, RegExpr) and n3.lhs.name == "A"
            and n3.op == "-=" and isinstance(n3.rhs, BinOp) and n3.rhs.op == "+"
            and isinstance(n3.rhs.lhs, RegExpr) and isinstance(n3.rhs.rhs, RegExpr)
            and n3.rhs.rhs.name == "C"):
        return None
    Rhi_sub = n3.rhs.lhs.name
    return (Rlo_sub, Rhi_min, Rhi_sub)


def _find_reggroup_name(nodes: List[HIRNode], before_idx: int, target_reg: str) -> Optional[str]:
    """Search nodes[:before_idx] backward for Assign(RegGroup, Name) containing target_reg."""
    for node in reversed(nodes[:before_idx]):
        if (isinstance(node, Assign) and isinstance(node.lhs, RegGroupExpr)
                and target_reg in node.lhs.regs and isinstance(node.rhs, NameExpr)):
            return node.rhs.name
    return None


def _simplify_carry_comparison(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    For each WhileNode with condition Reg("C"), scan body_nodes for the 4-node
    CLR C + SUBB lo + MOV A + SUBB hi sequence. If found and both operand names
    can be resolved via preceding RegGroup assignments, replace the condition with
    BinOp(Name(minuend), "<", Name(subtrahend)) and remove the 4 nodes.
    Recurses into nested structured nodes.
    """
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, WhileNode):
            # Recurse into body first (WhileNode needs special condition check)
            new_body = _simplify_carry_comparison(node.body_nodes)
            if isinstance(node.condition, RegExpr) and node.condition.name == "C":
                transformed = _try_collapse_subb16(node.condition, new_body)
                if transformed is not None:
                    new_cond, new_body = transformed
                    result.append(WhileNode(node.ea, new_cond, new_body))
                    continue
            result.append(WhileNode(node.ea, node.condition, new_body))
        else:
            result.append(node.map_bodies(_simplify_carry_comparison))
    return result


def _try_collapse_subb16(cond, body: List[HIRNode]):
    """
    Scan body for the 4-node SUBB16 sequence and resolve names from preceding
    RegGroup assigns. Returns (new_cond, new_body) or None.
    """
    for k in range(len(body)):
        match = _match_subb16(body, k)
        if match is None:
            continue
        Rlo_sub, Rhi_min, Rhi_sub = match

        # Resolve subtrahend (register pair containing Rhi_sub)
        subtrahend = _find_reggroup_name(body, k, Rhi_sub)
        if subtrahend is None:
            continue

        # Resolve minuend (register pair containing Rhi_min)
        minuend = _find_reggroup_name(body, k, Rhi_min)
        if minuend is None:
            continue

        # Remove the 4-node SUBB sequence from body
        new_body = body[:k] + body[k+4:]
        new_cond = BinOp(NameExpr(minuend), "<", NameExpr(subtrahend))
        dbg("typesimp",
            f"  [{hex(body[k].ea)}] subb16-collapse: {minuend} < {subtrahend}"
            f" (via {Rhi_min},{Rlo_sub} - {Rhi_sub},{Rlo_sub})")
        return (new_cond, new_body)
    return None


# ── ORL + JZ zero-check simplification ───────────────────────────────────────

def _extract_zero_cond(node: HIRNode, parent: str) -> Optional[BinOp]:
    """Return BinOp(Name(parent), op, 0) if node tests something against 0; else None."""
    cond = None
    if isinstance(node, IfNode):
        cond = node.condition
    elif isinstance(node, IfGoto):
        cond = node.cond
    if (isinstance(cond, BinOp)
            and cond.op in ("==", "!=")
            and isinstance(cond.rhs, Const) and cond.rhs.value == 0):
        return BinOp(NameExpr(parent), cond.op, Const(0))
    return None


def _replace_cond(node: HIRNode, new_cond) -> HIRNode:
    """Return a copy of node (IfNode or IfGoto) with new_cond substituted."""
    if isinstance(node, IfNode):
        return IfNode(node.ea, new_cond, node.then_nodes, node.else_nodes)
    if isinstance(node, IfGoto):
        return IfGoto(node.ea, new_cond, node.label)
    return node


def _simplify_orl_zero_check(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Recognise the 8051 ORL + JZ idiom for 16-bit zero tests.

    Pattern (consecutive):
      CompoundAssign(Reg("A"), "|=", Name(v))   where v has .hi/.lo/.bN suffix
      IfNode or IfGoto with BinOp(anything, "==" | "!=", Const(0)) condition

    Removes the ORL node and replaces the condition with
    BinOp(Name(parent_of_v), op, Const(0)).
    Recurses into WhileNode / IfNode / ForNode bodies first.
    """
    nodes = recurse_bodies(nodes, _simplify_orl_zero_check)
    result: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if (isinstance(node, CompoundAssign)
                and isinstance(node.lhs, RegExpr) and node.lhs.name == "A"
                and node.op == "|="
                and isinstance(node.rhs, NameExpr)):
            m = _RE_BYTE_FIELD.match(node.rhs.name)
            if m and i + 1 < len(nodes):
                parent = m.group(1)
                new_cond = _extract_zero_cond(nodes[i + 1], parent)
                if new_cond is not None:
                    result.append(_replace_cond(nodes[i + 1], new_cond))
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] orl-zero-check: A|={node.rhs.name} → {parent} {new_cond.op} 0")
                    i += 2
                    continue
        result.append(node)
        i += 1
    return result


def _dptr_live_after(nodes: List[HIRNode]) -> bool:
    """
    Return True if the current DPTR value is read by any downstream node,
    stopping early when an assignment kills it (Assign(DPTR, ...) or DPTR++).

    Flow-sensitive: unlike _collect_hir_name_refs, a DPTR write (e.g.
    DPTR = _dest) stops the scan — the old value is dead past that point.
    Recurses into IfNode/WhileNode/ForNode bodies conservatively.
    """
    for node in nodes:
        if (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr) and node.lhs.name == "DPTR"):
            return False  # DPTR overwritten; old value dead
        if _is_dptr_inc_node(node):
            return False  # Another DPTR++ also overwrites the old value
        if isinstance(node, IfNode):
            if (_dptr_live_after(node.then_nodes)
                    or _dptr_live_after(node.else_nodes)):
                return True
        elif isinstance(node, (WhileNode, ForNode)):
            if _dptr_live_after(node.body_nodes):
                return True
        else:
            if "DPTR" in _collect_hir_name_refs([node]):
                return True
    return False


def _prune_orphaned_dptr_inc(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Remove DPTR++ nodes that have no downstream DPTR reference.

    Must run after _propagate_values so that XRAM[DPTR] expressions have
    been resolved to XRAM[name] before we check for surviving references.
    Recurses into IfNode/WhileNode/ForNode bodies first so nested orphans
    are eliminated before outer nodes are evaluated.

    Uses _dptr_live_after (flow-sensitive) so that an intervening
    DPTR = sym assignment correctly kills a previous DPTR++ value.
    """
    nodes = recurse_bodies(nodes, _prune_orphaned_dptr_inc)
    result: List[HIRNode] = []
    for i, node in enumerate(nodes):
        if _is_dptr_inc_node(node) and not _dptr_live_after(nodes[i + 1:]):
            dbg("typesimp", f"  [{hex(node.ea)}] prune-orphaned-dptr++")
            continue
        result.append(node)
    return result
