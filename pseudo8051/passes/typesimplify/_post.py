"""
passes/typesimplify/_post.py — Post-simplify passes and shared traversal helper.
"""

import re
from typing import Callable, Dict, List, Optional

from pseudo8051.ir.hir    import (HIRNode, Statement, Assign, CompoundAssign,
                                   ExprStmt, ReturnStmt, IfGoto, IfNode, WhileNode, ForNode,
                                   SwitchNode)
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns._utils  import (
    VarInfo, _type_bytes, _byte_names, _walk_expr,
)
from pseudo8051.ir.expr import (Expr, UnaryOp, BinOp, Const, Call,
                                 Reg as RegExpr, RegGroup as RegGroupExpr, Name as NameExpr,
                                 XRAMRef, IRAMRef)

# ── Shared traversal helper ───────────────────────────────────────────────────

def recurse_bodies(nodes: List[HIRNode], fn: Callable) -> List[HIRNode]:
    """Recurse fn into IfNode/WhileNode/ForNode/SwitchNode bodies; pass other nodes through."""
    result = []
    for node in nodes:
        if isinstance(node, IfNode):
            result.append(IfNode(node.ea, node.condition,
                fn(node.then_nodes), fn(node.else_nodes)))
        elif isinstance(node, WhileNode):
            result.append(WhileNode(node.ea, node.condition, fn(node.body_nodes)))
        elif isinstance(node, ForNode):
            result.append(ForNode(node.ea, node.init, node.condition, node.update,
                fn(node.body_nodes)))
        elif isinstance(node, SwitchNode):
            new_cases = [
                (vals, fn(body) if isinstance(body, list) else body)
                for vals, body in node.cases
            ]
            new_default = fn(node.default_body) if isinstance(node.default_body, list) \
                          else node.default_body
            result.append(SwitchNode(node.ea, node.subject, new_cases,
                                     node.default_label, new_default))
        else:
            result.append(node)
    return result


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
                                    f"  xram-pair-consolidate: DPTR = {parent_nm}"
                                    f" (via {pair_key} + DPH)")
                            else:
                                out.append(Assign(node.ea,
                                                  RegGroupExpr(tuple(regs)),
                                                  NameExpr(parent_nm)))
                                dbg("typesimp",
                                    f"  xram-pair-consolidate: {pair_key} = {parent_nm}")
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
                        dbg("typesimp", f"  xram-single-load: {reg0} = {bname0}")
                        break
        # Recurse into structured nodes
        out.extend(recurse_bodies([node],
            lambda ns: _consolidate_xram_local_loads(ns, reg_map)))
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
        dbg("typesimp", f"  dpl-dph-collapse: DPTR = {pair_key}")

    return out


# ── Post-simplify call-arg fold and dead-setup pruning ────────────────────────

_RE_REG_TOKEN = re.compile(r'\b(R[0-7]+|DPTR|DPH|DPL|A)\b')


def _collect_hir_name_refs(nodes: List[HIRNode]) -> set:
    """Collect all Reg/Name .name strings from expression positions in nodes."""
    refs: set = set()

    def _add(expr: Expr) -> None:
        def _fn(e: Expr) -> Expr:
            if isinstance(e, (RegExpr, NameExpr)):
                refs.add(e.name)
            return e
        _walk_expr(expr, _fn)

    def _visit(node: HIRNode) -> None:
        if isinstance(node, Assign):
            _add(node.rhs)
            if not isinstance(node.lhs, (RegExpr, RegGroupExpr)):
                _add(node.lhs)
        elif isinstance(node, CompoundAssign):
            _add(node.rhs)
        elif isinstance(node, ExprStmt):
            _add(node.expr)
        elif isinstance(node, ReturnStmt) and node.value is not None:
            _add(node.value)
        elif isinstance(node, IfGoto):
            _add(node.cond)
        elif isinstance(node, Statement):
            for m in _RE_REG_TOKEN.finditer(node.text):
                refs.add(m.group(1))
        elif isinstance(node, IfNode):
            for sub in list(node.then_nodes) + list(node.else_nodes):
                _visit(sub)
        elif isinstance(node, (WhileNode, ForNode)):
            for sub in node.body_nodes:
                _visit(sub)

    for node in nodes:
        _visit(node)
    return refs


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
    if isinstance(node, Statement):
        new_text = re.sub(r'\b' + re.escape(reg) + r'\b', repl_str, node.text)
        return Statement(node.ea, new_text) if new_text != node.text else node
    return node


def _fold_and_prune_setups(nodes: List[HIRNode],
                            reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Post-simplify cleanup of register-setup lines before calls.

    1. Fold Assign(Reg, Const) into the next call node's args.
    2. Remove Assign(Reg/RegGroup, Name/Const) setup nodes whose LHS registers
       are not referenced in any subsequent node.
    3. Remove DPTR++ nodes whose DPTR value is not referenced afterwards.
    Recurses into IfNode / WhileNode / ForNode bodies.
    """
    # Recurse first so inner blocks are cleaned before the outer scan.
    recursed = recurse_bodies(nodes, lambda ns: _fold_and_prune_setups(ns, reg_map))

    work: List[HIRNode] = list(recursed)

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
                dbg("typesimp", f"  fold-const: {reg}={val.render()} into call")
            break
    work = [n for n in work if n is not None]

    # Phase 2: remove dead setup-assign and DPTR++ nodes.
    out: List[HIRNode] = []
    for i, node in enumerate(work):
        if _is_call_setup_assign(node):
            lhs_regs = _lhs_reg_names(node)
            if lhs_regs.isdisjoint(_collect_hir_name_refs(work[i + 1:])):
                dbg("typesimp",
                    f"  prune-setup: {node.lhs.render()} = {node.rhs.render()}")
                continue
        elif _is_dptr_inc_node(node):
            if "DPTR" not in _collect_hir_name_refs(work[i + 1:]):
                dbg("typesimp", "  prune-dptr++")
                continue
        out.append(node)
    return out


# ── Return-chain folding ──────────────────────────────────────────────────────

def _fold_return_chains(hir: List[HIRNode], ret_regs: tuple) -> List[HIRNode]:
    """Fold Assign(ret_reg, expr); ReturnStmt(ret_reg | None) → ReturnStmt(expr)."""
    ret_reg_set = set(ret_regs)

    def _fold(nodes: List[HIRNode]) -> List[HIRNode]:
        out: List[HIRNode] = []
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if (isinstance(node, Assign)
                    and isinstance(node.lhs, RegExpr)
                    and node.lhs.name in ret_reg_set
                    and i + 1 < len(nodes)
                    and isinstance(nodes[i + 1], ReturnStmt)
                    and (nodes[i + 1].value is None
                         or nodes[i + 1].value == RegExpr(node.lhs.name))):
                out.append(ReturnStmt(node.ea, node.rhs))
                i += 2
                continue
            # Recurse into structured bodies
            if isinstance(node, IfNode):
                node = IfNode(node.ea, node.condition,
                              _fold(node.then_nodes), _fold(node.else_nodes))
            elif isinstance(node, WhileNode):
                node = WhileNode(node.ea, node.condition, _fold(node.body_nodes))
            elif isinstance(node, ForNode):
                node = ForNode(node.ea, node.init, node.condition,
                               node.update, _fold(node.body_nodes))
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


# ── Forward single-use propagation ────────────────────────────────────────────

_RE_RETVAL_STMT = re.compile(
    r'^(?:\w[\w\s*]*?\s+)?(\w*retval\d+)\s*=\s*(.+?);$'
)


def _get_written_regs(node: HIRNode) -> frozenset:
    """Return the set of register/name strings written (defined) by node."""
    if isinstance(node, (Assign, CompoundAssign)):
        lhs = node.lhs
        if isinstance(lhs, RegExpr):
            return frozenset({lhs.name})
        if isinstance(lhs, RegGroupExpr):
            names = set(lhs.regs)
            names.add("".join(lhs.regs))
            return frozenset(names)
    if isinstance(node, Statement):
        # Written reg is the LHS token before " = "
        eq = node.text.find(" = ")
        if eq > 0:
            lhs_tok = node.text[:eq].split()[-1]
            return frozenset({lhs_tok})
    return frozenset()


def _count_reg_uses_in_node(r: str, node: HIRNode) -> int:
    """Count read-position occurrences of Reg/Name(r) in node."""
    count = [0]

    def _fn(e: Expr) -> Expr:
        if isinstance(e, (RegExpr, NameExpr)) and e.name == r:
            count[0] += 1
        return e

    if isinstance(node, Assign):
        _walk_expr(node.rhs, _fn)
        # Also count in compound LHS (e.g. XRAMRef inner)
        if not isinstance(node.lhs, (RegExpr, RegGroupExpr)):
            _walk_expr(node.lhs, _fn)
    elif isinstance(node, CompoundAssign):
        _walk_expr(node.rhs, _fn)
    elif isinstance(node, ExprStmt):
        _walk_expr(node.expr, _fn)
    elif isinstance(node, ReturnStmt) and node.value is not None:
        _walk_expr(node.value, _fn)
    elif isinstance(node, IfGoto):
        _walk_expr(node.cond, _fn)
    elif isinstance(node, Statement):
        return len(re.findall(r'\b' + re.escape(r) + r'\b', node.text))
    return count[0]


def _subst_reg_in_node(node: HIRNode, r: str,
                        replacement: Expr) -> Optional[HIRNode]:
    """
    Replace Reg/Name(r) → replacement in read positions of node.
    Returns updated node, or None if r does not appear.
    """
    def _fn(e: Expr) -> Expr:
        if isinstance(e, (RegExpr, NameExpr)) and e.name == r:
            return replacement
        return e

    if isinstance(node, Assign):
        new_rhs = _walk_expr(node.rhs, _fn)
        new_lhs = node.lhs
        if not isinstance(node.lhs, (RegExpr, RegGroupExpr)):
            new_lhs = _walk_expr(node.lhs, _fn)
        if new_rhs is node.rhs and new_lhs is node.lhs:
            return None
        return Assign(node.ea, new_lhs, new_rhs)

    if isinstance(node, CompoundAssign):
        new_rhs = _walk_expr(node.rhs, _fn)
        if new_rhs is node.rhs:
            return None
        return CompoundAssign(node.ea, node.lhs, node.op, new_rhs)

    if isinstance(node, ExprStmt):
        new_expr = _walk_expr(node.expr, _fn)
        if new_expr is node.expr:
            return None
        return ExprStmt(node.ea, new_expr)

    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _walk_expr(node.value, _fn)
        if new_val is node.value:
            return None
        return ReturnStmt(node.ea, new_val)

    if isinstance(node, IfGoto):
        new_cond = _walk_expr(node.cond, _fn)
        if new_cond is node.cond:
            return None
        return IfGoto(node.ea, new_cond, node.target)

    if isinstance(node, Statement):
        pat = re.compile(r'\b' + re.escape(r) + r'\b')
        if not pat.search(node.text):
            return None
        repl_str = replacement.render() if isinstance(replacement, Expr) else str(replacement)
        eq = node.text.find(" = ")
        if eq > 0:
            rhs = node.text[eq + 3:]
            if pat.search(rhs):
                return Statement(node.ea, node.text[:eq + 3] + pat.sub(repl_str, rhs))
            return None
        return Statement(node.ea, pat.sub(repl_str, node.text))

    return None


def _as_retval_stmt(node: HIRNode) -> Optional[tuple]:
    """Return (retval_name, call_expr_str) if node is a retval Statement; else None."""
    if not isinstance(node, Statement):
        return None
    m = _RE_RETVAL_STMT.match(node.text)
    if m:
        return (m.group(1), m.group(2))
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
                written = _get_written_regs(live[j])
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
                    dbg("typesimp", f"  prop-values: folded {r} into node {use_idx}")
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
            retval_name, call_expr_str = rv
            remaining = live[i + 1:]
            total_uses = _count_name_uses_in_nodes(retval_name, remaining)

            if total_uses == 1:
                # Find the target node.
                for j, tgt in enumerate(remaining):
                    if _count_reg_uses_in_node(retval_name, tgt) == 1:
                        abs_j = i + 1 + j
                        # Inline: build new node.
                        if (isinstance(tgt, Assign)
                                and isinstance(tgt.rhs, (NameExpr, RegExpr))
                                and tgt.rhs.name == retval_name):
                            lhs_str = tgt.lhs.render() if isinstance(tgt.lhs, Expr) else str(tgt.lhs)
                            live[abs_j] = Statement(tgt.ea,
                                                    f"{lhs_str} = {call_expr_str};")
                        else:
                            # Text substitution for Statement targets.
                            new_node = _subst_reg_in_node(
                                tgt, retval_name, NameExpr(call_expr_str))
                            if new_node is not None:
                                # Fix up: NameExpr renders with no parens; re-render
                                # by doing a text sub on the resulting text.
                                if isinstance(new_node, Statement):
                                    live[abs_j] = new_node
                                else:
                                    live[abs_j] = new_node
                        live[i] = None
                        dbg("typesimp",
                            f"  prop-values: inlined {retval_name} into node {abs_j}")
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
            # Recurse into body first
            new_body = _simplify_carry_comparison(node.body_nodes)
            # Check if condition is Reg("C")
            if isinstance(node.condition, RegExpr) and node.condition.name == "C":
                transformed = _try_collapse_subb16(node.condition, new_body)
                if transformed is not None:
                    new_cond, new_body = transformed
                    result.append(WhileNode(node.ea, new_cond, new_body))
                    continue
            result.append(WhileNode(node.ea, node.condition, new_body))
        elif isinstance(node, IfNode):
            result.append(IfNode(node.ea, node.condition,
                _simplify_carry_comparison(node.then_nodes),
                _simplify_carry_comparison(node.else_nodes)))
        elif isinstance(node, ForNode):
            result.append(ForNode(node.ea, node.init, node.condition, node.update,
                _simplify_carry_comparison(node.body_nodes)))
        else:
            result.append(node)
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
            f"  subb16-collapse: {minuend} < {subtrahend}"
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
        return IfGoto(node.ea, new_cond, node.target)
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
                        f"  orl-zero-check: A|={node.rhs.name} → {parent} {new_cond.op} 0")
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
            dbg("typesimp", "  prune-orphaned-dptr++")
            continue
        result.append(node)
    return result
