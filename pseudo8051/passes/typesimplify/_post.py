"""
passes/typesimplify/_post.py — Post-simplify passes and shared traversal helper.
"""

import re
from typing import Callable, Dict, List, Optional

from pseudo8051.ir.hir    import (HIRNode, Statement, Assign, CompoundAssign,
                                   ExprStmt, ReturnStmt, IfGoto, IfNode, WhileNode, ForNode)
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns._utils  import (
    VarInfo, _type_bytes, _byte_names, _walk_expr,
)
from pseudo8051.ir.expr import (Expr, UnaryOp, BinOp, Const, Call,
                                 Reg as RegExpr, RegGroup as RegGroupExpr, Name as NameExpr)

# ── Shared traversal helper ───────────────────────────────────────────────────

def recurse_bodies(nodes: List[HIRNode], fn: Callable) -> List[HIRNode]:
    """Recurse fn into IfNode/WhileNode/ForNode bodies; pass other nodes through."""
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
    _RE_BYTE_FIELD = re.compile(r'^(.+)\.(?:hi|lo|b\d+)$')

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
