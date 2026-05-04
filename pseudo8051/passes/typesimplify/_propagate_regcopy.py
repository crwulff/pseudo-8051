"""
passes/typesimplify/_propagate_regcopy.py — Sub-passes A0 and A.

A0: _fold_compound_assigns   — fold Assign(r,e)+CompoundAssign(r,op=,K) → Assign
A:  _propagate_register_copies — substitute single-use register/name assignments
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,
                                NodeAnnotation, GotoStatement, BreakStmt,
                                ContinueStmt, Label)
from pseudo8051.ir.expr import (BinOp, Const, Regs as RegExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import (
    VarInfo, _count_reg_uses_in_node, _subst_reg_in_node,
    _canonicalize_expr, _is_reg_free,
)
from pseudo8051.passes.patterns._node_utils import _fold_exprs_in_node
from pseudo8051.constants import dbg
from pseudo8051.passes.typesimplify._propagate_utils import (
    _COMPOUND_OPS,
    _has_xram_const_addr,
    _name_possibly_written_in,
    _expr_name_refs,
    _xram_pre_incr_delta,
    _expr_stmt_incr_delta,
    _collect_mid_writes,
    _collect_hir_name_refs,
    _dbg_node,
)


# ── Sub-pass A0: compound-assign expansion ────────────────────────────────────

def _fold_compound_assigns(live: List[HIRNode]) -> Tuple[List[HIRNode], bool]:
    """
    A0: fold  Assign(Reg(r), expr) + CompoundAssign(Reg(r), op=, rhs)
    into a single Assign(Reg(r), expr op rhs).
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
            if isinstance(live[j], (GotoStatement, BreakStmt, ContinueStmt)):
                break
            if isinstance(live[j], Label):
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
                if r in live[j].possibly_killed():
                    kill_idx = j
                    break
                if (r == 'C'
                        and isinstance(live[j], CompoundAssign)
                        and isinstance(live[j].lhs, RegExpr)
                        and live[j].lhs.is_single
                        and live[j].lhs.name == 'A'
                        and live[j].op in ('+=', '-=')):
                    kill_idx = j
                    break
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
                lhs_j = getattr(live[j], 'lhs', None)
                if isinstance(lhs_j, NameExpr) and lhs_j.name == r:
                    kill_idx = j
                    break
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
                _vinfo = reg_map.get(r)
                if _vinfo is None:
                    for _vi in reg_map.values():
                        if isinstance(_vi, VarInfo) and _vi.name == r:
                            _vinfo = _vi
                            break
                if isinstance(_vinfo, VarInfo):
                    if _vinfo.regs:
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
                        from pseudo8051.ir.expr import XRAMRef as _XRAMRef, Const as _Const
                        _sym = _vinfo.xram_sym
                        _blocked = any(
                            isinstance(_rv, _XRAMRef)
                            and isinstance(_rv.inner, _Const)
                            and _rv.inner.alias == _sym
                            for _rv in _use_ann.reg_exprs.values()
                        )
                        if _blocked:
                            dbg("propagate",
                                f"  [{hex(node.ea)}] prop-blocked: {r}={replacement.render()!r} "
                                f"— use-site has register holding {_sym}")
                            i += 1
                            continue

            repl_refs = _expr_name_refs(replacement)
            if repl_refs and use_idx > i + 1:
                mid_writes = _collect_mid_writes(live[i + 1:use_idx], reg_map)
                if repl_refs & mid_writes:
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] prop-values: blocked {r} — "
                        f"intermediate writes {repl_refs & mid_writes}")
                    i += 1
                    continue

            old_use_node = live[use_idx]
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
                new_node = _fold_exprs_in_node(new_node)
                define_node = live[i]
                new_node.source_nodes = [define_node] + list(old_use_node.source_nodes or [old_use_node])
                live[use_idx] = new_node
                live[i] = None
                dbg("typesimp", f"  [{hex(node.ea)}] prop-values: folded {r} into node {use_idx}")
                changed = True

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
                        if r in (_nk.written_regs if _nk is not None else frozenset()):
                            break

                if pre_incr_delta is not None:
                    new_r_val = Const((replacement.value + pre_incr_delta) & 0xFFFF)
                    has_more = any(
                        _count_reg_uses_in_node(r, live[k]) > 0
                        for k in range(use_idx + 1, len(live))
                        if not isinstance(live[k], (GotoStatement, BreakStmt, ContinueStmt))
                    )
                    if has_more:
                        defn_node = new_node.source_nodes[0]
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
                    tentative[j] = _fold_exprs_in_node(new_node)
            if all_ok and tentative:
                src_node = live[i]
                for j, new_node in tentative.items():
                    new_node.source_nodes = [src_node] + list(live[j].source_nodes or [live[j]])
                    live[j] = new_node
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
