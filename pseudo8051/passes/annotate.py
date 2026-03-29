"""
passes/annotate.py — AnnotationPass: forward + backward HIR annotation.

Runs on the flat-block HIR (after ChunkInliner, before RMWCollapser).
Annotates each HIR node with:
  - reg_names:    name/type info for registers live at that point
  - reg_consts:   known constant register values at that point
  - call_arg_ann: callee-param names back-propagated to pre-call assignments
  - callee_args:  callee reg_map stored on the call node itself
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir     import (HIRNode, NodeAnnotation, Assign, CompoundAssign,
                                    ExprStmt)
from pseudo8051.ir.expr    import Reg as RegExpr, RegGroup as RegGroupExpr, XRAMRef, Name as NameExpr
from pseudo8051.passes     import OptimizationPass
from pseudo8051.constants  import PARAM_REG_ORDER, dbg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_written_regs(node: HIRNode) -> frozenset:
    """Return the set of register names written as the primary LHS of this node."""
    if isinstance(node, (Assign, CompoundAssign)):
        lhs = node.lhs
        if isinstance(lhs, RegExpr):
            return frozenset({lhs.name})
        if isinstance(lhs, RegGroupExpr):
            return frozenset(lhs.regs)
    return frozenset()


def _is_call_node(node: HIRNode):
    """Return the Call expr if this node is a call statement/assignment, else None."""
    from pseudo8051.ir.expr import Call
    if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
        return node.expr
    if isinstance(node, Assign) and isinstance(node.rhs, Call):
        return node.rhs
    return None


def _rpo(func) -> list:
    """Return blocks in reverse post-order (entry first)."""
    visited, post = set(), []

    def _dfs(b):
        if b.start_ea in visited:
            return
        visited.add(b.start_ea)
        for s in b.successors:
            _dfs(s)
        post.append(b)

    _dfs(func.entry_block)
    return list(reversed(post))


def _meet_name_states(preds: List[Dict]) -> Dict:
    """Intersect predecessor name states: keep only keys present in all with same name."""
    if not preds:
        return {}
    keys = set(preds[0])
    for s in preds[1:]:
        keys &= s.keys()
    return {k: preds[0][k] for k in keys
            if all(s[k].name == preds[0][k].name for s in preds)}


def _resolve_name_addr(sym: str) -> Optional[int]:
    """Resolve a named symbol to its numeric address, or None if unknown."""
    try:
        import idc
        addr = idc.get_name_ea_simple(sym)
        if addr != idc.BADADDR:
            return addr
    except Exception:
        pass
    return None


def _propagate_const(node: HIRNode, const_state: Dict[str, int]) -> None:
    """Update const_state with any new constant value produced by this node.

    Called AFTER the kill step so stale values have already been removed.
    Mirrors propagate_insn() semantics but operates on HIR nodes.
    """
    from pseudo8051.ir.expr import Const as ConstExpr, Reg as RegExpr2, UnaryOp as UnaryOpExpr

    if isinstance(node, Assign):
        lhs, rhs = node.lhs, node.rhs
        if not isinstance(lhs, RegExpr):
            return
        reg = lhs.name

        # Resolve RHS to a numeric value if possible
        val: Optional[int] = None
        if isinstance(rhs, ConstExpr):
            val = rhs.value
        elif isinstance(rhs, NameExpr) and reg == "DPTR":
            # MOV DPTR, #sym — the symbol IS the address
            val = _resolve_name_addr(rhs.name)

        if val is not None:
            if reg == "DPTR":
                v16 = val & 0xFFFF
                const_state["DPTR"] = v16
                const_state["DPH"]  = v16 >> 8
                const_state["DPL"]  = v16 & 0xFF
            elif reg == "DPH":
                v8 = val & 0xFF
                const_state["DPH"] = v8
                lo = const_state.get("DPL")
                if lo is not None:
                    const_state["DPTR"] = (v8 << 8) | lo
            elif reg == "DPL":
                v8 = val & 0xFF
                const_state["DPL"] = v8
                hi = const_state.get("DPH")
                if hi is not None:
                    const_state["DPTR"] = (hi << 8) | v8
            else:
                const_state[reg] = val & 0xFF

        elif isinstance(rhs, RegExpr2):
            # Register copy: propagate known source value
            src = const_state.get(rhs.name)
            if src is not None:
                const_state[reg] = src

    elif isinstance(node, ExprStmt):
        expr = node.expr
        if isinstance(expr, UnaryOpExpr) and expr.op in ("++", "--"):
            if isinstance(expr.operand, RegExpr):
                reg = expr.operand.name
                v = const_state.get(reg)
                if v is None:
                    return
                if reg == "DPTR":
                    delta = 1 if expr.op == "++" else -1
                    nv = (v + delta) & 0xFFFF
                    const_state["DPTR"] = nv
                    const_state["DPH"]  = nv >> 8
                    const_state["DPL"]  = nv & 0xFF
                else:
                    delta = 1 if expr.op == "++" else -1
                    const_state[reg] = (v + delta) & 0xFF


def _backward_annotate_call(nodes: List[HIRNode], call_idx: int,
                             callee_rm: Dict) -> None:
    """Back-propagate callee param names to the assignments preceding a call."""
    remaining = set(callee_rm.keys())
    j = call_idx - 1
    while j >= 0 and remaining:
        written = _get_written_regs(nodes[j])
        for r in written & remaining:
            if nodes[j].ann is not None:
                nodes[j].ann.call_arg_ann.setdefault(r, callee_rm[r])
            remaining.discard(r)
        j -= 1


# ── Pass ───────────────────────────────────────────────────────────────────────

class AnnotationPass(OptimizationPass):
    """Annotate each flat-HIR node with name/type and constant information."""

    def run(self, func) -> None:
        from pseudo8051.prototypes import get_proto
        from pseudo8051.passes.typesimplify._regmap import (
            _build_reg_map, _augment_with_local_vars,
        )
        from pseudo8051.passes.patterns._utils import VarInfo

        proto   = get_proto(func.name)
        live_in = getattr(func.entry_block, "live_in", frozenset())

        # Build initial name state at function entry
        if proto is not None:
            entry_names = _build_reg_map(proto, live_in)
            dbg("annotate", f"{func.name}: proto found, entry_names={list(entry_names)}")
        else:
            entry_names: Dict = {}
            for idx, reg in enumerate(r for r in PARAM_REG_ORDER if r in live_in):
                info = VarInfo(f"arg{idx + 1}", "uint8_t", (reg,), is_param=True)
                entry_names[reg] = info
            dbg("annotate", f"{func.name}: no proto, inferred entry_names={list(entry_names)}")

        # Add XRAM local var info (indexed by xram_sym)
        xram_locals: Dict = {}
        augmented = _augment_with_local_vars(func.ea, {})
        for k, v in augmented.items():
            if isinstance(v, VarInfo) and v.xram_sym:
                xram_locals[v.xram_sym] = v

        # ── Forward walk ───────────────────────────────────────────────────
        block_exit_names: Dict[int, Dict] = {}  # start_ea → exit name_state
        call_sites: List[Tuple[List[HIRNode], int, Dict]] = []

        for block in _rpo(func):
            # Compute entry name_state from predecessor exits
            preds = block.predecessors
            pred_exits = [block_exit_names[p.start_ea]
                          for p in preds if p.start_ea in block_exit_names]

            if block.start_ea == func.entry_block.start_ea:
                name_state = dict(entry_names)
            elif pred_exits:
                name_state = _meet_name_states(pred_exits)
            else:
                name_state = {}

            # Initial const_state from block-entry CP state
            cp = getattr(block, "cp_entry", None)
            const_state: Dict[str, int] = dict(cp._d) if cp is not None else {}

            nodes = block.hir
            for idx, node in enumerate(nodes):
                # (a) Snapshot annotation
                ann = NodeAnnotation()
                ann.reg_names  = dict(name_state)
                ann.reg_consts = dict(const_state)
                node.ann = ann

                # (b) Detect call
                call_expr = _is_call_node(node)
                if call_expr is not None:
                    callee_proto = get_proto(call_expr.func_name)
                    if callee_proto is not None:
                        callee_rm = _build_reg_map(callee_proto)
                        ann.callee_args = callee_rm
                        for r, vi in callee_rm.items():
                            name_state.setdefault(r, vi)
                        call_sites.append((nodes, idx, callee_rm))
                    # Calls clobber all tracked regs
                    name_state.clear()
                    const_state.clear()
                    continue   # skip kill/xram steps for call nodes

                # (c) Detect XRAM load → annotate result reg with local var info
                xram_loaded_regs: frozenset = frozenset()
                if isinstance(node, Assign) and isinstance(node.rhs, XRAMRef):
                    inner = node.rhs.inner
                    sym = None
                    if isinstance(inner, NameExpr):
                        sym = inner.name
                    elif isinstance(inner, RegExpr) and inner.name == "DPTR":
                        dptr_val = const_state.get("DPTR")
                        if dptr_val is not None:
                            from pseudo8051.constants import resolve_ext_addr
                            sym = resolve_ext_addr(dptr_val)
                    if sym is not None and sym in xram_locals:
                        dst_regs = _get_written_regs(node)
                        for r in dst_regs:
                            name_state[r] = xram_locals[sym]
                        xram_loaded_regs = dst_regs

                # (d) Kill defined regs (unless already set by XRAM load above)
                for r in _get_written_regs(node):
                    if r not in xram_loaded_regs:
                        name_state.pop(r, None)
                    const_state.pop(r, None)
                    # DPTR write implicitly invalidates DPH/DPL and vice-versa
                    if r == "DPTR":
                        const_state.pop("DPH", None)
                        const_state.pop("DPL", None)
                    elif r in ("DPH", "DPL"):
                        const_state.pop("DPTR", None)

                # (e) Forward-propagate new constant produced by this node
                _propagate_const(node, const_state)

            block_exit_names[block.start_ea] = dict(name_state)

        # ── Backward pass: annotate pre-call assignments with callee param names ──
        for nodes, call_idx, callee_rm in call_sites:
            _backward_annotate_call(nodes, call_idx, callee_rm)

        func._annotation_pass_ran = True
        dbg("annotate", f"{func.name}: annotation complete")
