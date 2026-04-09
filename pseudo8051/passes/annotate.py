"""
passes/annotate.py — AnnotationPass: forward + backward HIR annotation.

Runs on the flat-block HIR (after ChunkInliner, before RMWCollapser).
Annotates each HIR node with:
  - reg_groups:   TypeGroup list for registers live at that point
  - reg_consts:   known constant register values at that point
  - call_arg_ann: callee-param TypeGroups back-propagated to pre-call assignments
  - callee_args:  callee TypeGroup list stored on the call node itself
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir     import (HIRNode, NodeAnnotation, Assign, CompoundAssign,
                                    ExprStmt)
from pseudo8051.ir.expr    import Reg, Regs as RegExpr, XRAMRef, Name as NameExpr
from pseudo8051.passes     import OptimizationPass
from pseudo8051.constants  import PARAM_REG_ORDER, DEBUG, dbg


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


def _kill_groups(groups: list, reg: str) -> list:
    """Remove *reg* from active_regs of all TypeGroups; discard groups that go empty."""
    result = []
    for g in groups:
        if reg in g.active_regs:
            ng = g.killed(reg)
            if ng is not None:
                result.append(ng)
        else:
            result.append(g)
    return result


def _meet_groups(pred_exits: List[list]) -> list:
    """Intersect predecessor TypeGroup lists.

    Keeps only groups whose (name, type, full_regs) key appears in ALL predecessors,
    with active_regs = intersection across all predecessors.
    """
    from pseudo8051.passes.patterns._utils import TypeGroup
    if not pred_exits:
        return []
    result = []
    first = pred_exits[0]
    for g in first:
        key = (g.name, g.type, g.full_regs)
        active = g.active_regs
        ok = True
        for other in pred_exits[1:]:
            match = next((og for og in other
                          if (og.name, og.type, og.full_regs) == key), None)
            if match is None:
                ok = False
                break
            active = active & match.active_regs
        if ok and active:
            result.append(TypeGroup(g.name, g.type, g.full_regs,
                                    active_regs=active,
                                    xram_sym=g.xram_sym, is_param=g.is_param,
                                    xram_addr=g.xram_addr))
    return result


def _varinfo_map_to_groups(reg_map: dict) -> list:
    """Convert a Dict[str, VarInfo] to a deduplicated List[TypeGroup].

    Skips XRAM-backed VarInfo (xram_sym set).
    """
    from pseudo8051.passes.patterns._utils import TypeGroup, VarInfo
    seen: dict = {}  # id(vi) -> TypeGroup
    result = []
    for vi in reg_map.values():
        if not isinstance(vi, VarInfo) or vi.xram_sym:
            continue
        if id(vi) not in seen:
            tg = TypeGroup(vi.name, vi.type, vi.regs,
                           is_param=vi.is_param)
            seen[id(vi)] = tg
            result.append(tg)
    return result


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
    from pseudo8051.ir.expr import Const as ConstExpr, Regs as RegExpr2, UnaryOp as UnaryOpExpr

    if isinstance(node, Assign):
        lhs, rhs = node.lhs, node.rhs
        if not (isinstance(lhs, RegExpr) and lhs.is_single):
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

        elif isinstance(rhs, RegExpr2) and rhs.is_single:
            # Register copy: propagate known source value
            src = const_state.get(rhs.name)
            if src is not None:
                const_state[reg] = src

    elif isinstance(node, ExprStmt):
        expr = node.expr
        if isinstance(expr, UnaryOpExpr) and expr.op in ("++", "--"):
            if isinstance(expr.operand, RegExpr) and expr.operand.is_single:
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


def _backward_annotate_call(block, nodes: List[HIRNode], call_idx: int,
                             callee_groups: list) -> None:
    """Back-propagate callee param TypeGroups to pre-call assignments.

    Only parameter TypeGroups (is_param=True) are propagated backward.
    Every node between the call and the start of the param-loading sequence
    is annotated with the TypeGroups that are *still pending* at that point:
    active_regs shrinks each time a param register is written (walking
    backward).  A TypeGroup is dropped from the pending list when all its
    registers have been written.

    Semantics: annotation at node N shows the params still needed from N
    onward (i.e. BEFORE N's writes take effect), so the node that writes
    register R carries R still active in its group.

    When the current block is exhausted and has a single linear predecessor
    (one predecessor, one successor), continues backward into that predecessor.
    """
    from pseudo8051.passes.patterns._utils import TypeGroup

    # Only back-propagate param groups (exclude retval)
    pending = [g for g in callee_groups if g.is_param]
    if not pending:
        return

    def _annotate_and_narrow(node, pending_in: list) -> list:
        """Annotate node with pending_in (BEFORE narrowing), return narrowed list."""
        from pseudo8051.ir.hir import NodeAnnotation as _NA
        if node.ann is None:
            node.ann = _NA()
        # Merge: replace any existing group for same name, else append
        existing_by_name = {ag.name: i for i, ag in enumerate(node.ann.call_arg_ann)}
        for g in pending_in:
            if g.name in existing_by_name:
                node.ann.call_arg_ann[existing_by_name[g.name]] = g
            else:
                node.ann.call_arg_ann.append(g)

        # Narrow: remove registers written by this node from each pending group
        written = node.written_regs
        result = []
        for g in pending_in:
            ng = g
            for r in g.active_regs & written:
                ng = ng.killed(r)
                if ng is None:
                    break
            if ng is not None:
                result.append(ng)
        return result

    j = call_idx - 1
    while j >= 0 and pending:
        pending = _annotate_and_narrow(nodes[j], pending)
        j -= 1

    if not pending:
        return
    # Continue into a single linear predecessor (unconditional fall-through only).
    preds = getattr(block, 'predecessors', [])
    if len(preds) != 1:
        return
    pred = preds[0]
    if len(getattr(pred, 'successors', [])) != 1:
        return
    pred_nodes = pred.hir
    j = len(pred_nodes) - 1
    while j >= 0 and pending:
        pending = _annotate_and_narrow(pred_nodes[j], pending)
        j -= 1


def _backward_annotate_xram_call(nodes: List[HIRNode], call_idx: int,
                                   xram_sym_map: Dict) -> None:
    """Back-propagate callee XRAM param names to XRAM writes preceding a call.

    xram_sym_map maps XRAM symbol string (e.g. 'EXT_DC44') → VarInfo.
    Annotates matching Assign(XRAMRef(...), rhs) nodes with a TypeGroup in
    call_arg_ann whose xram_sym matches the symbol.
    """
    from pseudo8051.passes.patterns._utils import TypeGroup, VarInfo
    from pseudo8051.ir.expr import Name as NameExpr2
    if not xram_sym_map:
        return
    j = call_idx - 1
    while j >= 0:
        node = nodes[j]
        if isinstance(node, Assign) and isinstance(node.lhs, XRAMRef):
            inner = node.lhs.inner
            sym = None
            if isinstance(inner, NameExpr2):
                sym = inner.name          # already "EXT_DC44" etc.
            elif inner == Reg("DPTR"):
                ann = node.ann
                if ann is not None:
                    dptr_val = ann.reg_consts.get("DPTR")
                    if dptr_val is not None:
                        from pseudo8051.constants import resolve_ext_addr
                        sym = resolve_ext_addr(dptr_val)
            if sym is not None and sym in xram_sym_map:
                vi = xram_sym_map[sym]
                if node.ann is not None:
                    existing_syms = {ag.xram_sym for ag in node.ann.call_arg_ann if ag.xram_sym}
                    if sym not in existing_syms:
                        tg = TypeGroup(vi.name, vi.type, (),
                                       xram_sym=sym, xram_addr=vi.xram_addr)
                        node.ann.call_arg_ann.append(tg)
        j -= 1


# ── Debug dump ────────────────────────────────────────────────────────────────

def _fmt_typegroup(g) -> str:
    """Compact single-line representation of a TypeGroup."""
    active = ",".join(sorted(g.active_regs)) if g.active_regs else "()"
    flags = []
    if g.is_param:   flags.append("param")
    if g.xram_sym:   flags.append(f"xram={g.xram_sym}")
    flag_str = f" [{','.join(flags)}]" if flags else ""
    return f"{g.type} {g.name}[{active}]{flag_str}"


def _dump_annotated_hir(func) -> None:
    """Print each block's annotated flat HIR to the debug console."""
    print(f"[pseudo8051:annotate] ── HIR dump: {func.name} ──")
    for block in _rpo(func):
        print(f"[pseudo8051:annotate]   block {hex(block.start_ea)}:")
        for node in block.hir:
            lines = node.render(indent=2)
            text = lines[0][1].strip() if lines else repr(node)
            ann  = node.ann
            ea   = getattr(node, "ea", None)
            ea_s = f"[{hex(ea)}] " if ea is not None else ""
            if ann is None:
                print(f"[pseudo8051:annotate]     {ea_s}{text}  (no ann)")
                continue
            print(f"[pseudo8051:annotate]     {ea_s}{text}")
            if ann.reg_groups:
                parts = ", ".join(_fmt_typegroup(g) for g in ann.reg_groups)
                print(f"[pseudo8051:annotate]       groups:   {parts}")
            if ann.reg_consts:
                parts = ", ".join(
                    f"{r}={hex(v)}" for r, v in sorted(ann.reg_consts.items())
                )
                print(f"[pseudo8051:annotate]       consts:   {parts}")
            if ann.call_arg_ann:
                parts = ", ".join(_fmt_typegroup(g) for g in ann.call_arg_ann)
                print(f"[pseudo8051:annotate]       call_arg: {parts}")
            if ann.callee_args:
                parts = ", ".join(_fmt_typegroup(g) for g in ann.callee_args)
                print(f"[pseudo8051:annotate]       callee:   {parts}")


# ── Pass ───────────────────────────────────────────────────────────────────────

class AnnotationPass(OptimizationPass):
    """Annotate each flat-HIR node with name/type and constant information."""

    def run(self, func) -> None:
        from pseudo8051.prototypes import get_proto, expand_regs, get_struct
        from pseudo8051.passes.typesimplify._regmap import (
            _build_reg_map, _build_type_groups, _split_struct_groups,
            _augment_with_local_vars, _split_struct_regs,
        )
        from pseudo8051.passes.patterns._utils import VarInfo, TypeGroup

        proto   = get_proto(func.name)
        live_in = getattr(func.entry_block, "live_in", frozenset())

        # Build initial TypeGroup list at function entry
        if proto is not None:
            entry_groups = _build_type_groups(proto, live_in)
            dbg("annotate", f"{func.name}: proto found, entry_groups={[g.name for g in entry_groups]}")
        else:
            entry_groups = []
            for idx, reg in enumerate(r for r in PARAM_REG_ORDER if r in live_in):
                tg = TypeGroup(f"arg{idx + 1}", "uint8_t", (reg,), is_param=True)
                entry_groups.append(tg)
            dbg("annotate", f"{func.name}: no proto, inferred entry_groups={[g.name for g in entry_groups]}")

        # Add XRAM local var info (indexed by xram_sym)
        xram_locals: Dict = {}
        augmented = _augment_with_local_vars(func.ea, {})
        for k, v in augmented.items():
            if isinstance(v, VarInfo) and v.xram_sym:
                xram_locals[v.xram_sym] = v

        # Load custom register annotations (indexed by insn_ea)
        _reg_anns: Dict[int, list] = {}
        try:
            from pseudo8051.reganns import get_reganns
            for ra in get_reganns(func.ea):
                _reg_anns.setdefault(ra.ea, []).append(ra)
        except Exception:
            pass  # not in IDA environment (unit tests)

        # ── Forward walk ───────────────────────────────────────────────────
        block_exit_groups: Dict[int, list] = {}   # start_ea → exit groups
        call_sites: List[Tuple] = []              # (block, nodes, idx, callee_groups)
        xram_call_sites: List[Tuple] = []         # (nodes, idx, xp_map)
        retval_count = 0                          # number of seeded return values so far

        for block in _rpo(func):
            # Compute entry groups from predecessor exits
            preds = block.predecessors
            pred_exits = [block_exit_groups[p.start_ea]
                          for p in preds if p.start_ea in block_exit_groups]

            if block.start_ea == func.entry_block.start_ea:
                groups = list(entry_groups)
            elif pred_exits:
                groups = _meet_groups(pred_exits)
            else:
                groups = []

            # Initial const_state from block-entry CP state
            cp = getattr(block, "cp_entry", None)
            const_state: Dict[str, int] = dict(cp._d) if cp is not None else {}

            nodes = block.hir
            for idx, node in enumerate(nodes):
                # (a) Inject custom register annotations at this instruction EA
                for ra in _reg_anns.get(getattr(node, 'ea', None) or 0, []):
                    tg = TypeGroup(ra.name, ra.type, tuple(ra.regs), is_param=False)
                    # Kill any existing groups covering those regs
                    for r in ra.regs:
                        groups = _kill_groups(groups, r)
                    groups.append(tg)

                # (b) Snapshot annotation
                ann = NodeAnnotation()
                ann.reg_groups = list(groups)
                ann.reg_consts = dict(const_state)
                node.ann = ann

                # (c) Detect call
                call_expr = _is_call_node(node)
                if call_expr is not None:
                    callee_proto = get_proto(call_expr.func_name)
                    if callee_proto is not None:
                        callee_groups = _build_type_groups(callee_proto)
                        ann.callee_args = callee_groups
                        # Seed callee param groups for forward propagation
                        for tg in callee_groups:
                            if tg.is_param and not any(
                                    tg.active_regs & g.active_regs for g in groups):
                                groups.append(tg)
                        call_sites.append((block, nodes, idx, callee_groups))

                    # Collect callee XRAM params for backward annotation
                    try:
                        callee_ea = _resolve_name_addr(call_expr.func_name)
                        if callee_ea is not None:
                            from pseudo8051.xram_params import get_xram_params
                            from pseudo8051.constants   import resolve_ext_addr
                            callee_xps = get_xram_params(callee_ea)
                            if callee_xps:
                                xp_map: Dict[str, object] = {}
                                for _p in callee_xps:
                                    _sym = resolve_ext_addr(_p.addr)
                                    _vi  = VarInfo(_p.name, _p.type, (),
                                                   xram_sym=_sym, xram_addr=_p.addr)
                                    xp_map[_sym] = _vi
                                xram_call_sites.append((nodes, idx, xp_map))
                    except Exception:
                        pass

                    # Calls clobber all tracked groups
                    groups = []
                    const_state.clear()
                    # Seed return value TypeGroup(s) for forward propagation
                    if callee_proto is not None and callee_proto.return_regs:
                        retval_count += 1
                        suffix = "" if retval_count == 1 else str(retval_count)
                        rv_name = f"retval{suffix}"
                        ret_regs = expand_regs(tuple(callee_proto.return_regs),
                                               callee_proto.return_type)
                        if get_struct(callee_proto.return_type) is not None:
                            groups.extend(
                                _split_struct_groups(rv_name, callee_proto.return_type,
                                                     ret_regs))
                        else:
                            groups.append(TypeGroup(rv_name, callee_proto.return_type,
                                                    ret_regs))
                    continue   # skip kill/xram steps for call nodes

                # (c) Detect XRAM load → annotate result reg with local var info
                xram_loaded_regs: frozenset = frozenset()
                if isinstance(node, Assign) and isinstance(node.rhs, XRAMRef):
                    inner = node.rhs.inner
                    sym = None
                    if isinstance(inner, NameExpr):
                        sym = inner.name
                    elif inner == Reg("DPTR"):
                        dptr_val = const_state.get("DPTR")
                        if dptr_val is not None:
                            from pseudo8051.constants import resolve_ext_addr
                            sym = resolve_ext_addr(dptr_val)
                    if sym is not None and sym in xram_locals:
                        dst_regs = node.written_regs
                        xram_vi = xram_locals[sym]
                        # Kill existing groups for those regs, then add XRAM-backed TypeGroup
                        for r in dst_regs:
                            groups = _kill_groups(groups, r)
                        xram_tg = TypeGroup(xram_vi.name, xram_vi.type,
                                            tuple(sorted(dst_regs)),
                                            xram_sym=xram_vi.xram_sym,
                                            xram_addr=xram_vi.xram_addr)
                        groups.append(xram_tg)
                        xram_loaded_regs = dst_regs

                # (d) Kill defined regs (unless just loaded from XRAM above)
                for r in node.written_regs:
                    if r not in xram_loaded_regs:
                        groups = _kill_groups(groups, r)
                    const_state.pop(r, None)
                    # DPTR write implicitly invalidates DPH/DPL and vice-versa
                    if r == "DPTR":
                        const_state.pop("DPH", None)
                        const_state.pop("DPL", None)
                    elif r in ("DPH", "DPL"):
                        const_state.pop("DPTR", None)

                # SUBB/ADDC pattern: CompoundAssign A-=... or A+=... whose RHS
                # references C means this instruction writes the carry flag C as
                # its output (borrow/carry result).  Kill any stale C=0 from a
                # preceding CLR C so it doesn't propagate past the arithmetic.
                if (isinstance(node, CompoundAssign)
                        and node.op in ("-=", "+=")
                        and "C" in const_state):
                    from pseudo8051.ir.expr import Regs as _RegsExpr
                    def _rhs_uses_carry(e) -> bool:
                        if isinstance(e, _RegsExpr) and e.names == ('C',):
                            return True
                        return any(_rhs_uses_carry(c) for c in e.children())
                    if _rhs_uses_carry(node.rhs):
                        const_state.pop("C", None)

                # (e) Forward-propagate new constant produced by this node
                _propagate_const(node, const_state)

            block_exit_groups[block.start_ea] = list(groups)

        # ── Backward pass: annotate pre-call assignments with callee param names ──
        for block, nodes, call_idx, callee_groups in call_sites:
            _backward_annotate_call(block, nodes, call_idx, callee_groups)
        for nodes, call_idx, xp_map in xram_call_sites:
            _backward_annotate_xram_call(nodes, call_idx, xp_map)

        func._annotation_pass_ran = True
        dbg("annotate", f"{func.name}: annotation complete")
        if DEBUG:
            from pseudo8051.passes.debug_dump import dump_pass_hir
            all_nodes = [n for block in _rpo(func) for n in block.hir]
            dump_pass_hir("annotate", all_nodes, func.name)
