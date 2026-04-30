"""
passes/annotate.py — AnnotationPass: forward + backward HIR annotation.

Runs on the flat-block HIR (after ChunkInliner, before RMWCollapser).
Annotates each HIR node with:
  - reg_groups:   TypeGroup list for registers live at that point
  - reg_consts:   known constant register values at that point
  - call_arg_ann: callee-param TypeGroups back-propagated to pre-call assignments
  - callee_args:  callee TypeGroup list stored on the call node itself
"""

import copy
from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir     import (HIRNode, NodeAnnotation, Assign, CompoundAssign,
                                    ExprStmt)
from pseudo8051.ir.expr    import (Reg, Regs as RegExpr, XRAMRef, IRAMRef, CROMRef,
                                    Name as NameExpr, Const as ConstExpr)
from pseudo8051.passes     import OptimizationPass
from pseudo8051.constants  import PARAM_REG_ORDER, DEBUG, dbg
from pseudo8051.passes.patterns._utils import _canonicalize_expr


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


def _expr_refs_reg(expr, reg: str) -> bool:
    """Return True if expr contains Reg(reg) anywhere in its tree."""
    from pseudo8051.ir.expr import Expr as _Expr
    if not isinstance(expr, _Expr):
        return False
    from pseudo8051.ir.expr import Regs as _Regs
    if isinstance(expr, _Regs) and reg in expr.reg_set():
        return True
    for child in expr.children():
        if _expr_refs_reg(child, reg):
            return True
    return False


def _meet_exprs(pred_exits: List[dict]) -> dict:
    """Intersect predecessor expr states: keep r→expr only when all preds agree."""
    if not pred_exits:
        return {}
    result = {}
    all_keys = set().union(*(set(d) for d in pred_exits))
    for k in all_keys:
        if any(k not in d for d in pred_exits):
            continue
        exprs = [d[k] for d in pred_exits]
        if all(e == exprs[0] for e in exprs[1:]):
            result[k] = exprs[0]
    return result


def _meet_reg_defs(pred_exits: List[Dict]) -> Dict:
    """Intersect predecessor reg_def_nodes: keep r→node only when all preds agree (same object).

    Unprocessed predecessors (back edges on first fixpoint pass) are skipped —
    they are treated as "top" (agree with everything) so the first pass
    propagates information optimistically.  Subsequent passes include them,
    dropping any register whose last-writer differs across paths.
    """
    if not pred_exits:
        return {}
    result = {}
    first = pred_exits[0]
    for reg, node in first.items():
        if all(pe.get(reg) is node for pe in pred_exits[1:]):
            result[reg] = node
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


def _folded_addr_consts(node: HIRNode) -> List[int]:
    """Return the list of constant values used as folded-in addresses in this node.

    Scans lhs and rhs of Assign/CompoundAssign and expr of ExprStmt for
    XRAMRef/IRAMRef/CROMRef nodes whose inner expression is a Const.
    Each such Const value is returned so the caller can look up which register
    had that value and link the defining node as a provenance source.
    """
    results: List[int] = []

    def _scan(expr) -> None:
        if expr is None:
            return
        if isinstance(expr, (XRAMRef, IRAMRef, CROMRef)):
            if isinstance(expr.inner, ConstExpr):
                results.append(expr.inner.value)
            else:
                _scan(expr.inner)
            return
        for child in getattr(expr, 'children', lambda: [])():
            _scan(child)

    if isinstance(node, (Assign, CompoundAssign)):
        _scan(node.lhs)
        _scan(node.rhs)
    elif isinstance(node, ExprStmt):
        _scan(node.expr)
    return results


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

        # Add XRAM local and param var info (indexed by xram_sym).
        # Both locals and params may be loaded via MOVX A,@DPTR; if the sym is
        # known, set a TypeGroup on the destination register so that subsequent
        # ACC.N bit tests (jb/jnb) can be resolved to the variable name.
        xram_locals: Dict = {}
        from pseudo8051.passes.typesimplify._regmap import _augment_with_xram_params
        augmented = _augment_with_local_vars(func.ea, {})
        augmented = _augment_with_xram_params(func.ea, augmented)
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
        block_exit_exprs:  Dict[int, dict] = {}   # start_ea → exit expr_state
        call_sites: List[Tuple] = []              # (block, nodes, idx, callee_groups)
        xram_call_sites: List[Tuple] = []         # (nodes, idx, xp_map)
        retval_count = 0                          # number of seeded return values so far

        # ── Phase 0: compute stable reg_def_nodes entry/exit states via fixpoint ──
        # At CFG merge points, only keep reg_def_nodes entries that are identical
        # (same HIR node object, is-comparison) on ALL processed incoming paths.
        # Unprocessed back-edge predecessors are treated as "top" (skipped) on the
        # first pass so information propagates optimistically; subsequent passes
        # include them and drop any register whose last-writer differs across paths.
        # Iteration terminates because the meet only removes entries (monotone).
        rpo_order = _rpo(func)
        block_exit_reg_defs: Dict[int, Dict[str, HIRNode]] = {}
        _fixpt_changed = True
        while _fixpt_changed:
            _fixpt_changed = False
            for _blk in rpo_order:
                _preds = _blk.predecessors
                _pred_exits = [block_exit_reg_defs[p.start_ea]
                               for p in _preds if p.start_ea in block_exit_reg_defs]
                if _blk.start_ea == func.entry_block.start_ea:
                    _entry_defs: Dict[str, HIRNode] = {}
                elif _pred_exits:
                    _entry_defs = _meet_reg_defs(_pred_exits)
                else:
                    _entry_defs = {}
                _exit_defs = dict(_entry_defs)
                for _node in _blk.hir:
                    for _r in _node.written_regs | _node.possibly_killed():
                        _exit_defs[_r] = _node
                _old = block_exit_reg_defs.get(_blk.start_ea)
                if _old is None or _old != _exit_defs:
                    block_exit_reg_defs[_blk.start_ea] = _exit_defs
                    _fixpt_changed = True

        for block in rpo_order:
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

            # reg_def_nodes: per-block entry state from stable fixpoint meet.
            # Only registers whose last-writer is identical on every processed
            # predecessor path survive to this block; the rest are dropped.
            _pred_def_exits = [block_exit_reg_defs[p.start_ea]
                               for p in preds if p.start_ea in block_exit_reg_defs]
            if block.start_ea == func.entry_block.start_ea:
                reg_def_nodes: Dict[str, HIRNode] = {}
            elif _pred_def_exits:
                reg_def_nodes = _meet_reg_defs(_pred_def_exits)
            else:
                reg_def_nodes = {}

            # Initial const_state from block-entry CP state
            cp = getattr(block, "cp_entry", None)
            const_state: Dict[str, int] = dict(cp._d) if cp is not None else {}

            # expr_state: meet of predecessor exit expression states
            pred_expr_exits = [block_exit_exprs[p.start_ea]
                                for p in preds if p.start_ea in block_exit_exprs]
            if block.start_ea == func.entry_block.start_ea:
                expr_state: dict = {}
            elif pred_expr_exits:
                expr_state = _meet_exprs(pred_expr_exits)
            else:
                expr_state = {}

            nodes = block.hir
            for idx, node in enumerate(nodes):

                # (a) Inject custom register annotations at this instruction EA
                user_tgs = []
                for ra in _reg_anns.get(getattr(node, 'ea', None) or 0, []):
                    tg = TypeGroup(ra.name, ra.type, tuple(ra.regs), is_param=False)
                    # Kill any existing groups covering those regs
                    for r in ra.regs:
                        groups = _kill_groups(groups, r)
                    groups.append(tg)
                    user_tgs.append(tg)

                # Register provenance: link the defining node for any register whose
                # value is no longer directly visible in the output expression.
                #
                # Case 1 — constant-folded address: a register's value was folded
                #   into a XRAMRef/IRAMRef/CROMRef Const inner, so the register
                #   name is gone from the expression.  Link whatever last defined
                #   that register so the address computation stays traceable.
                #
                # Case 2 — ExprStmt register reads: ExprStmt nodes (e.g. DPTR++)
                #   both read and write a register.  The old value is consumed
                #   (merged) to produce the new value, and no downstream fusion
                #   pattern will capture the previous-writer provenance.  For Assign
                #   nodes we skip this: explicit register reads are either still
                #   visible in the rendered expression or will be captured by the
                #   fusion pattern that consumes them (e.g. MOV A + MOVX → fused).
                _linked: set = set()
                for _addr_val in _folded_addr_consts(node):
                    for _reg, _val in const_state.items():
                        if _val == _addr_val and _reg in reg_def_nodes:
                            _src = reg_def_nodes[_reg]
                            if id(_src) not in _linked:
                                # Shallow-copy to freeze source_nodes at this point:
                                # later passes may prepend to _src.source_nodes via
                                # assignment, but the copy retains the current list.
                                node.source_nodes = [copy.copy(_src)] + list(node.source_nodes)
                                _linked.add(id(_src))

                if isinstance(node, ExprStmt):
                    for _reg in node.name_refs():
                        if _reg in reg_def_nodes:
                            _src = reg_def_nodes[_reg]
                            if id(_src) not in _linked:
                                node.source_nodes = [copy.copy(_src)] + list(node.source_nodes)
                                _linked.add(id(_src))


                # (b) Snapshot annotation
                ann = NodeAnnotation()
                ann.reg_groups = list(groups)
                ann.reg_consts = dict(const_state)
                ann.reg_exprs  = dict(expr_state)
                ann.user_anns  = user_tgs   # user annotations force-install in _build_node_eff
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

                        # Link the defining nodes for all callee parameter registers.
                        # When CP baked a constant directly into a call arg at lift
                        # time, the register name is absent from name_refs() and the
                        # existing ExprStmt case-2 path won't fire.  Iterate over
                        # every register in the callee prototype and link whatever
                        # last wrote it in this function's HIR.
                        for tg in callee_groups:
                            for _pr in tg.active_regs:
                                _src = reg_def_nodes.get(_pr)
                                if _src is not None and id(_src) not in _linked:
                                    node.source_nodes = [copy.copy(_src)] + list(node.source_nodes)
                                    _linked.add(id(_src))

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

                    # Calls clobber all tracked groups and expression state
                    groups = []
                    const_state.clear()
                    expr_state.clear()
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

                # Capture pre-kill state for self-referential Assign/CompoundAssign (step f)
                _compound_old_expr    = None
                _compound_canon_rhs   = None
                _assign_precanon_rhs  = None   # pre-kill canon for self-ref Assign
                _unary_reg            = None   # register modified by ExprStmt(UnaryOp)
                _unary_old_expr       = None
                _unary_delta          = None
                if (isinstance(node, CompoundAssign)
                        and isinstance(node.lhs, RegExpr) and node.lhs.is_single):
                    _compound_old_expr = expr_state.get(node.lhs.name)
                    if _compound_old_expr is not None:
                        _compound_canon_rhs = _canonicalize_expr(
                            node.rhs, const_state, groups, expr_state)
                elif (isinstance(node, Assign)
                        and isinstance(node.lhs, RegExpr) and node.lhs.is_single):
                    lhs_name = node.lhs.name
                    if (expr_state.get(lhs_name) is not None
                            and _expr_refs_reg(node.rhs, lhs_name)):
                        # RHS reads the LHS reg — pre-canonicalize before kill so the
                        # old value of the reg is substituted in (e.g. rl: A=rol8(A)
                        # with A=arg1 → A=rol8(arg1) after the kill).
                        _assign_precanon_rhs = _canonicalize_expr(
                            node.rhs, const_state, groups, expr_state)
                elif (isinstance(node, ExprStmt)):
                    from pseudo8051.ir.expr import UnaryOp as _UnaryOp
                    if (isinstance(node.expr, _UnaryOp)
                            and node.expr.op in ('++', '--')
                            and isinstance(node.expr.operand, RegExpr)
                            and node.expr.operand.is_single):
                        r = node.expr.operand.name
                        old = expr_state.get(r)
                        if old is not None:
                            _unary_reg      = r
                            _unary_old_expr = old
                            _unary_delta    = +1 if node.expr.op == '++' else -1
                        # Explicitly kill expr_state for r and stale references —
                        # written_regs returns empty for ExprStmt so step (d)
                        # won't do this; step (f) will rebuild the canonical value.
                        expr_state.pop(r, None)
                        stale = [k for k, v in expr_state.items()
                                 if _expr_refs_reg(v, r)]
                        for k in stale:
                            expr_state.pop(k)

                # (d) Kill defined regs (unless just loaded from XRAM above)
                for r in node.written_regs:
                    if r not in xram_loaded_regs:
                        groups = _kill_groups(groups, r)
                    const_state.pop(r, None)
                    expr_state.pop(r, None)
                    # Also evict any expr_state entry whose value references r,
                    # since that expression is now stale (e.g. R6=A becomes
                    # invalid once A is overwritten).
                    stale = [k for k, v in expr_state.items()
                             if _expr_refs_reg(v, r)]
                    for k in stale:
                        expr_state.pop(k)
                    # DPTR write implicitly invalidates DPH/DPL and vice-versa
                    if r == "DPTR":
                        const_state.pop("DPH", None)
                        const_state.pop("DPL", None)
                        expr_state.pop("DPH", None)
                        expr_state.pop("DPL", None)
                    elif r in ("DPH", "DPL"):
                        const_state.pop("DPTR", None)
                        expr_state.pop("DPTR", None)
                        # Also kill any TypeGroup that claims DPTR holds a parameter
                        # value: if only DPH or DPL changed, DPTR no longer equals
                        # the original parameter (e.g. offset) and substituting it
                        # downstream would produce a stale expression.
                        groups = _kill_groups(groups, "DPTR")

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

                # Update register provenance tracker for all registers written
                # or side-effected by this node.
                for _reg in node.written_regs | node.possibly_killed():
                    reg_def_nodes[_reg] = node

                # (f) Track defining expression for single-register assigns/compound-assigns
                if (isinstance(node, Assign)
                        and isinstance(node.lhs, RegExpr) and node.lhs.is_single):
                    if _assign_precanon_rhs is not None:
                        expr_state[node.lhs.name] = _assign_precanon_rhs
                    else:
                        expr_state[node.lhs.name] = _canonicalize_expr(
                            node.rhs, const_state, groups, expr_state)
                elif (_compound_old_expr is not None
                        and isinstance(node, CompoundAssign)
                        and isinstance(node.lhs, RegExpr) and node.lhs.is_single):
                    # Reconstruct: new_A = old_A op rhs  (both already canonical)
                    from pseudo8051.ir.expr import BinOp as _BinOp
                    stripped_op = node.op[:-1]   # "+=" → "+"
                    new_expr = _BinOp(_compound_old_expr, stripped_op, _compound_canon_rhs)
                    expr_state[node.lhs.name] = _canonicalize_expr(
                        new_expr, const_state, groups, expr_state)
                elif _unary_old_expr is not None:
                    # ExprStmt(A++ / A--): rebuild as old_A ± 1
                    from pseudo8051.ir.expr import BinOp as _BinOp, Const as _Const
                    op = '+' if _unary_delta == 1 else '-'
                    new_expr = _BinOp(_unary_old_expr, op, _Const(1))
                    expr_state[_unary_reg] = _canonicalize_expr(
                        new_expr, const_state, groups, expr_state)

            block_exit_groups[block.start_ea] = list(groups)
            block_exit_exprs[block.start_ea]  = dict(expr_state)

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
            dump_pass_hir("02.annotate", all_nodes, func.name)
