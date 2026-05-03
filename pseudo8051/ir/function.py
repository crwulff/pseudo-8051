"""
ir/function.py — Function: block graph + pass runner + final HIR.
"""

from typing import List, Dict, Optional

import ida_funcs
import ida_gdl
import ida_name
import idc

from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.ir.hir        import HIRNode, Label, RemovedNode
from pseudo8051.constants     import PARAM_REG_ORDER, dbg, reset_debug_session


class Function:
    """
    Represents a single 8051 function.

    __init__ builds the block graph, runs all analysis and optimisation passes,
    then assembles the final top-level HIR list that the viewer renders.
    """

    def __init__(self, func_ea: int):
        reset_debug_session()
        self.ea            = func_ea
        self.name          = ida_funcs.get_func_name(func_ea) or hex(func_ea)
        self.removed_nodes: List[RemovedNode] = []

        ida_func = ida_funcs.get_func(func_ea)
        if not ida_func:
            raise ValueError(f"No function at {hex(func_ea)}")

        # ── Build block graph ─────────────────────────────────────────────
        try:
            fc = ida_gdl.FlowChart(ida_func, flags=ida_gdl.FC_PREDS)
        except Exception as e:
            raise RuntimeError(f"FlowChart failed for {hex(func_ea)}: {e}") from e

        BADADDR = idc.BADADDR

        # Collect IDA blocks that belong to this function.
        # IDA's FlowChart follows unconditional jumps across function boundaries
        # (tail calls via ljmp), so we must filter out blocks owned by other
        # functions; otherwise tail-call targets appear as loop back-edges.
        raw_blocks = []
        for b in fc:
            if b.start_ea >= BADADDR:
                continue
            owner = ida_funcs.get_func(b.start_ea)
            if owner is not None and owner.start_ea != func_ea:
                dbg("func", f"  skipping external block {hex(b.start_ea)} "
                             f"(belongs to {ida_funcs.get_func_name(owner.start_ea)})")
                continue
            raw_blocks.append(b)

        # Create BasicBlock wrappers; _block_map is needed by predecessors/successors
        self._block_map: Dict[int, BasicBlock] = {}
        for rb in raw_blocks:
            self._block_map[rb.start_ea] = BasicBlock(rb, self._block_map, self.ea)

        # Order blocks by BFS from the entry block so that the function body
        # always renders before tail/dispatch blocks that happen to sit at lower
        # addresses (e.g. jump-table SJMP entries added as function chunks).
        # Any blocks not reachable from the entry (rare dead-code tails) are
        # appended at the end, sorted by address.
        from collections import deque
        _ordered: List[BasicBlock] = []
        _seen: set = set()
        _queue: deque = deque([func_ea])
        while _queue:
            _ea = _queue.popleft()
            if _ea in _seen or _ea not in self._block_map:
                continue
            _seen.add(_ea)
            _blk = self._block_map[_ea]
            _ordered.append(_blk)
            # Prefer forward edges (higher address) before backward edges
            # so that low-address tail blocks appear after the code that
            # dispatches to them, not before the function entry.
            for _s in sorted(_blk.successors,
                             key=lambda b, ea=_blk.start_ea:
                                 (b.start_ea < ea, b.start_ea)):
                if _s.start_ea not in _seen:
                    _queue.append(_s.start_ea)
        # Append blocks not reachable via BFS (e.g. blocks only reachable through
        # filtered external edges), sorted by address.
        for _blk in sorted(self._block_map.values(), key=lambda b: b.start_ea):
            if _blk.start_ea not in _seen:
                _ordered.append(_blk)
        self._blocks: List[BasicBlock] = _ordered

        dbg("func", f"{self.name} @ {hex(self.ea)} — {len(self._blocks)} block(s)")

        # ── Assign labels to blocks that need them ────────────────────────
        self._assign_labels()

        for blk in self._blocks:
            preds = [hex(p.start_ea) for p in blk.predecessors]
            succs = [hex(s.start_ea) for s in blk.successors]
            label_str = f"  label={blk.label!r}" if blk.label else ""
            loop_str  = "  [loop-header]"         if blk.is_loop_header else ""
            dbg("func", f"  block {hex(blk.start_ea)}-{hex(blk.end_ea)}"
                        f"  preds={preds}  succs={succs}{label_str}{loop_str}")

        # ── Run all analysis + optimisation passes ────────────────────────
        from pseudo8051.passes import run_all_passes
        run_all_passes(self)

        # ── Assemble top-level HIR from all blocks ────────────────────────
        # After structural passes the per-block hir lists have been modified;
        # blocks whose content was absorbed into a WhileNode / IfNode are
        # marked absorbed and skipped here.
        self.hir: List[HIRNode] = []
        for block in self._blocks:
            if not getattr(block, "_absorbed", False):
                self.hir.extend(block.hir)

        absorbed = [hex(b.start_ea) for b in self._blocks
                    if getattr(b, "_absorbed", False)]
        live     = [hex(b.start_ea) for b in self._blocks
                    if not getattr(b, "_absorbed", False)]
        dbg("func", f"  live blocks after passes: {live}")
        if absorbed:
            dbg("func", f"  absorbed blocks:          {absorbed}")

        # ── Fix return statements using prototype ─────────────────────────
        self._fix_return_statements()

        # ── Strip dead code after returns ─────────────────────────────────
        # After block assembly, blocks that were copied into IfNode bodies via
        # externally_ref inlining may still appear as live (non-absorbed) blocks
        # in the flat HIR.  Remove any run of nodes that follows a ReturnStmt
        # without an intervening Label — such code is unreachable.
        cleaned: List[HIRNode] = []
        skip = False
        from pseudo8051.ir.hir import ReturnStmt as _RetStmt, Label as _Lbl
        for node in self.hir:
            if isinstance(node, _Lbl):
                skip = False   # label makes subsequent code reachable
            if not skip:
                cleaned.append(node)
            if isinstance(node, _RetStmt):
                skip = True
        self.hir = cleaned

        # ── Inline singleton goto targets ─────────────────────────────────
        # After block assembly, some gotos (nested inside IfNode bodies) target
        # flat-level labels that the block-based IfElseStructurer couldn't fold.
        # Inline the label's code at the single goto site and remove the
        # flat-level block.
        from pseudo8051.passes._ifelse_helpers import _inline_singleton_goto_targets
        self.hir = _inline_singleton_goto_targets(self.hir)

        # ── CJNE chain → switch (runs on assembled func.hir) ─────────────
        from pseudo8051.passes.cjne_switch import CJNEChainToSwitch
        CJNEChainToSwitch().run(self)

        # ── Type-aware simplification (runs on assembled func.hir) ────────
        from pseudo8051.passes.typesimplify import TypeAwareSimplifier
        TypeAwareSimplifier().run(self)

        # ── Promote while(1){if(cond)break;...} → while(!cond){...} ──────
        from pseudo8051.passes.loops import promote_while1_to_while
        promote_while1_to_while(self)

        # ── Switch case enum annotations ──────────────────────────────────
        from pseudo8051.passes.switchcomment import SwitchCaseAnnotator
        SwitchCaseAnnotator().run(self)

    # ── Block graph accessors ─────────────────────────────────────────────

    @property
    def blocks(self) -> List[BasicBlock]:
        return self._blocks

    @property
    def entry_block(self) -> BasicBlock:
        return self._block_map[self.ea]

    # ── Parameter / return inference (from liveness results) ─────────────

    @property
    def parameters(self) -> List[str]:
        """Registers live at function entry (set by LivenessAnalysis)."""
        entry = self.entry_block
        live  = entry.live_in
        return [r for r in PARAM_REG_ORDER if r in live]

    @property
    def return_registers(self) -> List[str]:
        """Registers live at return-instruction blocks (from liveness)."""
        regs: set = set()
        for block in self._blocks:
            last = None
            for instr in block.instructions:
                last = instr
            if last and last.is_return():
                regs.update(block.live_in)
        return [r for r in PARAM_REG_ORDER if r in regs]

    # ── Label assignment ──────────────────────────────────────────────────

    def _assign_labels(self) -> None:
        """
        Give a label string to every block that needs one:
          • blocks with more than one predecessor (join points), or
          • blocks that are the target of a back-edge (loop headers).
        """
        BADADDR = idc.BADADDR

        pred_count: Dict[int, int] = {}
        loop_headers: set = set()

        for block in self._blocks:
            for succ in block._ida_block.succs():
                if succ.start_ea >= BADADDR:
                    continue
                pred_count[succ.start_ea] = pred_count.get(succ.start_ea, 0) + 1

        # Detect loop headers via DFS (correct for all loop shapes, including
        # loops whose header EA is higher than the tail EA).
        from pseudo8051.passes.loops import _dfs_back_edges
        for _, hdr in _dfs_back_edges(self._block_map[self.ea]):
            loop_headers.add(hdr.start_ea)
            if hdr.start_ea in self._block_map:
                self._block_map[hdr.start_ea].is_loop_header = True

        needs_label = loop_headers | {ea for ea, cnt in pred_count.items() if cnt > 1}

        # Any block that IDA has a symbol for must also get a label: branch
        # operands are rendered via ida_name.get_ea_name(GN_LOCAL), so the goto
        # target text must match block.label exactly — even for single-predecessor
        # blocks that structural criteria alone would not label.
        for block in self._blocks:
            if block.start_ea != self.ea and (
                    ida_name.get_ea_name(block.start_ea, ida_name.GN_LOCAL)
                    or ida_name.get_name(block.start_ea)):
                needs_label.add(block.start_ea)

        for ea in needs_label:
            block = self._block_map.get(ea)
            if block and ea != self.ea:   # entry block never gets a label
                # Prefer the IDA-assigned symbol name so that branch operands
                # (resolved via ida_name.get_name in Operand.render) produce
                # goto targets that match.  Fall back to a generated name only
                # when IDA has no symbol at this address.
                ida_lbl = ida_name.get_ea_name(ea, ida_name.GN_LOCAL) or ida_name.get_name(ea)
                block.label = ida_lbl if ida_lbl else f"label_{hex(ea).removeprefix('0x')}"

    # ── Return-statement fix ──────────────────────────────────────────────

    def _fix_return_statements(self) -> None:
        """
        If this function has a prototype with a non-void return type, fill every
        ReturnStmt(None) with the appropriate return expression:
        - single register  → Reg("R2")
        - multiple registers → RegGroup(("R2", "R1"))
        """
        from pseudo8051.prototypes import get_proto, expand_regs
        from pseudo8051.ir.expr import Reg, RegGroup
        proto = get_proto(self.name)
        if proto is None or not proto.return_regs:
            return
        regs = expand_regs(tuple(proto.return_regs), proto.return_type)
        ret_expr = Reg(regs[0]) if len(regs) == 1 else RegGroup(tuple(regs))
        self._walk_fix_returns(self.hir, ret_expr)

    def _walk_fix_returns(self, nodes: List[HIRNode], ret_expr) -> None:
        from pseudo8051.ir.hir import ReturnStmt

        def _recurse(ns: List[HIRNode]) -> List[HIRNode]:
            self._walk_fix_returns(ns, ret_expr)
            return ns   # bodies mutated in-place; return same list

        for node in nodes:
            if isinstance(node, ReturnStmt) and node.value is None:
                node.value = ret_expr
            node.map_bodies(_recurse)  # no-op for leaves; recurses for structured nodes

    # ── Rendering ─────────────────────────────────────────────────────────

    def render(self) -> List[tuple]:
        """
        Return a flat list of (ea, text) tuples for the viewer.
        Wraps the body in a C-style function definition using the prototype
        (if one exists) or inferred parameters from liveness analysis.
        """
        from pseudo8051.prototypes import get_proto
        proto = get_proto(self.name)

        if proto:
            ret_type   = proto.return_type
            params_str = ", ".join(f"{p.type} {p.name}" for p in proto.params) or "void"
        else:
            ret_type = "void"
            params   = self.parameters
            params_str = ", ".join(params) if params else "void"

        try:
            from pseudo8051.xram_params import get_xram_params
            xram_ps = get_xram_params(self.ea)
            if xram_ps:
                xram_str = ", ".join(f"{p.type} {p.name}" for p in xram_ps)
                params_str = f"{params_str}, {xram_str}" if params_str != "void" else xram_str
        except Exception:
            pass

        lines: List[tuple] = [
            (self.ea, f"{ret_type} {self.name}({params_str})"),
            (self.ea, "{"),
        ]
        for node in self.hir:
            lines.extend(node.render(indent=1))
        lines.append((self.ea, "}"))
        return lines

    def __repr__(self) -> str:
        return f"<Function name={self.name!r} ea={hex(self.ea)}>"
