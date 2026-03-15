"""
ir/basicblock.py — BasicBlock: a sequence of Instructions with analysis slots.
"""

from typing import List, Optional, TYPE_CHECKING

import ida_ua
import ida_bytes
import idc

from pseudo8051.ir.instruction import Instruction
from pseudo8051.ir.hir         import HIRNode, Statement, Label

if TYPE_CHECKING:
    from pseudo8051.ir.function        import Function
    from pseudo8051.analysis.constprop import CPState


class BasicBlock:
    """
    Wraps an IDA basic block (ida_gdl.BasicBlock) inside a Function context.

    Analysis passes populate the mutable slots (live_in, live_out, cp_entry).
    Structural passes populate / replace hir.
    """

    def __init__(self, ida_block, func: "Function"):
        self._ida_block   = ida_block
        self._func        = func

        self.start_ea: int = ida_block.start_ea
        self.end_ea:   int = ida_block.end_ea

        # ── Analysis slots (set by passes) ────────────────────────────────
        self.live_in:       frozenset = frozenset()
        self.live_out:      frozenset = frozenset()
        self.cp_entry:      "CPState" = None   # set by ConstantPropagation

        # ── Structural annotations ────────────────────────────────────────
        self.is_loop_header: bool          = False
        self.label:          Optional[str] = None  # e.g. "label_1234"

        # ── HIR (populated by Function.__init__ after passes) ─────────────
        self.hir: List[HIRNode] = []

        # ── Cached properties ─────────────────────────────────────────────
        self._instructions: Optional[List[Instruction]] = None
        self._upward_use:   Optional[frozenset] = None
        self._defined:      Optional[frozenset] = None

    # ── Graph edges ───────────────────────────────────────────────────────

    @property
    def predecessors(self) -> List["BasicBlock"]:
        bmap = self._func._block_map
        return [bmap[p.start_ea] for p in self._ida_block.preds()
                if p.start_ea in bmap]

    @property
    def successors(self) -> List["BasicBlock"]:
        bmap = self._func._block_map
        return [bmap[s.start_ea] for s in self._ida_block.succs()
                if s.start_ea in bmap]

    # ── Instruction list (cached) ─────────────────────────────────────────

    @property
    def instructions(self) -> List[Instruction]:
        if self._instructions is None:
            self._instructions = self._decode_instructions()
        return self._instructions

    def _decode_instructions(self) -> List[Instruction]:
        BADADDR = idc.BADADDR
        result  = []
        ea      = self.start_ea
        while ea < self.end_ea and ea < BADADDR:
            insn = ida_ua.insn_t()
            size = ida_ua.decode_insn(insn, ea)
            if size > 0:
                result.append(Instruction(ea))
                ea += size
            else:
                nxt = ida_bytes.next_head(ea, self.end_ea)
                if nxt <= ea or nxt >= BADADDR:
                    break
                ea = nxt
        return result

    # ── Use / def sets (cached) ───────────────────────────────────────────

    @property
    def upward_use(self) -> frozenset:
        """Registers used before they are defined anywhere in this block."""
        if self._upward_use is None:
            self._compute_use_def()
        return self._upward_use

    @property
    def defined(self) -> frozenset:
        """Registers written (defined) somewhere in this block."""
        if self._defined is None:
            self._compute_use_def()
        return self._defined

    def _compute_use_def(self) -> None:
        use  = set()
        def_ = set()
        for insn in self.instructions:
            u = insn.use()
            d = insn.defs()
            use  |= u - def_   # only count uses not already covered by defs
            def_ |= d
        self._upward_use = frozenset(use)
        self._defined    = frozenset(def_)

    # ── Initial HIR generation ────────────────────────────────────────────

    def initial_hir(self) -> List[HIRNode]:
        """
        Build the flat initial HIR for this block.

        If a label was assigned (by Function.__init__), a Label node is
        prepended.  Each instruction is lifted using the CP state threaded
        through the block, producing Statement nodes.  Branch instructions
        produce Goto/conditional Statement nodes verbatim at this stage;
        structural passes will later replace them with IfNode / WhileNode.
        """
        from pseudo8051.analysis.constprop import CPState, propagate_insn

        nodes: List[HIRNode] = []

        # Prepend label if this block needs one
        if self.label:
            nodes.append(Label(self.start_ea, self.label))

        # Thread CP state through the block's instructions
        state = self.cp_entry.copy() if self.cp_entry else CPState()

        for instr in self.instructions:
            raw_insn = instr.insn
            if raw_insn is None:
                continue
            stmts = instr.lift(state)
            propagate_insn(raw_insn, state)
            for s in stmts:
                nodes.append(Statement(instr.ea, s))

        return nodes

    def __repr__(self) -> str:
        return f"<BasicBlock start={hex(self.start_ea)} end={hex(self.end_ea)}>"
