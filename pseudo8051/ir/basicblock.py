"""
ir/basicblock.py — BasicBlock: a sequence of Instructions with analysis slots.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import ida_ua
import ida_bytes
import ida_funcs
import idc

from pseudo8051.ir.instruction import Instruction
from pseudo8051.ir.hir         import HIRNode, Label, ReturnStmt, ExprStmt
from pseudo8051.ir.expr        import Name, Call
from pseudo8051.ir.cpstate     import CPState, propagate_insn


class BasicBlock:
    """
    Wraps an IDA basic block (ida_gdl.BasicBlock) inside a Function context.

    Analysis passes populate the mutable slots (live_in, live_out, cp_entry).
    Structural passes populate / replace hir.
    """

    def __init__(self, ida_block,
                 block_map: Dict[int, BasicBlock],
                 func_ea: int):
        self._ida_block   = ida_block
        self._block_map   = block_map
        self._func_ea     = func_ea

        self.start_ea: int = ida_block.start_ea
        self.end_ea:   int = ida_block.end_ea

        # ── Analysis slots (set by passes) ────────────────────────────────
        self.live_in:       frozenset = frozenset()
        self.live_out:      frozenset = frozenset()
        self.cp_entry:      CPState = None   # set by ConstantPropagation

        # ── Structural annotations ────────────────────────────────────────
        self.is_loop_header: bool          = False
        self.label:          Optional[str] = None  # e.g. "label_1234"

        # ── HIR (populated by Function.__init__ after passes) ─────────────
        self.hir: List[HIRNode] = []

        # ── Cached properties ─────────────────────────────────────────────
        self._instructions: Optional[List[Instruction]] = None
        self._upward_use:   Optional[frozenset] = None
        self._defined:      Optional[frozenset] = None

        # ── Synthetic CFG edges (injected by fixup_jmptable_edges) ────────
        self._succ_extra: List[BasicBlock] = []
        self._pred_extra: List[BasicBlock] = []

    # ── Graph edges ───────────────────────────────────────────────────────

    @property
    def predecessors(self) -> List[BasicBlock]:
        base = [self._block_map[p.start_ea] for p in self._ida_block.preds()
                if p.start_ea in self._block_map]
        return base + self._pred_extra

    @property
    def successors(self) -> List[BasicBlock]:
        base = [self._block_map[s.start_ea] for s in self._ida_block.succs()
                if s.start_ea in self._block_map]
        return base + self._succ_extra

    def _add_successor(self, other: BasicBlock) -> None:
        """Add a synthetic CFG edge self → other (not present in IDA's graph)."""
        if other not in self._succ_extra:
            self._succ_extra.append(other)
        if self not in other._pred_extra:
            other._pred_extra.append(self)

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

        # Fall-through tail call: if this block has no in-function successors
        # and does not end with a branch/return, check whether end_ea is the
        # entry of another function.  The callee's parameter registers count as
        # "used" here, just like SjmpHandler.use() does for explicit jumps.
        ft = self._fallthrough_tail_call()
        if ft is not None:
            _, ft_proto = ft
            if ft_proto:
                from pseudo8051.prototypes import param_regs as _prs
                for r_tuple in _prs(ft_proto):
                    use |= set(r_tuple) - def_

        self._upward_use = frozenset(use)
        self._defined    = frozenset(def_)

    def _fallthrough_tail_call(self):
        """
        Return (target_name, proto_or_None) if this block falls through into
        another function, else return None.

        Conditions:
          • no in-function successors (fall-through target was filtered)
          • last instruction is not a return or branch
          • end_ea is the start_ea of a different IDA function
        """
        if self.successors:
            return None
        last = self.instructions[-1] if self.instructions else None
        if last is None or last.is_return() or last.is_branch():
            return None
        target_fn = ida_funcs.get_func(self.end_ea)
        if (target_fn is None
                or target_fn.start_ea != self.end_ea
                or target_fn.start_ea == self._func_ea):
            return None
        target_name = ida_funcs.get_func_name(self.end_ea) or hex(self.end_ea)
        from pseudo8051.prototypes import get_proto
        return (target_name, get_proto(target_name))

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
            nodes.extend(stmts)

        # Fall-through tail call: append a synthetic call/return statement.
        ft = self._fallthrough_tail_call()
        if ft is not None:
            target_name, proto = ft
            ea = self.instructions[-1].ea if self.instructions else self.start_ea
            from pseudo8051.prototypes import param_regs as _prs
            from pseudo8051.constants  import dbg
            if proto:
                regs_list = _prs(proto)
                args = [Name("".join(r)) if r else Name("?") for r in regs_list]
                call_node = Call(target_name, args)
                dbg("func", f"  fall-through tail call → {target_name}")
                if proto.return_type != "void":
                    nodes.append(ReturnStmt(ea, call_node))
                else:
                    nodes.append(ExprStmt(ea, call_node))
            else:
                dbg("func", f"  fall-through tail call (no proto) → {target_name}")
                nodes.append(ReturnStmt(ea, Call(target_name, []), comment="tail call"))

        return nodes

    def __repr__(self) -> str:
        return f"<BasicBlock start={hex(self.start_ea)} end={hex(self.end_ea)}>"
