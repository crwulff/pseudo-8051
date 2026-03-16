"""
analysis/constprop.py — CPState + ConstantPropagation pass.

Generalised forward constant-propagation over all tracked registers.
Migrated and generalised from CPState / propagate_insn() /
compute_entry_states() in the old monolithic script.
"""

from collections import deque
from typing import Dict, Optional, TYPE_CHECKING

import ida_ua
import ida_bytes
import idc

from pseudo8051.constants import REG_DPTR, PARAM_REGS
from pseudo8051.passes     import OptimizationPass

# Registers whose values we track through constant propagation
_REG_TRACKED = frozenset(["A", "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"])

if TYPE_CHECKING:
    from pseudo8051.ir.function import Function


# ── CPState ───────────────────────────────────────────────────────────────────

class CPState:
    """
    Tracks register values with known integer constants for a single program
    point.  Values are integers (known) or absent from the dict (unknown).

    The dict maps register-name string → integer value.  Any register name can
    be tracked; the propagation rules below cover DPTR / DPH / DPL explicitly
    and conservatively kill any register they don't understand.
    """

    __slots__ = ("_d",)

    def __init__(self, other: Optional["CPState"] = None):
        self._d: Dict[str, int] = dict(other._d) if other is not None else {}

    def get(self, reg: str) -> Optional[int]:
        """Return known integer value, or None if unknown."""
        return self._d.get(reg)

    def set(self, reg: str, val: int) -> None:
        self._d[reg] = val

    def kill(self, *regs: str) -> None:
        for r in regs:
            self._d.pop(r, None)

    def copy(self) -> "CPState":
        return CPState(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CPState) and self._d == other._d

    @staticmethod
    def meet(states) -> "CPState":
        """
        Combine predecessor exit states: a key's value is kept only when
        every predecessor agrees on the same value for it (two-point lattice).
        """
        result = CPState()
        if not states:
            return result
        all_keys = set().union(*(set(s._d) for s in states))
        for k in all_keys:
            if any(k not in s._d for s in states):
                continue   # at least one predecessor unknown
            vals = {s._d[k] for s in states}
            if len(vals) == 1:
                result._d[k] = vals.pop()
            # else: predecessors disagree — leave unknown
        return result


# ── Propagation rules ─────────────────────────────────────────────────────────

def propagate_insn(insn, state: CPState) -> None:
    """
    Update state in-place with the known effects of insn.
    Tracks DPTR, DPH, DPL, and general registers A/R0-R7.
    Conservative on calls (kills all tracked registers).
    """
    mnem = insn.get_canon_mnem().upper()
    op0  = insn.ops[0]
    op1  = insn.ops[1]

    if mnem == "MOV":
        if op0.type == ida_ua.o_reg and op0.reg == REG_DPTR:
            if op1.type == ida_ua.o_imm:
                v = op1.value & 0xFFFF
                state.set("DPTR", v)
                state.set("DPH",  v >> 8)
                state.set("DPL",  v & 0xFF)
            else:
                state.kill("DPTR", "DPH", "DPL")
        elif op0.type == ida_ua.o_mem and (op0.addr & 0xFF) == 0x83:
            # MOV DPH, #imm  (SFR 0x83)
            if op1.type == ida_ua.o_imm:
                hi = op1.value & 0xFF
                state.set("DPH", hi)
                lo = state.get("DPL")
                if lo is not None:
                    state.set("DPTR", (hi << 8) | lo)
                else:
                    state.kill("DPTR")
            else:
                state.kill("DPH", "DPTR")
        elif op0.type == ida_ua.o_mem and (op0.addr & 0xFF) == 0x82:
            # MOV DPL, #imm  (SFR 0x82)
            if op1.type == ida_ua.o_imm:
                lo = op1.value & 0xFF
                state.set("DPL", lo)
                hi = state.get("DPH")
                if hi is not None:
                    state.set("DPTR", (hi << 8) | lo)
                else:
                    state.kill("DPTR")
            else:
                state.kill("DPL", "DPTR")
        elif op0.type == ida_ua.o_reg:
            # MOV A/#imm or MOV Rn, #imm / MOV Rn, Rm
            op0_name = idc.print_operand(insn.ea, 0)
            if op0_name in _REG_TRACKED:
                if op1.type == ida_ua.o_imm:
                    state.set(op0_name, op1.value & 0xFF)
                elif op1.type == ida_ua.o_reg:
                    op1_name = idc.print_operand(insn.ea, 1)
                    if op1_name in _REG_TRACKED:
                        val = state.get(op1_name)
                        if val is not None:
                            state.set(op0_name, val)
                        else:
                            state.kill(op0_name)
                    else:
                        state.kill(op0_name)
                else:
                    state.kill(op0_name)

    elif mnem == "INC":
        if op0.type == ida_ua.o_reg and op0.reg == REG_DPTR:
            v = state.get("DPTR")
            if v is not None:
                nv = (v + 1) & 0xFFFF
                state.set("DPTR", nv)
                state.set("DPH",  nv >> 8)
                state.set("DPL",  nv & 0xFF)
        elif op0.type == ida_ua.o_reg:
            op0_name = idc.print_operand(insn.ea, 0)
            if op0_name in _REG_TRACKED:
                val = state.get(op0_name)
                if val is not None:
                    state.set(op0_name, (val + 1) & 0xFF)
                else:
                    state.kill(op0_name)

    elif mnem == "DEC":
        if op0.type == ida_ua.o_reg:
            op0_name = idc.print_operand(insn.ea, 0)
            if op0_name in _REG_TRACKED:
                val = state.get(op0_name)
                if val is not None:
                    state.set(op0_name, (val - 1) & 0xFF)
                else:
                    state.kill(op0_name)

    elif mnem in ("ADD", "ADDC", "SUBB", "DA", "MUL", "DIV"):
        # These always modify A (and B for MUL/DIV, but we don't track B)
        state.kill("A")

    elif mnem == "CLR":
        # CLR A clears the accumulator to zero; CLR C/bit doesn't affect A
        if op0.type == ida_ua.o_reg:
            op0_name = idc.print_operand(insn.ea, 0)
            if op0_name == "A":
                state.set("A", 0)

    elif mnem in ("ANL", "ORL", "XRL", "CPL", "RL", "RLC", "RR", "RRC", "SWAP"):
        # Only kill A when the accumulator is the destination
        if op0.type == ida_ua.o_reg:
            op0_name = idc.print_operand(insn.ea, 0)
            if op0_name == "A":
                state.kill("A")

    elif mnem == "MOVX":
        # MOVX A, @DPTR / MOVX A, @Ri — reading from XRAM kills A
        if op0.type == ida_ua.o_reg:
            op0_name = idc.print_operand(insn.ea, 0)
            if op0_name == "A":
                state.kill("A")

    elif mnem == "XCH":
        # XCH A, Rn — swap values; both become unknown
        state.kill("A")
        if op1.type == ida_ua.o_reg:
            op1_name = idc.print_operand(insn.ea, 1)
            if op1_name in _REG_TRACKED:
                state.kill(op1_name)

    elif mnem in ("LCALL", "ACALL", "CALL"):
        state.kill("DPTR", "DPH", "DPL")
        for r in _REG_TRACKED:
            state.kill(r)

    elif mnem == "POP":
        if op0.type == ida_ua.o_mem:
            addr8 = op0.addr & 0xFF
            if addr8 == 0x83:
                state.kill("DPH", "DPTR")
            elif addr8 == 0x82:
                state.kill("DPL", "DPTR")


# ── Analysis pass ─────────────────────────────────────────────────────────────

class ConstantPropagation(OptimizationPass):
    """
    Forward constant-propagation pass.
    Populates block.cp_entry (CPState at the START of each block).
    """

    def run(self, func: "Function") -> None:
        BADADDR = idc.BADADDR
        blocks_by_ea = {b.start_ea: b for b in func.blocks}

        entry_states: Dict[int, CPState] = {func.ea: CPState()}
        exit_states:  Dict[int, CPState] = {}

        worklist    = deque([func.ea])
        in_worklist = {func.ea}

        while worklist:
            bea = worklist.popleft()
            in_worklist.discard(bea)

            block = blocks_by_ea.get(bea)
            if block is None:
                continue

            # Entry state = meet of predecessor exit states
            if bea == func.ea:
                new_entry = CPState()
            else:
                pred_exits = [exit_states[p.start_ea]
                              for p in block.predecessors
                              if p.start_ea in exit_states]
                new_entry = CPState.meet(pred_exits) if pred_exits else CPState()

            if entry_states.get(bea) == new_entry:
                continue

            entry_states[bea] = new_entry

            # Thread state through the block's instructions
            cur = new_entry.copy()
            ea  = block.start_ea
            while ea < block.end_ea and ea < BADADDR:
                insn = ida_ua.insn_t()
                size = ida_ua.decode_insn(insn, ea)
                if size > 0:
                    propagate_insn(insn, cur)
                    ea += size
                else:
                    nxt = ida_bytes.next_head(ea, block.end_ea)
                    if nxt <= ea or nxt >= BADADDR:
                        break
                    ea = nxt

            old_exit = exit_states.get(bea)
            exit_states[bea] = cur
            if old_exit != cur:
                for succ in block.successors:
                    if succ.start_ea < BADADDR and succ.start_ea not in in_worklist:
                        worklist.append(succ.start_ea)
                        in_worklist.add(succ.start_ea)

        # Write results back to blocks
        for bea, block in blocks_by_ea.items():
            block.cp_entry = entry_states.get(bea, CPState())
