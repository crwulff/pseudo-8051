"""
handlers/mov.py — MOV / MOVX / MOVC / PUSH / POP / XCH / XCHD handlers.
"""

from typing import List, Optional, TYPE_CHECKING

import ida_ua
import idc

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.constants      import (
    REG_DPTR, PHRASE_AT_DPTR,
    PARAM_REGS, resolve_ext_addr,
)

# Registers whose constant values we can inline as immediates
_TRACKED_REGS = frozenset(["A", "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"])

if TYPE_CHECKING:
    from pseudo8051.analysis.constprop import CPState


def _op(insn, n: int, state=None) -> str:
    return Operand(insn, n).render(state)


class MovHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = set()
        op0 = insn.ops[0]
        r0  = Operand(insn, 0).reg_name()
        r1  = Operand(insn, 1).reg_name()
        pr1 = Operand(insn, 1).registers()

        if r0:                              # MOV reg, src
            if r1:      u.add(r1)
            elif pr1:   u.update(pr1)
        elif op0.type == ida_ua.o_phrase:   # MOV @Ri, src
            u.update(Operand(insn, 0).registers())
            if r1:      u.add(r1)
        elif op0.type == ida_ua.o_mem:      # MOV direct, src
            if r1:      u.add(r1)
            else:       u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        op0 = insn.ops[0]
        if op0.type == ida_ua.o_reg:
            r = Operand(insn, 0).reg_name()
            if r:
                return frozenset({r})
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        op0, op1 = insn.ops[0], insn.ops[1]
        # MOV DPTR, #imm — annotate with EXT symbol
        if (op0.type == ida_ua.o_reg and op0.reg == REG_DPTR
                and op1.type == ida_ua.o_imm):
            imm = op1.value & 0xFFFF
            sym = resolve_ext_addr(imm)
            if sym != hex(imm):
                return [f"DPTR = {sym};  /* {hex(imm)} */"]
        # When the source register has a known constant value, inline it.
        # This makes patterns like "MOV R6, A" (where A=0) emit "R6 = 0x00;"
        # so ConstGroupPattern can recognize multi-byte constant loads.
        if state is not None and op1.type == ida_ua.o_reg:
            src_name = idc.print_operand(insn.ea, 1)
            if src_name in _TRACKED_REGS:
                val = state.get(src_name)
                if val is not None:
                    dst = _op(insn, 0, state)
                    return [f"{dst} = {hex(val)};"]
        dst = _op(insn, 0, state)
        src = _op(insn, 1, state)
        return [f"{dst} = {src};"]


class MovxHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = set()
        op0, op1 = insn.ops[0], insn.ops[1]
        pr0 = Operand(insn, 0).registers()
        pr1 = Operand(insn, 1).registers()
        r0  = Operand(insn, 0).reg_name()
        r1  = Operand(insn, 1).reg_name()
        if op0.type == ida_ua.o_phrase:
            u.update(pr0)
            if r1: u.add(r1)
        elif op1.type == ida_ua.o_phrase:
            u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        op0 = insn.ops[0]
        r0  = Operand(insn, 0).reg_name()
        if op0.type != ida_ua.o_phrase and r0:
            return frozenset({r0})
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        op0, op1 = insn.ops[0], insn.ops[1]
        dptr_val  = state.get("DPTR") if state else None

        if op0.type == ida_ua.o_phrase and op0.phrase == PHRASE_AT_DPTR:
            mem = resolve_ext_addr(dptr_val) if dptr_val is not None else "DPTR"
            return [f"XRAM[{mem}] = {_op(insn, 1, state)};"]
        if op1.type == ida_ua.o_phrase and op1.phrase == PHRASE_AT_DPTR:
            mem = resolve_ext_addr(dptr_val) if dptr_val is not None else "DPTR"
            return [f"{_op(insn, 0, state)} = XRAM[{mem}];"]
        return [f"{_op(insn, 0, state)} = {_op(insn, 1, state)};"]


class MovcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "DPTR"})

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[str]:
        dst = _op(insn, 0, state)
        src = _op(insn, 1, state)
        return [f"{dst} = {src};  /* code ROM */"]


class PushHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        s = idc.print_operand(insn.ea, 0)
        if s in PARAM_REGS:
            return frozenset({s})
        r0 = Operand(insn, 0).reg_name()
        if r0:
            return frozenset({r0})
        return Operand(insn, 0).registers() & PARAM_REGS

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"push({_op(insn, 0, state)});"]


class PopHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        s = idc.print_operand(insn.ea, 0)
        if s in PARAM_REGS:
            return frozenset({s})
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)} = pop();"]


class XchHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r0 = Operand(insn, 0).reg_name()
        r1 = Operand(insn, 1).reg_name()
        pr1 = Operand(insn, 1).registers()
        if r0: u.add(r0)
        if r1: u.add(r1)
        else:  u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        d = {"A"}
        r0 = Operand(insn, 0).reg_name()
        r1 = Operand(insn, 1).reg_name()
        if r0: d.add(r0)
        if r1: d.add(r1)
        return frozenset(d & PARAM_REGS)

    def lift(self, insn, state=None) -> List[str]:
        return [f"swap(A, {_op(insn, 1, state)});"]


class XchdHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r0 = Operand(insn, 0).reg_name()
        pr1 = Operand(insn, 1).registers()
        if r0: u.add(r0)
        u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        d = {"A"}
        r0 = Operand(insn, 0).reg_name()
        r1 = Operand(insn, 1).reg_name()
        if r0: d.add(r0)
        if r1: d.add(r1)
        return frozenset(d & PARAM_REGS)

    def lift(self, insn, state=None) -> List[str]:
        return [f"swap_nibble(A, {_op(insn, 1, state)});  /* low nibble only */"]
