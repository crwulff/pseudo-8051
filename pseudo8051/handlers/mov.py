"""
handlers/mov.py — MOV / MOVX / MOVC / PUSH / POP / XCH / XCHD handlers.
"""

from typing import List

import ida_ua
import idc

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.ir.hir         import HIRNode, Assign, ExprStmt
from pseudo8051.ir.expr        import Reg, Name, Const, XRAMRef, CROMRef, Call
from pseudo8051.constants      import (
    REG_DPTR, PHRASE_AT_DPTR,
    PARAM_REGS, resolve_ext_addr,
)


def _op_expr(insn, n: int, state=None):
    return Operand(insn, n).to_expr(state)


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

    def lift(self, insn, state=None) -> List[HIRNode]:
        op0, op1 = insn.ops[0], insn.ops[1]
        ea = insn.ea
        # MOV DPTR, #imm — annotate with the best available symbol.
        # IDA's print_operand already resolves cross-segment references (e.g.
        # code-segment labels loaded into DPTR for MOVC), so prefer that name.
        # Only fall back to EXT-segment resolution when IDA has no symbol.
        if (op0.type == ida_ua.o_reg and op0.reg == REG_DPTR
                and op1.type == ida_ua.o_imm):
            imm = op1.value & 0xFFFF
            ida_sym = idc.print_operand(ea, 1)   # e.g. "#osd_table" or "#0F999h"
            # Strip the leading '#' that IDA adds to immediate operands
            if ida_sym.startswith('#'):
                ida_sym = ida_sym[1:]
            # Use the IDA symbol if it is a real name (not a plain hex/decimal literal)
            if ida_sym and not ida_sym[0].isdigit() and not ida_sym.startswith('0'):
                sym = ida_sym
            else:
                sym = resolve_ext_addr(imm)
            if sym != hex(imm):
                return [Assign(ea, Reg("DPTR"), Const(imm, alias=sym))]
        dst = _op_expr(insn, 0, state)
        src = _op_expr(insn, 1, state)
        return [Assign(ea, dst, src)]


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

    def lift(self, insn, state=None) -> List[HIRNode]:
        op0, op1 = insn.ops[0], insn.ops[1]
        dptr_val  = state.get("DPTR") if state else None
        ea        = insn.ea

        if op0.type == ida_ua.o_phrase and op0.phrase == PHRASE_AT_DPTR:
            if dptr_val is not None:
                mem_name = resolve_ext_addr(dptr_val)
                alias = mem_name if mem_name != hex(dptr_val) else None
                xram_ref = XRAMRef(Const(dptr_val, alias=alias))
            else:
                xram_ref = XRAMRef(Reg("DPTR"))
            src = _op_expr(insn, 1, state)
            return [Assign(ea, xram_ref, src)]

        if op1.type == ida_ua.o_phrase and op1.phrase == PHRASE_AT_DPTR:
            if dptr_val is not None:
                mem_name = resolve_ext_addr(dptr_val)
                alias = mem_name if mem_name != hex(dptr_val) else None
                xram_ref = XRAMRef(Const(dptr_val, alias=alias))
            else:
                xram_ref = XRAMRef(Reg("DPTR"))
            dst = _op_expr(insn, 0, state)
            return [Assign(ea, dst, xram_ref)]

        dst = _op_expr(insn, 0, state)
        src = _op_expr(insn, 1, state)
        return [Assign(ea, dst, src)]


class MovcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "DPTR"})

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        dst = _op_expr(insn, 0, state)
        src = _op_expr(insn, 1, state)
        return [Assign(insn.ea, dst, src)]


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

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [ExprStmt(insn.ea, Call("push", [_op_expr(insn, 0, state)]))]


class PopHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        s = idc.print_operand(insn.ea, 0)
        if s in PARAM_REGS:
            return frozenset({s})
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        dst = _op_expr(insn, 0, state)
        return [Assign(insn.ea, dst, Call("pop", []))]


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

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [ExprStmt(insn.ea, Call("swap", [Reg("A"), _op_expr(insn, 1, state)]))]


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

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [ExprStmt(insn.ea, Call("swap_nibble", [Reg("A"), _op_expr(insn, 1, state)]))]
