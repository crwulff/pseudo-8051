"""
handlers/logic.py — ANL / ORL / XRL / CLR / SETB / CPL / RL/RLC/RR/RRC / SWAP.
"""

from typing import List

import ida_ua

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.ir.hir         import HIRNode, Assign, CompoundAssign, ExprStmt
from pseudo8051.ir.expr        import Reg, Const, UnaryOp, BinOp, Call, Rot9Op
from pseudo8051.constants      import PARAM_REGS


def _op_expr(insn, n: int, state=None):
    return Operand(insn, n).to_expr(state)


class AnlHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        pr1 = Operand(insn, 1).registers()
        if r1:  u.add(r1)
        else:   u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [CompoundAssign(insn.ea, _op_expr(insn, 0, state), "&=", _op_expr(insn, 1, state))]


class OrlHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        pr1 = Operand(insn, 1).registers()
        if r1:  u.add(r1)
        else:   u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [CompoundAssign(insn.ea, _op_expr(insn, 0, state), "|=", _op_expr(insn, 1, state))]


class XrlHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        pr1 = Operand(insn, 1).registers()
        if r1:  u.add(r1)
        else:   u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [CompoundAssign(insn.ea, _op_expr(insn, 0, state), "^=", _op_expr(insn, 1, state))]


class ClrHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [Assign(insn.ea, _op_expr(insn, 0, state), Const(0))]


class SetbHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        if r0:
            return frozenset({r0})
        # C is not in PARAM_REGS so reg_name() returns None for SETB C.
        # Only o_reg operands can be C; bit-address operands are o_mem.
        if insn.ops[0].type == ida_ua.o_reg:
            if Operand(insn, 0).render() == "C":
                return frozenset({"C"})
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [Assign(insn.ea, _op_expr(insn, 0, state), Const(1))]


class CplHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        dst = _op_expr(insn, 0, state)
        return [Assign(insn.ea, dst, UnaryOp("~", dst))]


class RlHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        a = Reg("A")
        return [Assign(insn.ea, a,
                       BinOp(BinOp(a, "<<", Const(1)), "|", BinOp(a, ">>", Const(7))))]


class RlcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "C"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A", "C"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [Assign(insn.ea, Reg("A"), Rot9Op("rol9", Reg("A"), Reg("C")))]


class RrHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        a = Reg("A")
        return [Assign(insn.ea, a,
                       BinOp(BinOp(a, ">>", Const(1)), "|", BinOp(a, "<<", Const(7))))]


class RrcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "C"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A", "C"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [Assign(insn.ea, Reg("A"), Rot9Op("ror9", Reg("A"), Reg("C")))]


class SwapHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        # A = ((A & 0x0F) << 4) | ((A >> 4) & 0x0F)
        from pseudo8051.ir.expr import BinOp, Const
        lo_nibble = BinOp(BinOp(Reg("A"), "&", Const(0x0F)), "<<", Const(4))
        hi_nibble = BinOp(BinOp(Reg("A"), ">>", Const(4)), "&", Const(0x0F))
        return [Assign(insn.ea, Reg("A"), BinOp(lo_nibble, "|", hi_nibble))]
