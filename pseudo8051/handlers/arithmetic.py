"""
handlers/arithmetic.py — ADD / ADDC / SUBB / INC / DEC / MUL / DIV / DA.
"""

from typing import List

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.ir.hir         import HIRNode, Assign, CompoundAssign, ExprStmt
from pseudo8051.ir.expr        import Reg, Const, BinOp, UnaryOp, RegGroup, Call
from pseudo8051.constants      import PARAM_REGS


def _op_expr(insn, n: int, state=None):
    return Operand(insn, n).to_expr(state)


class AddHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        pr1 = Operand(insn, 1).registers()
        if r1:  u.add(r1)
        else:   u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        # ADD sets the carry flag C as well as A.
        return frozenset({"A", "C"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [CompoundAssign(insn.ea, Reg("A"), "+=", _op_expr(insn, 1, state))]


class AddcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        if r1: u.add(r1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        # ADDC writes A and also sets the carry flag C as output
        return frozenset({"A", "C"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        rhs = BinOp(_op_expr(insn, 1, state), "+", Reg("C"))
        return [CompoundAssign(insn.ea, Reg("A"), "+=", rhs)]


class SubbHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        if r1: u.add(r1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        # SUBB writes A and also sets the carry flag C as borrow output
        return frozenset({"A", "C"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        rhs = BinOp(_op_expr(insn, 1, state), "+", Reg("C"))
        return [CompoundAssign(insn.ea, Reg("A"), "-=", rhs)]


class IncHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [ExprStmt(insn.ea, UnaryOp("++", _op_expr(insn, 0, state), post=True))]


class DecHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [ExprStmt(insn.ea, UnaryOp("--", _op_expr(insn, 0, state), post=True))]


class MulHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        lhs = RegGroup(("B", "A"), brace=True)
        rhs = BinOp(Reg("A"), "*", Reg("B"))
        return [Assign(insn.ea, lhs, rhs)]


class DivHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        ea = insn.ea
        return [
            Assign(ea, Reg("A"), BinOp(Reg("A"), "/", Reg("B"))),
            Assign(ea, Reg("B"), BinOp(Reg("A"), "%", Reg("B"))),
        ]


class DaHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[HIRNode]:
        return [Assign(insn.ea, Reg("A"), Call("bcd_adjust", [Reg("A")]))]
