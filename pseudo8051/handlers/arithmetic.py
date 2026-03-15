"""
handlers/arithmetic.py — ADD / ADDC / SUBB / INC / DEC / MUL / DIV / DA.
"""

from typing import List

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.constants      import PARAM_REGS


def _op(insn, n: int, state=None) -> str:
    return Operand(insn, n).render(state)


class AddHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        pr1 = Operand(insn, 1).registers()
        if r1:  u.add(r1)
        else:   u.update(pr1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return [f"A += {_op(insn, 1, state)};"]


class AddcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        if r1: u.add(r1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return [f"A += {_op(insn, 1, state)} + C;"]


class SubbHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        u = {"A"}
        r1 = Operand(insn, 1).reg_name()
        if r1: u.add(r1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return [f"A -= {_op(insn, 1, state)} + C;  /* borrow */"]


class IncHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)}++;"]


class DecHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)}--;"]


class MulHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def lift(self, insn, state=None) -> List[str]:
        return ["{B, A} = A * B;  /* unsigned 16-bit result */"]


class DivHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A", "B"})

    def lift(self, insn, state=None) -> List[str]:
        return ["A = A / B;  B = A % B;"]


class DaHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return ["A = bcd_adjust(A);"]
