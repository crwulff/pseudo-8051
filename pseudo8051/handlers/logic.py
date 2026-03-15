"""
handlers/logic.py — ANL / ORL / XRL / CLR / SETB / CPL / RL/RLC/RR/RRC / SWAP.
"""

from typing import List

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.constants      import PARAM_REGS


def _op(insn, n: int, state=None) -> str:
    return Operand(insn, n).render(state)


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

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)} &= {_op(insn, 1, state)};"]


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

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)} |= {_op(insn, 1, state)};"]


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

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)} ^= {_op(insn, 1, state)};"]


class ClrHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)} = 0;"]


class SetbHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)} = 1;"]


class CplHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[str]:
        dst = _op(insn, 0, state)
        return [f"{dst} = ~{dst};"]


class RlHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return ["A = rol8(A);"]


class RlcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return ["A = rol9(A, C);  /* through carry */"]


class RrHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return ["A = ror8(A);"]


class RrcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return ["A = ror9(A, C);  /* through carry */"]


class SwapHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset({"A"})

    def lift(self, insn, state=None) -> List[str]:
        return ["A = ((A & 0x0F) << 4) | ((A >> 4) & 0x0F);"]
