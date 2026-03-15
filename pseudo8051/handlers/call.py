"""
handlers/call.py — LCALL / ACALL / CALL / RET / RETI / NOP.
"""

from typing import List

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.constants      import PARAM_REGS


def _op(insn, n: int, state=None) -> str:
    return Operand(insn, n).render(state)


class LcallHandler(MnemonicHandler):
    """LCALL / ACALL / CALL — subroutine call."""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        # Conservative: called function may clobber any tracked register.
        return frozenset(PARAM_REGS)

    def lift(self, insn, state=None) -> List[str]:
        return [f"{_op(insn, 0, state)}();"]


class RetHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})   # A holds the return value by convention

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return ["return;  /* A = return value */"]


class RetiHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return ["return;  /* interrupt return */"]


class NopHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return ["/* nop */"]
