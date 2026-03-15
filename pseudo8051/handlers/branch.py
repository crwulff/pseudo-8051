"""
handlers/branch.py — SJMP / LJMP / AJMP / JZ / JNZ / JC / JNC / JB / JNB /
                     JBC / CJNE / DJNZ.
"""

from typing import List

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.constants      import PARAM_REGS


def _op(insn, n: int, state=None) -> str:
    return Operand(insn, n).render(state)


class SjmpHandler(MnemonicHandler):
    """SJMP / LJMP / AJMP — unconditional jump."""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"goto {_op(insn, 0, state)};"]


class JzHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"if (A == 0) goto {_op(insn, 0, state)};"]


class JnzHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"if (A != 0) goto {_op(insn, 0, state)};"]


class JcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"if (C) goto {_op(insn, 0, state)};"]


class JncHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"if (!C) goto {_op(insn, 0, state)};"]


class JbHandler(MnemonicHandler):
    """JB bit, label — if (bit) goto label"""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"if ({_op(insn, 0, state)}) goto {_op(insn, 1, state)};"]


class JnbHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        return [f"if (!{_op(insn, 0, state)}) goto {_op(insn, 1, state)};"]


class JbcHandler(MnemonicHandler):
    """JBC bit, label — if (bit) { bit=0; goto label; }"""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        dst = _op(insn, 0, state)
        lbl = _op(insn, 1, state)
        return [
            f"if ({dst}) {{",
            f"    {dst} = 0;",
            f"    goto {lbl};",
            "}",
        ]


class CjneHandler(MnemonicHandler):
    """CJNE op0, op1, label — if (op0 != op1) goto label"""

    def use(self, insn) -> frozenset:
        u = set()
        r0  = Operand(insn, 0).reg_name()
        pr0 = Operand(insn, 0).registers()
        r1  = Operand(insn, 1).reg_name()
        if r0:  u.add(r0)
        else:   u.update(pr0)
        if r1:  u.add(r1)
        return frozenset(u & PARAM_REGS)

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        tgt = _op(insn, 2, state)
        return [f"if ({_op(insn, 0, state)} != {_op(insn, 1, state)}) goto {tgt};"]


class DjnzHandler(MnemonicHandler):
    """DJNZ Rn, label — decrement and jump if not zero."""

    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[str]:
        dst = _op(insn, 0, state)
        lbl = _op(insn, 1, state)
        return [f"if (--{dst} != 0) goto {lbl};"]
