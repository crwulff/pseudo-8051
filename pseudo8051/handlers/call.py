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
        from pseudo8051.prototypes import get_proto, return_expr, param_regs
        callee = _op(insn, 0, state)
        proto  = get_proto(callee)
        if proto:
            # Use register pair names as arguments so TypeAwareSimplifier can
            # substitute typed variable names or inline constants/expressions.
            # Fall back to the parameter name only when no registers are assigned.
            p_regs = param_regs(proto)
            args = []
            for p, regs in zip(proto.params, p_regs):
                args.append("".join(regs) if regs else p.name)
            args_str   = ", ".join(args)
            call_expr  = f"{callee}({args_str})"
            if proto.return_type != "void" and proto.return_regs:
                return [f"{return_expr(proto)} = {call_expr};"]
            return [f"{call_expr};"]
        return [f"{callee}();"]


class RetHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})   # conservative: A is the common return register

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        # Return expression is filled in by Function._fix_return_statements()
        # once the prototype is known; default to bare return.
        return ["return;"]


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
