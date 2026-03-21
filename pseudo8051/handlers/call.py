"""
handlers/call.py — LCALL / ACALL / CALL / RET / RETI / NOP.
"""

from typing import List

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.ir.hir         import HIRNode, Assign, ExprStmt, ReturnStmt
from pseudo8051.ir.expr        import Reg, Name, RegGroup, Call
from pseudo8051.constants      import PARAM_REGS


def _op_expr(insn, n: int, state=None):
    return Operand(insn, n).to_expr(state)


class LcallHandler(MnemonicHandler):
    """LCALL / ACALL / CALL — subroutine call."""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        # Conservative: called function may clobber any tracked register.
        return frozenset(PARAM_REGS)

    def lift(self, insn, state=None) -> List[HIRNode]:
        from pseudo8051.prototypes import get_proto, return_expr, param_regs
        callee_expr = _op_expr(insn, 0, state)
        callee = callee_expr.name if isinstance(callee_expr, Name) else callee_expr.render()
        proto  = get_proto(callee)
        ea     = insn.ea
        if proto:
            p_regs = param_regs(proto)
            args = []
            for p, regs in zip(proto.params, p_regs):
                if regs:
                    args.append(Name("".join(regs)))
                else:
                    args.append(Name(p.name))
            call_node = Call(callee, args)
            if proto.return_type != "void" and proto.return_regs:
                ret_regs = proto.return_regs
                if len(ret_regs) == 1:
                    lhs = Reg(ret_regs[0])
                else:
                    lhs = RegGroup(ret_regs)
                return [Assign(ea, lhs, call_node)]
            return [ExprStmt(ea, call_node)]
        return [ExprStmt(ea, Call(callee, []))]


class RetHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})   # conservative: A is the common return register

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        # Return expression is filled in by Function._fix_return_statements()
        # once the prototype is known; default to bare return.
        return [ReturnStmt(insn.ea, None)]


class RetiHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        from pseudo8051.ir.hir import Statement
        return [Statement(insn.ea, "return;  /* interrupt return */")]


class NopHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        return []
