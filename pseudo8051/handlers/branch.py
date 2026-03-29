"""
handlers/branch.py — SJMP / LJMP / AJMP / JZ / JNZ / JC / JNC / JB / JNB /
                     JBC / CJNE / DJNZ.
"""

from typing import List

import ida_funcs
import ida_ua

from pseudo8051.ir.instruction import MnemonicHandler
from pseudo8051.ir.operand     import Operand
from pseudo8051.ir.hir         import HIRNode, Assign, ExprStmt, ReturnStmt, IfGoto, GotoStatement
from pseudo8051.ir.expr        import Reg, Const, Name, BinOp, UnaryOp, Call
from pseudo8051.constants      import PARAM_REGS


def _op_expr(insn, n: int, state=None):
    return Operand(insn, n).to_expr(state)


def _label_str(insn, n: int, state=None) -> str:
    """Render a branch-target operand as a label string."""
    return Operand(insn, n).render(state)


def _tail_call_target(insn) -> str:
    """
    If the jump target is the entry of a *different* IDA function, return
    its name (tail call).  Otherwise return None (normal intra-function jump).
    """
    op = insn.ops[0]
    if op.type not in (ida_ua.o_near, ida_ua.o_far):
        return None
    page_base  = insn.ea & ~0xFFFF
    target_ea  = page_base | (op.addr & 0xFFFF)
    target_fn  = ida_funcs.get_func(target_ea)
    current_fn = ida_funcs.get_func(insn.ea)
    if (target_fn is not None
            and target_fn.start_ea == target_ea       # target IS a function entry
            and (current_fn is None
                 or target_fn.start_ea != current_fn.start_ea)):
        return ida_funcs.get_func_name(target_ea) or hex(target_ea)
    return None


class SjmpHandler(MnemonicHandler):
    """SJMP / LJMP / AJMP — unconditional jump."""

    def use(self, insn) -> frozenset:
        tail = _tail_call_target(insn)
        if tail:
            from pseudo8051.prototypes import get_proto, param_regs
            proto = get_proto(tail)
            if proto:
                regs: set = set()
                for r_tuple in param_regs(proto):
                    regs.update(r_tuple)
                return frozenset(regs)
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        tail = _tail_call_target(insn)
        if tail:
            from pseudo8051.prototypes import get_proto, param_regs
            proto = get_proto(tail)
            if proto:
                regs_list = param_regs(proto)
                args = [Name("".join(r)) if r else Name("?") for r in regs_list]
                call_node = Call(tail, args)
                if proto.return_type != "void":
                    return [ReturnStmt(insn.ea, call_node)]
                return [ExprStmt(insn.ea, call_node), ReturnStmt(insn.ea, None)]
            return [ExprStmt(insn.ea, Call(tail, [])), ReturnStmt(insn.ea, None)]
        label = _label_str(insn, 0, state)
        return [GotoStatement(insn.ea, label)]


class JzHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        label = _label_str(insn, 0, state)
        return [IfGoto(insn.ea, BinOp(Reg("A"), "==", Const(0)), label)]


class JnzHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset({"A"})

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        label = _label_str(insn, 0, state)
        return [IfGoto(insn.ea, BinOp(Reg("A"), "!=", Const(0)), label)]


class JcHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        label = _label_str(insn, 0, state)
        return [IfGoto(insn.ea, Reg("C"), label)]


class JncHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        label = _label_str(insn, 0, state)
        return [IfGoto(insn.ea, UnaryOp("!", Reg("C")), label)]


class JbHandler(MnemonicHandler):
    """JB bit, label — if (bit) goto label"""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        bit   = _op_expr(insn, 0, state)
        label = _label_str(insn, 1, state)
        return [IfGoto(insn.ea, bit, label)]


class JnbHandler(MnemonicHandler):
    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        bit   = _op_expr(insn, 0, state)
        label = _label_str(insn, 1, state)
        return [IfGoto(insn.ea, UnaryOp("!", bit), label)]


class JbcHandler(MnemonicHandler):
    """JBC bit, label — if (bit) { bit=0; goto label; }"""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        bit   = _op_expr(insn, 0, state)
        label = _label_str(insn, 1, state)
        return [
            IfGoto(insn.ea, bit, label),
            Assign(insn.ea, bit, Const(0)),
            GotoStatement(insn.ea, label),
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

    def lift(self, insn, state=None) -> List[HIRNode]:
        label = _label_str(insn, 2, state)
        cond  = BinOp(_op_expr(insn, 0, state), "!=", _op_expr(insn, 1, state))
        return [IfGoto(insn.ea, cond, label)]


class JmpAtADptrHandler(MnemonicHandler):
    """JMP @A+DPTR — indirect computed jump; produces a placeholder Statement.
    JmpTableStructurer will replace it with a SwitchNode."""

    def use(self, insn) -> frozenset:
        return frozenset({"A", "DPTR"})

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        from pseudo8051.ir.hir import Statement
        return [Statement(insn.ea, "JMP @A+DPTR")]


class DjnzHandler(MnemonicHandler):
    """DJNZ Rn, label — decrement and jump if not zero."""

    def use(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def defs(self, insn) -> frozenset:
        r0 = Operand(insn, 0).reg_name()
        return frozenset({r0}) if r0 else frozenset()

    def lift(self, insn, state=None) -> List[HIRNode]:
        dst   = _op_expr(insn, 0, state)
        label = _label_str(insn, 1, state)
        # if (--dst != 0) goto label;
        cond  = BinOp(UnaryOp("--", dst, post=False), "!=", Const(0))
        return [IfGoto(insn.ea, cond, label)]
