"""
ir/expr/bin_op.py — BinOp: binary operation expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr
from pseudo8051.ir.expr._prec import _BIN_PREC


class BinOp(Expr):
    """
    Binary operation: BinOp(Reg("A"), "+", Const(5)) → "A + 5".

    render() wraps in parens when outer_prec < this node's prec (the
    enclosing context binds more tightly).
    """

    __slots__ = ("lhs", "op", "rhs")

    def __init__(self, lhs: Expr, op: str, rhs: Expr):
        self.lhs = lhs
        self.op  = op
        self.rhs = rhs

    @property
    def prec(self) -> int:
        return _BIN_PREC.get(self.op, 99)

    def render(self, outer_prec: int = 99) -> str:
        my_prec = self.prec
        # Add parens when outer context binds more tightly (lower prec number).
        # Default outer_prec=99 (top-level / no parent) → no parens added.
        need_parens = outer_prec < my_prec
        inner = f"{self.lhs.render(my_prec)} {self.op} {self.rhs.render(my_prec)}"
        return f"({inner})" if need_parens else inner

    def children(self) -> List[Expr]:
        return [self.lhs, self.rhs]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return BinOp(new_children[0], self.op, new_children[1])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, BinOp)
                and self.lhs == other.lhs
                and self.op  == other.op
                and self.rhs == other.rhs)

    def __hash__(self) -> int:
        return hash(("BinOp", self.lhs, self.op, self.rhs))

    def __repr__(self) -> str:
        return f"BinOp({self.lhs!r}, {self.op!r}, {self.rhs!r})"
