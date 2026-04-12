"""
ir/expr/unary_op.py — UnaryOp: unary operation expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr
from pseudo8051.ir.expr._prec import _UNARY_PREC


class UnaryOp(Expr):
    """
    Unary operation.

    UnaryOp("--", Reg("R7"), post=False)  → "--R7"   (pre-decrement)
    UnaryOp("++", Reg("R7"), post=True)   → "R7++"   (post-increment)
    UnaryOp("!",  Reg("C"))               → "!C"
    UnaryOp("~",  Reg("A"))               → "~A"
    """

    __slots__ = ("op", "operand", "post")

    def __init__(self, op: str, operand: Expr, post: bool = False):
        self.op      = op
        self.operand = operand
        self.post    = post

    def render(self, outer_prec: int = 0) -> str:
        # Unary has tightest binding; parens only needed when outer is tighter
        # (which doesn't normally happen for standard C precedence).
        inner = self.operand.render(_UNARY_PREC)
        if self.post:
            return f"{inner}{self.op}"
        return f"{self.op}{inner}"

    def children(self) -> List[Expr]:
        return [self.operand]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return UnaryOp(self.op, new_children[0], self.post)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, UnaryOp)
                and self.op      == other.op
                and self.operand == other.operand
                and self.post    == other.post)

    def __hash__(self) -> int:
        return hash(("UnaryOp", self.op, self.operand, self.post))

    def __repr__(self) -> str:
        return f"UnaryOp({self.op!r}, {self.operand!r}, post={self.post})"
