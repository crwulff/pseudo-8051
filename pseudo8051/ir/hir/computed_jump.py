"""
ir/hir/computed_jump.py — ComputedJump node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode


class ComputedJump(HIRNode):
    """JMP @A+DPTR — computed table-jump placeholder; replaced by SwitchNode."""

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}JMP @A+DPTR")]
