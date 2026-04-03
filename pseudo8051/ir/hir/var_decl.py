"""
ir/hir/var_decl.py — VarDecl node.
"""

from typing import List, Optional, Tuple

from pseudo8051.ir.hir._base import HIRNode


class VarDecl(HIRNode):
    """type name;  — forward declaration of a local variable."""

    def __init__(self, ea: int, type_str: str, name: str,
                 xram_sym: Optional[str] = None, xram_addr: Optional[int] = None):
        super().__init__(ea)
        self.type_str  = type_str
        self.name      = name
        self.xram_sym  = xram_sym
        self.xram_addr = xram_addr

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        comment = (f"  /* {self.xram_sym} @ {hex(self.xram_addr)} */"
                   if self.xram_addr else "")
        return [(self.ea, f"{self._ind(indent)}{self.type_str} {self.name};{comment}")]

    def ann_lines(self) -> List[str]:
        out = ["VarDecl", f"  type: {self.type_str!r}", f"  name: {self.name!r}"]
        if self.xram_sym:
            out.append(f"  xram: {self.xram_sym!r}")
        return out
