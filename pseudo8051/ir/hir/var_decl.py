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
        from pseudo8051.passes.patterns._utils import _parse_array_type, _type_bytes
        arr = _parse_array_type(self.type_str)
        if arr is not None:
            elem_type, count = arr
            type_decl = f"{elem_type} {self.name}[{count}]"
            total_bytes = _type_bytes(self.type_str)
        else:
            type_decl = f"{self.type_str} {self.name}"
            total_bytes = _type_bytes(self.type_str)
        if self.xram_addr:
            if total_bytes > 1:
                end_addr = self.xram_addr + total_bytes - 1
                comment = f"  /* {self.xram_sym} @ {hex(self.xram_addr)}-{hex(end_addr)} */"
            else:
                comment = f"  /* {self.xram_sym} @ {hex(self.xram_addr)} */"
        else:
            comment = ""
        return [(self.ea, f"{self._ind(indent)}{type_decl};{comment}")]

    def ann_lines(self) -> List[str]:
        out = ["VarDecl", f"  type: {self.type_str!r}", f"  name: {self.name!r}"]
        if self.xram_sym:
            out.append(f"  xram: {self.xram_sym!r}")
        return out
