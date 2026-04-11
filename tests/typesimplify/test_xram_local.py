from unittest.mock import patch

from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto
from pseudo8051.locals import LocalVar
from pseudo8051.ir.hir import VarDecl

from ..helpers import make_single_block_func


class TestXramLocalDecl:
    def test_local_declaration_prepended(self):
        """Declared XRAM local → VarDecl prepended to hir."""
        PROTOTYPES["fn"] = FuncProto(return_type="void", params=[])
        func = make_single_block_func("fn", ["XRAM[EXT_DC8A] = R7;"])

        local = LocalVar(name="var1", type="uint16_t", addr=0xdc8a)

        with patch("pseudo8051.locals.get_locals", return_value=[local]), \
             patch("pseudo8051.constants.resolve_ext_addr",
                   side_effect=lambda a: f"EXT_{a:04X}"):
            TypeAwareSimplifier().run(func)

        decl_nodes = [n for n in func.hir if isinstance(n, VarDecl)
                      and n.name == "var1"]
        assert len(decl_nodes) == 1
        rendered = decl_nodes[0].render(0)[0][1]
        assert rendered.startswith("uint16_t var1;")
        assert "EXT_DC8A" in rendered
        assert "0xdc8a" in rendered
        assert "0xdc8b" in rendered  # end of range for uint16_t

    def test_local_decl_at_start(self):
        """Local variable declaration is the first node in hir."""
        PROTOTYPES["fn2"] = FuncProto(return_type="void", params=[])
        func = make_single_block_func("fn2", ["return;"])

        local = LocalVar(name="cnt", type="uint8_t", addr=0xdc00)

        with patch("pseudo8051.locals.get_locals", return_value=[local]), \
             patch("pseudo8051.constants.resolve_ext_addr",
                   return_value="EXT_DC00"):
            TypeAwareSimplifier().run(func)

        assert isinstance(func.hir[0], VarDecl)
        assert func.hir[0].name == "cnt"
        assert func.hir[0].type_str == "uint8_t"
