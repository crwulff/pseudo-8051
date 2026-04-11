"""
Tests for XRAM array local variable support.

Covers:
  - _parse_array_type helper
  - VarDecl rendering for array types
  - Static element access: XRAM[EXT_DC33] → foo[2]
  - Dynamic indexed access: XRAM[EXT_DC31 + R6R7] → foo[R6R7]
  - Dynamic indexed access with Const base: XRAM[0xdc31 + R6R7] → foo[R6R7]
  - _augment_with_local_vars registers array entries
"""

from unittest.mock import patch

from pseudo8051.passes.patterns._utils import (
    _parse_array_type, _subst_xram_in_expr, VarInfo,
)
from pseudo8051.ir.expr import (
    Name, Const, XRAMRef, BinOp, ArrayRef, RegGroup,
)
from pseudo8051.ir.hir import VarDecl
from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto
from pseudo8051.locals import LocalVar

from ..helpers import make_single_block_func


# ── _parse_array_type ─────────────────────────────────────────────────────────

class TestParseArrayType:
    def test_uint8_array(self):
        assert _parse_array_type("uint8_t[6]") == ("uint8_t", 6)

    def test_uint16_array(self):
        assert _parse_array_type("uint16_t[3]") == ("uint16_t", 3)

    def test_non_array(self):
        assert _parse_array_type("uint8_t") is None

    def test_non_array_uint16(self):
        assert _parse_array_type("uint16_t") is None

    def test_non_array_struct(self):
        assert _parse_array_type("my_struct") is None

    def test_whitespace_stripped(self):
        assert _parse_array_type(" uint8_t[4] ") == ("uint8_t", 4)


# ── VarDecl rendering ─────────────────────────────────────────────────────────

class TestVarDeclArrayRender:
    def test_array_type_renders_c_style(self):
        """uint8_t[6] foo should render as 'uint8_t foo[6];' with address range."""
        decl = VarDecl(0x1000, "uint8_t[6]", "buf",
                       xram_sym="EXT_DC31", xram_addr=0xdc31)
        rendered = decl.render(0)[0][1]
        assert rendered.startswith("uint8_t buf[6];")
        assert "EXT_DC31" in rendered
        assert "0xdc31" in rendered
        assert "0xdc36" in rendered  # 0xdc31 + 6 - 1

    def test_scalar_type_renders_unchanged(self):
        """uint16_t count should still render as 'uint16_t count;'."""
        decl = VarDecl(0x1000, "uint16_t", "count",
                       xram_sym="EXT_DC8A", xram_addr=0xdc8a)
        rendered = decl.render(0)[0][1]
        assert rendered.startswith("uint16_t count;")


# ── _subst_xram_in_expr ───────────────────────────────────────────────────────

def _make_array_reg_map(base_addr: int, name: str, elem_type: str,
                        count: int, base_sym: str) -> dict:
    """Build a reg_map as _augment_with_local_vars would produce for an array."""
    from pseudo8051.passes.patterns._utils import _type_bytes
    elem_bytes = _type_bytes(elem_type)
    reg_map = {}
    reg_map[base_sym] = VarInfo(name, f"{elem_type}[{count}]", (),
                                xram_sym=base_sym, xram_addr=base_addr,
                                array_size=count, elem_type=elem_type)
    for k in range(count):
        elem_addr = base_addr + k * elem_bytes
        elem_sym = f"EXT_{elem_addr:04X}"
        reg_map[f"_arr_{elem_sym}"] = VarInfo(f"{name}[{k}]", elem_type, (),
                                               xram_sym=elem_sym,
                                               is_byte_field=True)
    return reg_map


class TestSubstXramArray:

    def test_static_elem0_access(self):
        """XRAM[EXT_DC31] → foo[0] (element 0 of array at 0xdc31)."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint8_t", 6, "EXT_DC31")
        expr = XRAMRef(Name("EXT_DC31"))
        result = _subst_xram_in_expr(expr, reg_map)
        assert isinstance(result, Name)
        assert result.name == "foo[0]"

    def test_static_elem2_access(self):
        """XRAM[EXT_DC33] → foo[2] (element 2 of array at 0xdc31)."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint8_t", 6, "EXT_DC31")
        expr = XRAMRef(Name("EXT_DC33"))
        result = _subst_xram_in_expr(expr, reg_map)
        assert isinstance(result, Name)
        assert result.name == "foo[2]"

    def test_static_last_elem_access(self):
        """XRAM[EXT_DC36] → foo[5] (last element of uint8_t[6])."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint8_t", 6, "EXT_DC31")
        expr = XRAMRef(Name("EXT_DC36"))
        result = _subst_xram_in_expr(expr, reg_map)
        assert isinstance(result, Name)
        assert result.name == "foo[5]"

    def test_dynamic_index_sym_base(self):
        """XRAM[EXT_DC31 + R6R7] → foo[R6R7] (sym base + register index)."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint8_t", 6, "EXT_DC31")
        idx = RegGroup(("R6", "R7"))
        expr = XRAMRef(BinOp(Name("EXT_DC31"), "+", idx))
        result = _subst_xram_in_expr(expr, reg_map)
        assert isinstance(result, ArrayRef)
        assert result.base == Name("foo")
        assert result.index == idx
        assert result.render() == "foo[R6R7]"

    def test_dynamic_index_const_base(self):
        """XRAM[0xdc31 + R6R7] → foo[R6R7] (const base + register index)."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint8_t", 6, "EXT_DC31")
        idx = RegGroup(("R6", "R7"))
        expr = XRAMRef(BinOp(Const(0xdc31), "+", idx))
        result = _subst_xram_in_expr(expr, reg_map)
        assert isinstance(result, ArrayRef)
        assert result.base == Name("foo")
        assert result.index == idx

    def test_dynamic_index_multibyte_elem_not_simplified(self):
        """XRAM[base + R7] with uint16_t elements is not simplified (would need /2)."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint16_t", 3, "EXT_DC31")
        idx = RegGroup(("R7",))
        expr = XRAMRef(BinOp(Name("EXT_DC31"), "+", idx))
        result = _subst_xram_in_expr(expr, reg_map)
        # Not converted to ArrayRef — raw XRAMRef remains (or may hit sym_map as foo[0])
        # The key thing: no ArrayRef should appear for multi-byte element arrays
        assert not isinstance(result, ArrayRef)

    def test_non_array_sym_unchanged(self):
        """XRAM[EXT_DC40] with no local declared is left alone."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint8_t", 6, "EXT_DC31")
        expr = XRAMRef(Name("EXT_DC40"))
        result = _subst_xram_in_expr(expr, reg_map)
        assert result == expr

    def test_scalar_local_still_works(self):
        """Scalar locals in the same reg_map are unaffected."""
        reg_map = _make_array_reg_map(0xdc31, "foo", "uint8_t", 6, "EXT_DC31")
        reg_map["EXT_DC8A"] = VarInfo("count", "uint16_t", (),
                                      xram_sym="EXT_DC8A", xram_addr=0xdc8a)
        expr = XRAMRef(Name("EXT_DC8A"))
        result = _subst_xram_in_expr(expr, reg_map)
        assert isinstance(result, Name)
        assert result.name == "count"


# ── ArrayRef expr node ────────────────────────────────────────────────────────

class TestArrayRefExpr:
    def test_render_const_index(self):
        assert ArrayRef(Name("foo"), Const(2)).render() == "foo[2]"

    def test_render_reg_index(self):
        assert ArrayRef(Name("foo"), RegGroup(("R6", "R7"))).render() == "foo[R6R7]"

    def test_equality(self):
        a = ArrayRef(Name("foo"), Const(0))
        b = ArrayRef(Name("foo"), Const(0))
        assert a == b

    def test_inequality_different_index(self):
        assert ArrayRef(Name("foo"), Const(0)) != ArrayRef(Name("foo"), Const(1))

    def test_children(self):
        idx = Const(3)
        base = Name("arr")
        ar = ArrayRef(base, idx)
        assert ar.children() == [base, idx]

    def test_rebuild(self):
        ar = ArrayRef(Name("foo"), Const(0))
        ar2 = ar.rebuild([Name("bar"), Const(5)])
        assert ar2.render() == "bar[5]"


# ── Integration: TypeAwareSimplifier with array local ────────────────────────

class TestArrayLocalIntegration:
    def test_array_vardecl_rendered_c_style(self):
        """TypeAwareSimplifier produces 'uint8_t buf[6];' VarDecl for array locals."""
        PROTOTYPES["fn_arr"] = FuncProto(return_type="void", params=[])
        func = make_single_block_func("fn_arr", ["return;"])

        local = LocalVar(name="buf", type="uint8_t[6]", addr=0xdc31)

        with patch("pseudo8051.locals.get_locals", return_value=[local]), \
             patch("pseudo8051.constants.resolve_ext_addr",
                   side_effect=lambda a: f"EXT_{a:04X}"):
            TypeAwareSimplifier().run(func)

        decl_nodes = [n for n in func.hir if isinstance(n, VarDecl) and n.name == "buf"]
        assert len(decl_nodes) == 1
        rendered = decl_nodes[0].render(0)[0][1]
        assert "uint8_t buf[6];" in rendered
        assert "EXT_DC31" in rendered
