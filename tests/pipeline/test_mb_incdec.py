from pseudo8051.ir.hir import ExprStmt, IfGoto, Assign, Label, IfNode
from pseudo8051.ir.expr import Reg, Const, BinOp, UnaryOp, XRAMRef, Name


class TestMultiByteIncDecPattern:
    """Unit tests for MultiByteIncDecPattern (called directly, no IDA context)."""

    def _pat(self):
        from pseudo8051.passes.patterns.mb_incdec import MultiByteIncDecPattern
        return MultiByteIncDecPattern()

    def _noop(self):
        return lambda nodes, reg_map: nodes

    def _xram_unit(self, sym: str, op: str, base_ea: int):
        """Build the 3 HIR nodes for an XRAM-based unit."""
        xr = XRAMRef(Name(sym))
        return [
            Assign(base_ea,     Reg("A"), xr),
            ExprStmt(base_ea + 1, UnaryOp(op, Reg("A"), post=True)),
            Assign(base_ea + 2, xr,       Reg("A")),
        ]

    def _carry_check(self, carry_expr, op: str, label: str, ea: int):
        overflow = Const(0) if op == "++" else Const(0xFF)
        return IfGoto(ea, BinOp(carry_expr, "!=", overflow), label)

    def _xram_reg_map_16bit(self, var_name="var1",
                             sym_lo="EXT_LO", sym_hi="EXT_HI"):
        from pseudo8051.passes.patterns._utils import VarInfo
        lo = VarInfo(f"{var_name}.lo", "uint8_t", (),
                     xram_sym=sym_lo, is_byte_field=True, xram_addr=0xdc8b)
        hi = VarInfo(f"{var_name}.hi", "uint8_t", (),
                     xram_sym=sym_hi, is_byte_field=True, xram_addr=0xdc8a)
        return {f"_byte_{sym_lo}": lo, f"_byte_{sym_hi}": hi}

    def test_mb_inc_xram_16bit(self):
        """8-node XRAM 16-bit increment sequence collapses to 'var1++;'."""
        nodes = (
            self._xram_unit("EXT_LO", "++", 0x1000)
            + [self._carry_check(Reg("A"), "++", "skip", 0x1006)]
            + self._xram_unit("EXT_HI", "++", 0x1008)
            + [Label(0x100e, "skip")]
        )
        reg_map = self._xram_reg_map_16bit()
        result = self._pat().match(nodes, 0, reg_map, self._noop())
        assert result is not None
        replacement, new_i = result
        assert new_i == len(nodes)
        assert len(replacement) == 1
        assert isinstance(replacement[0], ExprStmt)
        assert replacement[0].render()[0][1] == "var1++;"

    def test_mb_dec_xram_16bit(self):
        """16-bit XRAM decrement collapses to 'var1--;'."""
        nodes = (
            self._xram_unit("EXT_LO", "--", 0x1000)
            + [self._carry_check(Reg("A"), "--", "skip", 0x1006)]
            + self._xram_unit("EXT_HI", "--", 0x1008)
            + [Label(0x100e, "skip")]
        )
        reg_map = self._xram_reg_map_16bit()
        result = self._pat().match(nodes, 0, reg_map, self._noop())
        assert result is not None
        assert result[0][0].render()[0][1] == "var1--;"

    def test_mb_inc_xram_16bit_dptr_prefix(self):
        """Same as test_mb_inc_xram_16bit but with DPTR setup nodes included."""
        lo_xr = XRAMRef(Name("EXT_LO"))
        hi_xr = XRAMRef(Name("EXT_HI"))
        nodes = [
            Assign(0x1000, Reg("DPTR"), Name("EXT_LO")),
            Assign(0x1003, Reg("A"), lo_xr),
            ExprStmt(0x1004, UnaryOp("++", Reg("A"), post=True)),
            Assign(0x1005, lo_xr, Reg("A")),
            self._carry_check(Reg("A"), "++", "skip", 0x1006),
            Assign(0x1008, Reg("DPTR"), Name("EXT_HI")),
            Assign(0x100b, Reg("A"), hi_xr),
            ExprStmt(0x100c, UnaryOp("++", Reg("A"), post=True)),
            Assign(0x100d, hi_xr, Reg("A")),
            Label(0x100e, "skip"),
        ]
        reg_map = self._xram_reg_map_16bit()
        result = self._pat().match(nodes, 0, reg_map, self._noop())
        assert result is not None
        replacement, new_i = result
        assert new_i == len(nodes)
        assert len(replacement) == 1
        assert isinstance(replacement[0], ExprStmt)
        assert replacement[0].render()[0][1] == "var1++;"

    def test_mb_inc_reg_16bit(self):
        """4-node register 16-bit increment: R7++;  if (R7 != 0) goto skip;  R6++;  skip: → 'count++;'."""
        from pseudo8051.passes.patterns._utils import VarInfo
        vinfo = VarInfo("count", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}

        nodes = [
            ExprStmt(0x1000, UnaryOp("++", Reg("R7"), post=True)),
            self._carry_check(Reg("R7"), "++", "skip", 0x1002),
            ExprStmt(0x1004, UnaryOp("++", Reg("R6"), post=True)),
            Label(0x1006, "skip"),
        ]
        result = self._pat().match(nodes, 0, reg_map, self._noop())
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        assert len(replacement) == 1
        assert isinstance(replacement[0], ExprStmt)
        assert replacement[0].render()[0][1] == "count++;"

    def test_mb_inc_reg_16bit_no_varinfo(self):
        """Without a VarInfo entry, fall back to concatenated register name 'R7R6'."""
        nodes = [
            ExprStmt(0x1000, UnaryOp("++", Reg("R7"), post=True)),
            self._carry_check(Reg("R7"), "++", "skip", 0x1002),
            ExprStmt(0x1004, UnaryOp("++", Reg("R6"), post=True)),
            Label(0x1006, "skip"),
        ]
        result = self._pat().match(nodes, 0, {}, self._noop())
        assert result is not None
        assert result[0][0].render()[0][1] == "R7R6++;"

    def test_mb_inc_single_unit_no_match(self):
        """A single XRAM unit without a carry check → pattern must not fire."""
        nodes = self._xram_unit("EXT_LO", "++", 0x1000)
        reg_map = self._xram_reg_map_16bit()
        result = self._pat().match(nodes, 0, reg_map, self._noop())
        assert result is None

    def test_mb_inc_two_units_no_label_no_match(self):
        """Two units with carry check but no trailing Label → pattern must not fire."""
        nodes = (
            self._xram_unit("EXT_LO", "++", 0x1000)
            + [self._carry_check(Reg("A"), "++", "skip", 0x1006)]
            + self._xram_unit("EXT_HI", "++", 0x1008)
        )
        reg_map = self._xram_reg_map_16bit()
        result = self._pat().match(nodes, 0, reg_map, self._noop())
        assert result is None

    def test_mb_inc_xram_32bit(self):
        """4-unit XRAM 32-bit increment: 16 nodes → 'var32++;'."""
        from pseudo8051.passes.patterns._utils import VarInfo
        syms = ["EXT_B0", "EXT_B1", "EXT_B2", "EXT_B3"]
        reg_map = {}
        for idx, sym in enumerate(syms):
            vinfo = VarInfo(f"var32.b{idx}", "uint8_t", (),
                            xram_sym=sym, is_byte_field=True, xram_addr=0xdc80 + idx)
            reg_map[f"_byte_{sym}"] = vinfo

        nodes = []
        for k, sym in enumerate(syms):
            nodes += self._xram_unit(sym, "++", 0x1000 + k * 4)
            if k < len(syms) - 1:
                nodes.append(self._carry_check(Reg("A"), "++", "skip32",
                                               0x1003 + k * 4))
        nodes.append(Label(0x1100, "skip32"))

        result = self._pat().match(nodes, 0, reg_map, self._noop())
        assert result is not None
        replacement, new_i = result
        assert new_i == len(nodes)
        assert len(replacement) == 1
        assert isinstance(replacement[0], ExprStmt)
        assert replacement[0].render()[0][1] == "var32++;"


class TestIfNodeIncDecPattern:

    def _noop(self):
        return lambda nodes, reg_map: nodes

    def _xram_unit(self, sym: str, op: str, base_ea: int):
        xr = XRAMRef(Name(sym))
        return [
            Assign(base_ea,     Reg("A"), xr),
            ExprStmt(base_ea + 1, UnaryOp(op, Reg("A"), post=True)),
            Assign(base_ea + 2, xr,       Reg("A")),
        ]

    def _xram_reg_map_16bit(self, var_name="var1",
                             sym_lo="EXT_LO", sym_hi="EXT_HI"):
        from pseudo8051.passes.patterns._utils import VarInfo
        lo = VarInfo(f"{var_name}.lo", "uint8_t", (),
                     xram_sym=sym_lo, is_byte_field=True, xram_addr=0xdc8b)
        hi = VarInfo(f"{var_name}.hi", "uint8_t", (),
                     xram_sym=sym_hi, is_byte_field=True, xram_addr=0xdc8a)
        return {f"_byte_{sym_lo}": lo, f"_byte_{sym_hi}": hi}

    def _pat(self):
        from pseudo8051.passes.patterns.mb_incdec import IfNodeIncDecPattern
        return IfNodeIncDecPattern()

    def _ifnode_carry(self, op, then_nodes, ea=0x1006):
        overflow = Const(0) if op == "++" else Const(0xFF)
        return IfNode(ea, BinOp(Reg("A"), "==", overflow), then_nodes, [])

    def test_16bit_inc_ifnode(self):
        """16-bit XRAM ++ via IfNode carry → 'var1++;'."""
        nodes = (
            self._xram_unit("EXT_LO", "++", 0x1000)
            + [self._ifnode_carry("++", self._xram_unit("EXT_HI", "++", 0x1008))]
        )
        result = self._pat().match(nodes, 0, self._xram_reg_map_16bit(), self._noop())
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        assert replacement[0].render()[0][1] == "var1++;"

    def test_16bit_dec_ifnode(self):
        """16-bit XRAM -- via IfNode carry → 'var1--;'."""
        nodes = (
            self._xram_unit("EXT_LO", "--", 0x1000)
            + [self._ifnode_carry("--", self._xram_unit("EXT_HI", "--", 0x1008))]
        )
        result = self._pat().match(nodes, 0, self._xram_reg_map_16bit(), self._noop())
        assert result is not None
        assert result[0][0].render()[0][1] == "var1--;"

    def test_16bit_inc_ifnode_dptr_prefix(self):
        """DPTR prefix inside IfNode then_nodes is consumed transparently."""
        lo_xr = XRAMRef(Name("EXT_LO"))
        hi_xr = XRAMRef(Name("EXT_HI"))
        then_nodes = [
            Assign(0x1008, Reg("DPTR"), Name("EXT_HI")),
            Assign(0x100b, Reg("A"), hi_xr),
            ExprStmt(0x100c, UnaryOp("++", Reg("A"), post=True)),
            Assign(0x100d, hi_xr, Reg("A")),
        ]
        nodes = (
            self._xram_unit("EXT_LO", "++", 0x1000)
            + [self._ifnode_carry("++", then_nodes)]
        )
        result = self._pat().match(nodes, 0, self._xram_reg_map_16bit(), self._noop())
        assert result is not None
        assert result[0][0].render()[0][1] == "var1++;"

    def test_single_unit_no_ifnode(self):
        """A single XRAM unit not followed by IfNode → no match."""
        nodes = self._xram_unit("EXT_LO", "++", 0x1000)
        result = self._pat().match(nodes, 0, self._xram_reg_map_16bit(), self._noop())
        assert result is None

    def test_ifnode_wrong_cond_no_match(self):
        """IfNode condition is A != 0 (not the carry form) → no match."""
        nodes = (
            self._xram_unit("EXT_LO", "++", 0x1000)
            + [IfNode(0x1006, BinOp(Reg("A"), "!=", Const(0)),
                      self._xram_unit("EXT_HI", "++", 0x1008), [])]
        )
        result = self._pat().match(nodes, 0, self._xram_reg_map_16bit(), self._noop())
        assert result is None

    def test_ifnode_nonempty_else_no_match(self):
        """IfNode with non-empty else_nodes → no match."""
        nodes = (
            self._xram_unit("EXT_LO", "++", 0x1000)
            + [IfNode(0x1006, BinOp(Reg("A"), "==", Const(0)),
                      self._xram_unit("EXT_HI", "++", 0x1008),
                      [ExprStmt(0x1010, Name("extra"))])]
        )
        result = self._pat().match(nodes, 0, self._xram_reg_map_16bit(), self._noop())
        assert result is None

    def test_no_varinfo_no_match(self):
        """Without reg_map byte-field entries, pattern returns None."""
        nodes = (
            self._xram_unit("EXT_LO", "++", 0x1000)
            + [self._ifnode_carry("++", self._xram_unit("EXT_HI", "++", 0x1008))]
        )
        result = self._pat().match(nodes, 0, {}, self._noop())
        assert result is None
