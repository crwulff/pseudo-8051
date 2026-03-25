from pseudo8051.passes.patterns._utils import VarInfo, _replace_single_regs, _replace_pairs


class TestReplaceSingleRegs:
    def test_rhs_only_substitution(self):
        """LHS stays raw; RHS gets param name substituted."""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",), is_param=True)}
        result = _replace_single_regs("R7 = R7;", reg_map)
        assert result == "R7 = H;"

    def test_non_param_not_substituted(self):
        """Entries without is_param=True are left alone."""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",))}  # is_param defaults False
        result = _replace_single_regs("R7 = R7;", reg_map)
        assert result == "R7 = R7;"

    def test_no_assignment_substitutes_everywhere(self):
        """In a non-assignment expression, all occurrences are substituted."""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",), is_param=True)}
        result = _replace_single_regs("return R7;", reg_map)
        assert result == "return H;"

    def test_xram_entry_skipped(self):
        """XRAM-local VarInfo (xram_sym set) must not be substituted."""
        vinfo = VarInfo("var1", "uint16_t", (), xram_sym="EXT_DC8A")
        reg_map = {"R7": vinfo}   # degenerate: xram entry keyed by reg name
        result = _replace_single_regs("foo = R7;", reg_map)
        assert result == "foo = R7;"


class TestReplacePairs:
    def test_basic_pair_substitution(self):
        """R6R7 replaced by variable name in expression."""
        vinfo = VarInfo("myvar", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        result = _replace_pairs("foo(R6R7)", reg_map)
        assert result == "foo(myvar)"

    def test_xram_pair_skipped(self):
        """XRAM-local pair (xram_sym set) is not substituted."""
        vinfo = VarInfo("local1", "uint16_t", ("R6", "R7"),
                        xram_sym="EXT_DC00")
        reg_map = {"R6R7": vinfo}
        result = _replace_pairs("foo(R6R7)", reg_map)
        assert result == "foo(R6R7)"

    def test_longer_key_wins(self):
        """4-byte pair replaced before 2-byte pair (longest-first)."""
        v4 = VarInfo("quad", "uint32_t", ("R4", "R5", "R6", "R7"))
        v2 = VarInfo("pair", "uint16_t", ("R6", "R7"))
        reg_map = {
            "R4R5R6R7": v4, "R4": v4, "R5": v4, "R6R7": v2,
        }
        result = _replace_pairs("f(R4R5R6R7)", reg_map)
        assert result == "f(quad)"

    def test_single_reg_key_ignored(self):
        """Keys of length ≤ 2 (single registers like 'R7') are skipped."""
        vinfo = VarInfo("H", "uint8_t", ("R7",), is_param=True)
        reg_map = {"R7": vinfo}
        result = _replace_pairs("foo(R7)", reg_map)
        assert result == "foo(R7)"


class TestReplaceSingleRegsMultiByte:
    def _make_reg_map(self):
        """2-byte param 'count' in R6:R7."""
        vinfo = VarInfo("count", "uint16_t", ("R6", "R7"), is_param=True)
        return {"R6": vinfo, "R7": vinfo, "R6R7": vinfo}

    def test_hi_byte_gets_suffix(self):
        """R6 → count.hi (high byte of 2-byte param)."""
        result = _replace_single_regs("foo = R6;", self._make_reg_map())
        assert result == "foo = count.hi;"

    def test_lo_byte_gets_suffix(self):
        """R7 → count.lo (low byte of 2-byte param)."""
        result = _replace_single_regs("foo = R7;", self._make_reg_map())
        assert result == "foo = count.lo;"

    def test_single_byte_param_no_suffix(self):
        """Single-byte param gets no suffix."""
        vinfo = VarInfo("src_type", "uint8_t", ("R3",), is_param=True)
        reg_map = {"R3": vinfo}
        result = _replace_single_regs("foo = R3;", reg_map)
        assert result == "foo = src_type;"

    def test_four_byte_param_suffixes(self):
        """4-byte param gets b0..b3 suffixes."""
        vinfo = VarInfo("big", "uint32_t", ("R4", "R5", "R6", "R7"), is_param=True)
        reg_map = {"R4": vinfo, "R5": vinfo, "R6": vinfo, "R7": vinfo}
        assert _replace_single_regs("x = R4;", reg_map) == "x = big.b0;"
        assert _replace_single_regs("x = R7;", reg_map) == "x = big.b3;"
