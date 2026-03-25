from pseudo8051.prototypes import _parse_type_string, _regs_from_loc_str


class TestProtoStringParser:
    """Tests for _parse_type_string and _regs_from_loc_str (no IDA needed)."""

    def test_regs_from_loc_str_colon_pair(self):
        """R6:R7 → ("R6", "R7")  and  R2:R1 → ("R2", "R1")."""
        assert _regs_from_loc_str("R6:R7") == ("R6", "R7")
        assert _regs_from_loc_str("R2:R1") == ("R2", "R1")

    def test_regs_from_loc_str_concatenated(self):
        """R6R7 (no colon) also parses correctly."""
        assert _regs_from_loc_str("R6R7") == ("R6", "R7")
        assert _regs_from_loc_str("R4R5R6R7") == ("R4", "R5", "R6", "R7")

    def test_regs_from_loc_str_colon_quad(self):
        """R4:R5:R6:R7 (colon-separated four registers) works for 32-bit types."""
        assert _regs_from_loc_str("R4:R5:R6:R7") == ("R4", "R5", "R6", "R7")
        assert _regs_from_loc_str("R0:R1:R2:R3") == ("R0", "R1", "R2", "R3")

    def test_regs_from_loc_str_single(self):
        assert _regs_from_loc_str("R3") == ("R3",)
        assert _regs_from_loc_str("A")  == ("A",)

    def test_parse_usercall_with_colon_regs(self):
        """
        void __usercall(__int16 count@<R6:R7>, __int8 src_type@<R3>, __int16 src@<R2:R1>)
        → 3 params with correct types and register tuples.
        """
        proto = _parse_type_string(
            "void __usercall("
            "__int16 count@<R6:R7>, "
            "__int8 src_type@<R3>, "
            "__int16 src@<R2:R1>)",
            "code_7_2D51"
        )
        assert proto is not None
        assert proto.return_type == "void"
        assert len(proto.params) == 3

        count, src_type, src = proto.params
        assert count.name == "count"
        assert count.type == "int16_t"
        assert count.regs == ("R6", "R7")

        assert src_type.name == "src_type"
        assert src_type.type == "int8_t"
        assert src_type.regs == ("R3",)

        assert src.name == "src"
        assert src.type == "int16_t"
        assert src.regs == ("R2", "R1")

    def test_parse_usercall_with_return_reg(self):
        """__int16@<R6R7> __usercall(...) → return_regs=('R6','R7')."""
        proto = _parse_type_string(
            "__int16@<R6R7> __usercall(__int8 x@<R7>)",
            "some_func"
        )
        assert proto is not None
        assert proto.return_type == "int16_t"
        assert proto.return_regs == ("R6", "R7")
        assert proto.params[0].regs == ("R7",)

    def test_parse_usercall_32bit_colon_regs(self):
        """__int32 param with R4:R5:R6:R7 colon annotation → 4-tuple."""
        proto = _parse_type_string(
            "void __usercall(__int32 val@<R4:R5:R6:R7>)",
            "f"
        )
        assert proto is not None
        assert proto.params[0].type == "int32_t"
        assert proto.params[0].regs == ("R4", "R5", "R6", "R7")

    def test_parse_semicolon_stripped(self):
        """Trailing semicolon in the type string is accepted."""
        proto = _parse_type_string(
            "void __usercall(__int8 x@<R7>);",
            "f"
        )
        assert proto is not None
        assert proto.params[0].regs == ("R7",)
