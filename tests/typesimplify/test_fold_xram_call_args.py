"""
tests/typesimplify/test_fold_xram_call_args.py

Tests for _fold_xram_call_args passthrough resolution: when a callee's
XRAM parameter address is also an xram param of the caller, the caller's
parameter name should appear as the call argument even when no explicit
xarg = ... assignment precedes the call.
"""
from unittest.mock import patch, MagicMock
import sys

from pseudo8051.passes.typesimplify._xram_call_args import _fold_xram_call_args
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.ir.hir import Assign, ExprStmt
from pseudo8051.ir.expr import Call, Name, Const, Reg


EA = 0x1000
BADADDR = 0xFFFFFFFF


def _make_call_node(func_name, args=()):
    return ExprStmt(EA, Call(func_name, list(args)))


def _run_with_mocks(nodes, reg_map=None, xps_by_func=None, resolve_map=None):
    """Run _fold_xram_call_args with mocked IDA and xram_params."""
    xps_by_func = xps_by_func or {}
    resolve_map = resolve_map or {}

    mock_ida_name = MagicMock()
    def _get_name_ea(badaddr, name):
        return 0x5000 if name in xps_by_func else BADADDR
    mock_ida_name.get_name_ea.side_effect = _get_name_ea

    mock_idc = MagicMock()
    mock_idc.BADADDR = BADADDR

    # get_xram_params is keyed by EA (0x5000 for any callee)
    def _get_xram_params(ea):
        for xps in xps_by_func.values():
            return xps
        return []

    def _resolve(addr):
        return resolve_map.get(addr, f"EXT_{addr:04X}")

    with patch.dict(sys.modules, {"ida_name": mock_ida_name, "idc": mock_idc}), \
         patch("pseudo8051.xram_params.get_xram_params", side_effect=_get_xram_params), \
         patch("pseudo8051.constants.resolve_ext_addr", side_effect=_resolve):
        return _fold_xram_call_args(list(nodes), reg_map)


class TestFoldXramCallArgsPassthrough:

    def test_explicit_assign_folded(self):
        """Explicit xarg1 = val before call is folded in as usual."""
        from pseudo8051.xram_params import XRAMParam

        vi_caller = VarInfo("xarg1", "uint8_t", (), xram_sym="EXT_DC8A",
                            xram_addr=0xdc8a, is_param=True)
        reg_map = {"EXT_DC8A": vi_caller}

        xarg_assign = Assign(EA - 1, Name("xarg1"), Name("val"))
        call_node = _make_call_node("callee")

        result = _run_with_mocks(
            [xarg_assign, call_node],
            reg_map=reg_map,
            xps_by_func={"callee": [XRAMParam("xarg1", "uint8_t", 0xdc8a)]},
            resolve_map={0xdc8a: "EXT_DC8A"},
        )

        call_nodes = [n for n in result
                      if isinstance(n, ExprStmt) and isinstance(n.expr, Call)]
        assert len(call_nodes) == 1
        assert len(call_nodes[0].expr.args) == 1
        assert call_nodes[0].expr.args[0].render() == "val"

    def test_passthrough_from_caller_xram_param(self):
        """Callee xram param whose address is a caller xram param should use
        the caller's parameter name even without an explicit assignment."""
        from pseudo8051.xram_params import XRAMParam

        vi_caller_xarg3 = VarInfo("xarg3", "uint8_t", (), xram_sym="EXT_DC3E",
                                   xram_addr=0xdc3e, is_param=True)
        reg_map = {"EXT_DC3E": vi_caller_xarg3}

        call_node = _make_call_node("something_osd_5")

        result = _run_with_mocks(
            [call_node],
            reg_map=reg_map,
            xps_by_func={
                "something_osd_5": [XRAMParam("xarg1_callee", "uint8_t", 0xdc3e)]
            },
            resolve_map={0xdc3e: "EXT_DC3E"},
        )

        call_nodes = [n for n in result
                      if isinstance(n, ExprStmt) and isinstance(n.expr, Call)]
        assert len(call_nodes) == 1
        args = call_nodes[0].expr.args
        assert len(args) == 1, f"Expected 1 passthrough arg, got {args!r}"
        assert args[0].render() == "xarg3", (
            f"Expected caller param name 'xarg3', got {args[0].render()!r}"
        )

    def test_mixed_explicit_and_passthrough(self):
        """When callee has two xram params: one set explicitly, one passed
        through from caller's own xram param."""
        from pseudo8051.xram_params import XRAMParam

        vi_caller_xarg4 = VarInfo("xarg4", "uint16_t", (), xram_sym="EXT_DC40",
                                   xram_addr=0xdc40, is_param=True)
        reg_map = {"EXT_DC40": vi_caller_xarg4}

        xarg_assign = Assign(EA - 1, Name("callee_p1"), Name("someVal"))
        call_node = _make_call_node("callee2")

        result = _run_with_mocks(
            [xarg_assign, call_node],
            reg_map=reg_map,
            xps_by_func={
                "callee2": [
                    XRAMParam("callee_p1", "uint8_t", 0xdc3e),   # explicit
                    XRAMParam("callee_p2", "uint16_t", 0xdc40),  # passthrough
                ]
            },
            resolve_map={0xdc3e: "EXT_DC3E", 0xdc40: "EXT_DC40"},
        )

        call_nodes = [n for n in result
                      if isinstance(n, ExprStmt) and isinstance(n.expr, Call)]
        assert len(call_nodes) == 1
        args = call_nodes[0].expr.args
        assert len(args) == 2, f"Expected 2 args, got {args!r}"
        assert args[0].render() == "someVal", f"Expected explicit arg, got {args[0].render()!r}"
        assert args[1].render() == "xarg4",   f"Expected passthrough 'xarg4', got {args[1].render()!r}"

    def test_no_double_fold_on_second_pass(self):
        """Running _fold_xram_call_args twice on a call that already has its
        passthrough args folded in must not append them a second time."""
        from pseudo8051.xram_params import XRAMParam

        vi_caller_xarg3 = VarInfo("xarg3", "uint8_t", (), xram_sym="EXT_DC3E",
                                   xram_addr=0xdc3e, is_param=True)
        reg_map = {"EXT_DC3E": vi_caller_xarg3}

        # Simulate the call after the first pass already folded in xarg3
        call_node = ExprStmt(EA, Call("callee", [Name("xarg3")]))

        result = _run_with_mocks(
            [call_node],
            reg_map=reg_map,
            xps_by_func={
                "callee": [XRAMParam("xarg1_callee", "uint8_t", 0xdc3e)]
            },
            resolve_map={0xdc3e: "EXT_DC3E"},
        )

        call_nodes = [n for n in result
                      if isinstance(n, ExprStmt) and isinstance(n.expr, Call)]
        assert len(call_nodes) == 1
        args = call_nodes[0].expr.args
        assert len(args) == 1, (
            f"Second pass must not re-append passthrough arg, got {args!r}"
        )
        assert args[0].render() == "xarg3"

    def test_no_passthrough_without_is_param(self):
        """A reg_map entry for the same symbol that is NOT is_param=True
        (e.g. a local var) should NOT be used as a passthrough arg."""
        from pseudo8051.xram_params import XRAMParam

        vi_local = VarInfo("_local1", "uint8_t", (), xram_sym="EXT_DC3E",
                           xram_addr=0xdc3e, is_param=False)
        reg_map = {"EXT_DC3E": vi_local}

        call_node = _make_call_node("callee3")

        result = _run_with_mocks(
            [call_node],
            reg_map=reg_map,
            xps_by_func={"callee3": [XRAMParam("xp1", "uint8_t", 0xdc3e)]},
            resolve_map={0xdc3e: "EXT_DC3E"},
        )

        call_nodes = [n for n in result
                      if isinstance(n, ExprStmt) and isinstance(n.expr, Call)]
        assert len(call_nodes) == 1
        assert len(call_nodes[0].expr.args) == 0, (
            f"Local var should not be folded as passthrough, got {call_nodes[0].expr.args!r}"
        )
