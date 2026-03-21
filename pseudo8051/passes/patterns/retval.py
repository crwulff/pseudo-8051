"""
passes/patterns/retval.py — RetvalPattern.

Renames the return value of a function call to a fresh ``retvalN`` variable.
Handles both Assign (expression-tree) and legacy Statement nodes.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement, Assign
from pseudo8051.ir.expr import Reg, Name, RegGroup, Call
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo, _replace_pairs, _replace_xram_syms

_RE_CALL_STMT = re.compile(r'^(\w+) = ([A-Za-z_]\w*)\((.*)\);$', re.DOTALL)


def _expr_to_str(expr) -> str:
    """Render an expr or return str as-is."""
    from pseudo8051.ir.expr import Expr as ExprCls
    if isinstance(expr, ExprCls):
        return expr.render()
    return str(expr)


class RetvalPattern(Pattern):
    """Rename call return registers to retvalN and emit a typed declaration."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        node = nodes[i]

        lhs_name = None
        callee   = None
        args_str = None

        # ── Expr-tree path ────────────────────────────────────────────────
        if isinstance(node, Assign):
            lhs = node.lhs
            rhs = node.rhs
            if isinstance(rhs, Call):
                callee   = rhs.func_name
                args_str = ", ".join(_expr_to_str(a) for a in rhs.args)
                if isinstance(lhs, (Reg, Name)):
                    lhs_name = lhs.render()
                elif isinstance(lhs, RegGroup):
                    lhs_name = lhs.render()
                else:
                    lhs_name = lhs.render()

        # ── Legacy Statement path ─────────────────────────────────────────
        if lhs_name is None:
            if not isinstance(node, Statement):
                return None
            m = _RE_CALL_STMT.match(node.text)
            if not m:
                return None
            lhs_name, callee, args_str = m.group(1), m.group(2), m.group(3)

        if lhs_name is None or callee is None:
            return None

        vinfo = reg_map.get(lhs_name)
        if not isinstance(vinfo, VarInfo):
            vinfo = next((v for v in reg_map.values()
                          if isinstance(v, VarInfo) and v.name == lhs_name and v.regs), None)
        if vinfo is None or vinfo.xram_sym or not vinfo.regs:
            return None

        from pseudo8051.prototypes import get_proto
        proto = get_proto(callee)
        if proto is None or proto.return_type in ("void", "") or not proto.return_regs:
            return None

        counter = reg_map.get("__n__")
        if not isinstance(counter, list):
            return None
        n = counter[0]
        counter[0] += 1
        retval_name = f"retval{n + 1}"

        subst_args = _replace_pairs(_replace_xram_syms(args_str, reg_map), reg_map)

        new_info = VarInfo(retval_name, proto.return_type, proto.return_regs)
        pair = "".join(proto.return_regs)
        reg_map[pair] = new_info
        for r in proto.return_regs:
            reg_map[r] = new_info
        if vinfo.pair_name and vinfo.pair_name != pair:
            reg_map[vinfo.pair_name] = new_info
        if lhs_name in reg_map and lhs_name != pair:
            reg_map[lhs_name] = new_info

        text = f"{proto.return_type} {retval_name} = {callee}({subst_args});"
        dbg("typesimp", f"  retval: {text}")
        return ([Statement(node.ea, text)], i + 1)
