"""
passes/patterns/retval.py — RetvalPattern.

Renames the return value of a function call to a fresh ``retvalN`` variable.
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign
from pseudo8051.ir.expr import Reg, RegGroup, Call
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo, _subst_all_expr


class RetvalPattern(Pattern):
    """Rename call return registers to retvalN and emit a typed declaration."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        node = nodes[i]

        if not isinstance(node, Assign) or isinstance(node, TypedAssign):
            return None
        rhs = node.rhs
        if not isinstance(rhs, Call):
            return None

        lhs = node.lhs
        lhs_name = lhs.render()
        callee   = rhs.func_name

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

        # Substitute args using the pre-retval reg_map (before return regs are remapped).
        subst_call = _subst_all_expr(rhs, reg_map)

        new_info = VarInfo(retval_name, proto.return_type, proto.return_regs)
        pair = "".join(proto.return_regs)
        reg_map[pair] = new_info
        for r in proto.return_regs:
            reg_map[r] = new_info
        if vinfo.pair_name and vinfo.pair_name != pair:
            reg_map[vinfo.pair_name] = new_info
        if lhs_name in reg_map and lhs_name != pair:
            reg_map[lhs_name] = new_info

        from pseudo8051.passes.typesimplify._regmap import _split_struct_regs
        _split_struct_regs(retval_name, proto.return_type, proto.return_regs, reg_map)

        regs = proto.return_regs
        lhs = Reg(regs[0], alias=retval_name) if len(regs) == 1 else RegGroup(regs, alias=retval_name)
        out_node: HIRNode = TypedAssign(node.ea, proto.return_type, lhs, subst_call)
        dbg("typesimp", f"  retval: {out_node.render(0)[0][1]}")
        return ([out_node], i + 1)
