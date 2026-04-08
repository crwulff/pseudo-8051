"""
passes/patterns/retval.py — RetvalPattern.

Renames the return value of a function call to a fresh ``retvalN`` variable.
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign
from pseudo8051.ir.expr import Reg, Regs, RegGroup, Call
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
        if vinfo is None:
            # LHS may be a raw multi-register group whose pair key was killed by an
            # intermediate write (e.g. divisor loading clobbering retval regs).
            # Single-register returns (A, R7…) intentionally stay as-is when not in reg_map.
            if isinstance(lhs, Regs) and len(lhs.names) > 1:
                vinfo = VarInfo(lhs_name, "", lhs.names)
            else:
                return None
        elif vinfo.xram_sym or not vinfo.regs:
            return None

        # Derive return type/regs from annotation's callee_args (non-param TypeGroup).
        # Fall back to get_proto only when annotation is absent (e.g. unit tests).
        return_type: str = ""
        return_regs: tuple = ()
        if node.ann is not None and node.ann.callee_args is not None:
            ret_tg = next((g for g in node.ann.callee_args
                           if not g.is_param and g.full_regs), None)
            if ret_tg is not None:
                return_type = ret_tg.type
                return_regs = ret_tg.full_regs
        if not return_type or not return_regs:
            from pseudo8051.prototypes import get_proto
            proto = get_proto(callee)
            if proto is None or proto.return_type in ("void", "") or not proto.return_regs:
                return None
            return_type = proto.return_type
            return_regs = proto.return_regs
        if return_type in ("void", ""):
            return None

        counter = reg_map.get("__n__")
        if not isinstance(counter, list):
            return None
        n = counter[0]
        counter[0] += 1
        retval_name = f"retval{n + 1}"

        # Substitute args using the pre-retval reg_map (before return regs are remapped).
        subst_call = _subst_all_expr(rhs, reg_map)

        new_info = VarInfo(retval_name, return_type, return_regs)
        for r in return_regs:
            reg_map[r] = new_info
        if lhs_name in reg_map:
            reg_map[lhs_name] = new_info

        from pseudo8051.passes.typesimplify._regmap import _split_struct_regs
        _split_struct_regs(retval_name, return_type, return_regs, reg_map)

        regs = return_regs
        lhs = Reg(regs[0], alias=retval_name) if len(regs) == 1 else RegGroup(regs, alias=retval_name)
        out_node: HIRNode = TypedAssign(node.ea, return_type, lhs, subst_call)
        out_node.ann = node.ann  # propagate annotation so downstream passes see callee_args
        dbg("typesimp", f"  retval: {out_node.render(0)[0][1]}")
        return ([out_node], i + 1)
