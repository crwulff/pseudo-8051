"""
passes/typesimplify/_xram_call_args.py — Fold XRAM param pre-call assignments into calls.

Exports:
  _fold_xram_call_args
"""

from typing import Dict, List

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign, ExprStmt
from pseudo8051.ir.expr import Call, Name as NameExpr
from pseudo8051.constants import dbg


def _fold_xram_call_args(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Fold XRAM parameter pre-call assignments into the call's argument list.

    Pattern: for each callee that has XRAM params declared,
      xarg1 = val1;
      xarg2 = val2;
      callee(reg_args...)
    →
      callee(reg_args..., val1, val2, ...)

    Recurses into structured bodies first.
    """
    nodes = [node.map_bodies(_fold_xram_call_args) for node in nodes]

    def _get_call_expr(node: HIRNode):
        if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
            return node.expr
        if isinstance(node, Assign) and isinstance(node.rhs, Call):
            return node.rhs
        return None

    def _patch_call_node(node: HIRNode, new_call: Call) -> HIRNode:
        if isinstance(node, ExprStmt):
            return ExprStmt(node.ea, new_call)
        if isinstance(node, TypedAssign):
            return TypedAssign(node.ea, node.type_str, node.lhs, new_call)
        if isinstance(node, Assign):
            return Assign(node.ea, node.lhs, new_call)
        return node

    try:
        import ida_name
        import idc
        _BADADDR = idc.BADADDR
    except Exception:
        return nodes

    work = list(nodes)
    removed: set = set()

    for i in range(len(work)):
        if i in removed:
            continue
        node = work[i]
        call_expr = _get_call_expr(node)
        if call_expr is None:
            continue

        try:
            callee_ea = ida_name.get_name_ea(_BADADDR, call_expr.func_name)
            if callee_ea == _BADADDR:
                continue
        except Exception:
            continue

        try:
            from pseudo8051.xram_params import get_xram_params
            xps = get_xram_params(callee_ea)
        except Exception:
            continue
        if not xps:
            continue

        param_names = [p.name for p in xps]

        # Scan backward from i-1 collecting Assign(Name(param_name), rhs)
        collected: Dict[str, tuple] = {}
        j = i - 1
        while j >= 0:
            if j in removed:
                j -= 1
                continue
            nd = work[j]
            if (isinstance(nd, Assign)
                    and isinstance(nd.lhs, NameExpr)
                    and nd.lhs.name in param_names
                    and nd.lhs.name not in collected):
                collected[nd.lhs.name] = (j, nd.rhs)
                j -= 1
            elif _get_call_expr(nd) is not None:
                break  # hit another call — stop
            else:
                j -= 1

        if len(collected) != len(param_names):
            continue

        extra_args = [collected[name][1] for name in param_names]
        new_call = Call(call_expr.func_name, list(call_expr.args) + extra_args)
        work[i] = _patch_call_node(node, new_call)

        for name in param_names:
            idx = collected[name][0]
            removed.add(idx)
            dbg("typesimp",
                f"  [{hex(work[i].ea)}] fold-xram-call-args: folded {name} into {call_expr.func_name}")

    return [n for i, n in enumerate(work) if i not in removed]
