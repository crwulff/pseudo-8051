"""
passes/typesimplify/_xram_call_args.py — Fold XRAM param pre-call assignments into calls.

Exports:
  _fold_xram_call_args
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign, ExprStmt
from pseudo8051.ir.expr import Call, Name as NameExpr
from pseudo8051.constants import dbg


def _fold_xram_call_args(nodes: List[HIRNode],
                          reg_map: Optional[Dict] = None) -> List[HIRNode]:
    """
    Fold XRAM parameter pre-call assignments into the call's argument list.

    Pattern: for each callee that has XRAM params declared,
      xarg1 = val1;
      xarg2 = val2;
      callee(reg_args...)
    →
      callee(reg_args..., val1, val2, ...)

    When reg_map is provided, caller-passthrough xram params (callee params whose
    XRAM address is also an xram param of the caller) are resolved to the caller's
    parameter name even when no explicit assignment node precedes the call.

    Recurses into structured bodies first.
    """
    nodes = [node.map_bodies(lambda ns: _fold_xram_call_args(ns, reg_map))
             for node in nodes]

    def _get_call_expr(node: HIRNode):
        if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
            return node.expr
        if isinstance(node, Assign) and isinstance(node.rhs, Call):
            return node.rhs
        return None

    def _patch_call_node(node: HIRNode, new_call: Call) -> HIRNode:
        if isinstance(node, ExprStmt):
            new_node = ExprStmt(node.ea, new_call)
        elif isinstance(node, TypedAssign):
            new_node = TypedAssign(node.ea, node.type_str, node.lhs, new_call)
        elif isinstance(node, Assign):
            new_node = Assign(node.ea, node.lhs, new_call)
        else:
            return node
        node.copy_meta_to(new_node)
        new_node.source_nodes = [node]
        return new_node

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

        # Resolve caller-passthrough params: when a callee xram param's address
        # is also an xram param of the caller (same XRAM symbol in reg_map with
        # is_param=True), use the caller's parameter name directly.  This handles
        # the case where no explicit "xarg3 = ..." assignment precedes the call
        # because the caller just passes its own xram param through unchanged.
        #
        # Guard against double-folding on the second _fold_xram_call_args pass:
        # if Name(vi.name) is already present in the call's args (added by the
        # first pass), skip it to avoid appending the same arg twice.
        existing_arg_names = {a.name for a in call_expr.args if isinstance(a, NameExpr)}
        passthrough: Dict[str, NameExpr] = {}
        if reg_map:
            from pseudo8051.constants import resolve_ext_addr
            for p in xps:
                if p.name in collected:
                    continue   # already found an explicit assignment
                sym = resolve_ext_addr(p.addr)
                vi = reg_map.get(sym)
                if vi is not None and getattr(vi, 'is_param', False) and vi.name:
                    if vi.name in existing_arg_names:
                        continue   # already folded on a previous pass
                    passthrough[p.name] = NameExpr(vi.name)
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] fold-xram-call-args: passthrough "
                        f"{p.name} → {vi.name} ({sym}) in {call_expr.func_name}")

        # Fold whatever xargs were collected (or resolved as passthrough) before this call.
        if collected or passthrough:
            extra_args = []
            for p in xps:
                if p.name in collected:
                    extra_args.append(collected[p.name][1])
                elif p.name in passthrough:
                    extra_args.append(passthrough[p.name])
            new_call = Call(call_expr.func_name, list(call_expr.args) + extra_args)
            work[i] = _patch_call_node(node, new_call)
            for p in xps:
                if p.name in collected:
                    removed.add(collected[p.name][0])
                    dbg("typesimp",
                        f"  [{hex(work[i].ea)}] fold-xram-call-args: folded {p.name} into {call_expr.func_name}")

        # Always extend callee_args with ALL xram params so that the renderer
        # can show placeholder comments (/* name */) for any params not yet folded.
        # Deduplicate: skip params whose names are already in callee_args.
        final_node = work[i]
        if final_node.ann is not None:
            from pseudo8051.passes.patterns._utils import TypeGroup
            existing = final_node.ann.callee_args or []
            existing_names = {tg.name for tg in existing}
            new_tgs = [TypeGroup(p.name, p.type, (), xram_sym=None, is_param=True)
                       for p in xps if p.name not in existing_names]
            if new_tgs:
                final_node.ann.callee_args = list(existing) + new_tgs

    return [n for i, n in enumerate(work) if i not in removed]
