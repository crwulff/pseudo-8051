"""
passes/typesimplify/_enum_resolve.py — Replace Const values with enum member names.

Exports:
  _resolve_enum_consts(nodes, reg_map) -> List[HIRNode]

Looks for Const nodes in positions where the type context is known to be an IDA
enum type and replaces them with Name(enum_member_name).

Contexts handled:
  1. Call arguments whose corresponding Param.type is a known enum.
  2. RHS of Assign(Name/Reg, Const) where the LHS name has an enum type in reg_map.
  3. Const operand of a BinOp comparison where the other operand is a named enum var.
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign, ExprStmt
from pseudo8051.ir.expr import (Expr, BinOp, Call, Const,
                                 Regs as RegExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.enum_resolve import resolve_enum_const, is_enum_type
from pseudo8051.constants import dbg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_resolve(type_str: str, value: int, context: str = "") -> Optional[Const]:
    """Return Const(value, alias=member_name) if value resolves in the enum, else None."""
    name = resolve_enum_const(type_str, value)
    if name is None:
        return None
    dbg("enum", f"  enum-resolve: {type_str}({value:#x}) → {name}  [{context}]")
    return Const(value, alias=name)


def _resolve_in_expr(expr: Expr, type_str: str) -> Expr:
    """If expr is an unaliased Const whose value resolves in type_str, return Const(alias). Else expr."""
    if isinstance(expr, Const) and not expr.alias:
        replacement = _try_resolve(type_str, expr.value, f"expr={expr.render()}")
        if replacement is not None:
            return replacement
    return expr


def _get_var_type(name: str, reg_map: dict) -> Optional[str]:
    """Return the type string for a named variable in reg_map, or None if not an enum type."""
    info = reg_map.get(name)
    if info is None:
        # XRAM locals are keyed by xram_sym (e.g. "EXT_DC68"), not by variable name;
        # fall back to scanning values by VarInfo.name.
        for v in reg_map.values():
            if isinstance(v, VarInfo) and v.name == name:
                info = v
                break
    if isinstance(info, VarInfo):
        if is_enum_type(info.type):
            return info.type
        dbg("enum", f"  _get_var_type({name!r}): type={info.type!r} is not an enum")
    elif info is not None:
        dbg("enum", f"  _get_var_type({name!r}): not a VarInfo ({type(info).__name__})")
    return None


def _resolve_binop_const(expr: Expr, reg_map: dict) -> Expr:
    """
    For BinOp(Name|Reg, op, Const) or BinOp(Const, op, Name|Reg) where the
    named side has an enum type, replace the Const with the enum member Name.
    """
    if not isinstance(expr, BinOp):
        return expr
    lhs, op, rhs = expr.lhs, expr.op, expr.rhs

    if isinstance(lhs, (NameExpr, RegExpr)) and isinstance(rhs, Const):
        type_str = _get_var_type(lhs.name, reg_map)
        if type_str:
            new_rhs = _resolve_in_expr(rhs, type_str)
            if new_rhs is not rhs:
                return BinOp(lhs, op, new_rhs)

    if isinstance(rhs, (NameExpr, RegExpr)) and isinstance(lhs, Const):
        type_str = _get_var_type(rhs.name, reg_map)
        if type_str:
            new_lhs = _resolve_in_expr(lhs, type_str)
            if new_lhs is not lhs:
                return BinOp(new_lhs, op, rhs)

    return expr


def _param_types_from_ann(ann) -> Optional[list]:
    """Return [(name, type), ...] for callee params from a NodeAnnotation, or None."""
    if ann is None or ann.callee_args is None:
        return None
    pairs = [(g.name, g.type) for g in ann.callee_args if g.is_param]
    return pairs if pairs else None


def _param_types_from_proto(func_name: str) -> Optional[list]:
    """Fetch proto from IDA and return [(name, type), ...] for params, or None."""
    try:
        from pseudo8051.prototypes import get_proto
    except Exception:
        return None
    proto = get_proto(func_name)
    if not proto:
        return None
    return [(p.name, p.type) for p in proto.params]


def _resolve_call_args(call_expr: Call, ann=None) -> Call:
    """
    For each arg in a Call where the corresponding param has an enum type,
    replace Const args with enum member Names.

    Uses ann.callee_args (TypeGroups) when available; falls back to get_proto.
    """
    param_types = _param_types_from_ann(ann) or _param_types_from_proto(call_expr.func_name)
    if not param_types:
        dbg("enum", f"  _resolve_call_args({call_expr.func_name!r}): no param info")
        return call_expr

    dbg("enum", f"  _resolve_call_args({call_expr.func_name!r}): "
        f"{len(param_types)} params, {len(call_expr.args)} args")

    new_args = list(call_expr.args)
    changed = False
    for i, ((pname, ptype), arg) in enumerate(zip(param_types, call_expr.args)):
        dbg("enum", f"    arg[{i}] {pname!r}: type={ptype!r}, "
            f"arg={arg.render()!r} (Const={isinstance(arg, Const)})")
        if not isinstance(arg, Const) or arg.alias:
            continue
        if not is_enum_type(ptype):
            dbg("enum", f"    → {ptype!r} is not an IDA enum, skipping")
            continue
        replacement = _try_resolve(ptype, arg.value,
                                   f"call {call_expr.func_name} arg {i} ({pname})")
        if replacement is not None:
            new_args[i] = replacement
            changed = True

    return Call(call_expr.func_name, new_args) if changed else call_expr


def _get_call_expr(node: HIRNode) -> Optional[Call]:
    if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
        return node.expr
    if isinstance(node, (Assign, TypedAssign)) and isinstance(node.rhs, Call):
        return node.rhs
    return None


def _patch_call(node: HIRNode, new_call: Call) -> HIRNode:
    if isinstance(node, ExprStmt):
        new_node = ExprStmt(node.ea, new_call)
    elif isinstance(node, TypedAssign):
        new_node = TypedAssign(node.ea, node.type_str, node.lhs, new_call)
    elif isinstance(node, Assign):
        new_node = Assign(node.ea, node.lhs, new_call)
    else:
        return node
    return node.copy_meta_to(new_node)


# ── Condition walker ─────────────────────────────────────────────────────────

def _resolve_in_condition(cond: Expr, reg_map: dict) -> Expr:
    """Recursively resolve enum Const values in a condition expression."""
    new_cond = _resolve_binop_const(cond, reg_map)
    if isinstance(new_cond, BinOp):
        new_lhs = _resolve_in_condition(new_cond.lhs, reg_map)
        new_rhs = _resolve_in_condition(new_cond.rhs, reg_map)
        if new_lhs is not new_cond.lhs or new_rhs is not new_cond.rhs:
            new_cond = BinOp(new_lhs, new_cond.op, new_rhs)
    return new_cond


# ── Main pass ─────────────────────────────────────────────────────────────────

def _resolve_enum_consts(nodes: List[HIRNode], reg_map: dict) -> List[HIRNode]:
    """
    Walk HIR and replace Const values with enum member Names where the type
    context is a known IDA enum.  Recurses into structured node bodies.
    """
    nodes = [node.map_bodies(lambda ns: _resolve_enum_consts(ns, reg_map))
             for node in nodes]

    dbg("enum", f"_resolve_enum_consts: {len(nodes)} nodes, "
        f"reg_map keys={[k for k in reg_map if k != '__n__']}")

    result = []
    for node in nodes:
        # 1. Call arguments
        call_expr = _get_call_expr(node)
        if call_expr is not None:
            new_call = _resolve_call_args(call_expr, ann=node.ann)
            if new_call is not call_expr:
                node = _patch_call(node, new_call)

        # 2. RHS of Assign where LHS has enum type in reg_map
        if isinstance(node, (Assign, TypedAssign)) and isinstance(node.rhs, Const):
            lhs = node.lhs
            if isinstance(lhs, (NameExpr, RegExpr)):
                type_str = _get_var_type(lhs.name, reg_map)
                if type_str:
                    new_rhs = _resolve_in_expr(node.rhs, type_str)
                    if new_rhs is not node.rhs:
                        old_ann = node.ann
                        if isinstance(node, TypedAssign):
                            node = TypedAssign(node.ea, node.type_str, lhs, new_rhs)
                        else:
                            node = Assign(node.ea, lhs, new_rhs)
                        node.ann = old_ann

        # 3. Comparison conditions: IfNode / IfGoto
        if hasattr(node, "condition") or hasattr(node, "cond"):
            cond = getattr(node, "condition", None) or getattr(node, "cond", None)
            if cond is not None:
                new_cond = _resolve_in_condition(cond, reg_map)
                if new_cond is not cond:
                    node = node.replace_condition(new_cond)

        result.append(node)
    return result
