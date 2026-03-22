"""
passes/typesimplify.py — TypeAwareSimplifier pass.

Builds a register → variable-name map from the function prototype and
liveness analysis, then walks the HIR applying registered patterns and
falling back to register-pair substitution.

Individual patterns live in passes/patterns/.  To add a new one, see the
instructions in passes/patterns/__init__.py.
"""

import re
from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir    import (HIRNode, Statement, Assign, CompoundAssign,
                                   ExprStmt, ReturnStmt, IfNode, WhileNode, ForNode)
from pseudo8051.passes    import OptimizationPass
from pseudo8051.constants import dbg

from pseudo8051.passes.patterns         import _PATTERNS
from pseudo8051.passes.patterns._utils  import (
    VarInfo, _replace_pairs, _replace_xram_syms, _replace_single_regs,
    _replace_pairs_in_node, _replace_single_regs_in_node,
    _subst_all_expr, _subst_xram_in_expr,
    _type_bytes, _byte_names,
)

from pseudo8051.ir.function import Function
from pseudo8051.prototypes  import FuncProto

_RE_CALL_NAME = re.compile(r'\b([A-Za-z_]\w*)\(')


# ── Standard 8051 calling-convention register assignment ─────────────────────

_REG_POOL = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]


def _assign_regs(params) -> List[Tuple[str, ...]]:
    """Assign registers R7-downward per standard 8051 convention."""
    pool = list(_REG_POOL)
    result = []
    for p in params:
        size = _type_bytes(p.type)
        if size == 0 or size > len(pool):
            result.append(())
            continue
        regs = tuple(pool[-size:])
        pool = pool[:-size]
        result.append(regs)
    return result


def _assign_regs_from_liveness(params, live_in: frozenset) -> List[Tuple[str, ...]]:
    """Infer parameter register assignment from liveness analysis."""
    empty = [() for _ in params]
    live_rn = sorted([r for r in _REG_POOL if r in live_in],
                     key=lambda r: int(r[1:]))
    total_bytes = sum(_type_bytes(p.type) for p in params)
    if not live_rn or len(live_rn) != total_bytes:
        return empty
    result = []
    pos = 0
    for p in params:
        size = _type_bytes(p.type)
        if size == 0:
            result.append(())
            continue
        group = tuple(live_rn[pos:pos + size])
        if len(group) != size:
            return empty
        nums = [int(r[1:]) for r in group]
        if nums != list(range(nums[0], nums[0] + size)):
            return empty
        result.append(group)
        pos += size
    return result


# ── Register map construction ─────────────────────────────────────────────────

def _build_reg_map(proto: FuncProto,
                   live_in: frozenset = frozenset()) -> Dict[str, VarInfo]:
    """Map register names → VarInfo for prototype params and return registers."""
    params   = proto.params
    assigned: List[Tuple[str, ...]] = [p.regs if p.regs else () for p in params]

    needs = [i for i, r in enumerate(assigned) if not r]
    if needs:
        live_inferred = _assign_regs_from_liveness(params, live_in) if live_in else []
        if live_inferred and any(r for r in live_inferred):
            dbg("typesimp", f"  register assignment: liveness-inferred "
                            f"{[v for v in live_inferred]}")
            for i in needs:
                if live_inferred[i]:
                    assigned[i] = live_inferred[i]
            still_needs = [i for i in needs if not assigned[i]]
            if still_needs:
                conv = _assign_regs(params)
                for i in still_needs:
                    assigned[i] = conv[i]
        else:
            conv = _assign_regs(params)
            dbg("typesimp", f"  register assignment: convention "
                            f"{[v for v in conv]}")
            for i in needs:
                assigned[i] = conv[i]

    reg_map: Dict[str, VarInfo] = {}
    for p, regs in zip(params, assigned):
        if not regs:
            continue
        info = VarInfo(p.name, p.type, regs, is_param=True)
        for r in regs:
            reg_map[r] = info
        if len(regs) > 1:
            reg_map[info.pair_name] = info

    if proto.return_regs:
        ret_info = next((reg_map[r] for r in proto.return_regs if r in reg_map), None)
        if ret_info is None:
            ret_info = VarInfo("retval", proto.return_type, proto.return_regs)
        pair = "".join(proto.return_regs)
        if pair not in reg_map:
            reg_map[pair] = ret_info
        for r in proto.return_regs:
            if r not in reg_map:
                reg_map[r] = ret_info

    return reg_map


# ── Callee register-map augmentation ─────────────────────────────────────────

def _collect_call_names(hir: List[HIRNode]) -> set:
    """Recursively collect all called function names from HIR."""
    from pseudo8051.ir.expr import Call
    names: set = set()
    for node in hir:
        if isinstance(node, Statement):
            for m in _RE_CALL_NAME.finditer(node.text):
                names.add(m.group(1))
        elif isinstance(node, (Assign, ExprStmt, ReturnStmt)):
            # Walk the expr for Call nodes
            expr = None
            if isinstance(node, Assign):
                expr = node.rhs
            elif isinstance(node, ExprStmt):
                expr = node.expr
            elif isinstance(node, ReturnStmt) and node.value is not None:
                expr = node.value
            if expr is not None:
                from pseudo8051.ir.expr import Call
                _collect_calls_from_expr(expr, names)
        elif isinstance(node, IfNode):
            names.update(_collect_call_names(node.then_nodes))
            names.update(_collect_call_names(node.else_nodes))
        elif isinstance(node, (WhileNode, ForNode)):
            names.update(_collect_call_names(node.body_nodes))
    return names


def _collect_calls_from_expr(expr, names: set) -> None:
    """Walk an Expr tree, adding any Call.func_name to names."""
    from pseudo8051.ir.expr import (Expr, Call, BinOp, UnaryOp,
                                     XRAMRef, IRAMRef, CROMRef)
    if isinstance(expr, Call):
        names.add(expr.func_name)
        for a in expr.args:
            _collect_calls_from_expr(a, names)
    elif isinstance(expr, BinOp):
        _collect_calls_from_expr(expr.lhs, names)
        _collect_calls_from_expr(expr.rhs, names)
    elif isinstance(expr, UnaryOp):
        _collect_calls_from_expr(expr.operand, names)
    elif isinstance(expr, (XRAMRef, IRAMRef, CROMRef)):
        _collect_calls_from_expr(expr.inner, names)


def _augment_with_local_vars(func_ea: int,
                              reg_map: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """Add VarInfo entries for XRAM local variables declared for this function."""
    from pseudo8051.locals    import get_locals
    from pseudo8051.constants import resolve_ext_addr
    locals_list = get_locals(func_ea)
    if not locals_list:
        return reg_map
    result = dict(reg_map)
    for lv in locals_list:
        base_sym = resolve_ext_addr(lv.addr)
        if base_sym not in result:
            result[base_sym] = VarInfo(lv.name, lv.type, (), xram_sym=base_sym,
                                       xram_addr=lv.addr)
            dbg("typesimp", f"  local: {lv.name} ({lv.type}) @ {base_sym}")
        n = _type_bytes(lv.type)
        if n > 1:
            bnames = _byte_names(lv.name, n)
            for k, byte_name in enumerate(bnames):
                byte_sym = resolve_ext_addr(lv.addr + k)
                bkey = f"_byte_{byte_sym}"
                if bkey not in result:
                    result[bkey] = VarInfo(byte_name, "uint8_t", (),
                                           xram_sym=byte_sym, is_byte_field=True)
                    dbg("typesimp", f"  local byte: {byte_name} @ {byte_sym}")
    return result


def _augment_with_callee_regs(hir: List[HIRNode],
                               reg_map: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """Scan the HIR for function calls and add callee parameter register mappings."""
    from pseudo8051.prototypes import get_proto
    result = dict(reg_map)
    for name in _collect_call_names(hir):
        proto = get_proto(name)
        if proto is None:
            continue
        callee_reg_map = _build_reg_map(proto)
        for r, info in callee_reg_map.items():
            if r not in result:
                result[r] = info
    return result


# ── Default node transformation ───────────────────────────────────────────────

_RE_DPTR_SETUP = re.compile(r'^DPTR = (.+?);')


def _subst_text(text: str, reg_map: Dict[str, VarInfo]) -> str:
    """Apply XRAM-symbol, register-pair, and single-reg-param substitutions to text."""
    text = _replace_xram_syms(text, reg_map)
    text = _replace_pairs(text, reg_map)
    text = _replace_single_regs(text, reg_map)
    return text


def _subst_expr(expr, reg_map: Dict[str, VarInfo]):
    """Apply all substitutions to an Expr node."""
    return _subst_all_expr(expr, reg_map)


def _transform_default(node: HIRNode,
                       reg_map: Dict[str, VarInfo]) -> Optional[HIRNode]:
    """
    Fallback for nodes not consumed by any pattern.

    • Drops ``DPTR = sym;`` lines when sym is a declared XRAM local.
    • Applies substitutions to Statement text or Expr nodes.
    • Recurses into children of structured nodes.

    Returns None to signal that the node should be dropped.
    """
    if isinstance(node, Statement):
        m = _RE_DPTR_SETUP.match(node.text)
        if m:
            sym = m.group(1).strip()
            if any(v.xram_sym == sym for v in reg_map.values()
                   if isinstance(v, VarInfo) and v.xram_sym):
                return None
        new_text = _subst_text(node.text, reg_map)
        return Statement(node.ea, new_text) if new_text != node.text else node

    if isinstance(node, Assign):
        # Check for DPTR = sym drop
        from pseudo8051.ir.expr import Reg as RegExpr, Name as NameExpr
        if isinstance(node.lhs, RegExpr) and node.lhs.name == "DPTR":
            sym = node.rhs.render()
            if any(v.xram_sym == sym for v in reg_map.values()
                   if isinstance(v, VarInfo) and v.xram_sym):
                return None
        new_rhs = _subst_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return Assign(node.ea, node.lhs, new_rhs)
        return node

    if isinstance(node, CompoundAssign):
        new_rhs = _subst_expr(node.rhs, reg_map)
        if new_rhs is not node.rhs:
            return CompoundAssign(node.ea, node.lhs, node.op, new_rhs)
        return node

    if isinstance(node, ExprStmt):
        new_expr = _subst_expr(node.expr, reg_map)
        if new_expr is not node.expr:
            return ExprStmt(node.ea, new_expr)
        return node

    if isinstance(node, ReturnStmt):
        if node.value is not None:
            new_val = _subst_expr(node.value, reg_map)
            if new_val is not node.value:
                return ReturnStmt(node.ea, new_val)
        return node

    if isinstance(node, IfNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _subst_text(cond, reg_map)
        else:
            new_cond = _subst_expr(cond, reg_map)
        return IfNode(
            ea         = node.ea,
            condition  = new_cond,
            then_nodes = _simplify(node.then_nodes, reg_map),
            else_nodes = _simplify(node.else_nodes, reg_map),
        )
    if isinstance(node, WhileNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _subst_text(cond, reg_map)
        else:
            new_cond = _subst_expr(cond, reg_map)
        return WhileNode(
            ea         = node.ea,
            condition  = new_cond,
            body_nodes = _simplify(node.body_nodes, reg_map),
        )
    if isinstance(node, ForNode):
        init = node.init
        cond = node.condition
        update = node.update
        if isinstance(init, str):
            new_init = _subst_text(init, reg_map)
        elif isinstance(init, Assign):
            new_rhs = _subst_expr(init.rhs, reg_map)
            new_init = Assign(init.ea, init.lhs, new_rhs) if new_rhs is not init.rhs else init
        else:
            new_init = init
        if isinstance(cond, str):
            new_cond = _subst_text(cond, reg_map)
        else:
            new_cond = _subst_expr(cond, reg_map)
        if isinstance(update, str):
            new_update = _subst_text(update, reg_map)
        else:
            new_update = _subst_expr(update, reg_map)
        return ForNode(
            ea         = node.ea,
            init       = new_init,
            condition  = new_cond,
            update     = new_update,
            body_nodes = _simplify(node.body_nodes, reg_map),
        )
    return node


# ── Core simplifier walk ──────────────────────────────────────────────────────

def _simplify_once(nodes: List[HIRNode], reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """Apply one round of pattern matching to each node."""
    out: List[HIRNode] = []
    for node in nodes:
        for pat in _PATTERNS:
            result = pat.match([node], 0, reg_map, _simplify)
            if result is not None:
                out.extend(result[0])
                break
        else:
            out.append(node)
    return out


def _simplify(nodes: List[HIRNode], reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Walk nodes, trying each registered Pattern in turn.  Falls back to
    _transform_default for nodes not consumed by any pattern.
    """
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        for pat in _PATTERNS:
            result = pat.match(nodes, i, reg_map, _simplify)
            if result is not None:
                replacement, i = result
                out.extend(_simplify_once(replacement, reg_map))
                break
        else:
            transformed = _transform_default(nodes[i], reg_map)
            if transformed is not None:
                out.append(transformed)
            i += 1
    return out


# ── Pass ──────────────────────────────────────────────────────────────────────

class TypeAwareSimplifier(OptimizationPass):
    """
    Replace register-level expressions with typed variable names and collapse
    common multi-byte patterns via the registered Pattern list.
    """

    def run(self, func: Function) -> None:
        from pseudo8051.prototypes import get_proto
        live_in = getattr(func.entry_block, "live_in", frozenset())

        proto = get_proto(func.name)
        if proto is not None:
            reg_map = _build_reg_map(proto, live_in)
            dbg("typesimp", f"{func.name}: caller proto found, "
                            f"live_in={sorted(live_in)}  "
                            f"reg_map keys={list(reg_map.keys())}")
        else:
            reg_map = {}
            dbg("typesimp", f"{func.name}: no caller proto, "
                            "scanning callee prototypes for register mappings")

        reg_map = _augment_with_local_vars(func.ea, reg_map)
        reg_map = _augment_with_callee_regs(func.hir, reg_map)

        if not reg_map:
            dbg("typesimp", f"{func.name}: no register mappings found, running structural patterns only")

        dbg("typesimp", f"{func.name}: final reg_map={list(reg_map.keys())}")
        reg_map["__n__"] = [0]
        func.hir = _simplify(func.hir, reg_map)

        # Prepend C-style declarations for any XRAM local variables.
        seen: set = set()
        local_decls = []
        for vinfo in reg_map.values():
            if (isinstance(vinfo, VarInfo)
                    and vinfo.xram_sym and not vinfo.is_byte_field
                    and vinfo.name not in seen):
                seen.add(vinfo.name)
                local_decls.append(vinfo)
        if local_decls:
            local_decls.sort(key=lambda v: v.xram_sym)
            func.hir = [
                Statement(func.ea,
                          f"{v.type} {v.name};"
                          + (f"  /* {v.xram_sym} @ {hex(v.xram_addr)} */"
                             if v.xram_addr else ""))
                for v in local_decls
            ] + func.hir
