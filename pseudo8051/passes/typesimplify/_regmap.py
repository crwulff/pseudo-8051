"""
passes/typesimplify/_regmap.py — Register-map construction and augmentation.
"""

import re
from typing import Dict, List, Tuple

from pseudo8051.ir.hir    import (HIRNode, Assign, ExprStmt, ReturnStmt,
                                   IfNode, WhileNode, ForNode, DoWhileNode)
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns._utils import (
    VarInfo, _type_bytes, _byte_names,
)
from pseudo8051.ir.expr import UnaryOp, BinOp, Call
from pseudo8051.ir.function import Function
from pseudo8051.prototypes  import FuncProto, expand_regs, get_struct

# ── Struct return splitting ───────────────────────────────────────────────────

def _split_struct_regs(retval_name: str, struct_type: str,
                        return_regs: Tuple[str, ...],
                        reg_map: Dict[str, VarInfo]) -> None:
    """Split a struct return type into per-field VarInfo entries in reg_map.

    For each field of the struct, creates a VarInfo with name
    'retval_name.field_name' backed by the appropriate subset of return_regs.
    The reg_map is mutated in place.
    """
    struct_def = get_struct(struct_type)
    if struct_def is None:
        return
    reg_idx = 0
    for field in struct_def.fields:
        field_bytes = _type_bytes(field.type)
        if field_bytes <= 0 or reg_idx + field_bytes > len(return_regs):
            break
        field_regs = return_regs[reg_idx:reg_idx + field_bytes]
        field_name = f"{retval_name}.{field.name}"
        pair = "".join(field_regs)
        vinfo = VarInfo(field_name, field.type, field_regs, is_retval_field=True)
        reg_map[pair] = vinfo
        for r in field_regs:
            reg_map[r] = vinfo
        dbg("typesimp", f"  struct-split: {field_name} ({field.type}) = {field_regs}")
        reg_idx += field_bytes


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
    assigned: List[Tuple[str, ...]] = [
        expand_regs(p.regs, p.type) if p.regs else () for p in params
    ]

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
        ret_regs = expand_regs(tuple(proto.return_regs), proto.return_type)
        ret_info = next((reg_map[r] for r in ret_regs if r in reg_map), None)
        if ret_info is None:
            ret_info = VarInfo("retval", proto.return_type, ret_regs)
        pair = "".join(ret_regs)
        if pair not in reg_map:
            reg_map[pair] = ret_info
        for r in ret_regs:
            if r not in reg_map:
                reg_map[r] = ret_info

    return reg_map


# ── Callee register-map augmentation ─────────────────────────────────────────

def _collect_call_names(hir: List[HIRNode]) -> set:
    """Recursively collect all called function names from HIR."""
    names: set = set()
    for node in hir:
        if isinstance(node, (Assign, ExprStmt, ReturnStmt)):
            expr = None
            if isinstance(node, Assign):
                expr = node.rhs
            elif isinstance(node, ExprStmt):
                expr = node.expr
            elif isinstance(node, ReturnStmt) and node.value is not None:
                expr = node.value
            if expr is not None:
                _collect_calls_from_expr(expr, names)
        elif isinstance(node, IfNode):
            names.update(_collect_call_names(node.then_nodes))
            names.update(_collect_call_names(node.else_nodes))
        elif isinstance(node, (WhileNode, ForNode, DoWhileNode)):
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


def _augment_with_xram_params(func_ea: int,
                              reg_map: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """Add VarInfo entries for XRAM parameters declared for this function.

    Like _augment_with_local_vars but marks entries as is_param=True so that
    VarDecl nodes are not emitted for them (they appear in the signature instead).
    """
    from pseudo8051.xram_params import get_xram_params
    from pseudo8051.constants   import resolve_ext_addr
    params = get_xram_params(func_ea)
    if not params:
        return reg_map
    result = dict(reg_map)
    for p in params:
        base_sym = resolve_ext_addr(p.addr)
        if base_sym not in result:
            result[base_sym] = VarInfo(p.name, p.type, (), xram_sym=base_sym,
                                       xram_addr=p.addr, is_param=True)
            dbg("typesimp", f"  xram_param: {p.name} ({p.type}) @ {base_sym}")
        n = _type_bytes(p.type)
        if n > 1:
            bnames = _byte_names(p.name, n)
            for k, byte_name in enumerate(bnames):
                byte_sym = resolve_ext_addr(p.addr + k)
                bkey = f"_byte_{byte_sym}"
                if bkey not in result:
                    result[bkey] = VarInfo(byte_name, "uint8_t", (),
                                           xram_sym=byte_sym, is_byte_field=True,
                                           is_param=True)
                    dbg("typesimp", f"  xram_param byte: {byte_name} @ {byte_sym}")
    return result


def _augment_with_callee_xram_params(hir: List[HIRNode],
                                      reg_map: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """Scan the HIR for function calls and add callee XRAM parameter mappings.

    When a callee has XRAM parameters declared, XRAM writes to those addresses
    in the caller are renamed with the callee's parameter name, mirroring the
    way register parameters are propagated via _augment_with_callee_regs.
    """
    from pseudo8051.xram_params import get_xram_params
    from pseudo8051.constants   import resolve_ext_addr
    result = dict(reg_map)
    for name in _collect_call_names(hir):
        try:
            import ida_name
            import idc
            ea = ida_name.get_name_ea(idc.BADADDR, name)
            if ea == idc.BADADDR:
                continue
        except Exception:
            continue
        for p in get_xram_params(ea):
            sym = resolve_ext_addr(p.addr)
            if sym not in result:
                result[sym] = VarInfo(p.name, p.type, (), xram_sym=sym,
                                      xram_addr=p.addr)
                dbg("typesimp", f"  callee xram_param: {name}.{p.name} @ {sym}")
            n = _type_bytes(p.type)
            if n > 1:
                bnames = _byte_names(p.name, n)
                for k, byte_name in enumerate(bnames):
                    byte_sym = resolve_ext_addr(p.addr + k)
                    bkey = f"_byte_{byte_sym}"
                    if bkey not in result:
                        result[bkey] = VarInfo(byte_name, "uint8_t", (),
                                               xram_sym=byte_sym, is_byte_field=True)
    return result


def _augment_with_callee_regs(hir: List[HIRNode],
                               reg_map: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """Scan the HIR for function calls and add callee parameter register mappings.

    Only Rn registers (R0-R7) are added.  Address/accumulator registers such as
    DPTR, DPH, DPL, A, and SP are excluded: they are used as scratch pointers
    throughout 8051 code and a global rename would produce wrong output.
    """
    from pseudo8051.prototypes import get_proto
    _SKIP = re.compile(r'^(?:DPTR|DPH|DPL|A|SP|PC)$')
    result = dict(reg_map)
    for name in _collect_call_names(hir):
        proto = get_proto(name)
        if proto is None:
            continue
        callee_reg_map = _build_reg_map(proto)
        for r, info in callee_reg_map.items():
            if r not in result and not _SKIP.match(r):
                result[r] = info
    return result
