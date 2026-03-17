"""
passes/typesimplify.py — TypeAwareSimplifier pass.

Builds a register → variable-name map from the function prototype and
liveness analysis, then walks the HIR applying registered patterns and
falling back to register-pair text substitution.

Individual patterns live in passes/patterns/.  To add a new one, see the
instructions in passes/patterns/__init__.py.
"""

import re
from typing import Dict, List, Tuple, TYPE_CHECKING

from pseudo8051.ir.hir    import HIRNode, Statement, IfNode, WhileNode, ForNode
from pseudo8051.passes    import OptimizationPass
from pseudo8051.constants import dbg

from pseudo8051.passes.patterns         import _PATTERNS
from pseudo8051.passes.patterns._utils  import VarInfo, _replace_pairs, _type_bytes

if TYPE_CHECKING:
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
    """
    Infer parameter register assignment from liveness analysis.

    Succeeds only when the live Rn count exactly matches total param bytes
    and each param's registers form a contiguous ascending sequence.
    """
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

def _build_reg_map(proto: "FuncProto",
                   live_in: frozenset = frozenset()) -> Dict[str, VarInfo]:
    """
    Map register names → VarInfo for prototype params and return registers.

    Assignment priority:
      1. Explicit Param.regs (manual PROTOTYPES override)
      2. Liveness inference (handles non-standard calling conventions)
      3. Standard 8051 convention (R7 downward)
    """
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
        info = VarInfo(p.name, p.type, regs)
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
    """Recursively collect all called function names from HIR text."""
    names: set = set()
    for node in hir:
        if isinstance(node, Statement):
            for m in _RE_CALL_NAME.finditer(node.text):
                names.add(m.group(1))
        elif isinstance(node, IfNode):
            names.update(_collect_call_names(node.then_nodes))
            names.update(_collect_call_names(node.else_nodes))
        elif isinstance(node, (WhileNode, ForNode)):
            names.update(_collect_call_names(node.body_nodes))
    return names


def _augment_with_local_vars(func_ea: int,
                              reg_map: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """
    Add VarInfo entries for XRAM local variables declared for this function
    (stored in the IDA netnode via set_local()).
    Keyed by the resolved XRAM symbol name (e.g. 'EXT_DC8A').
    These entries carry xram_sym so _replace_pairs skips them; they are
    consumed by XRAMLocalWritePattern instead.
    """
    from pseudo8051.locals    import get_locals
    from pseudo8051.constants import resolve_ext_addr
    locals_list = get_locals(func_ea)
    if not locals_list:
        return reg_map
    result = dict(reg_map)
    for lv in locals_list:
        sym = resolve_ext_addr(lv.addr)
        if sym not in result:
            result[sym] = VarInfo(lv.name, lv.type, (), xram_sym=sym)
            dbg("typesimp", f"  local: {lv.name} ({lv.type}) @ {sym}")
    return result


def _augment_with_callee_regs(hir: List[HIRNode],
                               reg_map: Dict[str, VarInfo]) -> Dict[str, VarInfo]:
    """
    Scan the HIR for function calls, look up each callee's prototype, and add
    their parameter register mappings to reg_map.

    Existing reg_map entries (from the caller's own prototype) take priority.
    This enables backward type propagation: patterns like ConstGroupPattern and
    XRAMGroupReadPattern can recognise and collapse argument-loading sequences
    even when the caller function itself has no prototype.
    """
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

def _transform_default(node: HIRNode, reg_map: Dict[str, VarInfo]) -> HIRNode:
    """
    Fallback for nodes not consumed by any pattern: apply pair substitution to
    Statement text, or recurse into the children of structured nodes.
    """
    if isinstance(node, Statement):
        new_text = _replace_pairs(node.text, reg_map)
        return Statement(node.ea, new_text) if new_text != node.text else node

    if isinstance(node, IfNode):
        return IfNode(
            ea         = node.ea,
            condition  = _replace_pairs(node.condition, reg_map),
            then_nodes = _simplify(node.then_nodes, reg_map),
            else_nodes = _simplify(node.else_nodes, reg_map),
        )
    if isinstance(node, WhileNode):
        return WhileNode(
            ea         = node.ea,
            condition  = _replace_pairs(node.condition, reg_map),
            body_nodes = _simplify(node.body_nodes, reg_map),
        )
    if isinstance(node, ForNode):
        return ForNode(
            ea         = node.ea,
            init       = _replace_pairs(node.init,      reg_map),
            condition  = _replace_pairs(node.condition,  reg_map),
            update     = _replace_pairs(node.update,     reg_map),
            body_nodes = _simplify(node.body_nodes, reg_map),
        )
    return node


# ── Core simplifier walk ──────────────────────────────────────────────────────

def _simplify(nodes: List[HIRNode], reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Walk `nodes`, trying each registered Pattern in turn.  Falls back to
    _transform_default (pair substitution + structural recursion) for nodes
    not consumed by any pattern.
    """
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        for pat in _PATTERNS:
            result = pat.match(nodes, i, reg_map, _simplify)
            if result is not None:
                replacement, i = result
                out.extend(replacement)
                break
        else:
            out.append(_transform_default(nodes[i], reg_map))
            i += 1
    return out


# ── Pass ──────────────────────────────────────────────────────────────────────

class TypeAwareSimplifier(OptimizationPass):
    """
    Replace register-level expressions with typed variable names and collapse
    common multi-byte patterns via the registered Pattern list.

    Runs on func.hir after structural passes.
    No-ops if the function has no known prototype.
    """

    def run(self, func: "Function") -> None:
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

        # Add XRAM local variable declarations for this function.
        reg_map = _augment_with_local_vars(func.ea, reg_map)

        # Add register mappings from any callee prototypes found in the HIR.
        # This propagates callee parameter types backward into the argument-
        # loading sequences that precede each call.
        reg_map = _augment_with_callee_regs(func.hir, reg_map)

        if not reg_map:
            dbg("typesimp", f"{func.name}: no register mappings found, skipping")
            return

        dbg("typesimp", f"{func.name}: final reg_map={list(reg_map.keys())}")
        func.hir = _simplify(func.hir, reg_map)

        # Prepend C-style declarations for any XRAM local variables.
        seen: set = set()
        local_decls = []
        for vinfo in reg_map.values():
            if vinfo.xram_sym and vinfo.name not in seen:
                seen.add(vinfo.name)
                local_decls.append(vinfo)
        if local_decls:
            local_decls.sort(key=lambda v: v.xram_sym)
            func.hir = [Statement(func.ea, f"{v.type} {v.name};")
                        for v in local_decls] + func.hir
