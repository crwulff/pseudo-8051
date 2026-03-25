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
                                   ExprStmt, ReturnStmt, IfGoto, IfNode, WhileNode, ForNode)
from pseudo8051.passes    import OptimizationPass
from pseudo8051.constants import dbg

from pseudo8051.passes.patterns         import _PATTERNS
from pseudo8051.passes.patterns.mb_assign import collapse_mb_assigns
from pseudo8051.passes.patterns._utils  import (
    VarInfo, _replace_pairs, _replace_xram_syms, _replace_single_regs,
    _replace_pairs_in_node, _replace_single_regs_in_node,
    _subst_all_expr, _subst_xram_in_expr,
    _type_bytes, _byte_names,
    _walk_expr,
)
from pseudo8051.ir.expr import (Expr, UnaryOp, BinOp, Const, Call,
                                 Reg as RegExpr, RegGroup as RegGroupExpr, Name as NameExpr)

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
    """Scan the HIR for function calls and add callee parameter register mappings.

    Only Rn registers (R0-R7) are added.  Address/accumulator registers such as
    DPTR, DPH, DPL, A, and SP are excluded: they are used as scratch pointers
    throughout 8051 code and a global rename would produce wrong output.
    """
    from pseudo8051.prototypes import get_proto
    _SKIP = re.compile(r'^(?:DPTR|DPH|DPL|SP|PC)$')
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


# ── Boolean condition simplification ─────────────────────────────────────────

_NEGATE_OP = {
    "==": "!=", "!=": "==",
    "<":  ">=", ">":  "<=",
    "<=": ">",  ">=": "<",
}

_RE_NOT_CMP = re.compile(r'^!\((.+?)\s+(!=|==|>=|<=|>|<)\s+(.+)\)$')


def _simplify_bool_expr(expr: Expr) -> Expr:
    """Push `!` inward through comparisons; eliminate double negation.

    !(lhs op rhs)  →  lhs ~op rhs   (e.g. !(A != 0) → A == 0)
    !!x            →  x
    """
    def _fn(e: Expr) -> Expr:
        if isinstance(e, UnaryOp) and e.op == "!":
            inner = e.operand
            if isinstance(inner, BinOp) and inner.op in _NEGATE_OP:
                return BinOp(inner.lhs, _NEGATE_OP[inner.op], inner.rhs)
            if isinstance(inner, UnaryOp) and inner.op == "!":
                return inner.operand
        return e
    return _walk_expr(expr, _fn)


def _simplify_bool_str(cond: str) -> str:
    """String-condition version of boolean simplification."""
    m = _RE_NOT_CMP.match(cond)
    if m:
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        if op in _NEGATE_OP:
            return f"{lhs} {_NEGATE_OP[op]} {rhs}"
    # !(!(expr)) → expr
    if cond.startswith("!(!(") and cond.endswith("))"):
        return cond[2:-1]
    return cond


# ── Default node transformation ───────────────────────────────────────────────

_RE_DPTR_SETUP = re.compile(r'^DPTR = (.+?);')


def _get_written_regs(node: HIRNode) -> frozenset:
    """Return the set of register names written as the primary LHS of this node."""
    if isinstance(node, (Assign, CompoundAssign)):
        lhs = node.lhs
        if isinstance(lhs, RegExpr):
            return frozenset({lhs.name})
        if isinstance(lhs, RegGroupExpr):
            return frozenset(lhs.regs)
    return frozenset()


def _kill_params(reg_map: Dict[str, VarInfo], killed: set) -> Dict[str, VarInfo]:
    """Return reg_map with entries whose param registers overlap *killed* removed."""
    if not killed:
        return reg_map
    result = {}
    for k, v in reg_map.items():
        if isinstance(v, VarInfo) and v.is_param and v.regs:
            if any(r in killed for r in v.regs):
                continue
        result[k] = v
    return result


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
                       reg_map: Dict[str, VarInfo],
                       simplify_fn=None) -> Optional[HIRNode]:
    """
    Fallback for nodes not consumed by any pattern.

    • Drops ``DPTR = sym;`` lines when sym is a declared XRAM local.
    • Applies substitutions to Statement text or Expr nodes.
    • Recurses into children of structured nodes.

    Returns None to signal that the node should be dropped.

    simplify_fn is called for nested node lists (IfNode/WhileNode/ForNode bodies)
    so that the caller's flow-sensitive kill state propagates inward.
    """
    if simplify_fn is None:
        simplify_fn = _simplify
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

    if isinstance(node, IfGoto):
        new_cond = _simplify_bool_expr(_subst_expr(node.cond, reg_map))
        if new_cond is not node.cond:
            return IfGoto(node.ea, new_cond, node.label)
        return node

    if isinstance(node, IfNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return IfNode(
            ea         = node.ea,
            condition  = new_cond,
            then_nodes = simplify_fn(node.then_nodes, reg_map),
            else_nodes = simplify_fn(node.else_nodes, reg_map),
        )
    if isinstance(node, WhileNode):
        cond = node.condition
        if isinstance(cond, str):
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        return WhileNode(
            ea         = node.ea,
            condition  = new_cond,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
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
            new_cond = _simplify_bool_str(_subst_text(cond, reg_map))
        else:
            new_cond = _simplify_bool_expr(_subst_expr(cond, reg_map))
        if isinstance(update, str):
            new_update = _subst_text(update, reg_map)
        else:
            new_update = _subst_expr(update, reg_map)
        return ForNode(
            ea         = node.ea,
            init       = new_init,
            condition  = new_cond,
            update     = new_update,
            body_nodes = simplify_fn(node.body_nodes, reg_map),
        )
    return node


# ── Core simplifier walk ──────────────────────────────────────────────────────

def _simplify_once(nodes: List[HIRNode], reg_map: Dict[str, VarInfo],
                   simplify_fn=None) -> List[HIRNode]:
    """Apply one round of pattern matching + default transforms to each node."""
    if simplify_fn is None:
        simplify_fn = _simplify
    out: List[HIRNode] = []
    for node in nodes:
        for pat in _PATTERNS:
            result = pat.match([node], 0, reg_map, simplify_fn)
            if result is not None:
                out.extend(result[0])
                break
        else:
            transformed = _transform_default(node, reg_map, simplify_fn)
            if transformed is not None:
                out.append(transformed)
    return out


def _simplify(nodes: List[HIRNode], reg_map: Dict[str, VarInfo],
              _killed: Optional[set] = None) -> List[HIRNode]:
    """
    Walk nodes, trying each registered Pattern in turn.  Falls back to
    _transform_default for nodes not consumed by any pattern.

    Flow-sensitive kill tracking: when a node assigns to a register that
    carries an is_param mapping, that mapping is suppressed for all
    subsequent nodes (including nested IfNode/WhileNode/ForNode bodies).
    """
    out: List[HIRNode] = []
    i = 0
    killed: set = set() if _killed is None else set(_killed)

    def _sub_simplify(ns: List[HIRNode], rm: Dict[str, VarInfo]) -> List[HIRNode]:
        """Recursive simplify that carries the current kill-set inward."""
        return _simplify(ns, rm, killed)

    def _update_killed(node: HIRNode) -> None:
        for r in _get_written_regs(node):
            v = reg_map.get(r)
            if isinstance(v, VarInfo) and v.is_param:
                killed.add(r)

    while i < len(nodes):
        eff = _kill_params(reg_map, killed)

        for pat in _PATTERNS:
            result = pat.match(nodes, i, eff, _sub_simplify)
            if result is not None:
                replacement, new_i = result
                # Gather kills from the consumed range, then recompute eff
                for j in range(i, new_i):
                    _update_killed(nodes[j])
                eff = _kill_params(reg_map, killed)
                i = new_i
                out.extend(_simplify_once(replacement, eff, _sub_simplify))
                break
        else:
            # Transform BEFORE killing so the node's own RHS uses its old mapping
            transformed = _transform_default(nodes[i], eff, _sub_simplify)
            if transformed is not None:
                out.append(transformed)
            # Kill AFTER transform
            _update_killed(nodes[i])
            i += 1
    return out


def _is_dptr_inc_node(node: HIRNode) -> bool:
    """True for ExprStmt(DPTR++) — a data-pointer advance node."""
    return (isinstance(node, ExprStmt)
            and isinstance(node.expr, UnaryOp)
            and node.expr.op == "++"
            and node.expr.operand == RegExpr("DPTR"))


# ── XRAM local load consolidation ────────────────────────────────────────────

def _consolidate_xram_local_loads(nodes: List[HIRNode],
                                   reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Consolidate post-relay XRAM local byte loads into register-pair assignments.

    Two patterns handled:
    1. Rhi=Name("X.hi"); [DPTR++;...]; Rlo=Name("X.lo"); → RegGroup(Rhi,Rlo)=Name("X")
       Mutates reg_map["RhiRlo"] = VarInfo("X", type, (Rhi, Rlo)) (non-param, no xram_sym).
    2. Rn=Name("local") where "local" is a 1-byte XRAM local (no byte-field suffix)
       Kept as-is; mutates reg_map[Rn] = VarInfo("local", type, (Rn,), is_param=True).
    """
    _RE_BYTE_FIELD = re.compile(r'^(.+)\.(?:hi|lo|b\d+)$')

    def _parent_name(s):
        m = _RE_BYTE_FIELD.match(s)
        return m.group(1) if m else None

    def _find_parent_vinfo(parent_nm):
        for v in reg_map.values():
            if (isinstance(v, VarInfo) and v.name == parent_nm
                    and v.xram_sym and not v.is_byte_field):
                return v
        return None

    def _as_byte_assign(n):
        """Return (reg_name, name_str) if n is 'Reg(r) = Name(s)'; else None."""
        if (isinstance(n, Assign)
                and isinstance(n.lhs, RegExpr)
                and isinstance(n.rhs, NameExpr)):
            return (n.lhs.name, n.rhs.name)
        return None

    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        ba = _as_byte_assign(node)
        if ba is not None:
            reg0, bname0 = ba
            parent_nm = _parent_name(bname0)
            if parent_nm is not None:
                parent_vinfo = _find_parent_vinfo(parent_nm)
                if parent_vinfo is not None:
                    n_bytes = _type_bytes(parent_vinfo.type)
                    expected_bnames = _byte_names(parent_nm, n_bytes)
                    if bname0 == expected_bnames[0] and n_bytes >= 2:
                        regs = [reg0]
                        j = i + 1
                        for k in range(1, n_bytes):
                            while j < len(nodes) and _is_dptr_inc_node(nodes[j]):
                                j += 1
                            if j >= len(nodes):
                                break
                            next_ba = _as_byte_assign(nodes[j])
                            if next_ba is None or next_ba[1] != expected_bnames[k]:
                                break
                            regs.append(next_ba[0])
                            j += 1
                        if len(regs) == n_bytes:
                            pair_key = "".join(regs)
                            new_vinfo = VarInfo(parent_nm, parent_vinfo.type, tuple(regs))
                            reg_map[pair_key] = new_vinfo
                            for r in regs:
                                reg_map[r] = new_vinfo
                            out.append(Assign(node.ea,
                                              RegGroupExpr(tuple(regs)),
                                              NameExpr(parent_nm)))
                            dbg("typesimp",
                                f"  xram-pair-consolidate: {pair_key} = {parent_nm}")
                            i = j
                            continue
            else:
                # No byte suffix — check if it's a 1-byte XRAM local
                for v in reg_map.values():
                    if (isinstance(v, VarInfo) and v.name == bname0
                            and v.xram_sym and not v.is_byte_field
                            and _type_bytes(v.type) == 1):
                        new_vinfo = VarInfo(bname0, v.type, (reg0,), is_param=False)
                        reg_map[reg0] = new_vinfo
                        dbg("typesimp", f"  xram-single-load: {reg0} = {bname0}")
                        break
        # Recurse into structured nodes
        if isinstance(node, IfNode):
            new_node = IfNode(
                ea         = node.ea,
                condition  = node.condition,
                then_nodes = _consolidate_xram_local_loads(node.then_nodes, reg_map),
                else_nodes = _consolidate_xram_local_loads(node.else_nodes, reg_map),
            )
            out.append(new_node)
        elif isinstance(node, WhileNode):
            new_node = WhileNode(
                ea         = node.ea,
                condition  = node.condition,
                body_nodes = _consolidate_xram_local_loads(node.body_nodes, reg_map),
            )
            out.append(new_node)
        elif isinstance(node, ForNode):
            new_node = ForNode(
                ea         = node.ea,
                init       = node.init,
                condition  = node.condition,
                update     = node.update,
                body_nodes = _consolidate_xram_local_loads(node.body_nodes, reg_map),
            )
            out.append(new_node)
        else:
            out.append(node)
        i += 1
    return out


# ── DPL/DPH → DPTR collapsing ────────────────────────────────────────────────

def _as_dph_assign(n) -> Optional[str]:
    """Return Rhi name if n is Assign(Reg('DPH'), Reg(Rhi)); else None."""
    if (isinstance(n, Assign)
            and isinstance(n.lhs, RegExpr) and n.lhs.name == "DPH"
            and isinstance(n.rhs, RegExpr)):
        return n.rhs.name
    return None


def _as_dpl_assign(n) -> Optional[str]:
    """Return Rlo name if n is Assign(Reg('DPL'), Reg(Rlo)); else None."""
    if (isinstance(n, Assign)
            and isinstance(n.lhs, RegExpr) and n.lhs.name == "DPL"
            and isinstance(n.rhs, RegExpr)):
        return n.rhs.name
    return None


def _collapse_dpl_dph(nodes: List[HIRNode],
                       reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Collapse paired DPH/DPL byte assignments into a single DPTR assignment.

    DPH = Rhi; [skippable...] DPL = Rlo;  →  DPTR = RhiRlo;  (or DPTR = var;)
    DPL = Rlo; [skippable...] DPH = Rhi;  →  same

    Skippable = _is_call_setup_assign or _is_dptr_inc_node.
    """
    # Recurse into structured nodes first.
    recursed: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, IfNode):
            recursed.append(IfNode(node.ea, node.condition,
                _collapse_dpl_dph(node.then_nodes, reg_map),
                _collapse_dpl_dph(node.else_nodes, reg_map)))
        elif isinstance(node, WhileNode):
            recursed.append(WhileNode(node.ea, node.condition,
                _collapse_dpl_dph(node.body_nodes, reg_map)))
        elif isinstance(node, ForNode):
            recursed.append(ForNode(node.ea, node.init, node.condition, node.update,
                _collapse_dpl_dph(node.body_nodes, reg_map)))
        else:
            recursed.append(node)

    out: List[HIRNode] = []
    dead: set = set()   # indices consumed as the partner half of a pair

    for i, node in enumerate(recursed):
        if i in dead:
            continue

        rhi = _as_dph_assign(node)   # DPH = Rhi case
        rlo = _as_dpl_assign(node)   # DPL = Rlo case
        if rhi is None and rlo is None:
            out.append(node)
            continue

        # Search forward for the partner, skipping setup/dptr++ nodes.
        partner_idx = None
        partner_val = None
        for k in range(i + 1, len(recursed)):
            if k in dead:
                continue
            if rhi is not None:
                rlo2 = _as_dpl_assign(recursed[k])
                if rlo2 is not None:
                    partner_idx, partner_val = k, rlo2
                    break
            else:
                rhi2 = _as_dph_assign(recursed[k])
                if rhi2 is not None:
                    partner_idx, partner_val = k, rhi2
                    break
            if not (_is_call_setup_assign(recursed[k]) or _is_dptr_inc_node(recursed[k])):
                break   # non-skippable node blocks the search

        if partner_idx is None:
            out.append(node)
            continue

        # Build the collapsed DPTR assignment.
        if rhi is not None:
            reg_hi, reg_lo = rhi, partner_val
        else:
            reg_hi, reg_lo = partner_val, rlo

        pair_key = reg_hi + reg_lo
        vinfo = reg_map.get(pair_key)
        rhs: Expr = (NameExpr(vinfo.name)
                     if isinstance(vinfo, VarInfo)
                     else RegGroupExpr((reg_hi, reg_lo)))
        out.append(Assign(node.ea, RegExpr("DPTR"), rhs))
        dead.add(partner_idx)
        dbg("typesimp", f"  dpl-dph-collapse: DPTR = {pair_key}")

    return out


# ── Post-simplify call-arg fold and dead-setup pruning ────────────────────────

_RE_REG_TOKEN = re.compile(r'\b(R[0-7]+|DPTR|DPH|DPL|A)\b')


def _collect_hir_name_refs(nodes: List[HIRNode]) -> set:
    """Collect all Reg/Name .name strings from expression positions in nodes."""
    refs: set = set()

    def _add(expr: Expr) -> None:
        def _fn(e: Expr) -> Expr:
            if isinstance(e, (RegExpr, NameExpr)):
                refs.add(e.name)
            return e
        _walk_expr(expr, _fn)

    def _visit(node: HIRNode) -> None:
        if isinstance(node, Assign):
            _add(node.rhs)
            if not isinstance(node.lhs, (RegExpr, RegGroupExpr)):
                _add(node.lhs)
        elif isinstance(node, CompoundAssign):
            _add(node.rhs)
        elif isinstance(node, ExprStmt):
            _add(node.expr)
        elif isinstance(node, ReturnStmt) and node.value is not None:
            _add(node.value)
        elif isinstance(node, IfGoto):
            _add(node.cond)
        elif isinstance(node, Statement):
            for m in _RE_REG_TOKEN.finditer(node.text):
                refs.add(m.group(1))
        elif isinstance(node, IfNode):
            for sub in list(node.then_nodes) + list(node.else_nodes):
                _visit(sub)
        elif isinstance(node, (WhileNode, ForNode)):
            for sub in node.body_nodes:
                _visit(sub)

    for node in nodes:
        _visit(node)
    return refs


def _is_call_setup_assign(node: HIRNode) -> bool:
    """True for Assign(Reg/RegGroup, Name/Const) — a consolidated register-setup node."""
    return (isinstance(node, Assign)
            and isinstance(node.lhs, (RegExpr, RegGroupExpr))
            and isinstance(node.rhs, (NameExpr, Const)))


def _lhs_reg_names(node: HIRNode) -> frozenset:
    """Return register name strings written by a setup-assign node."""
    if not isinstance(node, Assign):
        return frozenset()
    lhs = node.lhs
    if isinstance(lhs, RegExpr):
        return frozenset({lhs.name})
    if isinstance(lhs, RegGroupExpr):
        names = set(lhs.regs)
        names.add("".join(lhs.regs))
        return frozenset(names)
    return frozenset()


def _subst_reg_in_call_node(node: HIRNode, reg: str, replacement: Expr) -> HIRNode:
    """Replace Name(reg) with replacement in the call args of node."""
    repl_str = replacement.render()

    def _patch(call: Call) -> Call:
        new_args = [replacement if (isinstance(a, NameExpr) and a.name == reg) else a
                    for a in call.args]
        if any(na is not oa for na, oa in zip(new_args, call.args)):
            return Call(call.func_name, new_args)
        return call

    if isinstance(node, ExprStmt) and isinstance(node.expr, Call):
        new_call = _patch(node.expr)
        return ExprStmt(node.ea, new_call) if new_call is not node.expr else node
    if isinstance(node, Assign) and isinstance(node.rhs, Call):
        new_call = _patch(node.rhs)
        return Assign(node.ea, node.lhs, new_call) if new_call is not node.rhs else node
    if isinstance(node, Statement):
        new_text = re.sub(r'\b' + re.escape(reg) + r'\b', repl_str, node.text)
        return Statement(node.ea, new_text) if new_text != node.text else node
    return node


def _fold_and_prune_setups(nodes: List[HIRNode],
                            reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Post-simplify cleanup of register-setup lines before calls.

    1. Fold Assign(Reg, Const) into the next call node's args.
    2. Remove Assign(Reg/RegGroup, Name/Const) setup nodes whose LHS registers
       are not referenced in any subsequent node.
    3. Remove DPTR++ nodes whose DPTR value is not referenced afterwards.
    Recurses into IfNode / WhileNode / ForNode bodies.
    """
    # Recurse first so inner blocks are cleaned before the outer scan.
    recursed: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, IfNode):
            recursed.append(IfNode(
                node.ea, node.condition,
                _fold_and_prune_setups(node.then_nodes, reg_map),
                _fold_and_prune_setups(node.else_nodes, reg_map),
            ))
        elif isinstance(node, WhileNode):
            recursed.append(WhileNode(
                node.ea, node.condition,
                _fold_and_prune_setups(node.body_nodes, reg_map),
            ))
        elif isinstance(node, ForNode):
            recursed.append(ForNode(
                node.ea, node.init, node.condition, node.update,
                _fold_and_prune_setups(node.body_nodes, reg_map),
            ))
        else:
            recursed.append(node)

    work: List[HIRNode] = list(recursed)

    # Phase 1: fold Assign(Reg, Const) into the next call's args.
    for i in range(len(work)):
        node = work[i]
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr)
                and isinstance(node.rhs, Const)):
            continue
        reg = node.lhs.name
        val = node.rhs
        for j in range(i + 1, len(work)):
            nj = work[j]
            if _is_call_setup_assign(nj) or _is_dptr_inc_node(nj):
                continue
            new_nj = _subst_reg_in_call_node(nj, reg, val)
            if new_nj is not nj:
                work[j] = new_nj
                work[i] = None
                dbg("typesimp", f"  fold-const: {reg}={val.render()} into call")
            break
    work = [n for n in work if n is not None]

    # Phase 2: remove dead setup-assign and DPTR++ nodes.
    out: List[HIRNode] = []
    for i, node in enumerate(work):
        if _is_call_setup_assign(node):
            lhs_regs = _lhs_reg_names(node)
            if lhs_regs.isdisjoint(_collect_hir_name_refs(work[i + 1:])):
                dbg("typesimp",
                    f"  prune-setup: {node.lhs.render()} = {node.rhs.render()}")
                continue
        elif _is_dptr_inc_node(node):
            if "DPTR" not in _collect_hir_name_refs(work[i + 1:]):
                dbg("typesimp", "  prune-dptr++")
                continue
        out.append(node)
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
        func.hir = _consolidate_xram_local_loads(func.hir, reg_map)
        func.hir = _simplify_once(func.hir, reg_map)
        func.hir = _collapse_dpl_dph(func.hir, reg_map)
        func.hir = _fold_and_prune_setups(func.hir, reg_map)

        func.hir = collapse_mb_assigns(func.hir)

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
