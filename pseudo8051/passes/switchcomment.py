"""
passes/switchcomment.py — SwitchCaseAnnotator: add enum comments to case labels.

For switch subjects that are bitfield combinations of parameters with enum types,
this pass computes per-case comments of the form:

    case 4:  // dest_type == PtrType_RAM_FE && src_type == PtrType_Code

For merged cases with multiple values the comment uses set notation:

    case 0: case 2: case 8: case 10:
        // dest_type in {PtrType_RAM_FE, PtrType_RAM_00} && src_type in {PtrType_RAM_FE, PtrType_RAM_00}

Runs after TypeAwareSimplifier so variable names are already resolved in the HIR.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir  import (HIRNode, SwitchNode, Assign,
                                 IfNode, WhileNode, DoWhileNode, ForNode)
from pseudo8051.ir.expr import Expr, Reg, Regs, Const, BinOp, Name
from pseudo8051.passes  import OptimizationPass
from pseudo8051.constants import dbg


# ── Subject decomposition ─────────────────────────────────────────────────────

def _parse_or_terms(expr: Expr) -> List[Expr]:
    """Split expr on '|' recursively → flat list of additive terms."""
    if isinstance(expr, BinOp) and expr.op == "|":
        return _parse_or_terms(expr.lhs) + _parse_or_terms(expr.rhs)
    return [expr]


def _extract_reg_shift(term: Expr) -> Optional[Tuple[str, int]]:
    """
    Reg(x)                       → (x, 0)
    BinOp(Reg(x), '<<', Const(k)) → (x, k)
    Anything else                → None
    """
    if isinstance(term, Regs) and term.is_single:
        return (term.name, 0)
    if (isinstance(term, BinOp) and term.op == "<<"
            and isinstance(term.lhs, Regs) and term.lhs.is_single
            and isinstance(term.rhs, Const)):
        return (term.lhs.name, term.rhs.value)
    return None


def _decompose_bitfield_subject(expr: Expr) -> Optional[List[Tuple[str, int]]]:
    """
    Return [(reg_name, shift), …] if expr is a pure bitfield-OR of registers,
    sorted by shift descending, or None if the pattern doesn't match.

    E.g. A<<2 | DPL  →  [("A", 2), ("DPL", 0)]
    """
    terms = _parse_or_terms(expr)
    components = []
    for t in terms:
        rs = _extract_reg_shift(t)
        if rs is None:
            return None
        components.append(rs)
    if not components:
        return None
    components.sort(key=lambda c: c[1], reverse=True)
    return components


# ── Direct parametric decomposition (new form) ───────────────────────────────

def _decompose_named_subject(
    expr: Expr,
    name_type: Dict[str, str],
) -> Optional[Tuple[List[Tuple[str, int]], Dict[str, Tuple[str, int, str]]]]:
    """
    Decompose a subject of the form  (param ± k) << shift | …  directly, without
    needing to look up register assignments in the preceding context.

    Returns (components, reg_info) in the same format as the register-based path
    (so _case_comment is unchanged), using param_name as the dict key.
    Returns None if any term doesn't match the expected pattern or the param name
    is not in name_type.
    """
    terms = _parse_or_terms(expr)
    components: List[Tuple[str, int]] = []
    reg_info: Dict[str, Tuple[str, int, str]] = {}
    for term in terms:
        # Peel off optional left-shift
        shift = 0
        inner = term
        if (isinstance(term, BinOp) and term.op == "<<"
                and isinstance(term.rhs, Const)):
            shift = term.rhs.value
            inner = term.lhs
        param_name, addend = _extract_linear(inner)
        if param_name is None:
            return None
        type_str = name_type.get(param_name)
        if type_str is None:
            return None
        components.append((param_name, shift))
        reg_info[param_name] = (param_name, addend, type_str)
    if not components:
        return None
    components.sort(key=lambda c: c[1], reverse=True)
    return components, reg_info


# ── Context scanning ──────────────────────────────────────────────────────────

def _find_last_assign(nodes: List[HIRNode], reg_name: str) -> Optional[Assign]:
    """
    Return the last Assign(Reg(reg_name), rhs) in a flat node list, or None.
    Only looks at top-level Assign nodes (not inside if/loop bodies).
    """
    result = None
    for node in nodes:
        if (isinstance(node, Assign)
                and node.lhs == Reg(reg_name)):
            result = node
    return result


def _var_name(e: Expr) -> Optional[str]:
    """Return the human-readable variable name from a Name or aliased Regs node.

    After TypeAwareSimplifier, parameter reads appear as Reg nodes with an alias
    (e.g. Reg("R7", alias="src_type")) rather than plain Name nodes.  Both forms
    carry the same semantic name; this helper extracts it from either.
    """
    if isinstance(e, Name):
        return e.name
    if isinstance(e, Regs) and e.alias:
        return e.alias
    return None


def _extract_linear(rhs: Expr) -> Tuple[Optional[str], int]:
    """
    Try to parse rhs as  <var> ± Const  and return (name, addend_to_get_param).

    <var> may be Name("x") or Reg("rx", alias="x") (aliased after TypeAwareSimplifier).

    <var> + Const(k)  → ("x", -k)   param = reg_val - k
    Const(k) + <var>  → ("x", -k)
    <var> - Const(k)  → ("x",  k)   param = reg_val + k
    <var>             → ("x",  0)
    Else              → (None, 0)
    """
    n = _var_name(rhs)
    if n is not None:
        return (n, 0)
    if isinstance(rhs, BinOp):
        if rhs.op == "+":
            lname = _var_name(rhs.lhs)
            if lname is not None and isinstance(rhs.rhs, Const):
                return (lname, -rhs.rhs.value)
            rname = _var_name(rhs.rhs)
            if rname is not None and isinstance(rhs.lhs, Const):
                return (rname, -rhs.lhs.value)
        if rhs.op == "-":
            lname = _var_name(rhs.lhs)
            if lname is not None and isinstance(rhs.rhs, Const):
                return (lname, rhs.rhs.value)
    return (None, 0)


# ── Comment generation ────────────────────────────────────────────────────────

def _field_mask(components: List[Tuple[str, int]], idx: int) -> int:
    """
    Compute the bit mask for component[idx] given its shift and the next
    component's shift.  The highest-shift component uses mask 0xFF.
    """
    _, shift = components[idx]
    if idx == 0:                     # highest shift (components sorted desc)
        return 0xFF
    _, next_shift = components[idx - 1]
    return (1 << (next_shift - shift)) - 1


def _case_comment(
    values: List[int],
    components: List[Tuple[str, int]],   # sorted descending by shift
    reg_info: Dict[str, Tuple[str, int, str]],  # reg → (param_name, addend, type)
    get_enum_name,
) -> Optional[str]:
    """
    Build a comment string for a case group.

    For single-value groups:
        dest_type == PtrType_RAM_FE && src_type == PtrType_Code

    For multi-value groups, collect the set of enum names for each component:
        dest_type in {PtrType_RAM_FE, PtrType_RAM_00} && src_type in {PtrType_RAM_FE, PtrType_RAM_00}
    """
    # per-component: set of enum names seen across all case values
    component_names: List[Tuple[str, set]] = []   # [(param_name, {enum_name, …}), …]
    for idx, (reg_name, shift) in enumerate(components):
        param_name, addend, type_str = reg_info[reg_name]
        mask  = _field_mask(components, idx)
        names: set = set()
        for v in values:
            field_val = (v >> shift) & mask
            param_val = (field_val + addend) & 0xFF
            ename = get_enum_name(type_str, param_val)
            if ename is None:
                return None        # unknown enum value — skip entire comment
            names.add(ename)
        component_names.append((param_name, names))

    parts = []
    for param_name, names in component_names:
        sorted_names = sorted(names)
        if len(sorted_names) == 1:
            parts.append(f"{param_name} == {sorted_names[0]}")
        else:
            joined = ", ".join(sorted_names)
            parts.append(f"{param_name} in {{{joined}}}")
    return " && ".join(parts)


# ── Pass ──────────────────────────────────────────────────────────────────────

class SwitchCaseAnnotator(OptimizationPass):
    """
    Walk func.hir after TypeAwareSimplifier and fill SwitchNode.case_comments
    for switches whose subject is a bitfield combination of enum-typed parameters.
    """

    def run(self, func) -> None:
        from pseudo8051.prototypes import get_proto, get_enum_name
        proto = get_proto(func.name)
        if proto is None:
            dbg("switch", f"SwitchCaseAnnotator: no proto for {func.name!r}, skipping")
            return
        name_type: Dict[str, str] = {p.name: p.type for p in proto.params}
        dbg("switch", f"SwitchCaseAnnotator: {func.name!r} name_type={name_type}")
        if not name_type:
            return
        self._walk(func.hir, [], name_type, get_enum_name)

    # ── recursive HIR walker ──────────────────────────────────────────────────

    def _walk(self, nodes: List[HIRNode], context: List[HIRNode],
              name_type: Dict[str, str], get_enum_name) -> None:
        for i, node in enumerate(nodes):
            preceding = context + list(nodes[:i])
            if isinstance(node, SwitchNode):
                self._annotate(node, preceding, name_type, get_enum_name)
                for _, body in node.cases:
                    if isinstance(body, list):
                        self._walk(body, preceding, name_type, get_enum_name)
                if isinstance(node.default_body, list):
                    self._walk(node.default_body, preceding, name_type, get_enum_name)
            elif isinstance(node, IfNode):
                self._walk(node.then_nodes, preceding, name_type, get_enum_name)
                self._walk(node.else_nodes, preceding, name_type, get_enum_name)
            elif isinstance(node, (WhileNode, DoWhileNode, ForNode)):
                self._walk(node.body_nodes, preceding, name_type, get_enum_name)

    # ── per-switch annotation ─────────────────────────────────────────────────

    def _annotate(self, sw: SwitchNode, context: List[HIRNode],
                  name_type: Dict[str, str], get_enum_name) -> None:
        subj_str = sw.subject.render() if hasattr(sw.subject, "render") else repr(sw.subject)
        dbg("switch", f"SwitchCaseAnnotator: subject={subj_str!r}")

        # ── Strategy 1: direct parametric form  (param ± k) << shift | … ──────
        named = _decompose_named_subject(sw.subject, name_type)
        if named is not None:
            components, reg_info = named
            dbg("switch", f"  components (named)={components}")
        else:
            # ── Strategy 2: register-based form  Reg << shift | … ────────────
            components = _decompose_bitfield_subject(sw.subject)
            if not components:
                dbg("switch", f"  → subject not decomposable, skipping")
                return
            dbg("switch", f"  components (reg)={components}")

            reg_info: Dict[str, Tuple[str, int, str]] = {}
            for reg_name, _ in components:
                asgn = _find_last_assign(context, reg_name)
                if asgn is None:
                    dbg("switch", f"  → no assignment for reg {reg_name!r}, skipping")
                    return
                asgn_str = asgn.rhs.render() if hasattr(asgn.rhs, "render") else repr(asgn.rhs)
                dbg("switch", f"  {reg_name} = {asgn_str}")
                param_name, addend = _extract_linear(asgn.rhs)
                if param_name is None:
                    # One-level register-copy trace: DPL = A, A = dest_type + 2
                    if isinstance(asgn.rhs, Regs) and asgn.rhs.is_single:
                        src_asgn = _find_last_assign(context, asgn.rhs.name)
                        if src_asgn is not None:
                            param_name, addend = _extract_linear(src_asgn.rhs)
                            if param_name is not None:
                                dbg("switch", f"  {reg_name} traced via {asgn.rhs.name!r}")
                if param_name is None:
                    dbg("switch", f"  → rhs not linear (Name ± Const), skipping")
                    return
                type_str = name_type.get(param_name)
                if type_str is None:
                    dbg("switch", f"  → param {param_name!r} not in prototype, skipping")
                    return
                dbg("switch", f"  {reg_name} → param={param_name!r} addend={addend} type={type_str!r}")
                reg_info[reg_name] = (param_name, addend, type_str)

        comments: List[Optional[str]] = []
        for values, _ in sw.cases:
            c = _case_comment(values, components, reg_info, get_enum_name)
            dbg("switch", f"  case {values} → comment={c!r}")
            comments.append(c)

        if any(c is not None for c in comments):
            sw.case_comments = comments
