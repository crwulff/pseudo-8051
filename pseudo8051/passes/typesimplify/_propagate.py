"""
passes/typesimplify/_propagate.py — Forward single-use value propagation pass.

Exports:
  _propagate_values   master pass (calls three sub-passes in a fixed-point loop)

Sub-passes (also exported for testing):
  _fold_compound_assigns      A0: fold Assign(r, e) + CompoundAssign(r, op=, rhs) → Assign
  _propagate_register_copies  A:  substitute single-use Assign(Reg, expr) into its use
  _inline_retvals             B:  inline TypedAssign retval = call() into its single use
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,
                                ExprStmt, ReturnStmt, IfGoto, IfNode,
                                WhileNode, ForNode, DoWhileNode)
from pseudo8051.ir.expr import (Expr, BinOp, Call,
                                 Regs as RegExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import (
    VarInfo, _count_reg_uses_in_node, _subst_reg_in_node, _walk_expr,
)
from pseudo8051.constants import dbg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expr_name_refs(expr: Expr) -> frozenset:
    """Collect register/name identities referenced in an expression tree.

    For RegGroup nodes, emits the canonical joined key (e.g. "R4R5R6R7") so
    that conflict detection operates in register-name space regardless of
    whether the node carries an alias.
    """
    refs: set = set()

    def _collect(e: Expr) -> Expr:
        if isinstance(e, RegExpr) and not e.is_single:
            refs.add("".join(e.regs))
        elif isinstance(e, (NameExpr, RegExpr)):
            refs.add(e.name)
        return e

    _walk_expr(expr, _collect)
    return frozenset(refs)


def _collect_mid_writes(nodes: List[HIRNode], reg_map: Dict) -> frozenset:
    """
    Collect all names/regs written by a sequence of nodes, with three expansions:

    1. Register-to-variable: for each register key in written_regs, look up the
       corresponding variable name in reg_map and add it.  This catches
       Assign(RegGroup(R4R5R6R7), ...) clobbering a variable named 'divisor'.

    2. Name-lhs: TypedAssign / Assign with a Name lhs don't appear in written_regs,
       so we add lhs.name directly.  This catches TypedAssign(Name('divisor'), ...).

    3. Variable-to-register (reverse): for each Name lhs, scan reg_map for VarInfo
       entries with that name and add their backing register keys.  This catches
       TypedAssign(Name('divisor'), ...) clobbering the R0R1R2R3 that backs divisor.
    """
    result: set = set()
    for node in nodes:
        result.update(node.written_regs)
        # Expand register writes → variable names
        if reg_map:
            for reg in node.written_regs:
                info = reg_map.get(reg)
                if isinstance(info, VarInfo) and info.name:
                    result.add(info.name)
        # Name-lhs writes (TypedAssign / Assign with Name LHS)
        if isinstance(node, (Assign, TypedAssign)):
            lhs = getattr(node, 'lhs', None)
            if isinstance(lhs, NameExpr):
                result.add(lhs.name)
                # Reverse lookup: find backing register keys for this variable name
                if reg_map:
                    for reg_key, info in reg_map.items():
                        if isinstance(info, VarInfo) and info.name == lhs.name:
                            result.add(reg_key)
    return frozenset(result)


def _as_retval_stmt(node: HIRNode) -> Optional[Tuple[str, Call]]:
    """Return (retval_name, call_expr) if node is a TypedAssign retval node; else None."""
    if isinstance(node, TypedAssign) and isinstance(node.rhs, Call):
        return (node.lhs.render(), node.rhs)
    return None


def _count_name_uses_in_nodes(name: str, nodes: List[HIRNode]) -> int:
    """Count total occurrences of Name/Reg(name) in read positions across nodes."""
    return sum(_count_reg_uses_in_node(name, n) for n in nodes)


def _is_reg_free(expr: Expr) -> bool:
    """True if expr contains no Reg() leaf — safe to forward-substitute."""
    found_reg = [False]

    def _fn(e: Expr) -> Expr:
        if isinstance(e, RegExpr):
            found_reg[0] = True
        return e

    _walk_expr(expr, _fn)
    return not found_reg[0]


def _collect_hir_name_refs(nodes: List[HIRNode]) -> frozenset:
    result: set = set()
    for n in nodes:
        result |= n.name_refs()
    return frozenset(result)


def _dbg_node(n) -> str:
    try:
        return f"{type(n).__name__}:{n.render(0)[0][1][:50]!r}"
    except Exception:
        return type(n).__name__


# ── Sub-pass A0: compound-assign expansion ────────────────────────────────────

_COMPOUND_OPS = {"+=": "+", "-=": "-", "&=": "&", "|=": "|", "^=": "^"}


def _fold_compound_assigns(live: List[HIRNode]) -> Tuple[List[HIRNode], bool]:
    """
    A0: fold  Assign(Reg(r), expr) + CompoundAssign(Reg(r), op=, rhs)
    into a single Assign(Reg(r), expr op rhs).

    Needed because _count_reg_uses_in_node only counts RHS uses in a
    CompoundAssign, so the preceding Assign would appear unused without
    this expansion step.
    """
    changed = False
    live = list(live)
    i = 0
    while i < len(live):
        node = live[i]
        if (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr) and node.lhs.is_single
                and i + 1 < len(live)):
            nxt = live[i + 1]
            if (isinstance(nxt, CompoundAssign)
                    and nxt.lhs == node.lhs
                    and nxt.op in _COMPOUND_OPS):
                op_str = _COMPOUND_OPS[nxt.op]
                live[i + 1] = Assign(nxt.ea, nxt.lhs, BinOp(node.rhs, op_str, nxt.rhs))
                live[i] = None
                dbg("typesimp",
                    f"  [{hex(node.ea)}] fold-compound: "
                    f"{node.lhs.render()} = {node.rhs.render()} "
                    f"{nxt.op} {nxt.rhs.render()}")
                changed = True
        i += 1
    return [n for n in live if n is not None], changed


# ── Sub-pass A: register copy propagation ─────────────────────────────────────

def _propagate_register_copies(live: List[HIRNode],
                                reg_map: Dict = {}) -> Tuple[List[HIRNode], bool]:
    """
    A: For each Assign(Reg(r), expr) at index i with exactly one downstream use
    before r is written again, substitute the replacement into that use and remove
    the assignment.

    Multi-use propagation is limited to reg-free replacements (no Reg leaves) to
    avoid duplicating side-effecting expressions.  Single-use propagation allows
    any expression; the intermediate guard checks for register clobbers.
    """
    changed = False
    live = list(live)
    i = 0
    while i < len(live):
        node = live[i]
        if not (isinstance(node, Assign) and isinstance(node.lhs, RegExpr) and node.lhs.is_single):
            i += 1
            continue
        r = node.lhs.name
        replacement = node.rhs

        total_uses = 0
        use_idx = None
        kill_idx = None
        for j in range(i + 1, len(live)):
            written = live[j].written_regs
            uses_here = _count_reg_uses_in_node(r, live[j])
            total_uses += uses_here
            if uses_here > 0 and use_idx is None:
                use_idx = j
            if r in written:
                kill_idx = j
                break

        dbg("propagate", f"  sub-A: {r}={replacement.render()!r} "
            f"total_uses={total_uses} kill_idx={kill_idx} "
            f"reg_free={_is_reg_free(replacement)}")
        if total_uses > 0 or kill_idx is not None:
            _scan_end = (kill_idx + 1) if kill_idx is not None else min(len(live), i + 10)
            for _j in range(i + 1, _scan_end):
                _u = _count_reg_uses_in_node(r, live[_j])
                _wr = r in live[_j].written_regs
                dbg("propagate", f"    scan[{_j}]={_dbg_node(live[_j])} uses={_u} kill={_wr}")

        if total_uses == 1 and use_idx is not None:
            # Guard: don't propagate past nodes that write to names (or their backing
            # registers) used in replacement.
            repl_refs = _expr_name_refs(replacement)
            if repl_refs and use_idx > i + 1:
                mid_writes = _collect_mid_writes(live[i + 1:use_idx], reg_map)
                if repl_refs & mid_writes:
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] prop-values: blocked {r} — "
                        f"intermediate writes {repl_refs & mid_writes}")
                    i += 1
                    continue
            new_node = _subst_reg_in_node(live[use_idx], r, replacement)
            if new_node is not None:
                live[use_idx] = new_node
                live[i] = None
                dbg("typesimp", f"  [{hex(node.ea)}] prop-values: folded {r} into node {use_idx}")
                changed = True
        elif total_uses > 1 and _is_reg_free(replacement) and use_idx is not None:
            end = kill_idx if kill_idx is not None else len(live)
            any_subst = False
            for j in range(i + 1, end):
                if _count_reg_uses_in_node(r, live[j]) > 0:
                    new_node = _subst_reg_in_node(live[j], r, replacement)
                    if new_node is not None:
                        live[j] = new_node
                        any_subst = True
            if any_subst:
                remaining = _collect_hir_name_refs(live[i + 1:end])
                if r not in remaining:
                    live[i] = None
                dbg("typesimp",
                    f"  [{hex(node.ea)}] prop-values-multi: {r} = {replacement.render()!r}"
                    f" into {total_uses} use(s)")
                changed = True

        i += 1

    return [n for n in live if n is not None], changed


# ── Sub-pass B: retval inlining ───────────────────────────────────────────────

def _inline_retvals(live: List[HIRNode],
                    reg_map: Dict = {}) -> Tuple[List[HIRNode], bool]:
    """
    B: For each retval TypedAssign with exactly one downstream use of the
    retval name, inline the call expression into the target and remove the
    TypedAssign.
    """
    changed = False
    live = list(live)
    i = 0
    while i < len(live):
        rv = _as_retval_stmt(live[i])
        if rv is None:
            i += 1
            continue
        retval_name, call_expr = rv
        remaining = live[i + 1:]
        total_uses = _count_name_uses_in_nodes(retval_name, remaining)

        if total_uses == 1:
            for j, tgt in enumerate(remaining):
                if _count_reg_uses_in_node(retval_name, tgt) == 1:
                    abs_j = i + 1 + j
                    # Guard: don't inline past nodes that write to any name/reg
                    # (or their backing registers) referenced in the call expression.
                    dbg("propagate",
                        f"  inline-retval [{hex(live[i].ea)}]: {retval_name} "
                        f"j={j} abs_j={abs_j} "
                        f"intermediates={[_dbg_node(live[k]) for k in range(i+1, abs_j)]}")
                    if j > 0:
                        call_reads = _expr_name_refs(call_expr)
                        mid_writes = _collect_mid_writes(live[i + 1:abs_j], reg_map)
                        dbg("propagate",
                            f"    call_reads={sorted(call_reads)} "
                            f"mid_writes={sorted(mid_writes)} "
                            f"conflict={sorted(call_reads & mid_writes)}")
                        if call_reads & mid_writes:
                            dbg("typesimp",
                                f"  [{hex(live[i].ea)}] inline-retval: blocked — "
                                f"intermediate writes {call_reads & mid_writes}")
                            break
                    if (isinstance(tgt, Assign)
                            and isinstance(tgt.rhs, (NameExpr, RegExpr))
                            and (not isinstance(tgt.rhs, RegExpr) or tgt.rhs.is_single)
                            and tgt.rhs.name == retval_name):
                        live[abs_j] = Assign(tgt.ea, tgt.lhs, call_expr)
                    else:
                        new_node = _subst_reg_in_node(tgt, retval_name, call_expr)
                        if new_node is not None:
                            live[abs_j] = new_node
                    src_ea = hex(live[i].ea)
                    live[i] = None
                    dbg("typesimp",
                        f"  [{src_ea}] prop-values: inlined {retval_name} into node {abs_j}")
                    changed = True
                    break

        i += 1

    return [n for n in live if n is not None], changed


# ── Master propagation pass ───────────────────────────────────────────────────

def _propagate_values(nodes: List[HIRNode],
                       reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Forward single-use propagation pass.

    Recurses into structured nodes first, then applies three sub-passes in a
    fixed-point loop until no changes occur:
      A0: fold Assign + CompoundAssign into a single Assign
      A:  substitute single-use (or reg-free multi-use) register copies
      B:  inline single-use retval TypedAssign into its target
    """
    work = [node.map_bodies(lambda ns: _propagate_values(ns, reg_map))
            for node in nodes]

    dbg("propagate", f"_propagate_values flat({len(work)}): "
        + ", ".join(_dbg_node(n) for n in work))

    changed = True
    while changed:
        changed = False
        work, c0 = _fold_compound_assigns(work)
        work, cA = _propagate_register_copies(work, reg_map)
        work, cB = _inline_retvals(work, reg_map)
        changed = c0 or cA or cB

    return work
