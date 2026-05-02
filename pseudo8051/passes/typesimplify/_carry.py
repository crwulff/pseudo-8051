"""
passes/typesimplify/_carry.py — Carry-flag and comparison simplification passes.

Exports:
  _simplify_carry_comparison   collapse CLR C + SUBB16/SUBB8 sequence in while-loop conditions
  _simplify_subb_jc            fold CLR C + SUBB + JC/JNC into typed comparison (flat if)
  _simplify_cjne_jnc           fold CJNE(nop-goto) + JNC/JC into typed comparisons
  _simplify_orl_zero_check     fold ORL + JZ idiom into multi-byte zero test
  _simplify_arithmetic         remove identity +0/-0 and fold Const OP Const
  _simplify_acc_bit_test       fold A=expr; C=ACC.N; if(C/!C) → if(expr & mask)
"""

import re
from typing import List, Optional

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,
                                ExprStmt, ReturnStmt, IfNode, IfGoto,
                                WhileNode, ForNode, DoWhileNode, SwitchNode,
                                Label)
from pseudo8051.ir.expr import (Expr, BinOp, Const, UnaryOp,
                                 Reg, Regs as RegExpr,
                                 Name as NameExpr)
from pseudo8051.passes.patterns.base import ConditionFoldTransform
from pseudo8051.constants import dbg

_RE_BYTE_FIELD = re.compile(r'^(.+)\.(?:hi|lo|b\d+)$')
_ACC_BIT_RE    = re.compile(r'^ACC\.([0-7])$')


# ── 16-bit SUBB comparison collapsing ────────────────────────────────────────

def _match_subb16(nodes: List[HIRNode], k: int):
    """Try to match CLR C + SUBB lo + MOV A,Rhi + SUBB hi starting at index k.
    Returns (Rlo_sub, Rhi_min, Rhi_sub) or None."""
    if k + 3 >= len(nodes):
        return None
    n0, n1, n2, n3 = nodes[k], nodes[k+1], nodes[k+2], nodes[k+3]
    if not (isinstance(n0, Assign) and n0.lhs == Reg("C")
            and isinstance(n0.rhs, Const) and n0.rhs.value == 0):
        return None
    if not (isinstance(n1, CompoundAssign) and n1.lhs == Reg("A")
            and n1.op == "-=" and isinstance(n1.rhs, BinOp) and n1.rhs.op == "+"
            and isinstance(n1.rhs.lhs, RegExpr) and n1.rhs.lhs.is_single
            and n1.rhs.rhs == Reg("C")):
        return None
    Rlo_sub = n1.rhs.lhs.name
    if not (isinstance(n2, Assign) and n2.lhs == Reg("A")
            and isinstance(n2.rhs, RegExpr) and n2.rhs.is_single):
        return None
    Rhi_min = n2.rhs.name
    if not (isinstance(n3, CompoundAssign) and n3.lhs == Reg("A")
            and n3.op == "-=" and isinstance(n3.rhs, BinOp) and n3.rhs.op == "+"
            and isinstance(n3.rhs.lhs, RegExpr) and n3.rhs.lhs.is_single
            and n3.rhs.rhs == Reg("C")):
        return None
    Rhi_sub = n3.rhs.lhs.name
    return (Rlo_sub, Rhi_min, Rhi_sub)


def _find_reggroup_name(nodes: List[HIRNode], before_idx: int, target_reg: str) -> Optional[str]:
    """Search nodes[:before_idx] backward for Assign(RegGroup, Name) containing target_reg."""
    for node in reversed(nodes[:before_idx]):
        if (isinstance(node, Assign) and isinstance(node.lhs, RegExpr) and not node.lhs.is_single
                and target_reg in node.lhs.regs and isinstance(node.rhs, NameExpr)):
            return node.rhs.name
    return None


def _try_collapse_subb16(cond, body: List[HIRNode]):
    """
    Scan body for the 4-node SUBB16 sequence and resolve names from preceding
    RegGroup assigns. Returns (new_cond, new_body) or None.
    """
    for k in range(len(body)):
        match = _match_subb16(body, k)
        if match is None:
            continue
        Rlo_sub, Rhi_min, Rhi_sub = match
        subtrahend = _find_reggroup_name(body, k, Rhi_sub)
        if subtrahend is None:
            continue
        minuend = _find_reggroup_name(body, k, Rhi_min)
        if minuend is None:
            continue
        new_body = body[:k] + body[k+4:]
        new_cond = BinOp(NameExpr(minuend), "<", NameExpr(subtrahend))
        dbg("typesimp",
            f"  [{hex(body[k].ea)}] subb16-collapse: {minuend} < {subtrahend}"
            f" (via {Rhi_min},{Rlo_sub} - {Rhi_sub},{Rlo_sub})")
        return (new_cond, new_body)
    return None


# ── 8-bit SUBB comparison collapsing ─────────────────────────────────────────

def _get_a_var_name(node: HIRNode) -> Optional[str]:
    """Return the named variable that A holds at this node (from annotation), or None.

    Uses the TypeGroup annotation set by AnnotationPass.  Only returns a name when
    A maps to a TypeGroup with a non-empty name (not 'A'), ensuring we don't produce
    bogus conditions from raw/unnamed register groups.

    Two paths:
    1. A is directly the sole register in a TypeGroup (A=flags directly).
    2. A is aliased to a single register Rx via reg_exprs (A=R7), and Rx is the
       sole register in a named TypeGroup (R7=flags).  This covers the common case
       where the function parameter is in Rx and A was just loaded from it.
    """
    ann = getattr(node, 'ann', None)
    if ann is None:
        return None

    # Path 1: A directly in a named TypeGroup.
    for g in ann.reg_groups:
        if (hasattr(g, 'full_regs')
                and len(g.full_regs) == 1
                and g.full_regs[0] == 'A'
                and g.name
                and g.name != 'A'):
            return g.name

    # Path 2: A=Rx in reg_exprs, and Rx is directly in a named TypeGroup.
    a_expr = ann.reg_exprs.get('A') if ann.reg_exprs else None
    if a_expr is not None and isinstance(a_expr, RegExpr) and a_expr.is_single:
        rx = a_expr.name
        for g in ann.reg_groups:
            if (hasattr(g, 'full_regs')
                    and len(g.full_regs) == 1
                    and g.full_regs[0] == rx
                    and g.name
                    and g.name != rx):
                return g.name

    return None


def _c_is_used_after(body: List[HIRNode], start: int) -> bool:
    """Return True if Reg('C') appears in a read position in body[start:]
    before the next node that writes C.  A C-write stops the scan."""
    for node in body[start:]:
        if 'C' in node.name_refs():
            return True
        if 'C' in node.written_regs:
            return False   # C is overwritten before being read
    return False


def _try_collapse_subb8(body: List[HIRNode]):
    """
    Scan body for the 8-bit CLR-C + SUBB (single-byte comparison) pattern that
    forms the loop condition:

      C = 0;              (CLR C — not present if already propagated away)
      A -= operand + C;   (SUBB A, operand — rhs still contains Reg("C"))
      OR
      A -= operand;       (after _propagate_values substituted C=0)

    The node is identified as a condition-setting SUBB when its annotation reports
    that C was 0 immediately before the instruction (ann.reg_consts["C"] == 0)
    and A holds a named typed variable (ann.reg_groups has a TypeGroup for A).

    Returns (new_cond, new_body) or None.
    If C is used by a subsequent node before the next C-write, the CompoundAssign
    is kept in the body (its C side-effect is still needed); otherwise it is removed.
    """
    for k in range(len(body)):
        node = body[k]
        if not (isinstance(node, CompoundAssign)
                and node.lhs == Reg("A")
                and node.op == "-="):
            continue

        # Require that C was known to be 0 before this instruction (CLR C preceded
        # it), distinguishing a deliberate comparison from a body computation with
        # carry-in from a prior SUBB/ADDC.
        ann = getattr(node, 'ann', None)
        if ann is None or ann.reg_consts.get("C") != 0:
            continue

        a_name = _get_a_var_name(node)
        if a_name is None:
            continue

        # Extract the operand: either rhs directly (Const/Name) or the lhs of
        # BinOp(operand, "+", C/Const(0)) when C hasn't been fully folded yet.
        rhs = node.rhs
        if isinstance(rhs, BinOp) and rhs.op == '+':
            operand = rhs.lhs   # strip the "+ C" or "+ 0" tail
        else:
            operand = rhs

        new_cond = BinOp(NameExpr(a_name), "<", operand)

        # Remove the SUBB from the body only when C is not used by subsequent
        # nodes before the next C-write (i.e. no inner if(C) depends on it).
        c_needed = _c_is_used_after(body, k + 1)
        new_body = body if c_needed else (body[:k] + body[k + 1:])

        dbg("typesimp",
            f"  [{hex(node.ea)}] subb8-collapse: {a_name} < {operand.render()!r}"
            f" (c_needed_after={c_needed})")
        return (new_cond, new_body)

    return None


def _extract_carry1_operand(subtrahend: Expr):
    """
    If subtrahend has the form  k + 1  or  1 + k  or  k + C  (where C is Reg('C')),
    return (base_operand, carry_value) so that the full subtrahend = base_operand + 1.
    Returns None if the form is not recognised.

    Used to detect the SETB-C + SUBB A, #k pattern (carry=1) after propagation has
    folded `C = 1; A -= k + C` into `A = X - (k + 1)`.
    """
    if not (isinstance(subtrahend, BinOp) and subtrahend.op == '+'):
        return None
    lhs, rhs = subtrahend.lhs, subtrahend.rhs
    # k + Const(1)  or  k + Reg('C')
    if isinstance(rhs, Const) and rhs.value == 1:
        return lhs
    if isinstance(rhs, RegExpr) and rhs.is_single and rhs.name == 'C':
        return lhs
    # Const(1) + k  (less common but possible)
    if isinstance(lhs, Const) and lhs.value == 1:
        return rhs
    return None


def _try_collapse_setb_subb8(body: List[HIRNode]):
    """
    Scan body for the SETB-C + SUBB pattern that forms the do-while condition.

    Pre-propagate form (CompoundAssign still present):
      C = 1;
      A -= operand + C;          ann.reg_consts["C"] == 1

    Post-propagate form (folded into Assign by _propagate_values):
      A = X - (operand + 1);     structural match, no annotation needed

    Both correspond to: C = (X < operand + 1) = (X <= operand).
    When the DoWhile condition is !C → while (X > operand).

    Returns (new_cond, new_body) or None.
    """
    for k in range(len(body)):
        node = body[k]

        # ── Pre-propagate: CompoundAssign(A, '-=', operand + C) with C=1 ──
        if (isinstance(node, CompoundAssign)
                and node.lhs == Reg("A")
                and node.op == "-="):
            ann = getattr(node, 'ann', None)
            if ann is not None and ann.reg_consts.get("C") == 1:
                a_name = _get_a_var_name(node)
                if a_name is not None:
                    rhs = node.rhs
                    operand = _extract_carry1_operand(rhs)
                    if operand is None:
                        operand = rhs   # rhs is directly the operand
                    new_cond = BinOp(NameExpr(a_name), ">", operand)
                    new_body = body[:k] + body[k + 1:]
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] setb-subb8 (compound): "
                        f"{a_name} > {operand.render()!r}")
                    return (new_cond, new_body)

        # ── Post-propagate: Assign(A, X - (operand + 1)) ──
        if (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr) and node.lhs.is_single
                and node.lhs.name == 'A'
                and isinstance(node.rhs, BinOp)
                and node.rhs.op == '-'):
            X = node.rhs.lhs
            operand = _extract_carry1_operand(node.rhs.rhs)
            if operand is not None and not isinstance(X, Const):
                new_cond = BinOp(X, ">", operand)
                new_body = body[:k] + body[k + 1:]
                dbg("typesimp",
                    f"  [{hex(node.ea)}] setb-subb8 (assign): "
                    f"{X.render()} > {operand.render()!r}")
                return (new_cond, new_body)

    return None


def _simplify_carry_comparison(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    For each WhileNode with condition Reg("C"), scan body_nodes for:
      1. The 4-node CLR C + SUBB16 sequence (16-bit comparison), or
      2. A single CLR-C + SUBB8 node (8-bit comparison) identified via annotation.
    For each DoWhileNode with condition !C, scan body_nodes for:
      3. The SETB-C + SUBB8 Assign (decrement-while-positive) pattern.
    If found, replace the condition with the appropriate typed BinOp comparison.
    Recurses into nested structured nodes.
    """
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, WhileNode):
            # Recurse into body; map_bodies preserves ann + src_eas on the rebuilt node.
            rebuilt = node.map_bodies(_simplify_carry_comparison)
            if rebuilt.condition == Reg("C"):
                transformed = (_try_collapse_subb16(rebuilt.condition, rebuilt.body_nodes)
                               or _try_collapse_subb8(rebuilt.body_nodes))
                if transformed is not None:
                    new_cond, new_body = transformed
                    result.append(rebuilt.copy_meta_to(WhileNode(rebuilt.ea, new_cond, new_body)))
                    continue
            result.append(rebuilt)
        elif isinstance(node, DoWhileNode):
            rebuilt = node.map_bodies(_simplify_carry_comparison)
            if rebuilt.condition == UnaryOp("!", Reg("C")):
                transformed = _try_collapse_setb_subb8(rebuilt.body_nodes)
                if transformed is not None:
                    new_cond, new_body = transformed
                    result.append(rebuilt.copy_meta_to(
                        DoWhileNode(rebuilt.ea, new_cond, new_body)))
                    continue
            result.append(rebuilt)
        else:
            result.append(node.map_bodies(_simplify_carry_comparison))
    return result


# ── CLR-C + SUBB + JC/JNC flat if-statement simplification ──────────────────

def _a_name_before_subb(nodes: List[HIRNode], subb_idx: int) -> Optional[str]:
    """
    Return the named variable A held immediately before nodes[subb_idx] (SUBB).

    Three paths (tried in order):
    1. TypeGroup directly on A at the SUBB node.
    2. A=Rx in reg_exprs; TypeGroup on Rx at the SUBB node.
    3. Scan backward, skipping CLR/SETB C, Labels, and dec/inc ExprStmts, for:
       a. Assign(Reg("A"), Name(varname)) — direct load of named var into A.
       b. Assign(Name(_), Name(varname)) — xram-local-write that captured A.
    """
    node = nodes[subb_idx]
    # Paths 1 and 2: annotation-based.
    name = _get_a_var_name(node)
    if name is not None:
        return name

    # Path 3: backward scan.
    from pseudo8051.ir.hir import ExprStmt as _ExprStmt
    for k in range(subb_idx - 1, -1, -1):
        prev = nodes[k]
        if isinstance(prev, Label):
            continue
        # CLR C / SETB C
        if (isinstance(prev, Assign)
                and prev.lhs == Reg("C")
                and isinstance(prev.rhs, Const)
                and prev.rhs.value in (0, 1)):
            continue
        # --name / ++name  (dec/inc of an XRAM local — safe to skip)
        if (isinstance(prev, _ExprStmt)
                and isinstance(prev.expr, UnaryOp)
                and prev.expr.op in ("--", "++")
                and isinstance(prev.expr.operand, NameExpr)):
            continue
        # Assign(Reg("A"), Name(varname)) — A was loaded from a named variable.
        if (isinstance(prev, Assign)
                and prev.lhs == Reg("A")
                and isinstance(prev.rhs, NameExpr)):
            return prev.rhs.name
        # Assign(Name(_), Name(varname)) — xram-local-write captured A.
        if (isinstance(prev, Assign)
                and isinstance(prev.lhs, NameExpr)
                and isinstance(prev.rhs, NameExpr)):
            return prev.rhs.name
        break  # anything else — give up

    return None


def _hi_byte_to_parent_name(hi_expr) -> Optional[str]:
    """Extract the parent 16-bit variable name from a hi-byte expression.

    If hi_expr is Name("var.hi") or Regs with alias "var.hi", returns "var".
    Otherwise returns None.
    """
    if isinstance(hi_expr, NameExpr) and hi_expr.name.endswith('.hi'):
        return hi_expr.name[:-3]
    alias = getattr(hi_expr, 'alias', None)
    if alias and alias.endswith('.hi'):
        return alias[:-3]
    return None


def _is_carry_expr(expr) -> bool:
    """Return True if expr is equivalent to the carry flag C.

    Matches bare Reg("C") or a BinOp "+" with one side Const(0) and the
    other Reg("C") — the form left by _fold_compound_assigns when C=0 was
    not yet constant-folded (e.g. A = Rhi - (0 + C)).
    """
    if expr == Reg("C"):
        return True
    if isinstance(expr, BinOp) and expr.op == "+":
        if isinstance(expr.lhs, Const) and expr.lhs.value == 0 and expr.rhs == Reg("C"):
            return True
        if isinstance(expr.rhs, Const) and expr.rhs.value == 0 and expr.lhs == Reg("C"):
            return True
    return False


def _simplify_subb_jc(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Collapse CLR-C/SETB-C + SUBB A, #k + JC/JNC into a typed comparison.

    CLR-C pattern (C=0):
      A -= k;                → if (a_name < k) / if (a_name >= k)

    SETB-C pattern (C=1):
      A -= k + 1;            → if (a_name < k+1) / if (a_name >= k+1)
      (setb C; subb A,#k folds to A -= k+1; C set when a_name < k+1)

    16-bit CLR-C pattern (C=0, with hi-byte SUBB propagation):
      A -= lo_k;             (lo-byte comparison, C=0)
      A = hi_expr - C;       (hi-byte borrow propagation: A = var.hi - C)
      if (C) / if (!C)       → if (var16 < lo_k) / if (var16 >= lo_k)

    Also handles IfGoto variants.  Recurses into structured bodies first.
    The SUBB CompoundAssign (and hi-byte Assign if present) are consumed.
    """
    nodes = [n.map_bodies(_simplify_subb_jc) for n in nodes]

    result: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]

        # Match: CompoundAssign(A, -=, ...) with C=0 or C=1 in annotation.
        if not (isinstance(node, CompoundAssign)
                and node.lhs == Reg("A")
                and node.op == "-="):
            result.append(node)
            i += 1
            continue

        ann = getattr(node, 'ann', None)
        c_val = ann.reg_consts.get("C") if ann is not None else None

        a_name = _a_name_before_subb(nodes, i)

        # Skip optional Label gap nodes to find what follows the lo-SUBB.
        j = i + 1
        while j < len(nodes) and isinstance(nodes[j], Label):
            j += 1

        # Detect 16-bit hi-byte SUBB propagation step before the c_val bail,
        # because the inner SUBB of a nested 16-bit comparison may have no
        # CLR-C annotation (carry flows from the outer comparison).
        #   A = hi_expr - C  (or A = hi_expr - (0+C) before _simplify_arithmetic)
        hi_subb_idx = None   # index of the hi-byte Assign node, if present
        hi_var_name = None   # parent 16-bit variable name derived from hi_expr
        if j < len(nodes):
            nd = nodes[j]
            if (isinstance(nd, Assign)
                    and nd.lhs == Reg("A")
                    and isinstance(nd.rhs, BinOp)
                    and nd.rhs.op == "-"
                    and _is_carry_expr(nd.rhs.rhs)):
                hi_expr = nd.rhs.lhs
                parent = _hi_byte_to_parent_name(hi_expr)
                if parent is not None:
                    hi_subb_idx = j
                    hi_var_name = parent
                    j += 1
                    # Skip any labels between hi-SUBB and condition
                    while j < len(nodes) and isinstance(nodes[j], Label):
                        j += 1

        # For the 8-bit path require a known carry state and a named variable.
        # The 16-bit path (hi_var_name set) can proceed without c_val.
        if hi_var_name is None:
            if c_val not in (0, 1):
                result.append(node)
                i += 1
                continue
            if a_name is None:
                result.append(node)
                i += 1
                continue

        # Extract the comparison operand.
        # CLR-C (C=0): rhs = k [+ C/0] → operand = k.
        # SETB-C (C=1): rhs = k [+ C/1] → operand = k+1.
        # Unknown carry (16-bit only): strip the "+ C" tail, use base k.
        rhs = node.rhs
        if c_val == 0:
            # Strip "+ C" or "+ 0" tail.
            operand = rhs.lhs if isinstance(rhs, BinOp) and rhs.op == '+' else rhs
        elif c_val == 1:
            # SETB-C: subtrahend is k+1.  _extract_carry1_operand returns k for
            # rhs = k+C or k+1; total operand = k+1.  If rhs is Const(1) (k=0
            # fully folded), operand = Const(1) directly.
            base_k = _extract_carry1_operand(rhs)
            if base_k is not None:
                # Fold base_k + 1 now if both are constants, else keep as BinOp.
                if isinstance(base_k, Const):
                    operand = Const(base_k.value + 1)
                else:
                    operand = BinOp(base_k, "+", Const(1))
            elif isinstance(rhs, Const):
                operand = rhs   # already k+1 = Const(n), e.g. Const(1) when k=0
            else:
                result.append(node)
                i += 1
                continue
        else:
            # c_val unknown (only reached for 16-bit path where hi_var_name is set).
            # Strip the "+ C" tail to get the base constant as the threshold.
            if isinstance(rhs, BinOp) and rhs.op == '+':
                if _is_carry_expr(rhs.rhs):
                    operand = rhs.lhs
                elif _is_carry_expr(rhs.lhs):
                    operand = rhs.rhs
                else:
                    result.append(node)
                    i += 1
                    continue
            elif isinstance(rhs, Const):
                operand = rhs
            else:
                result.append(node)
                i += 1
                continue

        if j >= len(nodes):
            result.append(node)
            i += 1
            continue

        cond_node = nodes[j]
        if isinstance(cond_node, (IfNode, WhileNode, ForNode, DoWhileNode)):
            current_cond = cond_node.condition
        elif isinstance(cond_node, IfGoto):
            current_cond = cond_node.cond
        else:
            result.append(node)
            i += 1
            continue

        is_c = (current_cond == Reg("C"))
        is_not_c = (isinstance(current_cond, UnaryOp)
                    and current_cond.op == "!"
                    and current_cond.operand == Reg("C"))
        if not (is_c or is_not_c):
            result.append(node)
            i += 1
            continue

        # For 8-bit pattern a_name is required; 16-bit uses hi_var_name instead.
        if hi_var_name is None and a_name is None:
            result.append(node)
            i += 1
            continue

        # Build comparison.  For 16-bit pattern use parent var; for 8-bit use a_name.
        cmp_name = hi_var_name if hi_var_name is not None else a_name
        new_cond = BinOp(NameExpr(cmp_name), "<" if is_c else ">=", operand)
        if hi_var_name is not None:
            dbg("typesimp",
                f"  [{hex(node.ea)}] subb-jc16(C={c_val}): {cmp_name} "
                f"{'<' if is_c else '>='} {operand.render()!r}")
        else:
            dbg("typesimp",
                f"  [{hex(node.ea)}] subb-jc(C={c_val}): {cmp_name} "
                f"{'<' if is_c else '>='} {operand.render()!r}")

        repl = cond_node.replace_condition(new_cond)
        # Emit gap labels (between lo-SUBB and hi-SUBB or condition), skipping
        # the hi-byte Assign node itself (it is consumed alongside the lo-SUBB).
        gap_nodes = [nodes[k] for k in range(i + 1, j)
                     if k != hi_subb_idx]
        src_nodes = [node, cond_node]
        if hi_subb_idx is not None:
            src_nodes.insert(1, nodes[hi_subb_idx])
        repl.source_nodes = src_nodes
        result.extend(gap_nodes)
        result.append(repl)
        i = j + 1

    return result


# ── CJNE + JNC/JC carry-comparison simplification ────────────────────────────

class _CjneJncFold(ConditionFoldTransform):
    """
    Fold ExprStmt(expr != const) [Labels…] IfNode/IfGoto(!C/C)
    into IfNode/IfGoto(expr >= const / expr < const).

    CJNE sets C = (expr < const) as a side-effect; the nop-goto form
    (_remove_nop_gotos converts the IfGoto to ExprStmt) leaves that
    comparison visible.  This fold consumes the ExprStmt and rewrites
    the carry-branch condition into a typed comparison.
    """

    def match_setup(self, node: HIRNode) -> bool:
        return (isinstance(node, ExprStmt)
                and isinstance(node.expr, BinOp)
                and node.expr.op == "!="
                and isinstance(node.expr.rhs, Const))

    def new_condition(self, setup_node, cond_node, current_cond) -> Optional[Expr]:
        cond_expr = setup_node.expr
        is_not_c = (isinstance(current_cond, UnaryOp)
                    and current_cond.op == "!"
                    and current_cond.operand == Reg("C"))
        is_c = (current_cond == Reg("C"))
        if is_not_c:
            new_cond = BinOp(cond_expr.lhs, ">=", cond_expr.rhs)
        elif is_c:
            new_cond = BinOp(cond_expr.lhs, "<", cond_expr.rhs)
        else:
            return None
        dbg("typesimp",
            f"  [{hex(setup_node.ea)}] cjne-jnc: "
            f"{cond_expr.render()!r} + carry-branch → {new_cond.render()!r}")
        return new_cond


_cjne_jnc_fold = _CjneJncFold()


def _simplify_cjne_jnc(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Collapse the CJNE (nop-goto) + JNC/JC pattern:

      ExprStmt(expr != const)   ← CJNE: sets C = (expr < const); nop-goto removed
      [optional Label nodes]
      IfNode(!C, body)          ← JNC: jumps when C==0 → expr >= const
    →
      IfNode(expr >= const, body)

    Also handles IfGoto variants and JC (Reg("C")).
    """
    nodes = [n.map_bodies(_simplify_cjne_jnc) for n in nodes]
    return _cjne_jnc_fold.fold_sequence(nodes)


# ── ORL + JZ zero-check simplification ───────────────────────────────────────

class _OrlZeroCheckFold(ConditionFoldTransform):
    """
    Fold CompoundAssign(A, |=, v.field) + IfNode/IfGoto(A == 0 / A != 0)
    into IfNode/IfGoto(parent_var == 0 / parent_var != 0).

    The ORL instruction ORs a byte field (e.g. var.hi) into A to test
    whether the full multi-byte variable is zero.  After the ORL the
    condition is tested; this fold replaces both with a direct comparison.
    No label gap is permitted — the ORL and branch must be consecutive.
    """

    def can_gap(self, node: HIRNode) -> bool:
        return False   # ORL and its branch must be strictly consecutive

    def match_setup(self, node: HIRNode) -> bool:
        return (isinstance(node, CompoundAssign)
                and node.lhs == Reg("A")
                and node.op == "|="
                and isinstance(node.rhs, NameExpr)
                and bool(_RE_BYTE_FIELD.match(node.rhs.name)))

    def new_condition(self, setup_node, cond_node, current_cond) -> Optional[Expr]:
        m = _RE_BYTE_FIELD.match(setup_node.rhs.name)
        if not m:
            return None
        parent = m.group(1)
        if not (isinstance(current_cond, BinOp)
                and current_cond.op in ("==", "!=")
                and isinstance(current_cond.rhs, Const)
                and current_cond.rhs.value == 0):
            return None
        new_cond = BinOp(NameExpr(parent), current_cond.op, Const(0))
        dbg("typesimp",
            f"  [{hex(setup_node.ea)}] orl-zero-check: "
            f"A|={setup_node.rhs.name} → {parent} {current_cond.op} 0")
        return new_cond


_orl_zero_check_fold = _OrlZeroCheckFold()


def _simplify_orl_zero_check(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Recognise the 8051 ORL + JZ idiom for 16-bit zero tests.

    Pattern (consecutive):
      CompoundAssign(Reg("A"), "|=", Name(v))   where v has .hi/.lo/.bN suffix
      IfNode or IfGoto with BinOp(anything, "==" | "!=", Const(0)) condition

    Removes the ORL node and replaces the condition with
    BinOp(Name(parent_of_v), op, Const(0)).
    Recurses into WhileNode / IfNode / ForNode bodies first.
    """
    nodes = [n.map_bodies(_simplify_orl_zero_check) for n in nodes]
    return _orl_zero_check_fold.fold_sequence(nodes)


# ── Constant arithmetic folding ───────────────────────────────────────────────

def _fold_const_expr(expr: Expr) -> Expr:
    """Fold identity-zero and Const-OP-Const patterns in an expression (post-order)."""
    from pseudo8051.passes.patterns._utils import _walk_expr

    def _fold(e: Expr) -> Expr:
        if not isinstance(e, BinOp):
            return e
        l, r, op = e.lhs, e.rhs, e.op
        # Fold two literal constants (no alias — preserves enum names).
        if isinstance(l, Const) and isinstance(r, Const) and not l.alias and not r.alias:
            v, w = l.value, r.value
            if op == '+':  return Const(v + w)
            if op == '-':  return Const(v - w)
            if op == '|':  return Const(v | w)
            if op == '&':  return Const(v & w)
            if op == '^':  return Const(v ^ w)
        # x + 0  or  x - 0  →  x
        if isinstance(r, Const) and r.value == 0 and not r.alias and op in ('+', '-'):
            return l
        # 0 + x  →  x
        if isinstance(l, Const) and l.value == 0 and not l.alias and op == '+':
            return r
        return e

    return _walk_expr(expr, _fold)


def _simplify_arithmetic(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Recursively simplify constant arithmetic across all HIR nodes.

    Removes identity operations (+0, -0, 0+) and folds Const OP Const pairs
    that arise from constant-propagated carry/register values.
    """
    result = []
    for node in nodes:
        # Apply to conditions of structured nodes.
        if isinstance(node, (IfNode, WhileNode, ForNode, DoWhileNode)):
            if isinstance(node.condition, Expr):
                new_cond = _fold_const_expr(node.condition)
                if new_cond is not node.condition:
                    node = node.replace_condition(new_cond)
        # Apply to expression fields of leaf nodes.
        elif isinstance(node, (Assign, TypedAssign)):
            new_rhs = _fold_const_expr(node.rhs)
            new_lhs = node.lhs if isinstance(node.lhs, RegExpr) else _fold_const_expr(node.lhs)
            if new_rhs is not node.rhs or new_lhs is not node.lhs:
                if isinstance(node, TypedAssign):
                    node = node.copy_meta_to(TypedAssign(node.ea, node.type_str, new_lhs, new_rhs))
                else:
                    node = node.copy_meta_to(Assign(node.ea, new_lhs, new_rhs))
        elif isinstance(node, CompoundAssign):
            new_rhs = _fold_const_expr(node.rhs)
            if new_rhs is not node.rhs:
                node = node.copy_meta_to(CompoundAssign(node.ea, node.lhs, node.op, new_rhs))
        elif isinstance(node, ExprStmt):
            new_expr = _fold_const_expr(node.expr)
            if new_expr is not node.expr:
                node = node.copy_meta_to(ExprStmt(node.ea, new_expr))
        elif isinstance(node, ReturnStmt) and node.value is not None:
            new_val = _fold_const_expr(node.value)
            if new_val is not node.value:
                node = node.copy_meta_to(ReturnStmt(node.ea, new_val))
        elif isinstance(node, IfGoto):
            new_cond = _fold_const_expr(node.cond)
            if new_cond is not node.cond:
                node = node.copy_meta_to(IfGoto(node.ea, new_cond, node.label))
        # Recurse into structured bodies.
        node = node.map_bodies(_simplify_arithmetic)
        result.append(node)
    return result


# ── ACC bit-test simplification ───────────────────────────────────────────────

_STRUCTURED = (IfNode, WhileNode, ForNode, DoWhileNode, SwitchNode)


def _a_value_before(nodes: List[HIRNode], idx: int) -> Optional[Expr]:
    """Scan backward from nodes[idx] for the most recent simple assignment to A.

    Returns the RHS expression if found before any A-clobbering node, or None.
    Stops conservatively at structured nodes.
    """
    for k in range(idx - 1, -1, -1):
        node = nodes[k]
        if isinstance(node, _STRUCTURED):
            return None
        if isinstance(node, Assign) and node.lhs == Reg("A"):
            return node.rhs
        if "A" in node.written_regs:
            return None   # A overwritten (CompoundAssign, etc.) — value unknown
    return None


def _c_condition(cond: Expr):
    """Return True if cond is Reg('C'), False if !Reg('C'), else None."""
    if cond == Reg("C"):
        return True
    if isinstance(cond, UnaryOp) and cond.op == "!" and cond.operand == Reg("C"):
        return False
    return None


def _acc_bit_condition(cond: Expr):
    """Return (bit, polarity) if cond is Name('ACC.N') or !Name('ACC.N'), else None."""
    if isinstance(cond, NameExpr):
        m = _ACC_BIT_RE.match(cond.name)
        if m:
            return int(m.group(1)), True
    if isinstance(cond, UnaryOp) and cond.op == "!" and isinstance(cond.operand, NameExpr):
        m = _ACC_BIT_RE.match(cond.operand.name)
        if m:
            return int(m.group(1)), False
    return None


def _simplify_acc_bit_test(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Fold 8051 bit-test idioms into direct mask expressions:

    Indirect (via carry):
      A = expr;
      C = ACC.N;          ← MOV C, ACC.N copies bit N of A into the carry flag
      if (C) { ... }      →  if (expr & (1 << N)) { ... }
      if (!C) { ... }     →  if (!(expr & (1 << N))) { ... }
    The C = ACC.N node is removed; the A = expr node is kept unchanged.

    Direct (jb/jnb ACC.N):
      A = expr;
      if (ACC.N) { ... }  →  if (expr & (1 << N)) { ... }
      if (!ACC.N) { ... } →  if (!(expr & (1 << N))) { ... }

    Also handles WhileNode and IfGoto conditions.
    Works at the current nesting level; recurses into structured bodies.
    """
    # Recurse first so inner patterns are resolved before the outer scan.
    nodes = [n.map_bodies(_simplify_acc_bit_test) for n in nodes]

    remove: set = set()                     # indices of C = ACC.N nodes to drop
    replace: dict = {}                       # index → new condition for if/while/goto
    extra_src: dict = {}                     # index → extra src_eas to union into replaced node

    for i, node in enumerate(nodes):
        # ── Indirect path: C = ACC.N ──────────────────────────────────────────
        if (isinstance(node, Assign)
                and node.lhs == Reg("C")
                and isinstance(node.rhs, NameExpr)):
            m = _ACC_BIT_RE.match(node.rhs.name)
            if m:
                bit  = int(m.group(1))
                mask = 1 << bit

                # Find the value A held when the bit was sampled.
                a_expr = _a_value_before(nodes, i)
                if a_expr is None:
                    continue

                # Scan forward for the next use of C — must be an if/while/goto condition.
                for j in range(i + 1, len(nodes)):
                    succ = nodes[j]
                    if "C" in succ.written_regs:
                        break                        # C overwritten before any branch
                    cond = None
                    if isinstance(succ, (IfNode, WhileNode, ForNode, DoWhileNode)):
                        cond = succ.condition
                    elif isinstance(succ, IfGoto):
                        cond = succ.cond
                    if cond is not None:
                        polarity = _c_condition(cond)
                        if polarity is None:
                            break                    # C is the condition but in an unexpected form
                        bit_test = BinOp(a_expr, "&", Const(mask))
                        new_cond = bit_test if polarity else UnaryOp("!", bit_test)
                        remove.add(i)
                        replace[j] = new_cond
                        extra_src[j] = node   # C=ACC.N contributes to the condition
                        dbg("typesimp",
                            f"  [{hex(node.ea)}] acc-bit-test(indirect): "
                            f"C=ACC.{bit} → {'!' if not polarity else ''}"
                            f"({a_expr.render()} & {hex(mask)})")
                        break
                    # Non-C node: safe to skip over as long as it does not read C in an
                    # opaque way (structured nodes can hide C reads — stop conservatively).
                    if isinstance(succ, _STRUCTURED):
                        break
                    if "C" in succ.name_refs():
                        break                        # C read in a non-condition context
            continue

        # ── Direct path: if (ACC.N) / if (!ACC.N) ────────────────────────────
        cond = None
        if isinstance(node, (IfNode, WhileNode, ForNode, DoWhileNode)):
            cond = node.condition
        elif isinstance(node, IfGoto):
            cond = node.cond
        if cond is None:
            continue

        parsed = _acc_bit_condition(cond)
        if parsed is None:
            continue
        bit, polarity = parsed
        mask = 1 << bit

        # Find the value A held just before this branch.
        # Primary path: scan HIR backward for Assign(A, expr).
        a_expr = _a_value_before(nodes, i)
        # Fallback: use the annotation snapshot — the AnnotationPass forward-propagates
        # a TypeGroup for A when A is loaded from a known XRAM address.  This fires when
        # AccumRelay has already consumed the A= node (e.g. A=XRAM[x]; R7=A → R7=x).
        if a_expr is None:
            ann = getattr(node, 'ann', None)
            if ann is not None:
                tg = ann.group_for("A")
                if tg is not None and tg.name:
                    a_expr = NameExpr(tg.name)
        if a_expr is None:
            continue

        bit_test = BinOp(a_expr, "&", Const(mask))
        new_cond = bit_test if polarity else UnaryOp("!", bit_test)
        replace[i] = new_cond
        dbg("typesimp",
            f"  [{hex(node.ea)}] acc-bit-test(direct): "
            f"ACC.{bit} → {'!' if not polarity else ''}"
            f"({a_expr.render()} & {hex(mask)})")

    result = []
    for i, node in enumerate(nodes):
        if i in remove:
            continue
        if i in replace:
            orig = node
            node = node.replace_condition(replace[i])
            # replace_condition copies source_nodes via copy_meta_to;
            # also prepend the removed C=ACC.N node for the indirect path.
            if i in extra_src:
                node.source_nodes = [extra_src[i]] + list(orig.source_nodes or [orig])
        result.append(node)
    return result
