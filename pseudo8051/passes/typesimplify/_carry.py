"""
passes/typesimplify/_carry.py — Carry-flag and comparison simplification passes.

Exports:
  _simplify_carry_comparison   collapse CLR C + SUBB16/SUBB8 sequence in while-loop conditions
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
    A is the sole register in a TypeGroup that has a non-empty name, ensuring we
    don't produce bogus conditions from raw/unnamed register groups.
    """
    ann = getattr(node, 'ann', None)
    if ann is None:
        return None
    for g in ann.reg_groups:
        if (hasattr(g, 'full_regs')
                and len(g.full_regs) == 1
                and g.full_regs[0] == 'A'
                and g.name
                and g.name != 'A'):
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


def _simplify_carry_comparison(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    For each WhileNode with condition Reg("C"), scan body_nodes for:
      1. The 4-node CLR C + SUBB16 sequence (16-bit comparison), or
      2. A single CLR-C + SUBB8 node (8-bit comparison) identified via annotation.
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
        else:
            result.append(node.map_bodies(_simplify_carry_comparison))
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
