"""
passes/typesimplify/_dptr.py — DPTR-related post-simplify passes.

Exports:
  _is_dptr_inc_node           predicate: ExprStmt(DPTR++)
  _collapse_dpl_dph           merge DPH/DPL register-pair writes into DPTR assignment
  _collapse_dpl_dph_arithmetic collapse DPL=lo_base+R_lo + DPH=hi_base+R_hi+C → DPTR=16bit+pair
  _dptr_live_after            flow-sensitive: is current DPTR value read downstream?
  _prune_orphaned_dptr_inc    remove DPTR++ nodes with no downstream use
"""

import re
from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (HIRNode, Assign, ExprStmt, IfNode, WhileNode,
                                ForNode, DoWhileNode, SwitchNode)
from pseudo8051.ir.expr import (Expr, UnaryOp, BinOp,
                                 Reg as RegExpr, Regs as RegsExpr,
                                 RegGroup as RegGroupExpr, Name as NameExpr,
                                 Const)
from pseudo8051.passes.patterns._utils import VarInfo, _walk_expr
from pseudo8051.constants import dbg

_RE_BYTE_FIELD = re.compile(r'^(.+)\.(hi|lo|b\d+)$')


# ── Shared predicate ──────────────────────────────────────────────────────────

def _is_dptr_inc_node(node: HIRNode) -> bool:
    """True for ExprStmt(DPTR++) — a data-pointer advance node."""
    return (isinstance(node, ExprStmt)
            and isinstance(node.expr, UnaryOp)
            and node.expr.op == "++"
            and node.expr.operand == RegExpr("DPTR"))


# ── DPL/DPH → DPTR collapsing ────────────────────────────────────────────────

def _as_dph_assign(n) -> Optional[str]:
    """Return Rhi name if n is Assign(Reg('DPH'), Reg(Rhi)); else None."""
    if (isinstance(n, Assign)
            and n.lhs == RegExpr("DPH")
            and isinstance(n.rhs, RegsExpr) and n.rhs.is_single):
        return n.rhs.name
    return None


def _as_dpl_assign(n) -> Optional[str]:
    """Return Rlo name if n is Assign(Reg('DPL'), Reg(Rlo)); else None."""
    if (isinstance(n, Assign)
            and n.lhs == RegExpr("DPL")
            and isinstance(n.rhs, RegsExpr) and n.rhs.is_single):
        return n.rhs.name
    return None


def _is_call_setup_assign(node: HIRNode) -> bool:
    """True for Assign(Reg/RegGroup, Name/Const) — a consolidated register-setup node."""
    from pseudo8051.ir.expr import Const
    return (isinstance(node, Assign)
            and isinstance(node.lhs, RegsExpr)
            and isinstance(node.rhs, (NameExpr, Const)))


def _collapse_dpl_dph(nodes: List[HIRNode],
                       reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Collapse paired DPH/DPL byte assignments into a single DPTR assignment.

    DPH = Rhi; [skippable...] DPL = Rlo;  →  DPTR = RhiRlo;  (or DPTR = var;)
    DPL = Rlo; [skippable...] DPH = Rhi;  →  same

    Skippable = _is_call_setup_assign or _is_dptr_inc_node.
    """
    recursed = [node.map_bodies(lambda ns: _collapse_dpl_dph(ns, reg_map))
                for node in nodes]

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
        rhs = (NameExpr(vinfo.name)
               if isinstance(vinfo, VarInfo)
               else RegGroupExpr((reg_hi, reg_lo)))
        out.append(Assign(node.ea, RegExpr("DPTR"), rhs))
        dead.add(partner_idx)
        dbg("typesimp", f"  [{hex(node.ea)}] dpl-dph-collapse: DPTR = {pair_key}")

    return out


# ── DPL+DPH arithmetic → 16-bit DPTR collapsing ──────────────────────────────

def _contains_c(expr: Expr) -> bool:
    """True if Reg("C") appears anywhere in expr."""
    found = [False]

    def _fn(e: Expr) -> Expr:
        if isinstance(e, RegsExpr) and e.is_single and e.name == "C":
            found[0] = True
        return e

    _walk_expr(expr, _fn)
    return found[0]


def _strip_carry_top(expr: Expr) -> Optional[Expr]:
    """
    If C appears additively at the outermost level, return expr with C removed.

    Handles two forms produced by 8051 ADDC lifting + AccumFold:
      X + C             top-level carry  (when ADDC is a standalone compound)
      X + (Y + C)       nested carry     (when ADDC lifts as A += Y+C and AccumFold
                                          builds BinOp(base, "+", BinOp(Y, "+", C)))

    Returns None if C is not found in either form.
    """
    if not isinstance(expr, BinOp) or expr.op != "+":
        return None
    # Form 1: X + C
    if isinstance(expr.rhs, RegsExpr) and expr.rhs.is_single and expr.rhs.name == "C":
        return expr.lhs
    # Form 2: X + (Y + C)  — ADDC compound rhs is BinOp(Y, "+", C)
    if isinstance(expr.rhs, BinOp) and expr.rhs.op == "+":
        inner = expr.rhs
        if isinstance(inner.rhs, RegsExpr) and inner.rhs.is_single and inner.rhs.name == "C":
            # Reconstruct as BinOp(X, "+", Y) with C stripped
            return BinOp(expr.lhs, "+", inner.lhs)
    return None


def _split_const_operand(expr: Expr) -> Tuple[int, Optional[Expr]]:
    """
    Decompose expr into (const_part, operand_or_None).

    Returns (value, rest) for ``Const(value) + rest`` or ``rest + Const(value)``,
    (value, None) for a bare Const or a Const+Const pair (folded inline), and
    (0, expr) when no leading constant is found.

    The Const+Const fold handles the CLR-A + ADDC pattern before _simplify_arithmetic
    runs: ``CLR A; ADDC A, #0xDC`` lifts as ``A=0; A+=(0xDC+C)`` and AccumFold
    produces ``DPH = 0 + (0xDC + C)`` → ``_strip_carry_top`` → ``0 + 0xDC``.
    Without folding, the leading 0 would be taken as the const and 0xDC as the
    operand, breaking the 16-bit combination.
    """
    if isinstance(expr, Const) and not expr.alias:
        return (expr.value, None)
    if isinstance(expr, BinOp) and expr.op == "+":
        if isinstance(expr.lhs, Const) and not expr.lhs.alias:
            # Both constants: fold inline so c_hi/c_lo are correct even before
            # _simplify_arithmetic removes the spurious +0 prefix.
            if isinstance(expr.rhs, Const) and not expr.rhs.alias:
                return (expr.lhs.value + expr.rhs.value, None)
            return (expr.lhs.value, expr.rhs)
        if isinstance(expr.rhs, Const) and not expr.rhs.alias:
            return (expr.rhs.value, expr.lhs)
    return (0, expr)


def _form_pair_expr(op_hi: Expr, op_lo: Expr) -> Optional[Expr]:
    """
    Try to form a 16-bit pair expression from hi and lo byte operands.

    Recognises:
    - Standard 8051 register pair (R0R1, R2R3, R4R5, R6R7) → RegGroup((Rhi, Rlo))
    - Named byte fields sharing the same parent (var.hi / var.lo) → Name(parent)

    Returns None if the operands cannot be identified as a matching pair.
    """
    if (isinstance(op_hi, RegsExpr) and op_hi.is_single
            and isinstance(op_lo, RegsExpr) and op_lo.is_single):
        rhi, rlo = op_hi.name, op_lo.name
        if rhi.startswith("R") and rlo.startswith("R"):
            try:
                hn, ln = int(rhi[1:]), int(rlo[1:])
                if hn + 1 == ln and hn % 2 == 0:
                    return RegGroupExpr((rhi, rlo))
            except ValueError:
                pass
    if isinstance(op_hi, NameExpr) and isinstance(op_lo, NameExpr):
        m_hi = _RE_BYTE_FIELD.match(op_hi.name)
        m_lo = _RE_BYTE_FIELD.match(op_lo.name)
        if m_hi and m_lo and m_hi.group(1) == m_lo.group(1):
            return NameExpr(m_hi.group(1))
    return None


def _collapse_dpl_dph_arithmetic(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Recognise the 16-bit DPTR construction from a carry-linked ADD/ADDC pair:

      DPL = [c_lo +] operand_lo         (arithmetic rhs, no carry flag)
      DPH = [c_hi +] operand_hi + C     (uses carry from DPL's ADD)

    where operand_lo and operand_hi form a recognised 16-bit pair (standard
    register pair R0R1/R2R3/R4R5/R6R7, or named byte-fields var.lo / var.hi).

    Collapses to:
      DPTR = 0x{c_hi:02x}{c_lo:02x} + pair(operand_hi, operand_lo)

    The carry arithmetic is exact because a 16-bit unsigned addition naturally
    propagates the low-byte carry into the high byte:
      (c_hi<<8 | c_lo) + (R_hi<<8 | R_lo) == DPH<<8 | DPL  (mod 65536)

    Recurses into structured node bodies.
    """
    recursed = [node.map_bodies(_collapse_dpl_dph_arithmetic) for node in nodes]
    out: List[HIRNode] = []
    i = 0
    while i < len(recursed):
        node = recursed[i]

        # 1. Match DPL = BinOp(...) with no carry flag in rhs
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegsExpr) and node.lhs.is_single
                and node.lhs.name == "DPL"):
            out.append(node)
            i += 1
            continue
        if not (isinstance(node.rhs, BinOp) and node.rhs.op in ("+", "-")):
            dbg("typesimp",
                f"  [{hex(node.ea)}] dpl-dph-arith: DPL rhs not arithmetic BinOp"
                f" ({type(node.rhs).__name__} {getattr(node.rhs, 'op', '?')!r})"
                f" → {node.rhs.render()!r}")
            out.append(node)
            i += 1
            continue
        if _contains_c(node.rhs):
            dbg("typesimp",
                f"  [{hex(node.ea)}] dpl-dph-arith: DPL rhs contains C → skip")
            out.append(node)
            i += 1
            continue

        # 2. Next node must be DPH = ... (with C somewhere)
        if i + 1 >= len(recursed):
            dbg("typesimp",
                f"  [{hex(node.ea)}] dpl-dph-arith: DPL is last node → no DPH")
            out.append(node)
            i += 1
            continue
        dph_node = recursed[i + 1]
        if not (isinstance(dph_node, Assign)
                and isinstance(dph_node.lhs, RegsExpr) and dph_node.lhs.is_single
                and dph_node.lhs.name == "DPH"):
            dbg("typesimp",
                f"  [{hex(node.ea)}] dpl-dph-arith: next node is not DPH assign"
                f" ({type(dph_node).__name__})")
            out.append(node)
            i += 1
            continue
        hi_no_c = _strip_carry_top(dph_node.rhs)
        if hi_no_c is None:
            dbg("typesimp",
                f"  [{hex(node.ea)}] dpl-dph-arith: DPH rhs has no top-level +C"
                f" → {dph_node.rhs.render()!r}")
            out.append(node)
            i += 1
            continue

        # 3. Decompose both into (const_part, operand)
        c_lo, op_lo = _split_const_operand(node.rhs)
        c_hi, op_hi = _split_const_operand(hi_no_c)

        # 4. Build the 16-bit operand expression
        base_16 = (c_hi << 8) | c_lo
        if op_lo is not None and op_hi is not None:
            # Both sides have a variable operand: must form a recognised 16-bit pair.
            pair = _form_pair_expr(op_hi, op_lo)
            if pair is None:
                dbg("typesimp",
                    f"  [{hex(node.ea)}] dpl-dph-arith: operands don't form a pair"
                    f" hi={op_hi.render()!r} lo={op_lo.render()!r}")
                out.append(node)
                i += 1
                continue
            rhs: Expr = BinOp(Const(base_16), "+", pair) if base_16 != 0 else pair
        elif op_lo is None and op_hi is None:
            # Pure constant DPTR — carry from c_lo is zero for well-formed 8051 code
            # (c_lo is an 8-bit immediate, so always < 256).
            rhs = Const(base_16)
        elif op_lo is not None and op_hi is None:
            # Constant hi byte, variable lo byte: DPTR = base16 + op_lo.
            # e.g. DPL = XRAM[DPTR] + 0x39; DPH = 0xDC + C
            #   → DPTR = 0xDC39 + XRAM[DPTR]
            # The 16-bit math holds: c_hi*256 + (c_lo + op_lo) = base16 + op_lo.
            rhs = BinOp(Const(base_16), "+", op_lo) if base_16 != 0 else op_lo
        else:
            # op_hi is not None, op_lo is None — variable hi byte, constant lo byte.
            # DPL would need to be a pure constant (no BinOp rhs), which is blocked by
            # step 1, so this branch is not reachable for well-formed 8051 HIR.
            dbg("typesimp",
                f"  [{hex(node.ea)}] dpl-dph-arith: variable hi / constant lo — skip"
                f" op_hi={op_hi.render()!r}")
            out.append(node)
            i += 1
            continue

        from pseudo8051.ir.hir import NodeAnnotation as _NA
        new_node = Assign(node.ea, RegExpr("DPTR"), rhs)
        new_node.ann = _NA.merge(node, dph_node)
        out.append(new_node)
        dbg("typesimp",
            f"  [{hex(node.ea)}] dpl-dph-arith: DPTR = {rhs.render()!r}")
        i += 2  # consume both DPL and DPH nodes

    return out


# ── Flow-sensitive DPTR liveness ──────────────────────────────────────────────

def _dptr_live_after(nodes: List[HIRNode]) -> bool:
    """
    Return True if the current DPTR value is read by any downstream node,
    stopping early when an assignment kills it.

    Flow-sensitive: a DPTR write (e.g. DPTR = _dest) or DPTR++ stops the scan.
    Recurses into IfNode/WhileNode/ForNode bodies conservatively.
    """
    for node in nodes:
        if (isinstance(node, Assign)
                and node.lhs == RegExpr("DPTR")):
            return False  # DPTR overwritten; old value dead
        if _is_dptr_inc_node(node):
            return False  # Another DPTR++ also overwrites the old value
        if isinstance(node, IfNode):
            if (_dptr_live_after(node.then_nodes)
                    or _dptr_live_after(node.else_nodes)):
                return True
        elif isinstance(node, (WhileNode, ForNode)):
            if _dptr_live_after(node.body_nodes):
                return True
        else:
            if "DPTR" in node.name_refs():
                return True
    return False


# ── Orphaned DPTR++ pruning ───────────────────────────────────────────────────

def _prune_orphaned_dptr_inc(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Remove DPTR++ nodes that have no downstream DPTR reference.

    Also removes ExprStmt(Const(...)) nodes — pure constant statements with no
    side effects that arise when a DPTR++ (or similar increment) is substituted
    with a known constant value and then folded by _canonicalize_expr.

    Must run after _propagate_values so that XRAM[DPTR] expressions have
    been resolved to XRAM[name] before we check for surviving references.
    Recurses into bodies first so nested orphans are eliminated before outer
    nodes are evaluated.  Uses _dptr_live_after (flow-sensitive).
    """
    nodes = [node.map_bodies(_prune_orphaned_dptr_inc) for node in nodes]
    result: List[HIRNode] = []
    for i, node in enumerate(nodes):
        if _is_dptr_inc_node(node) and not _dptr_live_after(nodes[i + 1:]):
            dbg("typesimp", f"  [{hex(node.ea)}] prune-orphaned-dptr++")
            continue
        if isinstance(node, ExprStmt) and isinstance(node.expr, Const):
            dbg("typesimp", f"  [{hex(node.ea)}] prune-const-exprstmt ({node.expr.render()})")
            continue
        result.append(node)
    return result
