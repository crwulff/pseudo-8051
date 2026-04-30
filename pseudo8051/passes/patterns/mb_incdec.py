"""
passes/patterns/mb_incdec.py — MultiByteIncDecPattern.

Recognises the 8051 idiom for multi-byte increment/decrement:

    # XRAM-based unit (3 nodes each):
    A = xram_expr;
    A++;           (or A--)
    xram_expr = A;

    # Register-based unit (1 node each):
    Rn++;          (or Rn--)   where Rn != A

Between every two consecutive units there is a carry/borrow skip:
    if (A != 0) goto skip;    (for ++) — XRAM units
    if (Rn != 0) goto skip;   (for ++) — register units
    if (A != 0xFF) goto skip; (for --) — XRAM units
    if (Rn != 0xFF) goto skip;(for --) — register units

A terminal Label(skip) immediately follows the last unit.  The whole
chain collapses to a single "var++;" or "var--;" statement.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, ExprStmt, IfGoto, Label
from pseudo8051.ir.expr import Expr, Reg, Regs, Const, Name, XRAMRef, BinOp, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import CombineTransform, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo, _contains_a


def _try_unit(nodes: List[HIRNode], j: int,
              expected_op: Optional[str]) -> Optional[Tuple[Expr, str, int]]:
    """
    Try to parse a single increment/decrement unit starting at nodes[j].

    Returns (byte_expr, op, new_j) or None.
    byte_expr is the expression representing the byte being modified:
      - For XRAM units: the xram_expr (e.g. XRAMRef(Name(sym)))
      - For register units: the Reg(rn) node

    XRAM units may optionally be preceded by ``Assign(DPTR, Name(sym))``
    (the 8051 MOVX @DPTR addressing mode setup).  That prefix node is
    consumed transparently when present.
    """
    n_total = len(nodes)

    # XRAM-based unit: [Assign(DPTR, Name(sym))]? Assign(A, xram_e),
    #                   ExprStmt(A++/--), Assign(xram_e, A)
    xram_j = j
    if xram_j < n_total:
        n = nodes[xram_j]
        if isinstance(n, Assign) and n.lhs == Reg("DPTR"):
            if isinstance(n.rhs, Name):
                dptr_sym: Optional[str] = n.rhs.name
            elif isinstance(n.rhs, Const):
                dptr_sym = n.rhs.alias
            else:
                dptr_sym = None
            if dptr_sym is not None:
                # Only consume if the very next node is A = XRAM[dptr_sym]
                nxt = nodes[xram_j + 1] if xram_j + 1 < n_total else None
                if (nxt is not None
                        and isinstance(nxt, Assign)
                        and nxt.lhs == Reg("A")
                        and isinstance(nxt.rhs, XRAMRef)):
                    inner = nxt.rhs.inner
                    inner_sym = (inner.name if isinstance(inner, Name)
                                 else inner.alias if isinstance(inner, Const)
                                 else None)
                    if inner_sym == dptr_sym:
                        xram_j += 1  # skip the DPTR prefix

    if xram_j + 2 < n_total:
        n0, n1, n2 = nodes[xram_j], nodes[xram_j + 1], nodes[xram_j + 2]
        if (isinstance(n0, Assign)
                and n0.lhs == Reg("A")
                and not _contains_a(n0.rhs)
                and isinstance(n1, ExprStmt)
                and isinstance(n1.expr, UnaryOp)
                and n1.expr.post
                and n1.expr.operand == Reg("A")
                and n1.expr.op in ("++", "--")
                and isinstance(n2, Assign)
                and n2.lhs == n0.rhs
                and n2.rhs == Reg("A")):
            unit_op = n1.expr.op
            if expected_op is None or unit_op == expected_op:
                return (n0.rhs, unit_op, xram_j + 3)

    # Register-based unit: ExprStmt(Rn++/--)  where Rn != A
    if j < n_total:
        n0 = nodes[j]
        if (isinstance(n0, ExprStmt)
                and isinstance(n0.expr, UnaryOp)
                and n0.expr.post
                and n0.expr.op in ("++", "--")
                and isinstance(n0.expr.operand, Regs) and n0.expr.operand.is_single
                and n0.expr.operand != Reg("A")):
            unit_op = n0.expr.op
            if expected_op is None or unit_op == expected_op:
                return (n0.expr.operand, unit_op, j + 1)

    return None


def _carry_operand_matches(lhs: Expr, byte_expr: Expr) -> bool:
    """
    Check whether the IfGoto carry-check LHS matches the unit's byte expression.

    XRAM unit (byte_expr is not a plain Reg): lhs must be Reg("A").
    Register unit (byte_expr is a Reg): lhs must equal byte_expr.
    """
    if isinstance(byte_expr, Regs) and byte_expr.is_single:
        return lhs == byte_expr
    return lhs == Reg("A")


def _resolve_var_name(byte_exprs: List[Expr], op: str,
                      reg_map: Dict[str, VarInfo]) -> Optional[str]:
    """
    Resolve the multi-byte variable name from a list of per-byte expressions.

    XRAM case: all byte_exprs should be XRAMRef(Name(sym)) entries whose
    _byte_ VarInfo all share the same parent name.

    Register case: all byte_exprs are Reg instances that should all map
    to the same VarInfo (or we build a concatenated register name).
    """
    if not byte_exprs:
        return None

    if isinstance(byte_exprs[0], Regs) and byte_exprs[0].is_single:
        # Register-based: all regs should map to the same VarInfo
        first_rn = byte_exprs[0].name
        first_vinfo = reg_map.get(first_rn)
        if first_vinfo is not None and not first_vinfo.xram_sym:
            same = all(
                reg_map.get(e.name) is first_vinfo   # type: ignore[union-attr]
                for e in byte_exprs
                if isinstance(e, Regs) and e.is_single
            )
            if same:
                return first_vinfo.name

        # Try concatenated register name (e.g. "R6R7")
        concat = "".join(e.name for e in byte_exprs if isinstance(e, Regs) and e.is_single)  # type: ignore[union-attr]
        if concat in reg_map and not reg_map[concat].xram_sym:
            return reg_map[concat].name
        # Fall back to the concatenated register names themselves
        return concat if concat else None

    # XRAM-based: look up _byte_ VarInfo entries and check parent name
    parent: Optional[str] = None
    for expr in byte_exprs:
        if not isinstance(expr, XRAMRef):
            return None
        sym = expr.inner.render()
        binfo = reg_map.get(f"_byte_{sym}")
        if binfo is None or not binfo.is_byte_field:
            return None
        # Strip the per-byte suffix (.hi / .lo / .bN) to get the parent name
        dot = binfo.name.rfind(".")
        if dot < 0:
            return None
        p = binfo.name[:dot]
        if parent is None:
            parent = p
        elif p != parent:
            return None  # inconsistent parent names
    return parent


class IfNodeIncDecPattern(CombineTransform):
    """Collapse IfNode-based 16-bit inc/dec (produced by multi-tail loop structuring)."""

    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Tuple[HIRNode, int]]:
        from pseudo8051.ir.hir import IfNode as _IfNode

        # 1. Outer (lo-byte) unit
        unit = _try_unit(nodes, i, None)
        if unit is None:
            return None
        lo_expr, op, j = unit

        # 2. Must be followed by IfNode with empty else
        if j >= len(nodes) or not isinstance(nodes[j], _IfNode):
            return None
        ifn = nodes[j]
        if ifn.else_nodes:
            return None

        # 3. IfNode condition must be the carry check: A == 0 (for ++) or A == 0xFF (--)
        expected_overflow = Const(0) if op == "++" else Const(0xFF)
        cond = ifn.condition
        if not (isinstance(cond, BinOp)
                and cond.op == "=="
                and cond.lhs == Reg("A")
                and cond.rhs == expected_overflow):
            return None

        # 4. Inner (hi-byte) unit must be at position 0 of then_nodes, with nothing after
        inner = _try_unit(ifn.then_nodes, 0, op)
        if inner is None:
            return None
        hi_expr, _, inner_j = inner
        if inner_j != len(ifn.then_nodes):   # unexpected trailing nodes
            return None

        # 5. Resolve variable name from both byte exprs
        byte_exprs = [lo_expr, hi_expr]
        var_name = _resolve_var_name(byte_exprs, op, reg_map)
        if var_name is None:
            return None

        ea = nodes[i].ea
        dbg("typesimp", f"  ifnode-incdec: {var_name}{op};  (nodes {i}–{j}, ea={ea:#x})")
        return (ExprStmt(ea, UnaryOp(op, Name(var_name), post=True)), j + 1)


class MultiByteIncDecPattern(CombineTransform):
    """Collapse 8051 multi-byte increment/decrement sequences into var++/var--."""

    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Tuple[HIRNode, int]]:

        j = i
        units: List[Tuple[Expr, int]] = []   # (byte_expr, ea)
        op: Optional[str] = None
        skip_label: Optional[str] = None

        while j < len(nodes):
            unit = _try_unit(nodes, j, op)
            if unit is None:
                break
            byte_expr, unit_op, new_j = unit
            if op is None:
                op = unit_op
            units.append((byte_expr, nodes[j].ea))
            j = new_j

            # After each unit: check for carry IfGoto
            if j < len(nodes) and isinstance(nodes[j], IfGoto):
                ig = nodes[j]
                expected_overflow = Const(0) if op == "++" else Const(0xFF)
                if (isinstance(ig.cond, BinOp)
                        and ig.cond.op == "!="
                        and ig.cond.rhs == expected_overflow
                        and _carry_operand_matches(ig.cond.lhs, byte_expr)):
                    if skip_label is None:
                        skip_label = ig.label
                    elif ig.label != skip_label:
                        break   # inconsistent labels → stop
                    j += 1
                    continue    # try next unit
            break   # no carry check → last unit

        if len(units) < 2 or skip_label is None:
            return None

        # Must be followed by Label(skip_label)
        if j >= len(nodes) or not isinstance(nodes[j], Label):
            return None
        if nodes[j].name != skip_label:
            return None
        j += 1  # consume label

        # Resolve variable name
        byte_exprs = [u[0] for u in units]
        var_name = _resolve_var_name(byte_exprs, op, reg_map)
        if var_name is None:
            return None

        ea = units[0][1]
        dbg("typesimp", f"  mb-incdec: {var_name}{op};  (nodes {i}–{j-1}, ea={ea:#x})")
        return (ExprStmt(ea, UnaryOp(op, Name(var_name), post=True)), j)
