"""
passes/patterns/xch_copy.py — XchCopyPattern.

Recognises the 8051 XCH-based dual-pointer copy idiom in two forms:

  ROM → XRAM (movc source):
    [A = 0;]                           ← optional clr A (offset = 0)
    A = CROM[A + DPTR];                ← movc A, @A+DPTR  (or CROM[DPTR] if simplified)
    DPTR++;                            ← inc DPTR  (advance source)
    swap(A, Rlo); swap(A, DPL); swap(A, Rlo);   ┐
    swap(A, Rhi); swap(A, DPH); swap(A, Rhi);   ┘ XCH swap: DPTR ↔ Rhi:Rlo, A preserved
    XRAM[DPTR] = A;                    ← movx @DPTR, A
    DPTR++;                            ← inc DPTR  (advance destination)
    swap(A, Rlo); swap(A, DPL); swap(A, Rlo);   ┐
    swap(A, Rhi); swap(A, DPH); swap(A, Rhi);   ┘ XCH swap again: restore DPTR ↔ Rhi:Rlo

  → XRAM[Rhi:Rlo] = CROM[DPTR];  DPTR++;  Rhi:Rlo++;

  XRAM → XRAM (movx source):
    A = XRAM[DPTR];                    ← movx A, @DPTR
    DPTR++;                            ← inc DPTR  (advance source)
    <same 6-XCH swap + movx write + inc DPTR + 6-XCH swap>

  → XRAM[Rhi:Rlo] = XRAM[DPTR];  DPTR++;  Rhi:Rlo++;
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, ExprStmt
from pseudo8051.ir.expr import Reg, Regs, Const, BinOp, UnaryOp, Call, CROMRef, XRAMRef, RegGroup
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo


# ── Node-shape helpers ────────────────────────────────────────────────────────

def _is_swap(node: HIRNode, a_name: str, x_name: str) -> bool:
    """True if node is ExprStmt(Call("swap", [Reg(a_name), Reg(x_name)]))."""
    if not isinstance(node, ExprStmt):
        return False
    expr = node.expr
    return (isinstance(expr, Call) and expr.func_name == "swap"
            and len(expr.args) == 2
            and expr.args[0] == Reg(a_name)
            and expr.args[1] == Reg(x_name))


def _swap_target(node: HIRNode, a_name: str) -> Optional[str]:
    """Return the non-A argument name if node is swap(a_name, X), else None."""
    if not isinstance(node, ExprStmt):
        return None
    expr = node.expr
    if (isinstance(expr, Call) and expr.func_name == "swap"
            and len(expr.args) == 2
            and expr.args[0] == Reg(a_name)
            and isinstance(expr.args[1], Regs) and expr.args[1].is_single):
        return expr.args[1].name
    return None


def _is_inc_dptr(node: HIRNode) -> bool:
    """True if node is ExprStmt(DPTR++)."""
    return (isinstance(node, ExprStmt)
            and isinstance(node.expr, UnaryOp)
            and node.expr.op == "++"
            and node.expr.operand == Reg("DPTR")
            and node.expr.post)


def _detect_xch_swap(nodes: List[HIRNode], i: int) -> Optional[Tuple[str, str]]:
    """
    Try to match the 6-instruction XCH swap starting at nodes[i].

    The sequence is:
        swap(A, Rlo); swap(A, DPL); swap(A, Rlo);
        swap(A, Rhi); swap(A, DPH); swap(A, Rhi);

    Returns (rlo, rhi) if matched, None otherwise.
    Rlo and Rhi must not be DPL or DPH.
    """
    if i + 6 > len(nodes):
        return None
    rlo = _swap_target(nodes[i], "A")
    if rlo is None or rlo in ("DPL", "DPH"):
        return None
    if not _is_swap(nodes[i + 1], "A", "DPL"):
        return None
    if not _is_swap(nodes[i + 2], "A", rlo):
        return None
    rhi = _swap_target(nodes[i + 3], "A")
    if rhi is None or rhi in ("DPL", "DPH") or rhi == rlo:
        return None
    if not _is_swap(nodes[i + 4], "A", "DPH"):
        return None
    if not _is_swap(nodes[i + 5], "A", rhi):
        return None
    return (rlo, rhi)


def _check_xch_swap(nodes: List[HIRNode], i: int, rlo: str, rhi: str) -> bool:
    """Check that nodes[i:i+6] is an XCH swap for the given rlo/rhi pair."""
    if i + 6 > len(nodes):
        return False
    pairs = [(rlo, "DPL", rlo), (rhi, "DPH", rhi)]
    j = i
    for r, dpl_or_dph, r2 in pairs:
        if not (_is_swap(nodes[j], "A", r)
                and _is_swap(nodes[j + 1], "A", dpl_or_dph)
                and _is_swap(nodes[j + 2], "A", r2)):
            return False
        j += 3
    return True


def _try_match_read(nodes: List[HIRNode], i: int):
    """
    Try to match the source-read at nodes[i].

    Returns (src_expr, new_j, read_ea) where src_expr is CROMRef(Reg("DPTR"))
    or XRAMRef(Reg("DPTR")), new_j is the index after the consumed nodes, and
    read_ea is the EA of the read instruction.  Returns None if no read matches.

    Accepted forms
    --------------
    CROM path (movc):
      [A = 0;]  A = CROM[A + DPTR];     ← clr A + movc (offset explicitly 0)
      [A = 0;]  A = CROM[DPTR];         ← clr A + simplified movc
               A = CROM[DPTR];          ← movc already simplified, no clr needed

    XRAM path (movx):
               A = XRAM[DPTR];          ← movx A, @DPTR
    """
    j = i

    # ── Try XRAM source first (no clr A prefix) ──────────────────────────────
    if (j < len(nodes)
            and isinstance(nodes[j], Assign)
            and nodes[j].lhs == Reg("A")
            and isinstance(nodes[j].rhs, XRAMRef)
            and isinstance(nodes[j].rhs.inner, Regs) and nodes[j].rhs.inner.is_single
            and nodes[j].rhs.inner.name == "DPTR"):
        return (XRAMRef(Reg("DPTR")), j + 1, nodes[j].ea)

    # ── Try CROM source ───────────────────────────────────────────────────────
    # Optional clr A
    if (j < len(nodes)
            and isinstance(nodes[j], Assign)
            and nodes[j].lhs == Reg("A")
            and nodes[j].rhs == Const(0)):
        j += 1

    if j >= len(nodes) or not isinstance(nodes[j], Assign):
        return None
    node = nodes[j]
    if node.lhs != Reg("A") or not isinstance(node.rhs, CROMRef):
        return None
    inner = node.rhs.inner
    if isinstance(inner, BinOp):
        # CROM[A + DPTR] — valid because preceding clr A made A = 0
        if not (inner.lhs == Reg("A")
                and inner.op == "+"
                and inner.rhs == Reg("DPTR")):
            return None
    elif inner == Reg("DPTR"):
        pass   # already simplified form
    else:
        return None
    return (CROMRef(Reg("DPTR")), j + 1, node.ea)


# ── Pattern ───────────────────────────────────────────────────────────────────

class XchCopyPattern(Pattern):
    """
    Collapse the XCH-based dual-pointer copy idiom (ROM→XRAM or XRAM→XRAM) into:
        XRAM[Rhi:Rlo] = CROM[DPTR];   (or XRAM[DPTR] for XRAM source)
        DPTR++;
        Rhi:Rlo++;
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:

        # ── Source read (CROM or XRAM) ────────────────────────────────────────
        read = _try_match_read(nodes, i)
        if read is None:
            return None
        src_expr, j, read_ea = read

        # ── inc DPTR  (source pointer advance) ───────────────────────────────
        if j >= len(nodes) or not _is_inc_dptr(nodes[j]):
            return None
        j += 1

        # ── First XCH swap: DPTR ↔ Rhi:Rlo ──────────────────────────────────
        swap1 = _detect_xch_swap(nodes, j)
        if swap1 is None:
            return None
        rlo, rhi = swap1
        j += 6

        # ── XRAM[DPTR] = A  (movx @DPTR, A) ─────────────────────────────────
        if j >= len(nodes):
            return None
        movx = nodes[j]
        if not (isinstance(movx, Assign)
                and isinstance(movx.lhs, XRAMRef)
                and movx.lhs.inner == Reg("DPTR")
                and movx.rhs == Reg("A")):
            return None
        j += 1

        # ── inc DPTR  (destination pointer advance) ───────────────────────────
        if j >= len(nodes) or not _is_inc_dptr(nodes[j]):
            return None
        j += 1

        # ── Second XCH swap: DPTR ↔ Rhi:Rlo  (restore source) ───────────────
        if not _check_xch_swap(nodes, j, rlo, rhi):
            return None
        j += 6

        # ── Build replacement ─────────────────────────────────────────────────
        dst = RegGroup((rhi, rlo))
        src_kind = "CROM" if isinstance(src_expr, CROMRef) else "XRAM"
        dbg("typesimp",
            f"  [{hex(read_ea)}] xch_copy: XRAM[{rhi}{rlo}] = {src_kind}[DPTR]; "
            f"DPTR++; {rhi}{rlo}++;")
        return ([
            Assign(read_ea, XRAMRef(dst), src_expr),
            ExprStmt(read_ea, UnaryOp("++", Reg("DPTR"), post=True)),
            ExprStmt(read_ea, UnaryOp("++", RegGroup((rhi, rlo)), post=True)),
        ], j)
