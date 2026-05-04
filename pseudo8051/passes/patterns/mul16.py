"""
passes/patterns/mul16.py — Mul16Pattern.

Recognises the 8051 idiom for 16×16→16 unsigned multiply:

    A   = Rlo1            # load lo byte of first operand
    B   = lo2_expr        # load lo byte of second operand
    {B,A} = A * B         # MUL: lo1 * lo2
    Rtemp = B             # save hi(lo1*lo2)
    swap(A, Rlo1)         # A = Rlo1(orig); Rlo1 = lo(lo1*lo2)
    B   = hi2_expr        # load hi byte of second operand
    {B,A} = A * B         # MUL: lo1 * hi2
    A  += Rtemp           # A += hi(lo1*lo2)
    swap(A, Rhi1)         # A = Rhi1(orig); Rhi1 = accumulated mid-byte
    B   = lo2_expr        # reload lo byte of second operand
    {B,A} = A * B         # MUL: hi1 * lo2
    A  += Rhi1            # A += accumulated mid-byte

where Rhi1 and Rlo1 form a standard adjacent 8051 register pair
(R0R1, R2R3, R4R5, or R6R7).

After the sequence:
  - A    holds the high byte of the 16-bit result
  - Rlo1 holds the low  byte of the 16-bit result (stored by the XCH in step 5)

Produces:
    {A, Rlo1} = pair1 * pair2

where pair1 is the substituted Rhi1:Rlo1 expression and pair2 is derived from
hi2_expr / lo2_expr (full pair if they form a known reg-map pair, lo2_expr only
if hi2_expr is zero, or a byte-shift construct otherwise).
"""

from typing import Dict, List, Optional, Tuple  # noqa: F401 (Optional used in _pair2_expr)

from pseudo8051.ir.hir import HIRNode, Assign, CompoundAssign, ExprStmt
from pseudo8051.ir.expr import Expr, Reg, Regs, Const, BinOp, Call, RegGroup, XRAMRef, Name, Cast
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base import CombineTransform, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns._expr_utils import (
    _subst_all_expr,
    _contains_a,
)


# ── Node-shape helpers ────────────────────────────────────────────────────────

def _is_mul_ab(node: HIRNode) -> bool:
    """True if node is Assign({B,A}, A * B)."""
    return (isinstance(node, Assign)
            and isinstance(node.lhs, Regs)
            and not node.lhs.is_single
            and frozenset(node.lhs.names) == frozenset({"B", "A"})
            and isinstance(node.rhs, BinOp)
            and node.rhs.lhs == Reg("A")
            and node.rhs.op == "*"
            and node.rhs.rhs == Reg("B"))


def _b_expr(node: HIRNode) -> Optional[Expr]:
    """If node is Assign(B, expr) where expr does not contain A, return expr; else None."""
    if not (isinstance(node, Assign) and node.lhs == Reg("B")):
        return None
    if _contains_a(node.rhs):
        return None
    return node.rhs


def _swap_target(node: HIRNode) -> Optional[str]:
    """
    If node is ExprStmt(Call("swap", [A, Reg(X)])), return X; else None.
    X must be a single named register (Regs.is_single == True).
    """
    if not isinstance(node, ExprStmt):
        return None
    expr = node.expr
    if not (isinstance(expr, Call)
            and expr.func_name == "swap"
            and len(expr.args) == 2
            and expr.args[0] == Reg("A")
            and isinstance(expr.args[1], Regs)
            and expr.args[1].is_single):
        return None
    return expr.args[1].name


def _is_adjacent_hi_lo(hi: str, lo: str) -> bool:
    """True when hi/lo form a standard 8051 register pair: Rn / Rn+1 with n even."""
    if not (hi.startswith("R") and lo.startswith("R")):
        return False
    try:
        hi_n, lo_n = int(hi[1:]), int(lo[1:])
    except ValueError:
        return False
    return lo_n == hi_n + 1 and hi_n % 2 == 0


# ── Pair-source look-back ─────────────────────────────────────────────────────

def _find_pair_load_source(nodes: List[HIRNode], i: int,
                            Rhi: str, Rlo: str) -> Optional[Expr]:
    """
    Scan backwards from nodes[i-1] for a node that assigns the pair (Rhi, Rlo).

    Accepts:
      Assign(RegGroup((Rhi, Rlo)), src_expr)   — pair load from source
      Assign(Regs(...) with names == (Rhi, Rlo), src_expr)

    Stops (and returns None) if any node between that assignment and i writes
    either Rhi or Rlo, which would mean the registers are no longer holding the
    loaded value.

    Returns the RHS src_expr on success, or None.
    """
    _MAX_LOOK_BACK = 8
    start = max(0, i - _MAX_LOOK_BACK)
    load_idx: Optional[int] = None
    load_src: Optional[Expr] = None

    for k in range(i - 1, start - 1, -1):
        node = nodes[k]
        if isinstance(node, Assign):
            lhs = node.lhs
            if (isinstance(lhs, Regs)
                    and not lhs.is_single
                    and set(lhs.names) == {Rhi, Rlo}):
                load_idx = k
                load_src = node.rhs
                break
        # If something writes Rhi or Rlo before we find the load, give up
        written = node.written_regs if node is not None else frozenset()
        if Rhi in written or Rlo in written:
            break

    if load_src is None:
        return None

    # Verify nothing between load_idx+1 and i-1 writes Rhi or Rlo
    for k in range(load_idx + 1, i):
        written = nodes[k].written_regs if nodes[k] is not None else frozenset()
        if Rhi in written or Rlo in written:
            return None

    return load_src


# ── Second-operand expression builder ────────────────────────────────────────

def _pair2_expr(lo2: Expr, hi2: Expr,
                reg_map: Dict[str, VarInfo],
                const_state: Optional[Dict[str, int]] = None) -> Expr:
    """
    Build the 16-bit second-operand expression from its lo and hi byte values.

    Priority:
      1. Both are single registers whose reg_map entries are the same VarInfo
         and form an adjacent pair → use the full RegGroup (alias → variable name).
      2. lo2 is a single register that is the low byte of a 2-byte VarInfo,
         and hi2 is either the corresponding hi register or Const(0) (i.e. the
         hi byte was constant-propagated away) → use the full RegGroup.
      3. hi2 is Const(0), or hi2 is a register known to be 0 via const_state
         → zero-extension cast: (uint16_t)lo2.
      4. General → construct (hi2 << 8) | lo2.
    """
    lo2_reg = lo2.name if (isinstance(lo2, Regs) and lo2.is_single) else None
    hi2_reg = hi2.name if (isinstance(hi2, Regs) and hi2.is_single) else None

    # Case 1: explicit adjacent-pair registers with same VarInfo
    if lo2_reg and hi2_reg and _is_adjacent_hi_lo(hi2_reg, lo2_reg):
        vi_lo = reg_map.get(lo2_reg)
        vi_hi = reg_map.get(hi2_reg)
        if (isinstance(vi_lo, VarInfo) and isinstance(vi_hi, VarInfo)
                and vi_lo is vi_hi and not vi_lo.xram_sym):
            return _subst_all_expr(RegGroup((hi2_reg, lo2_reg)), reg_map)

    # Case 2: lo2 is part of a known 2-byte VarInfo pair; hi2 matches or is zero
    if lo2_reg:
        vi_lo = reg_map.get(lo2_reg)
        if (isinstance(vi_lo, VarInfo) and not vi_lo.xram_sym
                and len(vi_lo.regs) == 2 and vi_lo.regs[1] == lo2_reg):
            hi_reg = vi_lo.regs[0]
            if hi2_reg == hi_reg or hi2 == Const(0):
                return _subst_all_expr(RegGroup((hi_reg, lo2_reg)), reg_map)

    # Case 3: hi2 is zero (literal or annotation-known) → zero-extension cast
    hi2_is_zero = (hi2 == Const(0)
                   or (hi2_reg is not None
                       and const_state is not None
                       and const_state.get(hi2_reg) == 0))
    if hi2_is_zero:
        return Cast("uint16_t", _subst_all_expr(lo2, reg_map))

    # Case 4: general → byte-shift construct
    hi2_s = _subst_all_expr(hi2, reg_map)
    lo2_s = _subst_all_expr(lo2, reg_map)
    return BinOp(BinOp(hi2_s, "<<", Const(8)), "|", lo2_s)


# ── Pattern ───────────────────────────────────────────────────────────────────

class Mul16Pattern(CombineTransform):
    """
    Collapse the 8051 16×16→16 multiply idiom into a single assignment:

        {A, Rlo1} = pair1 * pair2

    Consumed: 12 nodes (see module docstring for the full sequence).
    """

    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Tuple[HIRNode, int]]:
        j = i
        n_total = len(nodes)

        def get(k: int) -> Optional[HIRNode]:
            return nodes[k] if k < n_total else None

        # ── Step 1: A = Rlo1 ─────────────────────────────────────────────────
        n1 = get(j)
        if not (isinstance(n1, Assign)
                and n1.lhs == Reg("A")
                and isinstance(n1.rhs, Regs) and n1.rhs.is_single):
            return None
        Rlo1 = n1.rhs.name
        j += 1

        # ── Step 2: B = lo2_expr ─────────────────────────────────────────────
        n2 = get(j)
        lo2 = _b_expr(n2) if n2 is not None else None
        if lo2 is None:
            return None
        j += 1

        # ── Step 3: {B,A} = A * B  (MUL lo1*lo2) ────────────────────────────
        if not (get(j) and _is_mul_ab(get(j))):
            return None
        j += 1

        # ── Step 4: Rtemp = B ────────────────────────────────────────────────
        n4 = get(j)
        if not (isinstance(n4, Assign)
                and n4.rhs == Reg("B")
                and isinstance(n4.lhs, Regs) and n4.lhs.is_single):
            return None
        Rtemp = n4.lhs.name
        j += 1

        # ── Step 5: swap(A, Rlo1) ────────────────────────────────────────────
        n5 = get(j)
        if _swap_target(n5) != Rlo1:
            return None
        j += 1

        # ── Step 6: B = hi2_expr ─────────────────────────────────────────────
        n6 = get(j)
        hi2 = _b_expr(n6) if n6 is not None else None
        if hi2 is None:
            return None
        j += 1

        # ── Step 7: {B,A} = A * B  (MUL lo1*hi2) ────────────────────────────
        if not (get(j) and _is_mul_ab(get(j))):
            return None
        j += 1

        # ── Step 8: A += Rtemp ───────────────────────────────────────────────
        n8 = get(j)
        if not (isinstance(n8, CompoundAssign)
                and n8.lhs == Reg("A")
                and n8.op == "+="
                and n8.rhs == Reg(Rtemp)):
            return None
        j += 1

        # ── Step 9: swap(A, Rhi1) ────────────────────────────────────────────
        Rhi1 = _swap_target(get(j))
        if Rhi1 is None:
            return None
        j += 1

        # Verify Rhi1/Rlo1 form a standard 8051 adjacent register pair
        if not _is_adjacent_hi_lo(Rhi1, Rlo1):
            return None

        # ── Step 10: B = lo2_expr (same as step 2) ───────────────────────────
        n10 = get(j)
        lo2_2 = _b_expr(n10) if n10 is not None else None
        if lo2_2 is None or lo2_2 != lo2:
            return None
        j += 1

        # ── Step 11: {B,A} = A * B  (MUL hi1*lo2) ───────────────────────────
        if not (get(j) and _is_mul_ab(get(j))):
            return None
        j += 1

        # ── Step 12: A += Rhi1 ───────────────────────────────────────────────
        n12 = get(j)
        if not (isinstance(n12, CompoundAssign)
                and n12.lhs == Reg("A")
                and n12.op == "+="
                and n12.rhs == Reg(Rhi1)):
            return None
        j += 1

        # ── Optional step 13: Rhi1 = A ───────────────────────────────────────
        # Consume an immediately following "Rhi1 = A" that saves the high byte
        # of the result into its natural pair register.
        n13 = get(j)
        if (n13 is not None
                and isinstance(n13, Assign)
                and n13.lhs == Reg(Rhi1)
                and n13.rhs == Reg("A")):
            j += 1
            result_hi = Rhi1
        else:
            result_hi = "A"

        # ── Build result ─────────────────────────────────────────────────────
        # Prefer load-source look-back, then annotation-based XRAM parent lookup,
        # then plain reg-map substitution on the register group.
        src = _find_pair_load_source(nodes, i, Rhi1, Rlo1)
        if src is not None:
            pair1 = _subst_all_expr(src, reg_map)
        else:
            # Try to identify the XRAM-backed variable from CProp annotations.
            # If nodes[i].ann.reg_exprs has Rhi1 → XRAM[EXT_Xnn] and there is a
            # non-byte-field VarInfo keyed by "EXT_Xnn" in reg_map, use its name.
            ann0 = nodes[i].ann
            pair1 = None
            if ann0 is not None and ann0.reg_exprs:
                r_hi_expr = ann0.reg_exprs.get(Rhi1)
                r_lo_expr = ann0.reg_exprs.get(Rlo1)
                if (isinstance(r_hi_expr, XRAMRef)
                        and isinstance(r_hi_expr.inner, Name)
                        and isinstance(r_lo_expr, XRAMRef)
                        and isinstance(r_lo_expr.inner, Name)):
                    hi_sym = r_hi_expr.inner.name
                    lo_sym = r_lo_expr.inner.name
                    vi_parent = reg_map.get(hi_sym)
                    if (isinstance(vi_parent, VarInfo)
                            and vi_parent.xram_sym
                            and not vi_parent.is_byte_field):
                        # Optionally verify lo byte is adjacent via _byte_ entry
                        vi_lo_byte = reg_map.get(f"_byte_{lo_sym}")
                        if (vi_lo_byte is None
                                or vi_lo_byte.xram_addr == vi_parent.xram_addr + 1):
                            pair1 = Name(vi_parent.name)
                            dbg("typesimp",
                                f"  [{hex(nodes[i].ea)}] mul16: pair1 from ann "
                                f"{Rhi1}→{hi_sym}={vi_parent.name}")
            if pair1 is None:
                pair1 = _subst_all_expr(RegGroup((Rhi1, Rlo1)), reg_map)
        # Use the step-5 (swap) node's annotation to resolve hi2 constants —
        # it carries reg_consts for R4 etc. that n6 (B = hi2) itself lacks.
        _n5_ann = n5.ann if n5 is not None else None
        _const_state = _n5_ann.reg_consts if _n5_ann is not None else None
        pair2 = _pair2_expr(lo2, hi2, reg_map, _const_state)
        # Use braces only when the hi register is A (non-pair position);
        # a natural Rhi1/Rlo1 pair renders cleanly without braces.
        lhs   = RegGroup((result_hi, Rlo1), brace=(result_hi == "A"))
        result = Assign(nodes[i].ea, lhs, BinOp(pair1, "*", pair2))

        dbg("typesimp",
            f"  [{hex(nodes[i].ea)}] mul16: "
            f"{{{Rhi1},{Rlo1}}} * ... → {result.render()[0][1]}")
        return (result, j)
