"""
passes/patterns/accum_fold.py — AccumFoldPattern.

Collapses 8051 A-expression chains such as:

    DPTR = sym;                  # optional DPTR prefix
    A = XRAM[sym];               # or any expr not containing A
    A &= 1;                      # 0 or more compound assigns
    if (A == 0) goto label;      # IfGoto / IfNode / Assign / ReturnStmt terminal

into a single node with A substituted:

    if ((XRAM[sym] & 1) == 0) goto label;

Registered *after* AccumRelayPattern so the pure 2-node relay
(A = expr; target = A; with no ops and no DPTR prefix) is still
owned by AccumRelayPattern.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement, Assign, CompoundAssign, ExprStmt, ReturnStmt, IfGoto, IfNode
from pseudo8051.ir.expr import Expr, Reg, Name, XRAMRef, BinOp, RegGroup, Const, Cast
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo,
    _subst_all_expr,
    _walk_expr,
)

# Map CompoundAssign op to the corresponding binary op
_OP_WITHOUT_EQ = {
    "+=": "+", "-=": "-", "*=": "*", "/=": "/",
    "&=": "&", "|=": "|", "^=": "^",
    "<<=": "<<", ">>=": ">>",
}


def _is_adjacent_hi_lo(hi: str, lo: str) -> bool:
    """True if hi/lo form a standard 8051 pair: Rn / Rn+1 with n even."""
    if not (hi.startswith("R") and lo.startswith("R")):
        return False
    try:
        hi_n, lo_n = int(hi[1:]), int(lo[1:])
    except ValueError:
        return False
    return lo_n == hi_n + 1 and hi_n % 2 == 0


def _try_mul_pair_lookahead(nodes, j, terminal, a_expr, full_product,
                            pair_expr, reg_map, a_start_node):
    """
    Attempt to consume the 8051 mul-result relay pattern:

        Rn = A          (j)    save lo byte        <- terminal we're at
        A  = B          (j+1)  load hi byte from B
        Rm = A          (j+2)  store hi byte
        A  = Rn         (j+3)  restore lo byte
        [CompoundAssigns on A]
        Rk = A          (jj)   store final lo byte

    If Rm and Rk form an adjacent pair (R2R3, R4R5, ...), returns:
        ([Assign(RegGroup((Rm, Rk)), pair_expr_subst)], jj + 1)
    Otherwise returns None (fall-through to existing behaviour).
    """
    rn = terminal.lhs

    # Scan forward past safe intermediate nodes to find A = B.
    # Track register assignments in the interleaved block so that compound
    # assigns referencing those registers use the new values, not reg_map.
    _MAX_SKIP = 10
    skip_end = j + 1
    interleaved_vals: Dict[str, Expr] = {}
    while skip_end < len(nodes) and (skip_end - j) <= _MAX_SKIP:
        sn = nodes[skip_end]
        if isinstance(sn, Assign) and sn.lhs == Reg("A") and sn.rhs == Reg("B"):
            break   # found A = B
        if isinstance(sn, ExprStmt):
            skip_end += 1; continue
        if isinstance(sn, Assign) and sn.lhs != Reg("B") and sn.lhs != rn:
            # Record: resolve rhs through already-seen interleaved vals
            if isinstance(sn.lhs, Reg):
                def _subst_iv(e: Expr, _iv=interleaved_vals) -> Expr:
                    if isinstance(e, Reg) and e.name in _iv:
                        return _iv[e.name]
                    return e
                interleaved_vals[sn.lhs.name] = _walk_expr(sn.rhs, _subst_iv)
            skip_end += 1; continue
        return None   # unsafe: B or rn clobbered, or unrecognised type
    else:
        return None   # hit limit or end without finding A = B

    # skip_end points at A = B; n2, n3 follow it
    if skip_end + 2 >= len(nodes):
        return None
    n2, n3 = nodes[skip_end + 1], nodes[skip_end + 2]

    # Rm = A  (hi-byte destination)
    if not (isinstance(n2, Assign) and n2.rhs == Reg("A") and n2.lhs != Reg("A")):
        return None
    rm = n2.lhs
    # A = Rn  (restore lo byte)
    if not (isinstance(n3, Assign) and n3.lhs == Reg("A") and n3.rhs == rn):
        return None

    # Optional compound assigns on A after restore
    jj = skip_end + 3
    lo_a_expr    = a_expr
    pair_expr_la = pair_expr
    while jj < len(nodes):
        cn = nodes[jj]
        if not (isinstance(cn, CompoundAssign) and cn.lhs == Reg("A")):
            break
        bin_op = _OP_WITHOUT_EQ.get(cn.op)
        if bin_op is None:
            break
        lo_a_expr    = BinOp(lo_a_expr,    bin_op, cn.rhs)
        pair_expr_la = BinOp(pair_expr_la, bin_op, cn.rhs)
        jj += 1

    # Rk = A  (lo-byte destination)
    if jj >= len(nodes):
        return None
    n_final = nodes[jj]
    if not (isinstance(n_final, Assign)
            and n_final.rhs == Reg("A") and n_final.lhs != Reg("A")):
        return None
    rk = n_final.lhs

    # Must form an adjacent standard pair
    if not (isinstance(rm, Reg) and isinstance(rk, Reg)
            and _is_adjacent_hi_lo(rm.name, rk.name)):
        return None

    # Apply interleaved overrides first (e.g. R5 reassigned inside the
    # interleaved block), then reg_map for any remaining Reg references.
    if interleaved_vals:
        def _apply_iv(e: Expr, _iv=interleaved_vals) -> Expr:
            if isinstance(e, Reg) and e.name in _iv:
                return _iv[e.name]
            return e
        pair_expr_la = _walk_expr(pair_expr_la, _apply_iv)
    pair_expr_subst = _subst_all_expr(pair_expr_la, reg_map)
    pair_group = RegGroup((rm.name, rk.name))
    dbg("typesimp",
        f"  [{hex(a_start_node.ea)}] accum_fold (mul pair): {rm.name}{rk.name} = {pair_expr_subst.render()}")
    return ([Assign(a_start_node.ea, pair_group, pair_expr_subst)], jj + 1)


_RE_HEX_CONST = re.compile(r'^0x[0-9a-fA-F]+$')
_RE_DEC_CONST = re.compile(r'^\d+$')
_RE_REG_TOKEN = re.compile(r'^(R[0-7]|A|B|DPTR|DPH|DPL|C)$')


def _parse_simple_expr(s: str) -> Optional[Expr]:
    """Parse a simple single-token string into an Expr node.

    Returns Const for numeric literals, Reg for known register names,
    Name for other bare identifiers, or None for complex expressions
    (spaces, operators, parens etc.) that should not be attempted.
    """
    s = s.strip()
    if not s or ' ' in s or '(' in s or ')' in s:
        return None
    if _RE_HEX_CONST.match(s):
        return Const(int(s, 16))
    if _RE_DEC_CONST.match(s):
        return Const(int(s))
    if _RE_REG_TOKEN.match(s):
        return Reg(s)
    # bare identifier (parameter name, local variable)
    if re.match(r'^[A-Za-z_]\w*$', s):
        return Name(s)
    return None


def _contains_a(expr: Expr) -> bool:
    """Return True if Reg("A") appears anywhere in the Expr tree."""
    found = [False]

    def _fn(e: Expr) -> Expr:
        if e == Reg("A"):
            found[0] = True
        return e

    _walk_expr(expr, _fn)
    return found[0]


def _subst_a(expr: Expr, replacement: Expr) -> Expr:
    """Replace all occurrences of Reg("A") in expr with replacement."""
    def _fn(e: Expr) -> Expr:
        if e == Reg("A"):
            return replacement
        return e
    return _walk_expr(expr, _fn)


class AccumFoldPattern(Pattern):
    """Collapse A-expression chains into a single terminal node."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        j = i
        dptr_consumed = False

        # ── 1. Optional DPTR prefix ───────────────────────────────────────────
        if j < len(nodes):
            n = nodes[j]
            if (isinstance(n, Assign)
                    and n.lhs == Reg("DPTR")
                    and isinstance(n.rhs, Name)):
                dptr_sym = n.rhs.name
                # Peek at next: must be A = XRAM[dptr_sym]
                if j + 1 < len(nodes):
                    nxt = nodes[j + 1]
                    if (isinstance(nxt, Assign)
                            and nxt.lhs == Reg("A")
                            and isinstance(nxt.rhs, XRAMRef)
                            and isinstance(nxt.rhs.inner, Name)
                            and nxt.rhs.inner.name == dptr_sym):
                        dptr_consumed = True
                        j += 1   # skip DPTR node; j now points at A = XRAM[sym]
                    else:
                        # DPTR node present but next is not the expected XRAM load
                        return None
                else:
                    return None

        # ── 2. A-chain start ──────────────────────────────────────────────────
        if j >= len(nodes):
            return None
        a_start_node = nodes[j]
        if not (isinstance(a_start_node, Assign)
                and a_start_node.lhs == Reg("A")
                and not _contains_a(a_start_node.rhs)):
            return None

        a_expr: Expr = a_start_node.rhs
        j += 1

        # ── 3. Compound assigns ───────────────────────────────────────────────
        num_compound = 0
        _full_product: Optional[Expr] = None
        _pair_expr:    Optional[Expr] = None
        skipped: List[HIRNode] = []   # safe interleaved non-A/non-B assigns
        _ACCUM_REGS = {"A", "B", "DPTR", "DPH", "DPL"}
        while j < len(nodes):
            cn = nodes[j]

            # Normalize A += A → A *= 2 for chain purposes (issue 2.2)
            if (isinstance(cn, CompoundAssign)
                    and cn.lhs == Reg("A")
                    and cn.op == "+=" and cn.rhs == Reg("A")):
                a_expr = BinOp(a_expr, "*", Const(2))
                num_compound += 1
                j += 1
                continue

            # MUL AB: Assign(B, b_val) + Assign({B,A}, A*B) → Cast(uint8_t, a_expr*b_val) (issue 2.1)
            if (isinstance(cn, Assign)
                    and isinstance(cn.lhs, Reg) and cn.lhs.name == "B"
                    and not _contains_a(cn.rhs)
                    and j + 1 < len(nodes)):
                nxt = nodes[j + 1]
                if (isinstance(nxt, Assign)
                        and isinstance(nxt.lhs, RegGroup)
                        and set(nxt.lhs.regs) == {"B", "A"}
                        and isinstance(nxt.rhs, BinOp)
                        and nxt.rhs.lhs == Reg("A")
                        and nxt.rhs.op == "*"
                        and nxt.rhs.rhs == Reg("B")):
                    b_val         = _subst_all_expr(cn.rhs, reg_map)
                    _full_product = BinOp(a_expr, "*", b_val)
                    a_expr        = Cast("uint8_t", _full_product)
                    _pair_expr    = _full_product
                    j += 2
                    num_compound += 1
                    continue

            if not (isinstance(cn, CompoundAssign)
                    and cn.lhs == Reg("A")
                    and not _contains_a(cn.rhs)):
                # Statement fallback: "A op= rhs;"
                if isinstance(cn, Statement):
                    m_stmt = re.match(r'^A\s*(\+=|-=|\*=|&=|\|=|\^=|<<=|>>=)\s*(.+);$',
                                      cn.text)
                    if m_stmt:
                        bin_op_stmt = _OP_WITHOUT_EQ.get(m_stmt.group(1))
                        if bin_op_stmt is not None:
                            rhs_node = _parse_simple_expr(m_stmt.group(2).strip())
                            if rhs_node is not None and not _contains_a(rhs_node):
                                a_expr = BinOp(a_expr, bin_op_stmt, rhs_node)
                                if _pair_expr is not None:
                                    _pair_expr = BinOp(_pair_expr, bin_op_stmt, rhs_node)
                                num_compound += 1
                                j += 1
                                continue
                # Safe interleaved: Assign to a non-accumulator register that
                # doesn't read A.  Collect and re-emit in the output so that
                # downstream pruning can handle them normally.
                if (isinstance(cn, Assign)
                        and isinstance(cn.lhs, Reg)
                        and cn.lhs.name not in _ACCUM_REGS
                        and not _contains_a(cn.rhs)):
                    skipped.append(cn)
                    j += 1
                    continue
                break
            bin_op = _OP_WITHOUT_EQ.get(cn.op)
            if bin_op is None:
                break
            a_expr = BinOp(a_expr, bin_op, cn.rhs)
            if _pair_expr is not None:
                _pair_expr = BinOp(_pair_expr, bin_op, cn.rhs)
            num_compound += 1
            j += 1

        # ── 4. Terminal node ──────────────────────────────────────────────────
        if j >= len(nodes):
            return None
        terminal = nodes[j]

        a_expr_subst = _subst_all_expr(a_expr, reg_map)

        # IfGoto: substitute A in condition
        if isinstance(terminal, IfGoto) and _contains_a(terminal.cond):
            new_cond = _subst_a(terminal.cond, a_expr_subst)
            dbg("typesimp", f"  [{hex(a_start_node.ea)}] accum_fold (IfGoto): folded {a_expr_subst.render()} into cond")
            return (skipped + [IfGoto(a_start_node.ea, new_cond, terminal.label)], j + 1)

        # IfNode: substitute A in condition (Expr or str)
        if isinstance(terminal, IfNode):
            cond = terminal.condition
            if isinstance(cond, Expr) and _contains_a(cond):
                new_cond: object = _subst_a(cond, a_expr_subst)
                new_then = simplify(terminal.then_nodes, reg_map)
                new_else = simplify(terminal.else_nodes, reg_map) if terminal.else_nodes else []
                dbg("typesimp", f"  [{hex(a_start_node.ea)}] accum_fold (IfNode expr): folded {a_expr_subst.render()} into cond")
                return (skipped + [IfNode(a_start_node.ea, new_cond, new_then, new_else)], j + 1)
            if isinstance(cond, str) and re.search(r'\bA\b', cond):
                rendered = a_expr_subst.render()
                new_cond_str = re.sub(r'\bA\b', rendered, cond)
                new_then = simplify(terminal.then_nodes, reg_map)
                new_else = simplify(terminal.else_nodes, reg_map) if terminal.else_nodes else []
                dbg("typesimp", f"  [{hex(a_start_node.ea)}] accum_fold (IfNode str): folded {rendered} into cond")
                return (skipped + [IfNode(a_start_node.ea, new_cond_str, new_then, new_else)], j + 1)

        # Assign(target, Reg("A")) where target != A:
        # only fold if there was at least one compound assign or a DPTR prefix
        # (pure 2-node relay without ops is left to AccumRelayPattern).
        if (isinstance(terminal, Assign)
                and terminal.rhs == Reg("A")
                and terminal.lhs != Reg("A")
                and (num_compound > 0 or dptr_consumed)):

            # Try to fold mul hi+lo bytes into a register pair
            if _full_product is not None:
                result = _try_mul_pair_lookahead(
                    nodes, j, terminal, a_expr, _full_product,
                    _pair_expr, reg_map, a_start_node)
                if result is not None:
                    replacement_nodes, new_j = result
                    return (skipped + replacement_nodes, new_j)

            dbg("typesimp", f"  [{hex(a_start_node.ea)}] accum_fold (Assign relay): folded {a_expr_subst.render()} into {terminal.lhs.render()}")
            new_node = Assign(a_start_node.ea, terminal.lhs, a_expr_subst)
            new_node.ann = terminal.ann   # preserve call_arg_ann for downstream pair folding
            return (skipped + [new_node], j + 1)

        # ReturnStmt(Reg("A")): only if compound > 0 or DPTR consumed
        if (isinstance(terminal, ReturnStmt)
                and terminal.value == Reg("A")
                and (num_compound > 0 or dptr_consumed)):
            dbg("typesimp", f"  [{hex(a_start_node.ea)}] accum_fold (ReturnStmt): folded {a_expr_subst.render()}")
            return (skipped + [ReturnStmt(a_start_node.ea, a_expr_subst)], j + 1)

        # Statement terminal: "Rn = A;" — only if compound > 0 or DPTR consumed
        if isinstance(terminal, Statement) and (num_compound > 0 or dptr_consumed):
            m_term = re.match(r'^(\w+)\s*=\s*A;$', terminal.text)
            if m_term and m_term.group(1) != "A":
                target_name = m_term.group(1)
                dbg("typesimp",
                    f"  [{hex(a_start_node.ea)}] accum_fold (Stmt relay): folded {a_expr_subst.render()} into {target_name}")
                return (skipped + [Statement(a_start_node.ea,
                                             f"{target_name} = {a_expr_subst.render()};")], j + 1)

        # Carry-comparison terminal: A=val; A-=sub; if(C)/while(C) → if(val < sub)
        # After SUBB, carry is set iff the subtraction produced a borrow (val < sub unsigned).
        # Only applies when the net accumulated expression is a single subtraction BinOp.
        if (num_compound > 0
                and isinstance(a_expr, BinOp) and a_expr.op == "-"):
            sub_raw = a_expr.rhs
            # Strip "+ 0" or "0 +" that CProp leaves when carry-in was cleared (CLR C before SUBB)
            if isinstance(sub_raw, BinOp) and sub_raw.op == "+":
                if isinstance(sub_raw.rhs, Const) and sub_raw.rhs.value == 0:
                    sub_raw = sub_raw.lhs
                elif isinstance(sub_raw.lhs, Const) and sub_raw.lhs.value == 0:
                    sub_raw = sub_raw.rhs
            minuend    = _subst_all_expr(a_expr.lhs, reg_map)
            subtrahend = _subst_all_expr(sub_raw,    reg_map)
            carry_cond = BinOp(minuend, "<",  subtrahend)
            no_carry   = BinOp(minuend, ">=", subtrahend)

            def _is_carry(c) -> bool:
                return isinstance(c, Reg) and c.name == "C"

            def _is_not_carry(c) -> bool:
                from pseudo8051.ir.expr import UnaryOp
                return (isinstance(c, UnaryOp) and c.op == "!" and _is_carry(c.operand))

            if isinstance(terminal, IfGoto):
                if _is_carry(terminal.cond):
                    dbg("typesimp",
                        f"  [{hex(a_start_node.ea)}] accum_fold (carry< IfGoto): {minuend.render()} < {subtrahend.render()}")
                    return (skipped + [IfGoto(a_start_node.ea, carry_cond, terminal.label)], j + 1)
                if _is_not_carry(terminal.cond):
                    dbg("typesimp",
                        f"  [{hex(a_start_node.ea)}] accum_fold (carry>= IfGoto): {minuend.render()} >= {subtrahend.render()}")
                    return (skipped + [IfGoto(a_start_node.ea, no_carry, terminal.label)], j + 1)

            if isinstance(terminal, IfNode):
                cond = terminal.condition
                new_then = simplify(terminal.then_nodes, reg_map)
                new_else = simplify(terminal.else_nodes, reg_map) if terminal.else_nodes else []
                if _is_carry(cond):
                    dbg("typesimp",
                        f"  [{hex(a_start_node.ea)}] accum_fold (carry< IfNode): {minuend.render()} < {subtrahend.render()}")
                    return (skipped + [IfNode(a_start_node.ea, carry_cond,
                                             new_then, new_else)], j + 1)
                if _is_not_carry(cond):
                    dbg("typesimp",
                        f"  [{hex(a_start_node.ea)}] accum_fold (carry>= IfNode): {minuend.render()} >= {subtrahend.render()}")
                    return (skipped + [IfNode(a_start_node.ea, no_carry,
                                             new_then, new_else)], j + 1)

        return None
