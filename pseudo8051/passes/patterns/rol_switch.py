"""
passes/patterns/rol_switch.py — RolSwitchPattern.

Collapses the 8051 indirect-jump preamble before a switch:

    A = rol8(A);   ← N prefix rotates  (encode A << N)
    ...            ← optional compound assigns on A (e.g. A |= DPL)
    A = rol8(A);   ← K step-size rotates (paired with >> K in the switch subject)
    switch (A >> K) { ... }

into:

    switch ((A << N) <compound ops>) { ... }

The step-size rotates (the last K rol8 calls) and the >> K in the switch subject
cancel each other, leaving the N prefix rotates folded into the discriminant.

Common 2-byte jump table (K=1):

    rl A; rl A; orl A, DPL; rl A; switch (A >> 1)  →  switch ((A << 2) | DPL)
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, CompoundAssign, SwitchNode
from pseudo8051.ir.expr import Expr, Reg, Const, BinOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo, _subst_all_expr

_OP_WITHOUT_EQ = {
    "+=": "+", "-=": "-", "*=": "*", "/=": "/",
    "&=": "&", "|=": "|", "^=": "^",
    "<<=": "<<", ">>=": ">>",
}


def _is_rol_a(node: HIRNode) -> bool:
    """True if node is Assign(Reg("A"), (A << 1) | (A >> 7))."""
    if not (isinstance(node, Assign) and node.lhs == Reg("A")):
        return False
    rhs = node.rhs
    # (A << 1) | (A >> 7)
    return (isinstance(rhs, BinOp) and rhs.op == "|"
            and isinstance(rhs.lhs, BinOp) and rhs.lhs.op == "<<"
            and rhs.lhs.lhs == Reg("A") and rhs.lhs.rhs == Const(1)
            and isinstance(rhs.rhs, BinOp) and rhs.rhs.op == ">>"
            and rhs.rhs.lhs == Reg("A") and rhs.rhs.rhs == Const(7))


class RolSwitchPattern(Pattern):
    """
    Collapse a pre-switch rol8(A) preamble into the switch discriminant.

    Pattern (starting at the first A = rol8(A) before the switch):
        [A = rol8(A)] * N_before    — prefix rotates (N_before ≥ 1)
        [A op= rhs  ] * C           — 0 or more compound assigns
        [A = rol8(A)] * N_after     — step-size rotates
        SwitchNode(A >> K, ...)     — switch subject

    Determines effective counts:
      • If N_after ≥ K: step = K,  prefix = N_before + (N_after - K)
      • If N_after < K and no compounds: step = K, prefix = N_before - (K - N_after)
        (borrows from N_before since they are contiguous with no compounds between)
      • Otherwise: no match

    Builds: switch((A << prefix) op1 rhs1 op2 rhs2 ...) substituting reg_map.
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:

        # Must start with A = rol8(A)
        if not _is_rol_a(nodes[i]):
            return None

        j = i

        # ── 1. Prefix rols (before any compound assigns) ──────────────────────
        n_before = 0
        while j < len(nodes) and _is_rol_a(nodes[j]):
            n_before += 1
            j += 1

        # ── 2. Compound assigns on A ──────────────────────────────────────────
        compounds: List[Tuple[str, Expr]] = []
        while j < len(nodes):
            cn = nodes[j]
            if not (isinstance(cn, CompoundAssign) and cn.lhs == Reg("A")):
                break
            bin_op = _OP_WITHOUT_EQ.get(cn.op)
            if bin_op is None:
                break
            compounds.append((bin_op, cn.rhs))
            j += 1

        # ── 3. Step-size rols (after compound assigns) ────────────────────────
        n_after = 0
        while j < len(nodes) and _is_rol_a(nodes[j]):
            n_after += 1
            j += 1

        # ── 4. Terminal: SwitchNode ───────────────────────────────────────────
        if j >= len(nodes) or not isinstance(nodes[j], SwitchNode):
            return None
        sw = nodes[j]

        # ── 5. Extract shift K from switch subject ────────────────────────────
        subj = sw.subject
        if isinstance(subj, BinOp) and subj.op == ">>" and subj.lhs == Reg("A"):
            if not isinstance(subj.rhs, Const):
                return None
            k = subj.rhs.value
        elif subj == Reg("A"):
            k = 0
        else:
            return None

        # ── 6. Determine effective prefix and step counts ─────────────────────
        if n_after >= k:
            n_step   = k
            n_prefix = n_before + (n_after - k)
        elif not compounds:
            # All rols are contiguous; borrow k - n_after from n_before
            n_step   = k
            n_prefix = n_before - (k - n_after)
            if n_prefix < 0:
                return None  # not enough rols to cancel the shift
        else:
            # Compounds separate prefix from step rols; can't safely borrow
            return None

        # ── 7. Build simplified switch subject ────────────────────────────────
        new_subj: Expr = Reg("A")
        if n_prefix > 0:
            new_subj = BinOp(new_subj, "<<", Const(n_prefix))
        for bin_op, rhs in compounds:
            new_subj = BinOp(new_subj, bin_op, rhs)
        new_subj = _subst_all_expr(new_subj, reg_map)

        new_sw = SwitchNode(sw.ea, new_subj, sw.cases,
                            sw.default_label, sw.default_body)
        dbg("typesimp",
            f"  [{hex(nodes[i].ea)}] rol_switch: {n_before}+{n_after} rols "
            f"(k={k}) → switch({new_subj.render()})")
        return ([new_sw], j + 1)
