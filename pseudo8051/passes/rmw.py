"""
passes/rmw.py — RMWCollapser: fold XRAM read-modify-write sequences.

Works within each BasicBlock's HIR list.  Migrated from collapse_rmw_patterns()
in the old monolithic script, but now operates on Statement nodes rather than
raw (ea, text) tuples.

Recognised pattern (consecutive Statement nodes at the same indent == 0,
since all per-block statements have no indent at this stage):

    A = XRAM[REG];
    A &= expr;          ← one or more: &= |= ^= += or A = ~A;
    A |= expr;
    XRAM[REG] = A;

Collapsed to:
    *REG = ((*REG) & expr) | expr;
"""

import re
from typing import List, TYPE_CHECKING

from pseudo8051.ir.hir   import HIRNode, Statement
from pseudo8051.passes   import OptimizationPass
from pseudo8051.constants import dbg

if TYPE_CHECKING:
    from pseudo8051.ir.function import Function


# Regex patterns — match on Statement.text (no leading whitespace at this stage)
_RE_READ  = re.compile(r'^A = XRAM\[(.+?)\];$')
_RE_OP    = re.compile(r'^A ([&|^+]=) (.+);$')   # &= |= ^= +=
_RE_CPL   = re.compile(r'^A = ~A;$')
_RE_WRITE = re.compile(r'^XRAM\[(.+?)\] = A;$')


def _build_expr(base: str, ops: list) -> str:
    """
    Apply a sequence of ops to base, building a C expression.
    Each entry: ("~",) for complement, or (op_char, rhs) for binary ops.
    """
    expr = base
    for op_info in ops:
        if op_info[0] == "~":
            expr = f"~({expr})"
        else:
            c_op, rhs = op_info
            expr = f"({expr}) {c_op} {rhs}"
    return expr


def _collapse_block_hir(hir: List[HIRNode]) -> List[HIRNode]:
    """Collapse RMW patterns within a single block's HIR list."""
    result: List[HIRNode] = []
    i  = 0
    n  = len(hir)

    while i < n:
        node = hir[i]

        # Only Statement nodes can start an RMW pattern
        if not isinstance(node, Statement):
            result.append(node)
            i += 1
            continue

        m_read = _RE_READ.match(node.text)
        if not m_read:
            result.append(node)
            i += 1
            continue

        reg  = m_read.group(1)
        ops  = []
        j    = i + 1

        while j < n and isinstance(hir[j], Statement):
            txt    = hir[j].text
            m_op   = _RE_OP.match(txt)
            m_cpl  = _RE_CPL.match(txt)
            if m_op:
                op_char = m_op.group(1)[0]   # first char of &=, |=, etc.
                ops.append((op_char, m_op.group(2)))
                j += 1
            elif m_cpl:
                ops.append(("~",))
                j += 1
            else:
                break

        if ops and j < n and isinstance(hir[j], Statement):
            m_w = _RE_WRITE.match(hir[j].text)
            if m_w and m_w.group(1) == reg:
                base = f"*{reg}"
                expr = _build_expr(base, ops)
                collapsed = f"{base} = {expr};"
                dbg("RMW", f"  {node.text!r}  →  {collapsed!r}")
                result.append(Statement(node.ea, collapsed))
                i = j + 1
                continue

        result.append(node)
        i += 1

    return result


class RMWCollapser(OptimizationPass):
    """Collapse XRAM read-modify-write sequences within each block's HIR."""

    def run(self, func: "Function") -> None:
        for block in func.blocks:
            if block.hir:
                before = len(block.hir)
                block.hir = _collapse_block_hir(block.hir)
                after = len(block.hir)
                if after < before:
                    dbg("RMW", f"block {hex(block.start_ea)}: "
                               f"{before - after} line(s) collapsed")
