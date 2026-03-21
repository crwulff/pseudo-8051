"""
passes/loops.py — LoopStructurer: back-edge loops → WhileNode / ForNode.

For each back-edge B → H (successor start_ea ≤ block start_ea):
  1. Collect all blocks in the natural loop body (blocks that can reach B
     while dominated by H).
  2. Find the exit block (successor of the loop header outside the body).
  3. Replace the HIR of the header block with a WhileNode (or ForNode for
     DJNZ patterns) that wraps the body blocks' HIR.

DJNZ loops produce:  for (Rn = N; Rn; Rn--)  when a constant value can be
determined, otherwise  while (--Rn != 0).
"""

import re
from typing import List, Set, Optional

from pseudo8051.ir.hir      import HIRNode, Statement, Assign, WhileNode, ForNode, Label, IfGoto, GotoStatement
from pseudo8051.ir.expr     import Expr, Reg, Const, BinOp, UnaryOp
from pseudo8051.passes      import OptimizationPass
from pseudo8051.constants import dbg

from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock


# Regex to detect DJNZ-style conditional at the bottom of a loop body (legacy Statement)
_RE_DJNZ = re.compile(r'^if \(--(\w+) != 0\) goto (\S+);$')

# Regex to detect a simple register assignment: Rn = value; (legacy Statement)
_RE_ASSIGN = re.compile(r'^(\w+) = (.+);$')


def _is_djnz_node(node: HIRNode) -> Optional[str]:
    """Return the register name if node is a DJNZ-style conditional, else None."""
    if isinstance(node, IfGoto):
        cond = node.cond
        if isinstance(cond, BinOp) and cond.op == "!=" and cond.rhs == Const(0):
            lhs = cond.lhs
            if (isinstance(lhs, UnaryOp) and lhs.op == "--" and not lhs.post
                    and isinstance(lhs.operand, Reg)):
                return lhs.operand.name
    if isinstance(node, Statement):
        m = _RE_DJNZ.match(node.text)
        if m:
            return m.group(1)
    return None


def _collect_loop_body(header: BasicBlock, tail: BasicBlock) -> Set[int]:
    """
    Collect the EAs of all blocks in the natural loop defined by back-edge
    tail → header using a simple backward reachability from tail up to header.
    """
    body: Set[int] = {header.start_ea, tail.start_ea}
    worklist = [tail]
    while worklist:
        blk = worklist.pop()
        if blk is header:
            # Never traverse the header's predecessors: they are loop entries,
            # not loop-body blocks.  This also handles self-loops correctly
            # (tail is header), preventing entry-block pull-in.
            continue
        for pred in blk.predecessors:
            if pred.start_ea not in body:
                body.add(pred.start_ea)
                worklist.append(pred)
    return body


class LoopStructurer(OptimizationPass):
    """
    Detect natural loops via back-edges and replace them with WhileNode /
    ForNode HIR nodes in the header block.

    Blocks absorbed into a loop are marked with block._absorbed = True so
    the Function.render() step skips them.
    """

    def run(self, func: Function) -> None:
        # Identify all back-edges: succ.start_ea <= block.start_ea
        back_edges = []
        for block in func.blocks:
            for succ in block.successors:
                if succ.start_ea <= block.start_ea:
                    back_edges.append((block, succ))   # (tail, header)

        if not back_edges:
            dbg("loops", "no back-edges found")
        for tail, header in back_edges:
            dbg("loops", f"back-edge: {hex(tail.start_ea)} → {hex(header.start_ea)}")
            self._structure_loop(func, header, tail)

    def _structure_loop(self, func: Function,
                        header: BasicBlock, tail: BasicBlock) -> None:
        body_eas = _collect_loop_body(header, tail)

        # Body blocks sorted by EA (excluding header)
        body_blocks: List[BasicBlock] = sorted(
            (func._block_map[ea] for ea in body_eas
             if ea != header.start_ea and ea in func._block_map),
            key=lambda b: b.start_ea)

        body_eas_str = [hex(b.start_ea) for b in body_blocks]
        dbg("loops", f"  header={hex(header.start_ea)}  body={body_eas_str}")

        # ── Detect DJNZ pattern ───────────────────────────────────────────
        # Look for "if (--Rn != 0) goto label;" as the last statement of tail
        djnz_reg: Optional[str] = None
        if tail.hir:
            djnz_reg = _is_djnz_node(tail.hir[-1])

        # ── Build body HIR (all blocks except header, stripping back-edge) ─
        body_hir: List[HIRNode] = []
        for blk in body_blocks:
            for node in blk.hir:
                # Skip the back-edge goto / DJNZ at the end of the tail block
                if blk is tail and node is tail.hir[-1]:
                    if isinstance(node, IfGoto):
                        continue
                    if isinstance(node, GotoStatement):
                        continue
                    if isinstance(node, Statement) and (
                            node.text.startswith("goto ") or
                            _RE_DJNZ.match(node.text)):
                        continue
                # Skip Label nodes pointing back to the header (redundant)
                if isinstance(node, Label) and node.name == header.label:
                    continue
                body_hir.append(node)

        # ── Header HIR: keep non-branch statements (loop-entry init) ────
        header_stmts: List[HIRNode] = []
        branch_node: Optional[HIRNode] = None
        for node in header.hir:
            if isinstance(node, (IfGoto, GotoStatement)):
                branch_node = node
                break
            if isinstance(node, Statement) and (
                    node.text.startswith("if ") or node.text.startswith("goto ")):
                branch_node = node
                break
            if isinstance(node, Label):
                continue   # will be re-added to outer scope if needed
            header_stmts.append(node)

        # ── Construct structured node ─────────────────────────────────────
        loop_ea = header.start_ea

        if djnz_reg is not None:
            reg = djnz_reg
            # Look for an immediate assignment before the loop for ForNode
            init_val = self._find_init_value(header, body_blocks, reg)
            if init_val is not None:
                dbg("loops", f"  → ForNode  reg={reg}  init={init_val}")
                loop_node: HIRNode = ForNode(
                    loop_ea,
                    init=f"{reg} = {init_val}",
                    condition=Reg(reg),
                    update=UnaryOp("--", Reg(reg), post=False),
                    body_nodes=body_hir,
                )
            else:
                dbg("loops", f"  → WhileNode (DJNZ)  reg={reg}")
                loop_node = WhileNode(
                    loop_ea,
                    condition=BinOp(UnaryOp("--", Reg(reg), post=False), "!=", Const(0)),
                    body_nodes=body_hir,
                )
        elif branch_node is not None:
            if isinstance(branch_node, IfGoto):
                cond: object = branch_node.cond  # Expr
            elif isinstance(branch_node, Statement):
                m = re.match(r'^if \((.+)\) goto \S+;$', branch_node.text)
                if m:
                    cond = m.group(1)
                else:
                    dbg("loops", f"  → can't parse branch {branch_node.text!r}, skipping")
                    return
            else:
                dbg("loops", f"  → unhandled branch node type, skipping")
                return
            dbg("loops", f"  → WhileNode  cond={cond!r}")
            # Body HIR = header_stmts body + loop body
            loop_node = WhileNode(
                loop_ea,
                condition=cond,
                body_nodes=header_stmts + body_hir,
            )
            header_stmts = []
        else:
            dbg("loops", f"  → WhileNode (infinite)")
            loop_node = WhileNode(loop_ea, condition="1",
                                  body_nodes=body_hir)

        # ── Replace header's HIR ─────────────────────────────────────────
        new_hir: List[HIRNode] = []
        # Preserve the header's label if it exists
        if header.label:
            from pseudo8051.ir.hir import Label as LabelNode
            new_hir.append(LabelNode(header.start_ea, header.label))
        new_hir.extend(header_stmts)
        new_hir.append(loop_node)
        header.hir = new_hir

        # Mark body blocks as absorbed
        for blk in body_blocks:
            blk._absorbed = True
        if body_blocks and tail not in body_blocks:
            tail._absorbed = True

    def _find_init_value(self, header: BasicBlock,
                         _body_blocks: List[BasicBlock], reg: str) -> Optional[str]:
        """
        Search the block immediately preceding the header for an assignment
        'reg = value;' that initialises the loop counter.
        """
        for pred in header.predecessors:
            if pred.start_ea < header.start_ea:   # not a back-edge
                for node in reversed(pred.hir):
                    if isinstance(node, Assign) and node.lhs == Reg(reg):
                        return node.rhs.render()
                    if isinstance(node, Statement):
                        m = _RE_ASSIGN.match(node.text)
                        if m and m.group(1) == reg:
                            return m.group(2)
                    break   # stop at first non-label node from the end
        return None
