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

Multiple back-edges to the same header are handled as a single loop:
  - The unconditional back-edge (GotoStatement) is the primary tail.
  - Conditional back-edges (IfGoto) become IfNodes gating the tail body.
"""

import re
from collections import defaultdict
from typing import List, Set, Optional

from pseudo8051.ir.hir      import HIRNode, Statement, Assign, WhileNode, ForNode, Label, IfGoto, GotoStatement, IfNode
from pseudo8051.ir.expr     import Expr, Reg, Const, BinOp, UnaryOp
from pseudo8051.passes      import OptimizationPass
from pseudo8051.constants import dbg

from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock


# Regex to detect DJNZ-style conditional at the bottom of a loop body (legacy Statement)
_RE_DJNZ = re.compile(r'^if \(--(\w+) != 0\) goto (\S+);$')

# Regex to detect a simple register assignment: Rn = value; (legacy Statement)
_RE_ASSIGN = re.compile(r'^(\w+) = (.+);$')


_FLIP_OP = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


def _invert_cond(cond):
    """Invert a condition Expr or str cleanly."""
    if isinstance(cond, BinOp) and cond.op in _FLIP_OP:
        return BinOp(cond.lhs, _FLIP_OP[cond.op], cond.rhs)
    from pseudo8051.passes.ifelse import _invert_condition
    return _invert_condition(cond)


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

    Multiple back-edges to the same header are merged into a single loop.
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
            return

        # Group by header so multiple tails → one loop
        by_header: dict = defaultdict(list)
        for tail, header in back_edges:
            by_header[header.start_ea].append(tail)

        for header_ea, tails in by_header.items():
            header = func._block_map[header_ea]
            dbg("loops", f"header={hex(header_ea)}  tails={[hex(t.start_ea) for t in tails]}")
            self._structure_loop(func, header, tails)

    def _structure_loop(self, func: Function,
                        header: BasicBlock, tails: List[BasicBlock]) -> None:
        # Union of all natural-loop bodies
        body_eas: Set[int] = set()
        for t in tails:
            body_eas |= _collect_loop_body(header, t)

        # Body blocks sorted by EA (excluding header)
        body_blocks: List[BasicBlock] = sorted(
            (func._block_map[ea] for ea in body_eas
             if ea != header.start_ea and ea in func._block_map),
            key=lambda b: b.start_ea)

        body_eas_str = [hex(b.start_ea) for b in body_blocks]
        dbg("loops", f"  header={hex(header.start_ea)}  body={body_eas_str}")

        # Identify primary tail (unconditional back-edge) vs secondary tails
        # (conditional IfGoto back-edges = early continue)
        primary_tail = None
        secondary_tails = []
        for t in tails:
            last = t.hir[-1] if t.hir else None
            if last is None or isinstance(last, GotoStatement) or (
                    isinstance(last, Statement) and last.text.startswith("goto ")):
                primary_tail = t
            else:
                secondary_tails.append(t)
        if primary_tail is None:   # all conditional → pick highest EA
            primary_tail = max(tails, key=lambda t: t.start_ea)
            secondary_tails = [t for t in tails if t is not primary_tail]

        # ── Detect DJNZ pattern ───────────────────────────────────────────
        # Look for "if (--Rn != 0) goto label;" as the last statement of primary tail
        djnz_reg: Optional[str] = None
        if primary_tail.hir:
            djnz_reg = _is_djnz_node(primary_tail.hir[-1])

        # ── Build body HIR ────────────────────────────────────────────────
        body_hir: List[HIRNode] = self._build_body_hir(
            body_blocks, header, primary_tail, secondary_tails)

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
        if body_blocks and primary_tail not in body_blocks:
            primary_tail._absorbed = True

    def _build_body_hir(self, body_blocks: List[BasicBlock],
                        header: BasicBlock,
                        primary_tail: BasicBlock,
                        secondary_tails: List[BasicBlock]) -> List[HIRNode]:
        """
        Build the loop body HIR list.

        Secondary tails (conditional back-edges) split the body: the code
        after the secondary tail is wrapped in an IfNode gated by the
        inverted condition, so:
          secondary_tail_code; if (inv_cond) { remaining_body; }
        """
        if not secondary_tails:
            # Simple case: single tail, strip the back-edge from its last node
            result: List[HIRNode] = []
            for blk in body_blocks:
                for node in blk.hir:
                    if blk is primary_tail and node is blk.hir[-1]:
                        if isinstance(node, (IfGoto, GotoStatement)):
                            continue
                        if isinstance(node, Statement) and (
                                node.text.startswith("goto ") or
                                _RE_DJNZ.match(node.text)):
                            continue
                    if isinstance(node, Label) and node.name == header.label:
                        continue
                    result.append(node)
            return result

        secondary_ids = {id(t) for t in secondary_tails}
        result = []
        i = 0
        while i < len(body_blocks):
            blk = body_blocks[i]
            if id(blk) in secondary_ids:
                # Emit this block's HIR minus the conditional back-edge (IfGoto)
                for node in blk.hir:
                    if node is blk.hir[-1]:
                        break   # drop the conditional back-edge goto
                    if isinstance(node, Label) and node.name == header.label:
                        continue
                    result.append(node)
                # Collect remaining body blocks as the if-then body
                then_hir: List[HIRNode] = []
                for j in range(i + 1, len(body_blocks)):
                    rem = body_blocks[j]
                    for node in rem.hir:
                        if rem is primary_tail and node is rem.hir[-1]:
                            if isinstance(node, (IfGoto, GotoStatement)):
                                break
                            if isinstance(node, Statement) and (
                                    node.text.startswith("goto ") or
                                    _RE_DJNZ.match(node.text)):
                                break
                        if isinstance(node, Label) and node.name == header.label:
                            continue
                        then_hir.append(node)
                # Wrap in IfNode with inverted condition
                last = blk.hir[-1]
                if isinstance(last, IfGoto):
                    inv = _invert_cond(last.cond)
                    result.append(IfNode(last.ea, inv, then_hir, []))
                elif isinstance(last, Statement):
                    m = re.match(r'^if \((.+)\) goto \S+;$', last.text)
                    if m:
                        inv = _invert_cond(m.group(1))
                        result.append(IfNode(last.ea, inv, then_hir, []))
                    else:
                        result.extend(then_hir)
                else:
                    result.extend(then_hir)
                break   # all remaining blocks consumed
            else:
                # Normal block: strip back-edge from primary tail's last node
                for node in blk.hir:
                    if blk is primary_tail and node is blk.hir[-1]:
                        if isinstance(node, (IfGoto, GotoStatement)):
                            continue
                        if isinstance(node, Statement) and (
                                node.text.startswith("goto ") or
                                _RE_DJNZ.match(node.text)):
                            continue
                    if isinstance(node, Label) and node.name == header.label:
                        continue
                    result.append(node)
            i += 1
        return result

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
