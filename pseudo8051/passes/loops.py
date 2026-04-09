"""
passes/loops.py — LoopStructurer: back-edge loops → WhileNode / ForNode / DoWhileNode.

For each back-edge B → H (detected via DFS from the function entry):
  1. Collect all blocks in the natural loop body (blocks that can reach B
     while dominated by H).
  2. Find the exit block (successor of the loop header outside the body).
  3. Replace the HIR of the header block with a WhileNode, ForNode, or
     DoWhileNode that wraps the body blocks' HIR.

DJNZ loops produce:  for (Rn = N; Rn; Rn--)  when a constant value can be
determined, otherwise  while (--Rn != 0).

Multiple back-edges to the same header are handled as a single loop:
  - The unconditional back-edge (GotoStatement) is the primary tail.
  - Conditional back-edges (IfGoto) become IfNodes gating the tail body.
"""

import re
from collections import defaultdict
from typing import List, Set, Optional, Tuple

from pseudo8051.ir.hir      import (HIRNode, Assign, CompoundAssign, ExprStmt,
                                    WhileNode, ForNode, DoWhileNode, Label,
                                    IfGoto, GotoStatement, IfNode)
from pseudo8051.ir.expr     import Expr, Reg, Regs, Const, BinOp, UnaryOp
from pseudo8051.passes      import OptimizationPass
from pseudo8051.constants import dbg

from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock


_FLIP_OP = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}

# Maps (lo_reg, hi_reg) → pair name for 16-bit DJNZ detection
_DJNZ_PAIR = {
    ("R7", "R6"): "R6R7",
    ("R5", "R4"): "R4R5",
    ("R3", "R2"): "R2R3",
    ("R1", "R0"): "R0R1",
}


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
                    and isinstance(lhs.operand, Regs) and lhs.operand.is_single):
                return lhs.operand.name
    return None


def _dfs_back_edges(entry: "BasicBlock") -> List[Tuple["BasicBlock", "BasicBlock"]]:
    """
    Iterative DFS from entry; returns list of (tail, header) back-edge pairs.
    A back-edge is an edge to a block already on the current DFS stack (gray node).
    """
    in_stack: set = set()
    visited:  set = set()
    back_edges: List = []
    stack = [(entry, iter(entry.successors))]
    in_stack.add(entry.start_ea)
    while stack:
        block, children = stack[-1]
        try:
            child = next(children)
            if child.start_ea in in_stack:
                back_edges.append((block, child))   # (tail, header)
            elif child.start_ea not in visited:
                in_stack.add(child.start_ea)
                stack.append((child, iter(child.successors)))
        except StopIteration:
            block, _ = stack.pop()
            in_stack.discard(block.start_ea)
            visited.add(block.start_ea)
    return back_edges


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


def _extract_cond_reg(cond) -> Optional[str]:
    """Extract the primary register name from a loop condition."""
    if isinstance(cond, Regs) and cond.is_single:
        return cond.name
    if isinstance(cond, BinOp) and isinstance(cond.lhs, Regs) and cond.lhs.is_single:
        return cond.lhs.name
    if isinstance(cond, str):
        m = re.match(r'^(\w+)\s*(?:!=|==|<|>|<=|>=)', cond)
        if m:
            return m.group(1)
    return None


def _node_to_update_if_writes(node: HIRNode, reg: str):
    """Return update expression/string if node writes reg as a simple side-effect, else None."""
    if isinstance(node, CompoundAssign):
        if node.lhs == Reg(reg):
            return f"{node.lhs.render()} {node.op} {node.rhs.render()}"
    if isinstance(node, ExprStmt):
        expr = node.expr
        if (isinstance(expr, UnaryOp)
                and expr.operand == Reg(reg)):
            return expr
    return None


def _try_promote_to_for(while_node: WhileNode) -> HIRNode:
    """
    Promote a WhileNode to ForNode when the last body statement is a simple
    update of the loop condition variable.
    """
    body = while_node.body_nodes
    if not body:
        return while_node
    reg = _extract_cond_reg(while_node.condition)
    if reg is None:
        return while_node
    update = _node_to_update_if_writes(body[-1], reg)
    if update is None:
        return while_node
    dbg("loops", f"  → promote WhileNode to ForNode  reg={reg}  update={update!r}")
    return ForNode(
        ea         = while_node.ea,
        init       = None,
        condition  = while_node.condition,
        update     = update,
        body_nodes = body[:-1],
    )


def _extract_dowhile_cond(tail: "BasicBlock", header: "BasicBlock",
                           body_eas: Set[int], func) -> Optional[object]:
    """
    Return the do-while loop condition if the primary tail ends with a
    conditional back-edge to the loop body, else None.
    """
    if not tail.hir:
        return None
    last = tail.hir[-1]
    if _is_djnz_node(last):
        return None   # DJNZ handled separately

    label_map = {b.label: b.start_ea
                 for b in func._block_map.values() if b.label}

    if isinstance(last, IfGoto):
        target_ea = label_map.get(last.label)
        if target_ea is None:
            return None
        if target_ea in body_eas:
            return last.cond           # goto header → continue when true
        return _invert_cond(last.cond) # goto exit   → continue when not cond

    return None


class LoopStructurer(OptimizationPass):
    """
    Detect natural loops via back-edges and replace them with WhileNode /
    ForNode HIR nodes in the header block.

    Blocks absorbed into a loop are marked with block._absorbed = True so
    the Function.render() step skips them.

    Multiple back-edges to the same header are merged into a single loop.
    """

    def run(self, func: Function) -> None:
        # Identify all back-edges via DFS from the function entry block
        back_edges = _dfs_back_edges(func.entry_block)

        # Also DFS from orphan blocks (no predecessors; e.g. switch case targets
        # whose CFG predecessors IDA doesn't record for JMP @A+DPTR)
        entry_ea = func.entry_block.start_ea
        for block in func.blocks:
            if (not getattr(block, "_absorbed", False)
                    and block.start_ea != entry_ea
                    and not block.predecessors):
                back_edges.extend(_dfs_back_edges(block))

        if not back_edges:
            dbg("loops", "no back-edges found")
            from pseudo8051.passes.debug_dump import dump_pass_hir
            all_nodes = [n for b in func.blocks
                         if not getattr(b, "_absorbed", False) for n in b.hir]
            dump_pass_hir("loops", all_nodes, func.name)
            return

        # Group by header so multiple tails → one loop
        by_header: dict = defaultdict(list)
        for tail, header in back_edges:
            by_header[header.start_ea].append(tail)

        for header_ea, tails in by_header.items():
            header = func._block_map[header_ea]
            dbg("loops", f"header={hex(header_ea)}  tails={[hex(t.start_ea) for t in tails]}")
            self._structure_loop(func, header, tails)

        from pseudo8051.passes.debug_dump import dump_pass_hir
        all_nodes = [n for b in func.blocks
                     if not getattr(b, "_absorbed", False) for n in b.hir]
        dump_pass_hir("loops", all_nodes, func.name)

    def _structure_loop(self, func: Function,
                        header: BasicBlock, tails: List[BasicBlock]) -> None:
        # Union of all natural-loop bodies
        body_eas: Set[int] = set()
        for t in tails:
            body_eas |= _collect_loop_body(header, t)

        # Forward extension: also include blocks whose all predecessors are
        # already in the body but that exit the loop early (e.g. an early-break
        # block that jumps to the loop exit via a GotoStatement).
        # Only unlabeled blocks are eligible — labeled blocks are jump targets
        # (goto/IfGoto destinations) which are either explicit loop exits or
        # if-else branch targets; their placement is deliberate.
        # Also exclude the loop header's exit targets explicitly.
        header_exits: Set[int] = {s.start_ea for s in header.successors
                                   if s.start_ea not in body_eas}
        changed = True
        while changed:
            changed = False
            for blk in func.blocks:
                ea = blk.start_ea
                if ea in body_eas or ea in header_exits or blk.label:
                    continue
                if blk.predecessors and all(p.start_ea in body_eas
                                             for p in blk.predecessors):
                    body_eas.add(ea)
                    changed = True
                    dbg("loops", f"  forward-extend body: +{hex(ea)}")

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
            if last is None or isinstance(last, GotoStatement):
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
            if isinstance(node, Label):
                continue   # will be re-added to outer scope if needed
            header_stmts.append(node)

        # ── Construct structured node ─────────────────────────────────────
        loop_ea = header.start_ea

        if djnz_reg is not None:
            reg = djnz_reg  # outer (hi) register

            # Check for multi-byte DJNZ: branch_node is a DJNZ on the lo byte
            # and the header itself is a secondary tail (lo-byte self-loop).
            inner_reg = (_is_djnz_node(branch_node)
                         if isinstance(branch_node, IfGoto) else None)
            pair = None
            if inner_reg and any(t is header for t in secondary_tails):
                pair = _DJNZ_PAIR.get((inner_reg, reg))

            if pair is not None:
                dbg("loops", f"  → DoWhileNode (multi-byte DJNZ)  lo={inner_reg}  hi={reg}  pair={pair}")
                cond_mb = BinOp(UnaryOp("--", Reg(pair), post=False), "!=", Const(0))
                loop_node: HIRNode = DoWhileNode(loop_ea, cond_mb, header_stmts + body_hir)
                header_stmts = []
            else:
                # Single-byte DJNZ.
                # Self-loop (tail IS header): body lives in header_stmts; DJNZ is
                # check-last so use DoWhileNode.
                if primary_tail is header:
                    djnz_body = header_stmts + body_hir
                    header_stmts = []
                    dbg("loops", f"  → DoWhileNode (DJNZ self-loop)  reg={reg}")
                    cond_djnz = BinOp(UnaryOp("--", Reg(reg), post=False), "!=", Const(0))
                    loop_node = DoWhileNode(loop_ea, cond_djnz, djnz_body)
                else:
                    # Separate body block — use ForNode when init is known, else WhileNode
                    init_val = self._find_init_value(header, body_eas, reg)
                    if init_val is not None:
                        dbg("loops", f"  → ForNode  reg={reg}  init={init_val}")
                        loop_node = ForNode(
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
                branch_target = branch_node.label
            else:
                dbg("loops", f"  → unhandled branch node type, skipping")
                return

            # Determine direction: forward exit (invert) vs back-edge continue (use directly).
            label_map = {b.label: b.start_ea for b in func._block_map.values() if b.label}
            target_ea = label_map.get(branch_target)
            is_exit = (target_ea is not None) and (target_ea not in body_eas)
            if is_exit:
                cond = _invert_cond(cond)
            dbg("loops", f"  → WhileNode  cond={cond!r}  exit_inverted={is_exit}")
            # Body HIR = header_stmts body + loop body
            loop_node: HIRNode = WhileNode(
                loop_ea,
                condition=cond,
                body_nodes=header_stmts + body_hir,
            )
            header_stmts = []
            # Try to promote to ForNode if body ends with a condition update
            loop_node = _try_promote_to_for(loop_node)
        else:
            # No branch in header — check primary tail for do-while pattern
            do_cond = _extract_dowhile_cond(primary_tail, header, body_eas, func)
            if do_cond is not None:
                dbg("loops", f"  → DoWhileNode  cond={do_cond!r}")
                loop_node = DoWhileNode(
                    loop_ea,
                    condition=do_cond,
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
                        if isinstance(node, Label) and node.name == header.label:
                            continue
                        then_hir.append(node)
                # Wrap in IfNode with inverted condition
                last = blk.hir[-1]
                if isinstance(last, IfGoto):
                    inv = _invert_cond(last.cond)
                    result.append(IfNode(last.ea, inv, then_hir, []))
                else:
                    result.extend(then_hir)
                break   # all remaining blocks consumed
            else:
                # Normal block: strip back-edge from primary tail's last node
                for node in blk.hir:
                    if blk is primary_tail and node is blk.hir[-1]:
                        if isinstance(node, (IfGoto, GotoStatement)):
                            continue
                    if isinstance(node, Label) and node.name == header.label:
                        continue
                    result.append(node)
            i += 1
        return result

    def _find_init_value(self, header: BasicBlock,
                         body_eas: Set[int], reg: str) -> Optional[str]:
        """
        Search the block immediately preceding the header for an assignment
        'reg = value;' that initialises the loop counter.
        """
        for pred in header.predecessors:
            if pred.start_ea in body_eas:   # skip back-edge predecessors
                continue
            for node in reversed(pred.hir):
                if isinstance(node, Assign) and node.lhs == Reg(reg):  # Reg factory still works
                    return node.rhs.render()
                break   # stop at first non-label node from the end
        return None
