"""passes/simple_inline.py — SimpleExternalInliner pass.

Inline very simple external callees at their call sites.  A callee is
"simple" if it consists of a single basic block with no branches or nested
calls, at most MAX_INSTRUCTIONS assembly instructions, and ends with a plain
RET.  The callee's raw HIR (minus the ReturnStmt) is substituted for the
call node so the TypeAwareSimplifier can work on the combined code in context.

Runs immediately after ChunkInliner, before structural passes.
"""

import copy
from typing import Dict, List, Optional

from pseudo8051.passes import OptimizationPass
from pseudo8051.ir.hir import ReturnStmt, Assign, ExprStmt
from pseudo8051.ir.cpstate import CPState, propagate_insn
from pseudo8051.ir.instruction import Instruction
from pseudo8051.ir.expr import Reg, RegGroup, Name, Const
from pseudo8051.constants import dbg

MAX_INSTRUCTIONS = 50


class SimpleExternalInliner(OptimizationPass):
    """
    For each LCALL/ACALL to a simple external function, replace the call
    node with the callee's raw HIR body (ReturnStmt stripped).
    Runs after ChunkInliner, before structural passes.
    """

    def run(self, func) -> None:
        # Cache lifted callee bodies (None = not simple / already checked)
        _cache: Dict[int, Optional[List]] = {}

        for block in list(func.blocks):
            if not block.instructions:
                continue

            # Collect external call sites in this block, in order
            ext_calls = []
            for insn in block.instructions:
                if not insn.is_call():
                    continue
                targets = insn.branch_targets()
                if not targets:
                    continue
                target_ea = targets[0]
                if _is_chunk(target_ea, func.ea):
                    continue   # already handled by ChunkInliner
                ext_calls.append((insn.ea, target_ea))

            if not ext_calls:
                continue

            # Process in reverse order so earlier replacements don't shift later indices
            for call_ea, callee_ea in reversed(ext_calls):
                if callee_ea not in _cache:
                    _cache[callee_ea] = _lift_simple_callee(callee_ea)
                inline_nodes = _cache[callee_ea]
                if inline_nodes is None:
                    continue

                # Find the HIR node produced by this call (matched by EA and Call content)
                for idx, node in enumerate(block.hir):
                    if node.ea == call_ea and _contains_call(node):
                        block.hir[idx : idx + 1] = [copy.deepcopy(n) for n in inline_nodes]
                        dbg("func", f"  SimpleInliner: inlined {len(inline_nodes)} nodes "
                                    f"from {hex(callee_ea)} at call {hex(call_ea)}")
                        break

        from pseudo8051.passes.debug_dump import dump_pass_hir
        all_nodes = [n for b in func.blocks
                     if not getattr(b, "_absorbed", False) for n in b.hir]
        dump_pass_hir("01b.simple_inline", all_nodes, func.name)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_chunk(target_ea: int, func_ea: int) -> bool:
    """True if target_ea is an intra-function tail chunk (not an external callee)."""
    import ida_funcs
    target_fn = ida_funcs.get_func(target_ea)
    return (target_fn is not None
            and target_fn.start_ea == func_ea
            and target_ea != func_ea)


def _contains_call(node) -> bool:
    """True if node is an Assign or ExprStmt whose expression is a Call."""
    from pseudo8051.ir.expr import Call
    if isinstance(node, Assign):
        return isinstance(node.rhs, Call)
    if isinstance(node, ExprStmt):
        return isinstance(node.expr, Call)
    return False


def _make_tail_call_node(ea: int, tail_name: str, state: CPState):
    """Build a properly-annotated HIR node for a tail call to tail_name.

    Consults the callee prototype to build argument expressions and, when the
    callee has a non-void return type, wraps the Call in an Assign to the
    appropriate return register(s).  Falls back to a bare ExprStmt(Call(..., []))
    when no prototype is available.
    """
    from pseudo8051.prototypes import get_proto, param_regs, expand_regs
    from pseudo8051.ir.expr import Call
    proto = get_proto(tail_name)
    if proto:
        p_regs = param_regs(proto)
        args = []
        for p, regs in zip(proto.params, p_regs):
            if regs:
                if state is not None and len(regs) == 1:
                    val = state.get(regs[0])
                    if val is not None:
                        args.append(Const(val))
                        continue
                elif state is not None and len(regs) > 1:
                    vals = [state.get(r) for r in regs]
                    if all(v is not None for v in vals):
                        combined = 0
                        for v in vals:
                            combined = (combined << 8) | (v & 0xFF)
                        args.append(Const(combined))
                        continue
                args.append(RegGroup(regs) if len(regs) > 1 else Name("".join(regs)))
            else:
                args.append(Name(p.name))
        call_node = Call(tail_name, args)
        if proto.return_type != "void" and proto.return_regs:
            ret_regs = expand_regs(tuple(proto.return_regs), proto.return_type)
            lhs = Reg(ret_regs[0]) if len(ret_regs) == 1 else RegGroup(tuple(ret_regs))
            return Assign(ea, lhs, call_node)
        return ExprStmt(ea, call_node)
    return ExprStmt(ea, Call(tail_name, []))


def _lift_simple_callee(callee_ea: int) -> Optional[List]:
    """
    Lift the callee's assembly into raw HIR nodes.

    Returns the body nodes if the callee is simple (no conditional branches,
    no nested calls, at most MAX_INSTRUCTIONS instructions), or None otherwise.

    Returns None (skips inlining) if the callee has a type signature — typed
    functions should remain as named call nodes so TypeAwareSimplifier can
    substitute their register arguments with named parameters.

    Termination:
      • RET  — strip the ReturnStmt and return the body.
      • Unconditional direct jump (LJMP/SJMP/AJMP to a known address) — treat
        as a tail call: append ExprStmt(Call(target_name)) and return the body.
        The tail call returns to the original caller, so no ReturnStmt is needed.

    Does not require callee_ea to be the IDA function entry — works even when
    the target is a tail chunk owned by a different function.
    """
    import ida_name
    from pseudo8051.ir.expr import Call
    from pseudo8051.prototypes import get_proto
    callee_name = ida_name.get_name(callee_ea) or f"sub_{hex(callee_ea).removeprefix('0x')}"
    if get_proto(callee_name) is not None:
        return None   # has a type signature — keep as a named call

    state = CPState()
    nodes: List = []
    ea = callee_ea
    count = 0

    while count <= MAX_INSTRUCTIONS:
        instr = Instruction(ea)
        if not instr.insn:
            return None
        if instr.is_call():
            return None   # nested call — not simple
        if instr.is_branch():
            # Allow a single direct unconditional jump as a tail call.
            if not instr.is_unconditional_branch():
                return None   # conditional branch — not simple
            targets = instr.branch_targets()
            if len(targets) != 1:
                return None   # computed jump (JMP @A+DPTR) — not simple
            tail_ea = targets[0]
            tail_name = ida_name.get_name(tail_ea) or f"sub_{hex(tail_ea).removeprefix('0x')}"
            nodes.append(_make_tail_call_node(instr.ea, tail_name, state))
            return nodes
        stmts = instr.lift(state)
        propagate_insn(instr.insn, state)
        nodes.extend(stmts)
        ea += instr.size
        count += 1
        if nodes and isinstance(nodes[-1], ReturnStmt):
            return nodes[:-1]   # strip ReturnStmt; caller deep-copies before insertion

    return None   # exceeded MAX_INSTRUCTIONS without finding RET or tail call
