"""
passes/__init__.py — OptimizationPass ABC + BlockStructurer + run_all_passes().
"""

from abc import ABC, abstractmethod
from typing import Callable

from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock


class OptimizationPass(ABC):
    """Base class for all IR transformation / analysis passes."""

    @abstractmethod
    def run(self, func: Function) -> None:
        """Transform / annotate func in-place."""
        ...


# ── Standalone block-iteration helpers ────────────────────────────────────────

def run_blocks_until_stable(func: Function,
                             try_fn: Callable[[Function, BasicBlock], bool],
                             *,
                             reverse: bool = False) -> None:
    """
    Iterate func.blocks (skipping absorbed blocks), calling try_fn(func, block)
    for each.  Restart from the beginning whenever try_fn returns True.
    Stops when a full pass produces no changes.
    """
    blocks = func.blocks
    changed = True
    while changed:
        changed = False
        seq = reversed(blocks) if reverse else iter(blocks)
        for block in seq:
            if getattr(block, "_absorbed", False):
                continue
            if try_fn(func, block):
                changed = True
                break


def dump_hir(func: Function, pass_name: str) -> None:
    """Dump all non-absorbed HIR nodes to the per-pass debug file."""
    from pseudo8051.passes.debug_dump import dump_pass_hir
    all_nodes = [n for b in func.blocks
                 if not getattr(b, "_absorbed", False) for n in b.hir]
    dump_pass_hir(pass_name, all_nodes, func.name)


# ── BlockStructurer ABC ───────────────────────────────────────────────────────

class BlockStructurer(OptimizationPass, ABC):
    """
    Base for CFG-level structuring passes that:
      1. Iterate func.blocks (forward or reverse).
      2. Call try_block() on each non-absorbed block.
      3. Restart whenever try_block() returns True.
      4. Stop when a full pass produces no changes.
      5. Call post_run() for any cleanup after the stable point.
      6. Emit a debug dump (if pass_name is set).

    Subclass contract:
      • Set block_order = "forward" (default) or "reverse".
      • Set pass_name to the debug-dump key (e.g. "07.ifelse"); "" skips dump.
      • Implement try_block(func, block) -> bool.
      • Override post_run(func) for cleanup after the stable point.
    """

    block_order: str = "forward"
    pass_name:   str = ""

    @abstractmethod
    def try_block(self, func: Function, block: BasicBlock) -> bool:
        """
        Attempt to detect and apply one structuring transform starting at block.
        Return True if the block graph was changed (triggers a restart).
        Absorbed-block skipping is handled by run().
        """

    def post_run(self, func: Function) -> None:
        """Called after the stable point. Override for post-structuring cleanup."""

    def run(self, func: Function) -> None:
        run_blocks_until_stable(func, self.try_block,
                                reverse=(self.block_order == "reverse"))
        self.post_run(func)
        if self.pass_name:
            dump_hir(func, self.pass_name)


def run_all_passes(func: Function) -> None:
    """
    Execute all passes on func in the correct order.
    Called from Function.__init__ after the block graph is built.
    """
    from pseudo8051.analysis.constprop  import ConstantPropagation
    from pseudo8051.analysis.liveness   import LivenessAnalysis
    from pseudo8051.passes.rmw          import RMWCollapser
    from pseudo8051.passes.loops        import LoopStructurer
    from pseudo8051.passes.jmptable     import (JmpTableStructurer,
                                                fixup_jmptable_edges)
    from pseudo8051.passes.switch       import SwitchStructurer, SwitchBodyAbsorber
    from pseudo8051.passes.ifelse       import IfElseStructurer

    ConstantPropagation().run(func)
    LivenessAnalysis().run(func)

    # Build initial flat HIR (needs cp_entry to be populated first)
    for block in func.blocks:
        block.hir = block.initial_hir()

    from pseudo8051.passes.chunk_inline  import ChunkInliner
    from pseudo8051.passes.simple_inline import SimpleExternalInliner
    ChunkInliner().run(func)
    SimpleExternalInliner().run(func)

    # Add synthetic CFG edges for JMP @A+DPTR blocks (HIR-level, more reliable
    # than raw instruction scan).  Re-run ConstantPropagation so that AnnotationPass
    # sees correct block.cp_entry with the full, corrected CFG.
    fixup_jmptable_edges(func)
    ConstantPropagation().run(func)

    from pseudo8051.passes.annotate import AnnotationPass
    AnnotationPass().run(func)

    RMWCollapser().run(func)
    # JmpTableStructurer and SwitchStructurer must run BEFORE LoopStructurer so
    # that their absorbed blocks are excluded from the loop body (LoopStructurer
    # filters _absorbed blocks when building body_blocks).  If LoopStructurer ran
    # first it would call _structure_flat_ifelse on the loop body and convert the
    # switch-step IfGoto nodes to IfNodes before SwitchStructurer could see them.
    JmpTableStructurer().run(func)
    SwitchStructurer().run(func)
    LoopStructurer().run(func)
    IfElseStructurer().run(func)
    SwitchBodyAbsorber().run(func)
