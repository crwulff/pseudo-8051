"""
passes/__init__.py — OptimizationPass ABC + run_all_passes().
"""

from abc import ABC, abstractmethod

from pseudo8051.ir.function import Function


class OptimizationPass(ABC):
    """Base class for all IR transformation / analysis passes."""

    @abstractmethod
    def run(self, func: Function) -> None:
        """Transform / annotate func in-place."""
        ...


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
    LoopStructurer().run(func)
    JmpTableStructurer().run(func)
    SwitchStructurer().run(func)
    IfElseStructurer().run(func)
    SwitchBodyAbsorber().run(func)
