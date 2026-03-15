"""
passes/__init__.py — OptimizationPass ABC + run_all_passes().
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pseudo8051.ir.function import Function


class OptimizationPass(ABC):
    """Base class for all IR transformation / analysis passes."""

    @abstractmethod
    def run(self, func: "Function") -> None:
        """Transform / annotate func in-place."""
        ...


def run_all_passes(func: "Function") -> None:
    """
    Execute all passes on func in the correct order.
    Called from Function.__init__ after the block graph is built.
    """
    from pseudo8051.analysis.constprop  import ConstantPropagation
    from pseudo8051.analysis.liveness   import LivenessAnalysis
    from pseudo8051.passes.rmw          import RMWCollapser
    from pseudo8051.passes.loops        import LoopStructurer
    from pseudo8051.passes.ifelse       import IfElseStructurer

    ConstantPropagation().run(func)
    LivenessAnalysis().run(func)

    # Build initial flat HIR (needs cp_entry to be populated first)
    for block in func.blocks:
        block.hir = block.initial_hir()

    RMWCollapser().run(func)
    LoopStructurer().run(func)
    IfElseStructurer().run(func)
