"""
passes/typesimplify/ — TypeAwareSimplifier package.

Re-exports maintain backward compatibility with all existing import paths:
  from pseudo8051.passes.typesimplify import TypeAwareSimplifier
  from pseudo8051.passes.typesimplify import _collapse_dpl_dph
"""

from pseudo8051.passes.typesimplify._pass import TypeAwareSimplifier
from pseudo8051.passes.typesimplify._post import _collapse_dpl_dph

__all__ = ["TypeAwareSimplifier", "_collapse_dpl_dph"]
