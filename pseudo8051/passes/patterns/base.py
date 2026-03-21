"""
passes/patterns/base.py — Pattern ABC and shared type aliases.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode
from pseudo8051.passes.patterns._utils import VarInfo

# Recursive-simplify callback.  Patterns that need to transform nested HIR
# (e.g. IfNode branches) receive this instead of importing the walker directly,
# which would create a circular dependency.
Simplify = Callable[[List[HIRNode], Dict], List[HIRNode]]

# Return type for Pattern.match on success.
Match = Tuple[List[HIRNode], int]   # (replacement_nodes, new_i)


class Pattern(ABC):
    """
    A statement-sequence pattern recognised by the TypeAwareSimplifier.

    To add a new pattern:
      1. Subclass this in a new file under passes/patterns/.
      2. Implement match().
      3. Append an instance to _PATTERNS in passes/patterns/__init__.py.
    """

    @abstractmethod
    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        """
        Try to match a sequence starting at nodes[i].

        Parameters
        ----------
        nodes    : full node list currently being walked
        i        : current position in nodes
        reg_map  : register → VarInfo mapping for the current function
        simplify : recursive-simplify callback — use this to transform nested
                   node lists (e.g. IfNode.then_nodes / else_nodes) rather
                   than importing the walker directly.

        Returns
        -------
        (replacement_nodes, new_i) on success, where new_i is the index of
        the first node NOT consumed by the match.
        None if the pattern does not apply at this position.
        """
