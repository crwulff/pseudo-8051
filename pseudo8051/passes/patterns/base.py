"""
passes/patterns/base.py — Transform ABC and the typed sub-hierarchies.

The hierarchy encodes the arity of each transform so that src_eas propagation
and annotation merging happen automatically in base-class match() implementations:

    SubstituteTransform   1 → 1     apply(node, ...) → Optional[HIRNode]
    InlineTransform       N → 1     produce(nodes, i, ...) → Optional[(HIRNode, int)]
    CombineTransform      N → 1     produce(nodes, i, ...) → Optional[(HIRNode, int)]
    EliminateTransform    N → 0     detect(nodes, i, ...) → Optional[int]
    RestructureTransform  N → M≥2   produce(nodes, i, ...) → Optional[(List[HIRNode], int)]

Patterns that don't fit a standard arity (e.g. AccumFoldPattern, which re-emits
pass-through "skipped" nodes alongside newly-created terminal nodes) subclass
Transform directly and implement match() in full.

Pattern = Transform   # backward-compatible alias
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, NodeAnnotation
from pseudo8051.passes.patterns._utils import VarInfo

# Recursive-simplify callback.  Transforms that need to recurse into nested HIR
# (e.g. IfNode branches) receive this instead of importing the walker directly,
# which would create a circular dependency.
Simplify = Callable[[List[HIRNode], Dict], List[HIRNode]]

# Return type for Transform.match on success.
Match = Tuple[List[HIRNode], int]   # (replacement_nodes, new_i)


class Transform(ABC):
    """
    A statement-sequence transform recognised by the TypeAwareSimplifier.

    To add a new transform:
      1. Choose the appropriate typed sub-class (SubstituteTransform,
         InlineTransform, CombineTransform, EliminateTransform, or
         RestructureTransform) and subclass it in a new file under
         passes/patterns/.
      2. Implement the required abstract method (apply / produce / detect).
      3. Append an instance to _PATTERNS in passes/patterns/__init__.py.

    For transforms that don't fit any standard arity, subclass Transform
    directly and implement match() in full.
    """

    @abstractmethod
    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        """
        Try to match a sequence starting at nodes[i].

        Returns (replacement_nodes, new_i) on success, where new_i is the index
        of the first node NOT consumed by the match.
        Returns None if the transform does not apply at this position.
        """


# ── Typed sub-hierarchies ─────────────────────────────────────────────────────

class SubstituteTransform(Transform):
    """1 → 1: rewrite expressions within a single node.

    Implement apply().  Return a HIRNode (modified or the same object) when the
    transform applies, or None when it does not match.

    The base match() automatically:
      - copies src_eas from the input node to any newly-created output node
      - copies ann from the input if the new node has no ann set
    (Both are skipped when apply() returns the same node object.)
    """

    def match(self, nodes, i, reg_map, simplify):
        node = nodes[i]
        result = self.apply(node, reg_map, simplify)
        if result is None:
            return None
        if result is not node:
            result.src_eas = node.src_eas
            if result.ann is None:
                result.ann = node.ann
        return ([result], i + 1)

    @abstractmethod
    def apply(self,
              node:     HIRNode,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[HIRNode]:
        """Return the (possibly new) node if the transform applies, or None."""


class _NTo1Transform(Transform):
    """Private base for N → 1 transforms.

    Subclasses implement produce(), which returns (out_node, new_i) or None.

    The base match() automatically:
      - unions src_eas from all consumed nodes (nodes[i:new_i]) into out_node
      - merges annotations from the first and last consumed nodes into out_node
        (only when produce() has not already set out_node.ann)
    """

    def match(self, nodes, i, reg_map, simplify):
        result = self.produce(nodes, i, reg_map, simplify)
        if result is None:
            return None
        out_node, new_i = result
        out_node.src_eas = frozenset().union(
            *(n.src_eas for n in nodes[i:new_i]))
        if out_node.ann is None:
            out_node.ann = NodeAnnotation.merge(nodes[i], nodes[new_i - 1])
        return ([out_node], new_i)

    @abstractmethod
    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Tuple[HIRNode, int]]:
        """Return (out_node, new_i) if the transform applies, or None."""


class InlineTransform(_NTo1Transform):
    """N → 1: fold expressions from auxiliary nodes into a host node.

    The host node is returned in modified form with expressions from the
    auxiliary nodes embedded into it.  All consumed nodes are replaced by the
    single modified host.

    Examples: AccumRelayPattern (A as relay), RegPostIncPattern,
              RegPreIncPattern, SignBitTestPattern.
    """


class CombineTransform(_NTo1Transform):
    """N → 1: synthesise a fresh node from the combined effect of N input nodes.

    All N consumed nodes are replaced by a single brand-new node.

    Examples: XRAMLocalWritePattern, ConstGroupPattern, XRAMGroupReadPattern,
              MultiByteAddPattern, MultiByteIncDecPattern, Neg16Pattern,
              RolSwitchPattern.
    """


class EliminateTransform(Transform):
    """N → 0: consume nodes and produce no output.

    Implement detect(), which returns the new position (past all consumed nodes)
    when the transform applies, or None otherwise.

    src_eas from consumed nodes are intentionally discarded.

    Example: RegCopyGroupPattern — drops register-copy sequences and propagates
             the source variable name forward via reg_map mutation.
    """

    def match(self, nodes, i, reg_map, simplify):
        new_i = self.detect(nodes, i, reg_map, simplify)
        if new_i is None:
            return None
        return ([], new_i)

    @abstractmethod
    def detect(self,
               nodes:    List[HIRNode],
               i:        int,
               reg_map:  Dict[str, VarInfo],
               simplify: Simplify) -> Optional[int]:
        """Return the new position past all consumed nodes, or None."""


class RestructureTransform(Transform):
    """N → M (M ≥ 2): restructure a node sequence into a different sequence.

    Implement produce(), which returns (out_nodes, new_i) or None.

    The base match() unions src_eas from all consumed nodes into every output
    node (the union is identical for all outputs).

    Example: XchCopyPattern (N consumed → 3 streamlined output nodes).
    """

    def match(self, nodes, i, reg_map, simplify):
        result = self.produce(nodes, i, reg_map, simplify)
        if result is None:
            return None
        out_nodes, new_i = result
        all_src = frozenset().union(*(n.src_eas for n in nodes[i:new_i]))
        for out in out_nodes:
            out.src_eas = out.src_eas | all_src
        return (out_nodes, new_i)

    @abstractmethod
    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Tuple[List[HIRNode], int]]:
        """Return (out_nodes, new_i) if the transform applies, or None."""


# ── Backward-compatible alias ─────────────────────────────────────────────────

Pattern = Transform
