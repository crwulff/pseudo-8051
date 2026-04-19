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

from pseudo8051.ir.hir import HIRNode, NodeAnnotation, RemovedNode
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
            node.copy_meta_to(result)
            result.source_nodes = [node]
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
        out_node.source_nodes = list(nodes[i:new_i])
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

    Consumed nodes are recorded in self._pending_removed (drained by
    TypeAwareSimplifier into func.removed_nodes after each pattern pass).

    Example: RegCopyGroupPattern — drops register-copy sequences and propagates
             the source variable name forward via reg_map mutation.
    """

    def __init__(self):
        self._pending_removed: List[RemovedNode] = []

    def match(self, nodes, i, reg_map, simplify):
        new_i = self.detect(nodes, i, reg_map, simplify)
        if new_i is None:
            return None
        for n in nodes[i:new_i]:
            self._pending_removed.append(RemovedNode(n, type(self).__name__))
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
        consumed = list(nodes[i:new_i])
        for out in out_nodes:
            out.source_nodes = consumed
        return (out_nodes, new_i)

    @abstractmethod
    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Tuple[List[HIRNode], int]]:
        """Return (out_nodes, new_i) if the transform applies, or None."""


class ConditionFoldTransform(Transform):
    """
    Consume a 'setup' node, skip optional gap nodes (Labels by default),
    and replace the following condition node's condition expression.

    Pattern (at nodes[i]):
      nodes[i]     — setup node (matched by match_setup)
      nodes[i+1:j] — gap nodes (all satisfy can_gap; Labels by default)
      nodes[j]     — condition node with replace_condition (IfNode, IfGoto, …)

    Output: [gap_nodes…, modified_condition_node]
    The setup node is consumed; the condition node's src_eas absorbs the
    setup node's src_eas so both source instructions appear in the detail view.

    For post-processing use (outside _PATTERNS), call fold_sequence() which
    drives the scan loop without needing reg_map / simplify arguments.
    """

    def can_gap(self, node: HIRNode) -> bool:
        """Return True if node may appear between setup and condition (default: Label)."""
        # Import inside method to avoid circular imports at module load time.
        from pseudo8051.ir.hir.label import Label
        return isinstance(node, Label)

    @abstractmethod
    def match_setup(self, node: HIRNode) -> bool:
        """Return True if node is the setup node for this fold."""

    @abstractmethod
    def new_condition(self,
                      setup_node:   HIRNode,
                      cond_node:    HIRNode,
                      current_cond: "Expr") -> "Optional[Expr]":
        """
        Return the replacement condition expression, or None if this combination
        doesn't match.  current_cond is the existing condition of cond_node.
        """

    # ------------------------------------------------------------------
    # Transform ABC implementation
    # ------------------------------------------------------------------

    def match(self, nodes, i, reg_map, simplify):
        node = nodes[i]
        if not self.match_setup(node):
            return None
        # Skip gap nodes (Labels by default).
        j = i + 1
        while j < len(nodes) and self.can_gap(nodes[j]):
            j += 1
        if j >= len(nodes):
            return None
        cond_node = nodes[j]
        if not hasattr(cond_node, "replace_condition"):
            return None
        # Extract the existing condition from the target node.
        from pseudo8051.ir.hir import IfNode, WhileNode, ForNode, DoWhileNode, IfGoto
        if isinstance(cond_node, (IfNode, WhileNode, ForNode, DoWhileNode)):
            current_cond = cond_node.condition
        elif isinstance(cond_node, IfGoto):
            current_cond = cond_node.cond
        else:
            return None
        new_cond = self.new_condition(node, cond_node, current_cond)
        if new_cond is None:
            return None
        repl = cond_node.replace_condition(new_cond)
        # Record setup node + original condition node as sources.
        repl.source_nodes = [node, cond_node]
        gap_nodes = list(nodes[i + 1:j])
        return (gap_nodes + [repl], j + 1)

    # ------------------------------------------------------------------
    # Convenience runner for post-processing (no reg_map / simplify needed)
    # ------------------------------------------------------------------

    def fold_sequence(self, nodes: List[HIRNode]) -> List[HIRNode]:
        """
        Apply this fold to a flat node list.
        Does not recurse into structured-node bodies — callers should do so
        via map_bodies before or after if needed.
        """
        result: List[HIRNode] = []
        i = 0
        while i < len(nodes):
            m = self.match(nodes, i, {}, lambda x, _: x)
            if m is not None:
                result.extend(m[0])
                i = m[1]
            else:
                result.append(nodes[i])
                i += 1
        return result


# ── Backward-compatible alias ─────────────────────────────────────────────────

Pattern = Transform
