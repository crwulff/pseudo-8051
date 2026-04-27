"""
ir/hir/switch_node.py — SwitchNode structured control-flow node.
"""

from typing import Callable, List, Optional, Tuple, Union

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _ann_field, _killed_by_seq, _possibly_killed_by_seq, _refs_from_expr
from pseudo8051.ir.expr import Expr


class SwitchNode(HIRNode):
    """
    switch (subject) {
        case 2: goto label_a;
        case 4: case 8: goto label_b;
        default: goto label_default;
    }

    cases is a list of (values_list, body) pairs where body is either:
      - str: goto label (pre-absorption)
      - List[HIRNode]: inlined body (post-absorption by SwitchBodyAbsorber)

    default_label is the target for unmatched values (from a trailing jnz), or None.
    default_body is the inlined default body (post-absorption), or None.
    """

    def __init__(self, ea: int, subject: Expr,
                 cases: List[Tuple[List[int], Union[str, List['HIRNode']]]],
                 default_label: Optional[str] = None,
                 default_body: Optional[List['HIRNode']] = None,
                 case_comments: Optional[List[Optional[str]]] = None,
                 case_src_eas: Optional[List[frozenset]] = None,
                 default_src_eas: Optional[frozenset] = None,
                 case_enum_names: Optional[List[Optional[List[str]]]] = None):
        super().__init__(ea)
        self.subject         = subject
        self.cases           = cases
        self.default_label   = default_label
        self.default_body    = default_body
        self.case_comments:  List[Optional[str]] = case_comments or []
        # Per-case enum name lists: when set, case labels use enum names instead of
        # integer constants.  None entry = fall back to integer + comment.
        self.case_enum_names: Optional[List[Optional[List[str]]]] = case_enum_names
        # Per-case instruction EAs: one frozenset per entry in self.cases (same order).
        # None means "not tracked" — fall back to the whole-switch src_eas.
        self.case_src_eas:   Optional[List[frozenset]] = case_src_eas
        # EAs for the default arm's comparison/branch instruction(s).
        self.default_src_eas: Optional[frozenset] = default_src_eas

    def map_bodies(self, fn: Callable[[List[HIRNode]], List[HIRNode]]) -> "SwitchNode":
        new_cases = [
            (vals, fn(body) if isinstance(body, list) else body)
            for vals, body in self.cases
        ]
        new_default_body = fn(self.default_body) if isinstance(self.default_body, list) else self.default_body
        return self.copy_meta_to(SwitchNode(self.ea, self.subject, new_cases, self.default_label, new_default_body,
                                             case_comments=list(self.case_comments),
                                             case_src_eas=self.case_src_eas,
                                             default_src_eas=self.default_src_eas,
                                             case_enum_names=list(self.case_enum_names) if self.case_enum_names is not None else None))

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind  = self._ind(indent)
        ind1 = self._ind(indent + 1)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}switch ({_render_expr(self.subject)}) {{"))
        # Pre-compute case prefixes and find max width for comment alignment.
        def _case_prefix(i: int, values: List[int]) -> str:
            enum_names = (self.case_enum_names[i]
                          if self.case_enum_names is not None and i < len(self.case_enum_names)
                          else None)
            if enum_names:
                return " ".join(f"case {n}:" for n in enum_names)
            return " ".join(f"case {v}:" for v in values)

        prefixes = [_case_prefix(i, values) for i, (values, _) in enumerate(self.cases)]
        # Only add comments for cases that don't already use enum name labels.
        commented = [i for i in range(len(self.cases))
                     if i < len(self.case_comments) and self.case_comments[i]
                     and (self.case_enum_names is None
                          or i >= len(self.case_enum_names)
                          or not self.case_enum_names[i])]
        comment_col = (max(len(prefixes[i]) for i in commented) + 2
                       if commented else 0)
        for i, (values, body) in enumerate(self.cases):
            case_prefix = prefixes[i]
            has_enum_labels = (self.case_enum_names is not None
                               and i < len(self.case_enum_names)
                               and self.case_enum_names[i])
            if not has_enum_labels and i < len(self.case_comments) and self.case_comments[i]:
                pad = " " * (comment_col - len(case_prefix))
                comment = f"{pad}// {self.case_comments[i]}"
            else:
                comment = ""
            if isinstance(body, str):
                lines.append((self.ea, f"{ind1}{case_prefix} goto {body};{comment}"))
            else:
                lines.append((self.ea, f"{ind1}{case_prefix}{comment}"))
                for node in body:
                    lines.extend(node.render(indent + 2))
        if self.default_body is not None:
            lines.append((self.ea, f"{ind1}default:"))
            for node in self.default_body:
                lines.extend(node.render(indent + 2))
        elif self.default_label is not None:
            lines.append((self.ea, f"{ind1}default: goto {self.default_label};"))
        lines.append((self.ea, f"{ind}}}"))
        return lines

    def definitely_killed(self) -> frozenset:
        """Registers killed on ALL case paths (intersection across all inlined bodies)."""
        inlined = [body for _, body in self.cases if isinstance(body, list)]
        if self.default_body is not None:
            inlined.append(self.default_body)
        if not inlined:
            return frozenset()
        result = _killed_by_seq(inlined[0])
        for body in inlined[1:]:
            result &= _killed_by_seq(body)
        return result

    def possibly_killed(self) -> frozenset:
        """Registers killed on ANY case path (union across all inlined bodies)."""
        result: frozenset = frozenset()
        for _, body in self.cases:
            if isinstance(body, list):
                result |= _possibly_killed_by_seq(body)
        if self.default_body is not None:
            result |= _possibly_killed_by_seq(self.default_body)
        return result

    def name_refs(self) -> frozenset:
        subject_refs = _refs_from_expr(self.subject)
        body_refs: frozenset = frozenset()
        for _, body in self.cases:
            if isinstance(body, list):
                body_refs = body_refs.union(*(n.name_refs() for n in body))
        if self.default_body is not None:
            body_refs = body_refs.union(*(n.name_refs() for n in self.default_body))
        return subject_refs | body_refs

    def ann_lines(self) -> List[str]:
        out = (["SwitchNode"] + _ann_field("subject", self.subject)
               + [f"  cases: {len(self.cases)}"])
        if self.default_label is not None:
            out.append(f"  default: {self.default_label!r}")
        return out
