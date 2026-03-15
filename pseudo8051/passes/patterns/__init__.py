"""
passes/patterns — Pattern ABC and built-in pattern registry.

To add a new pattern:
  1. Create a new .py file in this directory.
  2. Subclass Pattern, implement match(), and add a docstring.
  3. Import the class below and append an instance to _PATTERNS.
"""

from typing import List

from pseudo8051.passes.patterns.base        import Pattern, Match, Simplify       # noqa: F401
from pseudo8051.passes.patterns._utils      import VarInfo, _replace_pairs        # noqa: F401
from pseudo8051.passes.patterns.sign_bit    import SignBitTestPattern
from pseudo8051.passes.patterns.neg16       import Neg16Pattern
from pseudo8051.passes.patterns.const_group import ConstGroupPattern

_PATTERNS: List[Pattern] = [
    SignBitTestPattern(),
    ConstGroupPattern(),
    Neg16Pattern(),
]
