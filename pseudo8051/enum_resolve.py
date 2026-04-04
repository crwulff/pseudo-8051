"""
enum_resolve.py — IDA enum constant lookup helpers.

Exports:
  resolve_enum_const(type_str, value) -> Optional[str]
      Return the IDA enum member name for (type_str, value), or None if
      type_str is not a known IDA enum or value has no member.

  is_enum_type(type_str) -> bool
      True if type_str names a known IDA enum.
"""

from typing import Optional

from pseudo8051.constants import dbg

# type_str → enum_id (int) or -1 if not an enum
_ENUM_ID_CACHE: dict = {}

# type_str → {value → name_or_None}
_ENUM_VALUE_CACHE: dict = {}


def _enum_id(type_str: str) -> int:
    """Return the IDA enum_id for type_str, or -1 if not found."""
    if type_str in _ENUM_ID_CACHE:
        return _ENUM_ID_CACHE[type_str]
    try:
        import idc
        eid = idc.get_enum(type_str)
        result = eid if eid != idc.BADADDR else -1
        dbg("enum", f"  _enum_id({type_str!r}) → {result:#x}" if result != -1
            else f"  _enum_id({type_str!r}) → not found")
    except Exception as e:
        dbg("enum", f"  _enum_id({type_str!r}) → exception: {e}")
        result = -1
    _ENUM_ID_CACHE[type_str] = result
    return result


def resolve_enum_const(type_str: str, value: int) -> Optional[str]:
    """
    Look up the IDA enum member name for (type_str, value).

    Returns the member name string (e.g. "FORWARD") or None when:
      - type_str is not a known IDA enum, or
      - value has no matching member in that enum.
    """
    val_map = _ENUM_VALUE_CACHE.setdefault(type_str, {})
    if value in val_map:
        return val_map[value]

    eid = _enum_id(type_str)
    if eid == -1:
        val_map[value] = None
        return None

    try:
        import idc
        # serial=0 tries the first member with this value; -1 = DEFMASK (any mask)
        mid = idc.get_enum_member(eid, value, 0, -1)
        if mid == idc.BADADDR:
            dbg("enum", f"  resolve_enum_const({type_str!r}, {value:#x}) → no member")
            val_map[value] = None
            return None
        name = idc.get_enum_member_name(mid)
        dbg("enum", f"  resolve_enum_const({type_str!r}, {value:#x}) → {name!r}")
        val_map[value] = name or None
        return val_map[value]
    except Exception as e:
        dbg("enum", f"  resolve_enum_const({type_str!r}, {value:#x}) → exception: {e}")
        val_map[value] = None
        return None


def is_enum_type(type_str: str) -> bool:
    """Return True if type_str names a known IDA enum."""
    return _enum_id(type_str) != -1


def clear_cache() -> None:
    """Invalidate the enum lookup cache (call after IDA type changes)."""
    _ENUM_ID_CACHE.clear()
    _ENUM_VALUE_CACHE.clear()
