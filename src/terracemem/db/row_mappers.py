"""Type-safe runtime decoders for psycopg tuple[object, ...] database rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any
from uuid import UUID


def as_uuid(val: object) -> UUID:
    """Decode a UUID column value."""
    if isinstance(val, UUID):
        return val
    if isinstance(val, str):
        return UUID(val)
    raise TypeError(f"Expected UUID column, got {type(val).__name__}")


def as_opt_uuid(val: object) -> UUID | None:
    """Decode a nullable UUID column value."""
    if val is None:
        return None
    return as_uuid(val)


def as_int(val: object, *, default: int = 0) -> int:
    """Decode an integer column value."""
    if val is None:
        return default
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, (float, str)):
        return int(val)
    raise TypeError(f"Expected integer column, got {type(val).__name__}")


def as_opt_int(val: object) -> int | None:
    """Decode a nullable integer column value."""
    if val is None:
        return None
    return as_int(val)


def as_float(val: object, *, default: float = 0.0) -> float:
    """Decode a float/numeric column value."""
    if val is None:
        return default
    if isinstance(val, (int, float, str)):
        return float(val)
    raise TypeError(f"Expected float column, got {type(val).__name__}")


def as_str(val: object, *, default: str = "") -> str:
    """Decode a text column value."""
    if val is None:
        return default
    return str(val)


def as_opt_str(val: object) -> str | None:
    """Decode a nullable text column value."""
    if val is None:
        return None
    return str(val)


def as_str_list(val: object) -> list[str]:
    """Decode a TEXT[] or JSONB array column into list[str]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(item) for item in val]
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return [str(item) for item in val]
    raise TypeError(f"Expected sequence column, got {type(val).__name__}")


def as_datetime(val: object) -> datetime:
    """Decode a TIMESTAMPTZ column value."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    raise TypeError(f"Expected datetime column, got {type(val).__name__}")


def as_opt_datetime(val: object) -> datetime | None:
    """Decode a nullable TIMESTAMPTZ column value."""
    if val is None:
        return None
    return as_datetime(val)


def as_dict(val: object) -> dict[str, Any]:
    """Decode a JSONB object column into dict[str, Any]."""
    if val is None:
        return {}
    if isinstance(val, Mapping):
        return {str(k): v for k, v in val.items()}
    raise TypeError(f"Expected JSON mapping column, got {type(val).__name__}")
