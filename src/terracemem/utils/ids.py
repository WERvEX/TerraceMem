"""Monotonic RFC 9562 UUIDv7 generator using Python standard library only."""

from __future__ import annotations

import secrets
import threading
import time
from uuid import UUID

_lock = threading.Lock()
_last_ms = 0
_seq = 0


def new_uuid7() -> UUID:
    """Generate a time-ordered RFC 9562 UUIDv7 suitable for B-tree primary keys."""
    global _last_ms, _seq

    with _lock:
        now_ms = time.time_ns() // 1_000_000
        if now_ms > _last_ms:
            _last_ms = now_ms
            _seq = secrets.randbits(12)
        else:
            _seq = (_seq + 1) & 0xFFF
            if _seq == 0:
                _last_ms += 1
            now_ms = _last_ms

        rand_a = _seq
        rand_b = secrets.randbits(62)

    value = (now_ms & 0xFFFFFFFFFFFF) << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0x2 << 62
    value |= rand_b
    return UUID(int=value)
