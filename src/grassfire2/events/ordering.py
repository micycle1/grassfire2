from __future__ import annotations
from ..model import Event


def compare_event_by_time(one: Event, other: Event) -> int:
    # Legacy:
    # 1) time asc
    # 2) triangle type desc
    # 3) id(triangle) as final tie-break
    #
    # New deterministic tie-break:
    # 3) triangle.uid (stable)
    if one.time < other.time:
        return -1
    if one.time > other.time:
        return 1

    if -one.triangle_tp < -other.triangle_tp:
        return -1
    if -one.triangle_tp > -other.triangle_tp:
        return 1

    if one.tri.uid < other.tri.uid:
        return -1
    if one.tri.uid > other.tri.uid:
        return 1
    return 0