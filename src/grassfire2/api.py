from __future__ import annotations

import logging
from typing import Optional, Protocol

from . import adapters
from .events.loop import DebugHook, event_loop, init_event_list
from .init import init_skeleton, internal_only_skeleton
from .model import Skeleton
from .transform import get_box, get_transform

logger = logging.getLogger(__name__)


class PointsAndSegments(Protocol):
    points: list[tuple[float, float]]
    infos: object
    segments: object


def compute_skeleton(
    conv: PointsAndSegments,
    *,
    shrink: bool = True,
    internal_only: bool = False,
    debug_hook: Optional[DebugHook] = None,
) -> Skeleton:
    if shrink:
        box = get_box(conv.points)
        transform = get_transform(box)
        pts = list(map(transform.forward, conv.points))
    else:
        pts = conv.points
        transform = None

    dt = adapters.triangulate_with_tri(pts, conv.infos, conv.segments)
    mesh = adapters.from_tri_delaunay(dt)

    skel = init_skeleton(mesh)
    if internal_only:
        skel = internal_only_skeleton(skel)
    if shrink:
        skel.transform = transform

    queue = init_event_list(skel)
    _last = event_loop(queue, skel, debug_hook=debug_hook)
    return skel


def compute_segments(
    conv: PointsAndSegments,
    *,
    shrink: bool = True,
    internal_only: bool = False,
):
    return compute_skeleton(conv, shrink=shrink, internal_only=internal_only).segments()


def calc_skel(conv: PointsAndSegments, *, shrink: bool = True, internal_only: bool = False) -> Skeleton:
    # backwards-ish compatibility (drops pause/output flags)
    return compute_skeleton(conv, shrink=shrink, internal_only=internal_only)
