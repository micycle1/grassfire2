from __future__ import annotations

import logging
from typing import Optional, Callable

from tri.delaunay.insert_kd import triangulate
from tri.delaunay.helpers import ToPointsAndSegments

from .transform import get_box, get_transform
from .init import init_skeleton, internal_only_skeleton
from .events.loop import init_event_list, event_loop, DebugHook
from .model import Skeleton

logger = logging.getLogger(__name__)


def compute_skeleton(
    conv: ToPointsAndSegments,
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

    dt = triangulate(pts, conv.infos, conv.segments, False)

    skel = init_skeleton(dt)
    if internal_only:
        skel = internal_only_skeleton(skel)
    if shrink:
        skel.transform = transform

    queue = init_event_list(skel)
    _last = event_loop(queue, skel, debug_hook=debug_hook)
    return skel


def compute_segments(
    conv: ToPointsAndSegments,
    *,
    shrink: bool = True,
    internal_only: bool = False,
):
    return compute_skeleton(conv, shrink=shrink, internal_only=internal_only).segments()


def calc_skel(conv: ToPointsAndSegments, *, shrink: bool = True, internal_only: bool = False) -> Skeleton:
    # backwards-ish compatibility (drops pause/output flags)
    return compute_skeleton(conv, shrink=shrink, internal_only=internal_only)