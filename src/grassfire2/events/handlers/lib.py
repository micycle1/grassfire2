from __future__ import annotations

import logging
import math
from typing import Optional

from ..queue import OrderedSequence
from ...tolerances import near_zero
from ...collapse import compute_collapse_time, compute_new_edge_collapse_event
from ...model import KineticTriangle, KineticVertex, SkeletonNode, Event
from ...linalg import add, mul, norm

from ...line import LineLineIntersector, LineLineIntersectionResult, make_vector  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def is_infinitely_fast(fan: list[KineticTriangle], now: float) -> bool:
    times = [tri.event.time if tri.event is not None else -1 for tri in fan]
    return bool(fan) and all(near_zero(t - now) for t in times)


def stop_kvertices(V: list[KineticVertex], step: int, now: float, pos: Optional[tuple[float, float]] = None):
    sk_node = None

    for v in V:
        stopped = v.stops_at is not None
        time_close = near_zero(v.starts_at - now)  # type: ignore[operator]
        if stopped:
            sk_node = v.stop_node
        elif time_close:
            sk_node = v.start_node
        else:
            v.stops_at = now

    if sk_node is not None:
        for v in V:
            v.stop_node = sk_node
            v.stops_at = now
        is_new_node = False
    else:
        if pos is None:
            pts = [v.position_at(now) for v in V]
            sumx = sum(p[0] for p in pts)
            sumy = sum(p[1] for p in pts)
            pos = (sumx / len(pts), sumy / len(pts))
        sk_node = SkeletonNode(pos=pos, step=step)
        for v in V:
            v.stop_node = sk_node
        is_new_node = True

    for v in V:
        assert v.stop_node is not None and v.stops_at == now
    return sk_node, is_new_node


def compute_new_kvertex(ul, ur, now: float, sk_node: SkeletonNode, info: int, internal: bool) -> KineticVertex:
    kv = KineticVertex()
    kv.info = info
    kv.starts_at = now
    kv.start_node = sk_node
    kv.internal = internal

    from ...linalg import add as vadd
    from ...linalg import dot as vdot

    u1, u2 = ul.w, ur.w
    direction = vadd(u1, u2)

    # legacy angle_unit behavior
    d = vdot(u1, u2)
    d = max(-1.0, min(1.0, d))
    acos_d = math.acos(d)

    if (near_zero(direction[0]) and near_zero(direction[1])) or near_zero(acos_d - math.pi) or d < math.cos(math.radians(179.999999)):
        bi = (0.0, 0.0)
        pos_at_t0 = sk_node.pos
    else:
        intersect = LineLineIntersector(ul, ur)
        tp = intersect.intersection_type()
        if tp == LineLineIntersectionResult.NO_INTERSECTION:
            bi = (0.0, 0.0)
            pos_at_t0 = sk_node.pos
        elif tp == LineLineIntersectionResult.POINT:
            pos_at_t0 = intersect.result
            ul_t = ul.translated(ul.w)
            ur_t = ur.translated(ur.w)
            intersect_t = LineLineIntersector(ul_t, ur_t)
            assert intersect_t.intersection_type() == LineLineIntersectionResult.POINT
            bi = make_vector(end=intersect_t.result, start=pos_at_t0)
        elif tp == LineLineIntersectionResult.LINE:
            bi = tuple(ul.w[:])
            neg_velo = mul(mul(bi, -1.0), now)
            pos_at_t0 = add(sk_node.pos, neg_velo)
        else:
            raise RuntimeError(f"Unknown intersection type: {tp}")

    kv.velocity = (float(bi[0]), float(bi[1]))
    if kv.velocity == (0.0, 0.0):
        kv.inf_fast = True
        kv.origin = sk_node.pos
    else:
        kv.origin = pos_at_t0

    kv.ul = ul
    kv.ur = ur
    return kv


def get_fan(t: KineticTriangle, v: KineticVertex, direction):
    fan = []
    start = t
    while t is not None:
        side = t.vertices.index(v)
        fan.append(t)
        t = t.neighbours[direction(side)]
        assert t is not start
    return fan


def update_circ(v_left: Optional[KineticVertex], v_right: Optional[KineticVertex], now: float) -> None:
    if v_left is not None:
        v_left.right = (v_right, now)  # type: ignore[arg-type]
    if v_right is not None:
        v_right.left = (v_left, now)  # type: ignore[arg-type]


def replace_in_queue(t: KineticTriangle, now: float, queue: OrderedSequence[Event], immediate: list[Event]) -> None:
    if t.event is not None:
        queue.discard(t.event)
        if t.event in immediate:
            immediate.remove(t.event)

    e = compute_collapse_time(t, now)
    if e is not None:
        queue.add(e)


def replace_kvertex(t: Optional[KineticTriangle], v: KineticVertex, newv: KineticVertex, now: float, direction, queue, immediate):
    fan = []
    while t is not None:
        side = t.vertices.index(v)
        fan.append(t)
        t.vertices[side] = newv

        if newv.inf_fast and t.event is not None:
            queue.discard(t.event)
            if t.event in immediate:
                immediate.remove(t.event)
        else:
            replace_in_queue(t, now, queue, immediate)

        t = t.neighbours[direction(side)]
    return tuple(fan)


def schedule_immediately(tri: KineticTriangle, now: float, queue, immediate) -> None:
    if tri.event is not None:
        queue.discard(tri.event)
        if tri.event in immediate:
            immediate.remove(tri.event)
    E = compute_new_edge_collapse_event(tri, now)
    tri.event = E
    if tri.neighbours.count(None) == 3:
        tri.event.side = (0, 1, 2)  # type: ignore[misc]
    assert len(tri.event.side) > 0  # type: ignore[arg-type]
    immediate.append(tri.event)