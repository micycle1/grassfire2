from __future__ import annotations

import logging
import math
from typing import Optional
from ...topology import apex, ccw, dest, orig

from ...collapse import compute_collapse_time, compute_new_edge_collapse_event
from ...model import Event, KineticTriangle, KineticVertex, SkeletonNode
from ...tolerances import near_zero
from ..queue import EventQueue
from ...line import Line2

logger = logging.getLogger(__name__)

COS_179_999999 = math.cos(math.radians(179.999999))


def is_infinitely_fast(fan: list[KineticTriangle], now: float) -> bool:
    times = [tri.event.time if tri.event is not None else -1 for tri in fan]
    return bool(fan) and all(near_zero(t - now) for t in times)


def stop_kvertices(V: list[KineticVertex], step: int, now: float, pos: Optional[tuple[float, float]] = None):
    sk_node = None

    for v in V:
        stopped = v.stops_at is not None
        time_close = near_zero(v.starts_at - now)
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


def compute_new_kvertex(ul: Line2, ur: Line2, now: float, sk_node: SkeletonNode, info: int, internal: bool) -> KineticVertex:
    kv = KineticVertex()
    kv.info = info
    kv.starts_at = now
    kv.start_node = sk_node
    kv.internal = internal

    u1x, u1y = ul.w
    u2x, u2y = ur.w

    # "direction" is sum of normals (your legacy check)
    dirx, diry = (u1x + u2x), (u1y + u2y)

    # dot(u1,u2) (unit vectors => in [-1,1])
    d = u1x * u2x + u1y * u2y
    d = max(-1.0, min(1.0, d))

    # preserve your old degeneracy tests but cheaper (avoid acos)
    # old: near_zero(acos(d)-pi)  <=> d near -1
    # plus your explicit cosine threshold
    if (near_zero(dirx) and near_zero(diry)) or near_zero(d + 1.0) or (d < COS_179_999999):
        bi = (0.0, 0.0)
        pos_at_t0 = sk_node.pos
    else:
        # Determine if ul and ur intersect, are parallel distinct, or coincident.
        denom = u1x * u2y - u2x * u1y
        if near_zero(denom):
            # Parallel: test coincident using minors (valid because Line2 is normalized)
            # If normals are parallel, lines coincide iff (a1,b1,c1) and (a2,b2,c2) represent same line.
            # With normalization, check:
            x1 = u1x * ur.b - u2x * ul.b
            x2 = u1y * ur.b - u2y * ul.b
            if near_zero(x1) and near_zero(x2):
                # LINE (coincident): legacy behavior => velocity along ul.w
                bi = (u1x, u1y)
                # choose origin so that kv.position_at(now) == sk_node.pos
                pos_at_t0 = (sk_node.pos[0] - bi[0] * now, sk_node.pos[1] - bi[1] * now)
            else:
                # NO_INTERSECTION (distinct parallel)
                bi = (0.0, 0.0)
                pos_at_t0 = sk_node.pos
        else:
            # POINT intersection at line-time t=0 and t=1
            p0 = ul.intersect_at_time(ur, 0.0)
            p1 = ul.intersect_at_time(ur, 1.0)
            if p0 is None or p1 is None:
                bi = (0.0, 0.0)
                pos_at_t0 = sk_node.pos
            else:
                pos_at_t0 = p0
                bi = (p1[0] - p0[0], p1[1] - p0[1])

    kv.velocity = (float(bi[0]), float(bi[1]))
    if kv.velocity == (0.0, 0.0):
        kv.inf_fast = True
        kv.origin = sk_node.pos
    else:
        kv.origin = (float(pos_at_t0[0]), float(pos_at_t0[1]))

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
        v_left.right = (v_right, now)
    if v_right is not None:
        v_right.left = (v_left, now)


def replace_in_queue(t: KineticTriangle, now: float, queue: EventQueue[Event], immediate: list[Event]) -> None:
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
        tri.event.side = (0, 1, 2)
    assert len(tri.event.side) > 0
    immediate.append(tri.event)
    

def flip(t0: KineticTriangle, side0: int, t1: KineticTriangle, side1: int) -> None:
    apex0, orig0, dest0 = apex(side0), orig(side0), dest(side0)
    apex1, orig1, dest1 = apex(side1), orig(side1), dest(side1)

    assert t0.vertices[orig0] is t1.vertices[dest1]
    assert t0.vertices[dest0] is t1.vertices[orig1]
    assert t0.neighbours[apex0] is not None
    assert t1.neighbours[apex1] is not None

    A, B, C, D = t0.vertices[apex0], t0.vertices[orig0], t1.vertices[apex1], t0.vertices[dest0]
    AB, BC, CD, DA = t0.neighbours[dest0], t1.neighbours[orig1], t1.neighbours[dest1], t0.neighbours[orig0]

    apex_around = []
    for neighbour, corner in zip([AB, BC, CD, DA], [A, B, C, D]):
        if neighbour is None:
            apex_around.append(None)
        else:
            apex_around.append(ccw(neighbour.vertices.index(corner)))

    for neighbour, side, t in zip([AB, BC, CD, DA], apex_around, [t0, t0, t1, t1]):
        if neighbour is not None:
            neighbour.neighbours[side] = t

    t0.vertices = [A, B, C]
    t0.neighbours = [BC, t1, AB]
    t1.vertices = [C, D, A]
    t1.neighbours = [DA, t0, CD]
