from __future__ import annotations

import logging
import math

from tri.delaunay.tds import cw, ccw, Edge

from ...model import Event
from .lib import (
    stop_kvertices,
    compute_new_kvertex,
    update_circ,
    replace_kvertex,
    schedule_immediately,
)
from ...tolerances import near_zero
from .parallel import handle_parallel_fan
from ...line import WaveFrontIntersector

logger = logging.getLogger(__name__)


def handle_edge_event(evt: Event, step: int, skel, queue, immediate) -> None:
    t = evt.tri
    assert len(evt.side) == 1
    e = evt.side[0]
    is_wavefront_collapse = t.neighbours[e] is None
    now = evt.time

    v1 = t.vertices[ccw(e)]
    v2 = t.vertices[cw(e)]
    assert v1 is not None and v2 is not None

    if is_wavefront_collapse and (not v1.is_stopped) and (not v2.is_stopped):
        assert v1.right is v2
        assert v2.left is v1

    a = v1.wfl
    c = v2.wfr
    assert a is not None and c is not None

    pos_at_now = None
    try:
        pos_at_now = WaveFrontIntersector(a, c).get_intersection_at_t(now)
    except ValueError:
        pass

    sk_node, newly_made = stop_kvertices([v1, v2], step, now, pos=pos_at_now)
    if newly_made:
        skel.sk_nodes.append(sk_node)

    kv = compute_new_kvertex(v1.ul, v2.ur, now, sk_node, len(skel.vertices) + 1, v1.internal or v2.internal)
    kv.wfl = v1.wfl
    kv.wfr = v2.wfr

    skel.vertices.append(kv)

    update_circ(v1.left, kv, now)    # type: ignore[arg-type]
    update_circ(kv, v2.right, now)   # type: ignore[arg-type]

    assert kv.left is None or kv.wfl is kv.left.wfr
    assert kv.right is None or kv.wfr is kv.right.wfl

    a_tri = t.neighbours[ccw(e)]
    b_tri = t.neighbours[cw(e)]
    n_tri = t.neighbours[e]

    fan_a = []
    fan_b = []

    if a_tri is not None:
        a_idx = a_tri.neighbours.index(t)
        a_tri.neighbours[a_idx] = b_tri
        fan_a = replace_kvertex(a_tri, v2, kv, now, cw, queue, immediate)
        if fan_a:
            ed = Edge(fan_a[-1], cw(fan_a[-1].vertices.index(kv)))
            orig, dest = ed.segment
            if near_zero(math.sqrt(orig.distance2_at(dest, now))):
                schedule_immediately(fan_a[-1], now, queue, immediate)

    if b_tri is not None:
        b_idx = b_tri.neighbours.index(t)
        b_tri.neighbours[b_idx] = a_tri
        fan_b = replace_kvertex(b_tri, v1, kv, now, ccw, queue, immediate)
        if fan_b:
            ed = Edge(fan_b[-1], ccw(fan_b[-1].vertices.index(kv)))
            orig, dest = ed.segment
            if near_zero(math.sqrt(orig.distance2_at(dest, now))):
                schedule_immediately(fan_b[-1], now, queue, immediate)

    if n_tri is not None:
        n_tri.neighbours[n_tri.neighbours.index(t)] = None
        if n_tri.event is not None and n_tri.stops_at is None:
            schedule_immediately(n_tri, now, queue, immediate)

    t.stops_at = now

    if kv.inf_fast:
        if fan_a and fan_b:
            fan_a = list(fan_a)
            fan_a.reverse()
            fan_a.extend(fan_b)
            handle_parallel_fan(fan_a, kv, now, ccw, step, skel, queue, immediate)
        elif fan_a:
            handle_parallel_fan(list(fan_a), kv, now, cw, step, skel, queue, immediate)
        elif fan_b:
            handle_parallel_fan(list(fan_b), kv, now, ccw, step, skel, queue, immediate)


def handle_edge_event_3sides(evt: Event, step: int, skel, queue, immediate) -> None:
    now = evt.time
    t = evt.tri
    assert len(evt.side) == 3

    sk_node, newly_made = stop_kvertices([v for v in t.vertices if v is not None], step, now)
    if newly_made:
        skel.sk_nodes.append(sk_node)

    for n in t.neighbours:
        if n is not None and n.event is not None and n.stops_at is None:
            n.neighbours[n.neighbours.index(t)] = None
            schedule_immediately(n, now, queue, immediate)

    t.stops_at = now


def handle_edge_event_1side(evt: Event, step: int, skel, queue, immediate) -> None:
    t = evt.tri
    assert len(evt.side) == 1
    e = evt.side[0]
    now = evt.time

    v0 = t.vertices[e]
    v1 = t.vertices[ccw(e)]
    v2 = t.vertices[cw(e)]
    assert v0 is not None and v1 is not None and v2 is not None

    sk_node, newly_made = stop_kvertices([v1, v2], step, now)
    if newly_made:
        skel.sk_nodes.append(sk_node)

    kv = compute_new_kvertex(v1.ul, v2.ur, now, sk_node, len(skel.vertices) + 1, v1.internal or v2.internal)
    skel.vertices.append(kv)

    sk_node2, newly_made2 = stop_kvertices([v0, kv], step, now)
    if newly_made2:
        skel.sk_nodes.append(sk_node2)

    t.stops_at = now