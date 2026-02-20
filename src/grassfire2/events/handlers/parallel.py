from __future__ import annotations

import logging

from tri.delaunay.tds import ccw, cw, Edge
from tri.delaunay.tds import apex, orig, dest

from ...tolerances import near_zero
from ...linalg import dist, norm
from ...model import KineticTriangle, KineticVertex
from .lib import stop_kvertices, update_circ, compute_new_kvertex, replace_kvertex, schedule_immediately

logger = logging.getLogger(__name__)


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
            neighbour.neighbours[side] = t  # type: ignore[index]
    t0.vertices = [A, B, C]
    t0.neighbours = [BC, t1, AB]
    t1.vertices = [C, D, A]
    t1.neighbours = [DA, t0, CD]


def handle_parallel_fan(
    fan: list[KineticTriangle],
    pivot: KineticVertex,
    now: float,
    direction,
    step: int,
    skel,
    queue,
    immediate,
) -> None:
    if not fan:
        raise ValueError("expected fan of triangles")

    assert pivot.inf_fast
    first_tri = fan[0]
    last_tri = fan[-1]

    if first_tri.neighbours.count(None) == 3:
        assert first_tri is last_tri
        dists = []
        for side in range(3):
            e = Edge(first_tri, side)
            dists.append(dist(*map(lambda x: x.position_at(now), e.segment)))
        dists_sub_min = [near_zero(d - min(dists)) for d in dists]
        if near_zero(min(dists)) and dists_sub_min.count(True) == 1:
            side = dists_sub_min.index(True)
            pivot2 = first_tri.vertices[side]
            assert isinstance(pivot2, KineticVertex)
            handle_parallel_edge_event_even_legs(first_tri, side, pivot2, now, step, skel, queue, immediate)
            return
        handle_parallel_edge_event_3tri(first_tri, first_tri.vertices.index(pivot), pivot, now, step, skel, queue, immediate)
        return

    if direction is cw:
        left = fan[0]
        right = fan[-1]
    else:
        left = fan[-1]
        right = fan[0]

    left_leg_idx = ccw(left.vertices.index(pivot))
    left_leg = Edge(left, left_leg_idx)
    left_dist = dist(*map(lambda x: x.position_at(now), left_leg.segment))

    right_leg_idx = cw(right.vertices.index(pivot))
    right_leg = Edge(right, right_leg_idx)
    right_dist = dist(*map(lambda x: x.position_at(now), right_leg.segment))

    dists = [left_dist, right_dist]
    dists_sub_min = [near_zero(d - min(dists)) for d in dists]
    unique = dists_sub_min.count(True)

    if unique == 2:
        if len(fan) == 1:
            handle_parallel_edge_event_even_legs(first_tri, first_tri.vertices.index(pivot), pivot, now, step, skel, queue, immediate)
        elif len(fan) == 2:
            all_2 = True
            for t in fan:
                left_leg_idx = ccw(t.vertices.index(pivot))
                right_leg_idx = cw(t.vertices.index(pivot))
                ld = dist(*map(lambda x: x.position_at(now), Edge(t, left_leg_idx).segment))
                rd = dist(*map(lambda x: x.position_at(now), Edge(t, right_leg_idx).segment))
                u = [near_zero(ld - min(ld, rd)), near_zero(rd - min(ld, rd))].count(True)
                if u != 2:
                    all_2 = False
            if all_2:
                for t in fan:
                    handle_parallel_edge_event_even_legs(t, t.vertices.index(pivot), pivot, now, step, skel, queue, immediate)
            else:
                t0, t1 = fan[0], fan[1]
                side0 = t0.neighbours.index(t1)
                side1 = t1.neighbours.index(t0)
                flip(t0, side0, t1, side1)
                if any(getattr(v, "inf_fast", False) for v in t0.vertices if v is not None):
                    handle_parallel_edge_event_even_legs(t0, t0.vertices.index(pivot), pivot, now, step, skel, queue, immediate)
                if any(getattr(v, "inf_fast", False) for v in t1.vertices if v is not None):
                    handle_parallel_edge_event_even_legs(t1, t1.vertices.index(pivot), pivot, now, step, skel, queue, immediate)
        else:
            raise NotImplementedError("More than 2 triangles in equal-legs parallel fan")

        return

    shortest_idx = dists_sub_min.index(True)
    if shortest_idx == 1:
        handle_parallel_edge_event_shorter_leg(right_leg.triangle, right_leg.side, pivot, now, step, skel, queue, immediate)
    else:
        handle_parallel_edge_event_shorter_leg(left_leg.triangle, left_leg.side, pivot, now, step, skel, queue, immediate)


def handle_parallel_edge_event_shorter_leg(
    t: KineticTriangle,
    e: int,
    pivot: KineticVertex,
    now: float,
    step: int,
    skel,
    queue,
    immediate,
) -> None:
    assert pivot.inf_fast
    v1 = t.vertices[ccw(e)]
    v2 = t.vertices[cw(e)]
    assert isinstance(v1, KineticVertex) and isinstance(v2, KineticVertex)

    to_stop = [v for v in (v1, v2) if not v.inf_fast]
    sk_node, newly_made = stop_kvertices(to_stop, step, now)
    if newly_made:
        skel.sk_nodes.append(sk_node)

    if pivot.stop_node is None:
        pivot.stop_node = sk_node
        pivot.stops_at = now

    t.stops_at = now

    kv = compute_new_kvertex(v1.ul, v2.ur, now, sk_node, len(skel.vertices) + 1, v1.internal or v2.internal)
    kv.wfl = v1.left.wfr if v1.left is not None else None  # type: ignore[union-attr]
    kv.wfr = v2.right.wfl if v2.right is not None else None  # type: ignore[union-attr]
    skel.vertices.append(kv)

    update_circ(v1.left, kv, now)   # type: ignore[arg-type]
    update_circ(kv, v2.right, now)  # type: ignore[arg-type]

    a = t.neighbours[ccw(e)]
    b = t.neighbours[cw(e)]
    n = t.neighbours[e]

    fan_a = []
    fan_b = []

    if a is not None:
        a_idx = a.neighbours.index(t)
        a.neighbours[a_idx] = b
        fan_a = replace_kvertex(a, v2, kv, now, cw, queue, immediate)

    if b is not None:
        b_idx = b.neighbours.index(t)
        b.neighbours[b_idx] = a
        fan_b = replace_kvertex(b, v1, kv, now, ccw, queue, immediate)

    if n is not None:
        n.neighbours[n.neighbours.index(t)] = None
        if n.event is not None and n.stops_at is None:
            schedule_immediately(n, now, queue, immediate)

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


def handle_parallel_edge_event_even_legs(
    t: KineticTriangle,
    e: int,
    pivot: KineticVertex,
    now: float,
    step: int,
    skel,
    queue,
    immediate,
) -> None:
    assert t.vertices.index(pivot) == e
    v1 = t.vertices[ccw(e)]
    v2 = t.vertices[cw(e)]
    assert isinstance(v1, KineticVertex) and isinstance(v2, KineticVertex)

    sk_node, newly_made = stop_kvertices([v1, v2], step, now)
    if newly_made:
        skel.sk_nodes.append(sk_node)

    pivot.stop_node = sk_node
    pivot.stops_at = now
    t.stops_at = now

    n = t.neighbours[e]
    if n is not None:
        n.neighbours[n.neighbours.index(t)] = None
        if n.event is not None and n.stops_at is None:
            schedule_immediately(n, now, queue, immediate)


def handle_parallel_edge_event_3tri(
    t: KineticTriangle,
    e: int,
    pivot: KineticVertex,
    now: float,
    step: int,
    skel,
    queue,
    immediate,
) -> None:
    assert t.vertices.index(pivot) == e
    v1 = t.vertices[ccw(e)]
    v2 = t.vertices[cw(e)]
    assert isinstance(v1, KineticVertex) and isinstance(v2, KineticVertex)

    magn_v1 = norm(v1.velocity_at(now))
    magn_v2 = norm(v2.velocity_at(now))

    if magn_v2 < magn_v1:
        sk_node, newly_made = stop_kvertices([v2], step, now)
        if newly_made:
            skel.sk_nodes.append(sk_node)
        v1.stop_node = sk_node
        v1.stops_at = now
    else:
        sk_node, newly_made = stop_kvertices([v1], step, now)
        if newly_made:
            skel.sk_nodes.append(sk_node)
        v2.stop_node = sk_node
        v2.stops_at = now

    pivot.stop_node = sk_node
    pivot.stops_at = now
    t.stops_at = now