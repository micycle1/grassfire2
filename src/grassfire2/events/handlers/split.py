from __future__ import annotations

from tri.delaunay.tds import cw, ccw

from ...model import Event
from .lib import stop_kvertices, compute_new_kvertex, update_circ, replace_kvertex
from .parallel import handle_parallel_fan


def handle_split_event(evt: Event, step: int, skel, queue, immediate) -> None:
    t = evt.tri
    assert len(evt.side) == 1
    e = evt.side[0]
    now = evt.time

    v = t.vertices[e % 3]
    n = t.neighbours[e]
    assert n is None
    v1 = t.vertices[(e + 1) % 3]
    v2 = t.vertices[(e + 2) % 3]
    assert v is not None and v1 is not None and v2 is not None

    assert v1.wfr is v2.wfl

    sk_node, newly_made = stop_kvertices([v], step, now)
    if newly_made:
        skel.sk_nodes.append(sk_node)

    assert v1.ur is v2.ul

    vb = compute_new_kvertex(v.ul, v2.ul, now, sk_node, len(skel.vertices) + 1, v.internal or v2.internal)
    vb.wfl = v.wfl
    vb.wfr = v2.wfl
    skel.vertices.append(vb)

    va = compute_new_kvertex(v1.ur, v.ur, now, sk_node, len(skel.vertices) + 1, v.internal or v1.internal)
    va.wfl = v1.wfr
    va.wfr = v.wfr
    skel.vertices.append(va)

    update_circ(v.left, vb, now)     # type: ignore[arg-type]
    update_circ(vb, v2, now)         # type: ignore[arg-type]

    update_circ(v1, va, now)         # type: ignore[arg-type]
    update_circ(va, v.right, now)    # type: ignore[arg-type]

    b = t.neighbours[(e + 1) % 3]
    assert b is not None
    b.neighbours[b.neighbours.index(t)] = None
    fan_b = replace_kvertex(b, v, vb, now, ccw, queue, immediate)

    a = t.neighbours[(e + 2) % 3]
    assert a is not None
    a.neighbours[a.neighbours.index(t)] = None
    fan_a = replace_kvertex(a, v, va, now, cw, queue, immediate)

    t.stops_at = now

    if va.inf_fast:
        handle_parallel_fan(list(fan_a), va, now, cw, step, skel, queue, immediate)
    if vb.inf_fast:
        handle_parallel_fan(list(fan_b), vb, now, ccw, step, skel, queue, immediate)