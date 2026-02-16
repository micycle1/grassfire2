from __future__ import annotations

from tri.delaunay.tds import apex, orig, dest, ccw
from ...model import Event, KineticTriangle
from .lib import replace_in_queue


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


def handle_flip_event(evt: Event, step: int, skel, queue, immediate) -> None:
    now = evt.time
    assert len(evt.side) == 1
    t, t_side = evt.tri, evt.side[0]
    n = t.neighbours[t_side]
    assert n is not None
    n_side = n.neighbours.index(t)
    flip(t, t_side, n, n_side)
    replace_in_queue(t, now, queue, immediate)
    replace_in_queue(n, now, queue, immediate)