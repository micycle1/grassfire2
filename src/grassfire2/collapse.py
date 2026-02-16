from __future__ import annotations

import bisect
import logging
import math
from typing import Callable, Optional

from tri.delaunay.tds import cw, ccw, Edge

from .tolerances import get_unique_times, near_zero
from .model import Event, InfiniteVertex, KineticTriangle, KineticVertex
from .linalg import dot, sub

try:
    from predicates import orient2d_xy as _orient2d_xy  # type: ignore
    def orient2d(p0, p1, p2) -> float:
        return float(_orient2d_xy(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]))
except Exception:  # pragma: no cover
    from tri.delaunay.tds import orient2d as orient2d  # type: ignore

logger = logging.getLogger(__name__)

Sieve = Callable[[list[float], float], Optional[float]]


def find_gt(a: list[Optional[float]], x: float) -> Optional[float]:
    a2 = [v for v in a if v is not None]
    a2 = [v for v in a2 if not near_zero(v - x)]
    L = sorted(a2)
    i = bisect.bisect_right(L, x)
    if i != len(L):
        return L[i]
    return None


def find_gte(a: list[Optional[float]], x: float) -> Optional[float]:
    L = sorted([v for v in a if v is not None])
    i = bisect.bisect_left(L, x)
    if i != len(L):
        return L[i]
    return None


def vertex_crash_time(org: KineticVertex, dst: KineticVertex, apx: KineticVertex) -> Optional[float]:
    Mv = tuple(sub(apx.origin, org.origin))  # type: ignore[arg-type]
    assert org.ur is not None
    assert org.ur == dst.ul
    n = tuple(org.ur.w)  # type: ignore[attr-defined]
    s = apx.velocity  # type: ignore[assignment]
    dist_v_e = dot(Mv, n)
    s_proj = dot(s, n)  # type: ignore[arg-type]
    denom = 1.0 - s_proj
    if not near_zero(denom):
        return dist_v_e / denom
    return None


def solve_quadratic(A: float, B: float, C: float) -> list[float]:
    if near_zero(A) and not near_zero(B):
        return [-C / B]
    if near_zero(A) and near_zero(B):
        return []

    T = -B / A
    D = C / A
    centre = T * 0.5
    under = 0.25 * (T ** 2) - D
    if near_zero(under):
        return [centre]
    if under < 0:
        return []
    plus_min = math.sqrt(under)
    return [centre - plus_min, centre + plus_min]


def area_collapse_time_coeff(kva: KineticVertex, kvb: KineticVertex, kvc: KineticVertex) -> tuple[float, float, float]:
    pa, shifta = kva.origin, kva.velocity
    pb, shiftb = kvb.origin, kvb.velocity
    pc, shiftc = kvc.origin, kvc.velocity
    xaorig, yaorig = pa[0], pa[1]
    xborig, yborig = pb[0], pb[1]
    xcorig, ycorig = pc[0], pc[1]
    dxa, dya = shifta[0], shifta[1]
    dxb, dyb = shiftb[0], shiftb[1]
    dxc, dyc = shiftc[0], shiftc[1]

    A = dxa * dyb - dxb * dya + dxb * dyc - dxc * dyb + dxc * dya - dxa * dyc
    B = (
        xaorig * dyb
        - xborig * dya
        + xborig * dyc
        - xcorig * dyb
        + xcorig * dya
        - xaorig * dyc
        + dxa * yborig
        - dxb * yaorig
        + dxb * ycorig
        - dxc * yborig
        + dxc * yaorig
        - dxa * ycorig
    )
    C = (
        xaorig * yborig
        - xborig * yaorig
        + xborig * ycorig
        - xcorig * yborig
        + xcorig * yaorig
        - xaorig * ycorig
    )
    return (A, B, C)


def area_collapse_times(o: KineticVertex, d: KineticVertex, a: KineticVertex) -> list[float]:
    coeff = area_collapse_time_coeff(o, d, a)
    sol = solve_quadratic(coeff[0], coeff[1], coeff[2])
    sol.sort()
    return sol


def collapse_time_edge(v1: KineticVertex, v2: KineticVertex) -> float:
    s1 = v1.velocity
    s2 = v2.velocity
    o1 = v1.origin
    o2 = v2.origin
    dv = sub(s1, s2)
    denominator = dot(dv, dv)
    if not near_zero(denominator):
        w0 = sub(o2, o1)
        nominator = dot(dv, w0)
        collapse_time = nominator / denominator
        logger.debug("edge collapse time: %s", collapse_time)
        return float(collapse_time)
    logger.debug("%s|%s", v1, v2)
    return -1.0


def compute_event_0triangle(tri: KineticTriangle, now: float, sieve: Sieve) -> Optional[Event]:
    assert tri.neighbours.count(None) == 0
    o, d, a = tri.vertices  # type: ignore[misc]
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, KineticVertex)

    times_area = area_collapse_times(o, d, a)
    for time in times_area:
        if near_zero(abs(time - now)):
            dists = [d.distance2_at(a, now), a.distance2_at(o, now), o.distance2_at(d, now)]
            indices = [i for i, val in enumerate(map(math.sqrt, dists)) if near_zero(val)]
            if len(indices) == 1:
                return Event(time=now, tri=tri, side=(indices[0],), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
            if len(indices) == 3:
                raise ValueError("0-triangle collapsing to point")
            side = dists.index(max(dists))
            return Event(time=now, tri=tri, side=(side,), tp="flip", triangle_tp=tri.type)  # type: ignore[arg-type]

    times_edge = [collapse_time_edge(o, d), collapse_time_edge(d, a), collapse_time_edge(a, o)]
    dists = [o.distance2_at(d, times_edge[0]), d.distance2_at(a, times_edge[1]), a.distance2_at(o, times_edge[2])]
    indices = [i for i, val in enumerate(map(math.sqrt, dists)) if near_zero(val)]
    t_e = [times_edge[i] for i in indices]
    time_edge = sieve(t_e, now)
    time_area = sieve(times_area, now)

    if time_edge is None and time_area is None:
        return None

    if time_edge is not None and time_area is not None:
        if near_zero(abs(time_area - time_edge)):
            time = time_edge
            dists2 = [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]
            zeros = [near_zero(v - min(dists2)) for v in dists2]
            ct = zeros.count(True)
            if ct == 3:
                return Event(time=time, tri=tri, side=(0, 1, 2), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
            if ct == 1:
                return Event(time=time, tri=tri, side=(zeros.index(True),), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
            time = time_area
            dists3 = [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]
            side = dists3.index(max(dists3))
            return Event(time=time, tri=tri, side=(side,), tp="flip", triangle_tp=tri.type)  # type: ignore[arg-type]

        if time_area < time_edge:
            time = time_area
            dists3 = [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]
            side = dists3.index(max(dists3))
            return Event(time=time, tri=tri, side=(side,), tp="flip", triangle_tp=tri.type)  # type: ignore[arg-type]

        time = time_edge
        dists3 = [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]
        zeros = [near_zero(v) for v in dists3]
        ct = zeros.count(True)
        if ct == 3:
            return Event(time=time, tri=tri, side=(0, 1, 2), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
        if ct == 1:
            return Event(time=time, tri=tri, side=(zeros.index(True),), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
        raise ValueError("can this happen?")

    if time_edge is not None:
        time = time_edge
        dists3 = [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]
        zeros = [near_zero(v) for v in dists3]
        ct = zeros.count(True)
        if ct == 3:
            return Event(time=time, tri=tri, side=(0, 1, 2), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
        if ct == 1:
            return Event(time=time, tri=tri, side=(zeros.index(True),), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
        raise ValueError("0 triangle with 2 or 0 side collapse while edge collapse time computed?")

    time = time_area
    dists3 = [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]
    side = dists3.index(max(dists3))
    return Event(time=time, tri=tri, side=(side,), tp="flip", triangle_tp=tri.type)  # type: ignore[arg-type]


def compute_event_1triangle(tri: KineticTriangle, now: float, sieve: Sieve) -> Optional[Event]:
    assert tri.neighbours.count(None) == 1
    wavefront_side = tri.neighbours.index(None)

    o, d, a = tri.vertices  # type: ignore[misc]
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, KineticVertex)

    ow, dw, aw = (tri.vertices[ccw(wavefront_side)], tri.vertices[cw(wavefront_side)], tri.vertices[wavefront_side])  # type: ignore[misc]
    assert isinstance(ow, KineticVertex) and isinstance(dw, KineticVertex) and isinstance(aw, KineticVertex)

    times_vertex_crash = [vertex_crash_time(ow, dw, aw)]
    for time in times_vertex_crash:
        if time is None:
            continue
        if near_zero(abs(time - now)):
            time = now
            dists = [d.distance2_at(a, now), a.distance2_at(o, now), o.distance2_at(d, now)]
            indices = [i for i, val in enumerate(map(math.sqrt, dists)) if near_zero(val)]
            if len(indices) == 1:
                return Event(time=now, tri=tri, side=(indices[0],), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
            d2 = [math.sqrt(d.distance2_at(a, time)), math.sqrt(a.distance2_at(o, time)), math.sqrt(o.distance2_at(d, time))]
            longest = d2.index(max(d2))
            tp = "split" if longest == wavefront_side else "flip"
            return Event(time=time, tri=tri, side=(longest,), tp=tp, triangle_tp=tri.type)  # type: ignore[arg-type]

    time_vertex = sieve([t for t in times_vertex_crash if t is not None], now)
    time_area = sieve(area_collapse_times(o, d, a), now)
    time_edge = sieve([collapse_time_edge(ow, dw)], now)

    if time_edge is None and time_vertex is None:
        time = sieve(solve_quadratic(*area_collapse_time_coeff(*tri.vertices)), now)  # type: ignore[arg-type]
        if time is None:
            return None
        if near_zero(time - now):
            return Event(time=now, tri=tri, side=(wavefront_side,), tp="split", triangle_tp=tri.type)  # type: ignore[arg-type]

        dists = [
            d.distance2_at(a, time) if tri.neighbours[0] is not None else -1,
            a.distance2_at(o, time) if tri.neighbours[1] is not None else -1,
            o.distance2_at(d, time) if tri.neighbours[2] is not None else -1,
        ]
        side = (dists.index(max(dists)),)
        return Event(time=time, tri=tri, side=side, tp="flip", triangle_tp=tri.type)  # type: ignore[arg-type]

    if time_edge is None and time_vertex is not None:
        if time_area is not None and time_area < time_vertex:
            time = time_area
        else:
            time = time_vertex

        d2 = [math.sqrt(d.distance2_at(a, time)), math.sqrt(a.distance2_at(o, time)), math.sqrt(o.distance2_at(d, time))]
        longest_sides = [i for i, val in enumerate(d2) if near_zero(val - max(d2))]
        if wavefront_side in longest_sides and len(longest_sides) == 1:
            return Event(time=time_vertex, tri=tri, side=(wavefront_side,), tp="split", triangle_tp=tri.type)  # type: ignore[arg-type]

        zeros = [near_zero(val) for val in d2]
        ct = zeros.count(True)
        if ct == 1:
            return Event(time=time, tri=tri, side=(d2.index(min(d2)),), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
        return Event(time=time, tri=tri, side=(d2.index(max(d2)),), tp="flip", triangle_tp=tri.type)  # type: ignore[arg-type]

    if time_edge is not None and time_vertex is None:
        return Event(time=time_edge, tri=tri, side=(wavefront_side,), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]

    assert time_edge is not None and time_vertex is not None
    if time_edge <= time_vertex:
        time = time_edge
        dists_sq = [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]
        side = (dists_sq.index(min(dists_sq)),)
        return Event(time=time, tri=tri, side=side, tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]

    time = time_vertex
    d2 = [math.sqrt(d.distance2_at(a, time)), math.sqrt(a.distance2_at(o, time)), math.sqrt(o.distance2_at(d, time))]
    zeros = [near_zero(val) for val in d2]
    if True in zeros and zeros.count(True) == 1:
        return Event(time=time, tri=tri, side=(zeros.index(True),), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
    if True in zeros and zeros.count(True) == 3:
        return Event(time=time, tri=tri, side=(0, 1, 2), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
    max_side = d2.index(max(d2))
    tp = "split" if tri.neighbours[max_side] is None else "flip"
    return Event(time=time, tri=tri, side=(max_side,), tp=tp, triangle_tp=tri.type)  # type: ignore[arg-type]


def compute_event_2triangle(tri: KineticTriangle, now: float, sieve: Sieve) -> Optional[Event]:
    assert tri.neighbours.count(None) == 2
    o, d, a = tri.vertices  # type: ignore[misc]
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, KineticVertex)

    times: list[Optional[float]] = []
    if tri.neighbours[2] is None:
        times.append(collapse_time_edge(o, d))
    if tri.neighbours[0] is None:
        times.append(collapse_time_edge(d, a))
    if tri.neighbours[1] is None:
        times.append(collapse_time_edge(a, o))

    uniq = get_unique_times(times)
    time = sieve(uniq, now)
    if time is None:
        time = sieve(area_collapse_times(o, d, a), now)

    if time is None:
        return None

    d2 = [math.sqrt(d.distance2_at(a, time)), math.sqrt(a.distance2_at(o, time)), math.sqrt(o.distance2_at(d, time))]
    d2 = [val - min(d2) for val in d2]
    zeros = [near_zero(val) for val in d2]
    ct = zeros.count(True)
    if ct == 3:
        return Event(time=time, tri=tri, side=(0, 1, 2), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
    if ct == 2:
        raise ValueError(f"This is not possible with this type of triangle [{tri.info}]")
    if ct == 1:
        side = d2.index(min(d2))
        return Event(time=time, tri=tri, side=(side,), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
    return None


def compute_event_3triangle(tri: KineticTriangle, now: float, sieve: Sieve) -> Optional[Event]:
    a, o, d = tri.vertices  # type: ignore[misc]
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, KineticVertex)

    t_e = [collapse_time_edge(o, d), collapse_time_edge(d, a), collapse_time_edge(a, o)]
    dists = [o.distance2_at(d, t_e[0]), d.distance2_at(a, t_e[1]), a.distance2_at(o, t_e[2])]
    indices = [i for i, val in enumerate(map(math.sqrt, dists)) if near_zero(val)]
    assert tri.neighbours.count(None) == 3

    time_edge = sieve(t_e, now)
    time_area = sieve(area_collapse_times(o, d, a), now)

    if time_edge is not None:
        sides = tuple(indices) if indices else (0, 1, 2)
        if len(sides) in (2, 0):
            sides = (0, 1, 2)
        return Event(time=time_edge, tri=tri, side=sides, tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
    if time_area is not None:
        return Event(time=time_area, tri=tri, side=(0, 1, 2), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
    return None


def compute_event_inftriangle(tri: KineticTriangle, now: float, sieve: Sieve) -> Optional[Event]:
    inf_idx = None
    for idx, v in enumerate(tri.vertices):
        if isinstance(v, InfiniteVertex):
            inf_idx = idx
            break
    assert inf_idx is not None
    side = inf_idx
    o, d, a = tri.vertices[cw(side)], tri.vertices[ccw(side)], tri.vertices[side]  # type: ignore[misc]
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, InfiniteVertex)

    if tri.neighbours[side] is None:
        assert tri.type == 1
        time = find_gt([collapse_time_edge(o, d)], now)
        if time is not None:
            if near_zero(o.distance2_at(d, time)):
                return Event(time=time, tri=tri, side=(side,), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
        return None

    time = sieve(area_collapse_times(o, d, a), now)  # type: ignore[arg-type]
    if time is None:
        return None
    if near_zero(o.distance2_at(d, time)):
        return Event(time=time, tri=tri, side=(side,), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]

    # flip fallback
    dists2 = []
    for func in (cw, ccw):
        start, end = Edge(tri, func(side)).segment
        dists2.append(start.distance2_at(end, time))
    idx = dists2.index(min(dists2))
    min_side = (cw, ccw)[idx](side)
    return Event(time=time, tri=tri, side=(min_side,), tp="flip", triangle_tp=tri.type)  # type: ignore[arg-type]


def compute_collapse_time(tri: KineticTriangle, now: float = 0.0, sieve: Sieve = find_gte) -> Optional[Event]:
    if tri.stops_at is not None:
        return None

    event: Optional[Event] = None
    if tri.is_finite:
        tp = tri.type
        if tp == 0:
            event = compute_event_0triangle(tri, now, sieve)
        elif tp == 1:
            event = compute_event_1triangle(tri, now, sieve)
        elif tp == 2:
            event = compute_event_2triangle(tri, now, sieve)
        elif tp == 3:
            event = compute_event_3triangle(tri, now, sieve)

        # legacy orientation warnings preserved as debug-only
        if event is not None and all(not v.inf_fast for v in tri.vertices if isinstance(v, KineticVertex)):
            verts = [v.position_at(((event.time - now) * 0.5) + now) for v in tri.vertices]  # type: ignore[union-attr]
            if orient2d(verts[0], verts[1], verts[2]) < 0:
                logger.warning("TRIANGLE possibly wrong orientation; might miss event")

    else:
        event = compute_event_inftriangle(tri, now, sieve)

    if event is not None:
        tri.event = event
    return event


def compute_new_edge_collapse_event(tri: KineticTriangle, time: float) -> Event:
    o, d, a = tri.vertices  # type: ignore[misc]
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, KineticVertex)
    dists = list(map(math.sqrt, [d.distance2_at(a, time), a.distance2_at(o, time), o.distance2_at(d, time)]))
    zeros = [near_zero(val - min(dists)) for val in dists]
    sides = tuple(i for i, z in enumerate(zeros) if z)
    return Event(time=time, tri=tri, side=sides, tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]