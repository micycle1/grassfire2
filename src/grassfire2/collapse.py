from __future__ import annotations

import bisect
import logging
import math
from typing import Callable, Optional

from predicates import orient2d
from tri.delaunay.tds import cw, ccw, Edge

from .tolerances import get_unique_times, near_zero
from .model import Event, InfiniteVertex, KineticTriangle, KineticVertex, VertexRef
from .linalg import dot, sub

Vec2 = tuple[float, float]

STOP_EPS = 1e-9

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


def vertex_crash_time(
    org: KineticVertex,
    dst: KineticVertex,
    apx: KineticVertex,
    now: float,
) -> Optional[float]:
    """
    Predict absolute time T when apex hits the moving wavefront edge (org-dst),
    computed robustly from positions/velocities at `now`.

    Returns absolute time or None if undefined / not in the future.
    """
    assert org.ur is not None
    assert org.ur == dst.ul
    n = tuple(org.ur.w)  # type: ignore[attr-defined]  # unit normal

    Por = org.position_at(now)
    Pap = apx.position_at(now)
    s = apx.velocity_at(now)

    # Signed distance along edge normal at current time
    dist_v_e = dot(sub(Pap, Por), n)
    s_proj = dot(s, n)
    denom = 1.0 - s_proj
    if near_zero(denom):
        return None

    tau = dist_v_e / denom
    if tau < -STOP_EPS:
        return None
    return now + max(0.0, float(tau))


def cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def area_collapse_time_coeff_tau(o: VertexRef, d: VertexRef, a: VertexRef, now: float) -> tuple[float, float, float]:
    Ax, Ay = o.position_at(now)
    Bx, By = d.position_at(now)
    Cx, Cy = a.position_at(now)

    Vax, Vay = o.velocity_at(now)
    Vbx, Vby = d.velocity_at(now)
    Vcx, Vcy = a.velocity_at(now)

    dP10x, dP10y = (Bx - Ax), (By - Ay)
    dP20x, dP20y = (Cx - Ax), (Cy - Ay)
    dV10x, dV10y = (Vbx - Vax), (Vby - Vay)
    dV20x, dV20y = (Vcx - Vax), (Vcy - Vay)

    A2 = dV10x * dV20y - dV10y * dV20x
    A1 = (dP10x * dV20y - dP10y * dV20x) + (dV10x * dP20y - dV10y * dP20x)
    A0 = dP10x * dP20y - dP10y * dP20x
    return (float(A2), float(A1), float(A0))


def solve_quadratic(A2: float, A1: float, A0: float) -> list[float]:
    # (same as your existing solve_quadratic, but coefficients order matches A2,A1,A0)
    if near_zero(A2) and not near_zero(A1):
        return [-A0 / A1]
    if near_zero(A2) and near_zero(A1):
        return []

    # Solve A2*t^2 + A1*t + A0 = 0 in a numerically reasonable way
    # (Keeping your existing method style)
    T = -A1 / A2
    D = A0 / A2
    centre = T * 0.5
    under = 0.25 * (T * T) - D
    if near_zero(under):
        return [centre]
    if under < 0.0:
        return []
    s = math.sqrt(under)
    return [centre - s, centre + s]


def pick_future_root_tau(A2: float, A1: float, A0: float) -> float:
    """
    Choose a τ >= 0 root.
    This borrows the explicit selection idea from Surfer2 (don’t just 'min positive' blindly).

    Returns +inf if no future root.
    """
    roots = solve_quadratic(A2, A1, A0)
    if not roots:
        return math.inf

    # Filter to τ >= 0 (with tolerance)
    fut = [r for r in roots if r >= -STOP_EPS]
    if not fut:
        return math.inf

    if len(fut) == 1:
        return max(0.0, fut[0])

    r0, r1 = (min(fut), max(fut))

    # Optional heuristic matching the Java excerpt:
    # - if parabola opens upward (A2 > 0): earliest non-negative crossing
    # - if opens downward (A2 < 0): latest non-negative crossing
    if A2 > 0.0:
        return max(0.0, r0)
    if A2 < 0.0:
        return max(0.0, r1)

    # Linear-ish fallback (shouldn’t happen due to near_zero(A2) earlier)
    return max(0.0, r0)



def area_collapse_times(o: KineticVertex, d: KineticVertex, a: KineticVertex, now: float) -> list[float]:
    """
    Replacement for your existing area_collapse_times(o,d,a) that returns ABSOLUTE times,
    computed stably from the current state at 'now'.
    """
    A2, A1, A0 = area_collapse_time_coeff_tau(o, d, a, now)
    roots_tau = solve_quadratic(A2, A1, A0)
    out: list[float] = []
    for tau in roots_tau:
        if tau >= -STOP_EPS:
            out.append(now + max(0.0, tau))
    out.sort()
    return out


def area_collapse_time_first(o: KineticVertex, d: KineticVertex, a: KineticVertex, now: float) -> float:
    """
    If you prefer a single chosen time instead of all times.
    """
    A2, A1, A0 = area_collapse_time_coeff_tau(o, d, a, now)
    tau = pick_future_root_tau(A2, A1, A0)
    if not math.isfinite(tau):
        return math.inf
    return now + tau


def collapse_time_edge(v1: VertexRef, v2: VertexRef, now: float) -> Optional[float]:
    """
    Predict absolute time T of closest approach between v1 and v2, computed from
    state at `now` (positions + effective velocities).

    This avoids cancellation from absolute-time formulas and avoids stale velocities
    for stopped vertices (velocity_at => (0,0) after stop).
    """
    P1 = v1.position_at(now)
    P2 = v2.position_at(now)
    
    V1 = v1.velocity_at(now)
    V2 = v2.velocity_at(now)

    dv = sub(V1, V2)
    denom = dot(dv, dv)
    if near_zero(denom):
        return None

    w0 = sub(P2, P1)
    tau = dot(dv, w0) / denom
    if tau < -STOP_EPS:
        return None

    T = now + max(0.0, float(tau))
    logger.debug("edge collapse time (abs): %s", T)
    return T


def compute_event_0triangle(tri: KineticTriangle, now: float, sieve: Sieve) -> Optional[Event]:
    assert tri.neighbours.count(None) == 0
    o, d, a = tri.vertices  # type: ignore[misc]
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, KineticVertex)

    times_area = area_collapse_times(o, d, a, now)
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

    times_edge: list[Optional[float]] = [
        collapse_time_edge(o, d, now),
        collapse_time_edge(d, a, now),
        collapse_time_edge(a, o, now),
    ]

    # If time is None (no meaningful closest-approach), treat as "no edge event"
    pairs = [(o, d), (d, a), (a, o)]
    dists: list[float] = []
    for (p, q), t in zip(pairs, times_edge):
        dists.append(math.inf if t is None else p.distance2_at(q, t))

    indices = [i for i, val in enumerate(map(math.sqrt, dists)) if near_zero(val)]
    t_e = [times_edge[i] for i in indices if times_edge[i] is not None]
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

    times_vertex_crash = [vertex_crash_time(ow, dw, aw, now)]
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
    time_area = sieve(area_collapse_times(o, d, a, now), now)
    time_edge = sieve([t for t in [collapse_time_edge(ow, dw, now)] if t is not None], now)

    if time_edge is None and time_vertex is None:
        A2, A1, A0 = area_collapse_time_coeff_tau(*tri.vertices, now)  # type: ignore[arg-type]
        taus = solve_quadratic(A2, A1, A0)
        abs_times = [now + max(0.0, t) for t in taus if t >= -STOP_EPS]
        time = sieve(abs_times, now)
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
        times.append(collapse_time_edge(o, d, now))
    if tri.neighbours[0] is None:
        times.append(collapse_time_edge(d, a, now))
    if tri.neighbours[1] is None:
        times.append(collapse_time_edge(a, o, now))

    uniq = get_unique_times(times)
    time = sieve(uniq, now)
    if time is None:
        time =  sieve(area_collapse_times(o, d, a, now), now)

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

    t_e: list[Optional[float]] = [
        collapse_time_edge(o, d, now),
        collapse_time_edge(d, a, now),
        collapse_time_edge(a, o, now),
    ]

    dists: list[float] = []
    for (p, q), t in zip([(o, d), (d, a), (a, o)], t_e):
        dists.append(math.inf if t is None else p.distance2_at(q, t))

    indices = [i for i, val in enumerate(map(math.sqrt, dists)) if near_zero(val)]
    assert tri.neighbours.count(None) == 3

    time_edge = sieve([t for t in t_e if t is not None], now)
    time_area = sieve(area_collapse_times(o, d, a, now), now)

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
        time = find_gt([collapse_time_edge(o, d, now)], now)
        if time is not None:
            if near_zero(o.distance2_at(d, time)):
                return Event(time=time, tri=tri, side=(side,), tp="edge", triangle_tp=tri.type)  # type: ignore[arg-type]
        return None

    time = sieve(area_collapse_times(o, d, a, now), now)  # type: ignore[arg-type]
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
    return Event(time=time, tri=tri, side=sides, tp="edge", triangle_tp=tri.type)