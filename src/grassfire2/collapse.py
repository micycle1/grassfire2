from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from .topology import ccw, cw

from .linalg import dot, sub
from .model import Event, InfiniteVertex, KineticTriangle, KineticVertex, VertexRef
from .tolerances import get_unique_times, near_zero

STOP_EPS = 1e-9 # Allow small negative τ due to floating error; clamp to "now".

Sieve = Callable[[list[Optional[float]], float], Optional[float]]


def find_gt(a: list[Optional[float]], x: float) -> Optional[float]:
    best: Optional[float] = None
    for v in a:
        if v is None:
            continue
        if near_zero(v - x):
            continue
        if v <= x:
            continue
        if best is None or v < best:
            best = v
    return best


def find_gte(a: list[Optional[float]], x: float) -> Optional[float]:
    best: Optional[float] = None
    for v in a:
        if v is None or v < x:
            continue
        if best is None or v < best:
            best = v
    return best


def vertex_crash_time(
    org: KineticVertex,
    dst: KineticVertex,
    apx: KineticVertex,
    now: float,
) -> Optional[float]:
    """
    Absolute time when apex hits the moving wavefront edge (org-dst),
    robustly from state at `now`.
    """
    assert org.ur is not None
    assert org.ur == dst.ul
    n = org.ur.w

    Por = org.position_at(now)
    Pap = apx.position_at(now)
    s = apx.velocity_at(now)

    dist_v_e = dot(sub(Pap, Por), n)
    s_proj = dot(s, n)
    denom = 1.0 - s_proj
    if near_zero(denom):
        return None

    tau = dist_v_e / denom
    if tau < -STOP_EPS:
        return None
    return now + max(0.0, float(tau))


def area_collapse_time_coeff_tau(
    o: VertexRef, d: VertexRef, a: VertexRef, now: float
) -> tuple[float, float, float]:
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
    if near_zero(A2) and not near_zero(A1):
        return [-A0 / A1]
    if near_zero(A2) and near_zero(A1):
        return []

    # Solve A2*t^2 + A1*t + A0 = 0 stably
    T = -A1 / A2
    D = A0 / A2
    centre = T * 0.5
    under = 0.25 * (T * T) - D
    if near_zero(under):
        return [centre]
    if under < 0.0:
        return []
    s = math.sqrt(under)

    if centre > 0:
        r1 = centre + s
        r2 = D / r1
    else:
        r1 = centre - s
        r2 = D / r1 if r1 != 0 else 0.0
    return sorted([float(r1), float(r2)])


def area_collapse_times(o: VertexRef, d: VertexRef, a: VertexRef, now: float) -> list[float]:
    A2, A1, A0 = area_collapse_time_coeff_tau(o, d, a, now)
    roots_tau = solve_quadratic(A2, A1, A0)
    out: list[float] = []
    for tau in roots_tau:
        if tau >= -STOP_EPS:
            out.append(now + max(0.0, tau))
    out.sort()
    return out


def collapse_time_edge(v1: VertexRef, v2: VertexRef, now: float) -> Optional[float]:
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
    return T


def _finite_vertices(tri: KineticTriangle) -> tuple[KineticVertex, KineticVertex, KineticVertex]:
    o, d, a = tri.vertices
    assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, KineticVertex)
    return o, d, a


def _side_d2(o: KineticVertex, d: KineticVertex, a: KineticVertex, t: float) -> list[float]:
    # side 0: edge (d,a), side 1: edge (a,o), side 2: edge (o,d)
    return [d.distance2_at(a, t), a.distance2_at(o, t), o.distance2_at(d, t)]


def _side_lengths(o: KineticVertex, d: KineticVertex, a: KineticVertex, t: float) -> list[float]:
    d2 = _side_d2(o, d, a, t)
    return [math.sqrt(v) for v in d2]


def _edge_pair_for_side(o: KineticVertex, d: KineticVertex, a: KineticVertex, side: int) -> tuple[KineticVertex, KineticVertex]:
    if side == 0:
        return d, a
    if side == 1:
        return a, o
    return o, d  # side == 2


def _edge_event(tri: KineticTriangle, time: float, sides: tuple[int, ...]) -> Event:
    return Event(time=time, tri=tri, side=sides, tp="edge", triangle_tp=tri.type)


def _flip_event(tri: KineticTriangle, time: float, side: int) -> Event:
    return Event(time=time, tri=tri, side=(side,), tp="flip", triangle_tp=tri.type)


def _split_event(tri: KineticTriangle, time: float, side: int) -> Event:
    return Event(time=time, tri=tri, side=(side,), tp="split", triangle_tp=tri.type)


@dataclass(slots=True)
class CollapseEventComputer:
    sieve: Sieve = find_gte

    def compute(self, tri: KineticTriangle, now: float = 0.0) -> Optional[Event]:
        if tri.stops_at is not None:
            return None

        # tri.neighbours[i] is None means side i is on the wavefront/boundary.
        # Finite types are effectively classified by count(None).
        if tri.is_finite:
            none_ct = tri.neighbours.count(None)
            if none_ct == 0:
                event = self._finite_0(tri, now)
            elif none_ct == 1:
                event = self._finite_1(tri, now)
            elif none_ct == 2:
                event = self._finite_2(tri, now)
            elif none_ct == 3:
                event = self._finite_3(tri, now)
            else:
                event = None
        else:
            event = self._infinite(tri, now)

        if event is not None:
            tri.event = event
        return event

    def _finite_0(self, tri: KineticTriangle, now: float) -> Optional[Event]:
        o, d, a = _finite_vertices(tri)
        assert tri.neighbours.count(None) == 0

        times_area = area_collapse_times(o, d, a, now)
        for t in times_area:
            if near_zero(abs(t - now)):
                d2_now = _side_d2(o, d, a, now)
                zero_len_sides = tuple(i for i, v in enumerate(d2_now) if near_zero(math.sqrt(v)))
                if len(zero_len_sides) == 1:
                    return _edge_event(tri, now, (zero_len_sides[0],))
                if len(zero_len_sides) == 3:
                    raise ValueError("0-triangle collapsing to point")
                side = d2_now.index(max(d2_now))
                return _flip_event(tri, now, side)

        # Edge events only if the closest-approach time actually yields ~0 length
        edge_zero_times: list[float] = []
        for side in (0, 1, 2):
            p, q = _edge_pair_for_side(o, d, a, side)
            t = collapse_time_edge(p, q, now)
            if t is None:
                continue
            if near_zero(math.sqrt(p.distance2_at(q, t))):
                edge_zero_times.append(t)

        time_edge = self.sieve(edge_zero_times, now)
        time_area = self.sieve(times_area, now)

        if time_edge is None and time_area is None:
            return None

        if time_edge is not None and time_area is not None:
            if near_zero(abs(time_area - time_edge)):
                # tie: try to classify as edge at time_edge using relative-min test; otherwise flip
                d2_t = _side_d2(o, d, a, time_edge)
                m = min(d2_t)
                rel_zeros = [near_zero(v - m) for v in d2_t]
                ct = rel_zeros.count(True)
                if ct == 3:
                    return _edge_event(tri, time_edge, (0, 1, 2))
                if ct == 1:
                    return _edge_event(tri, time_edge, (rel_zeros.index(True),))

                d2_a = _side_d2(o, d, a, time_area)
                return _flip_event(tri, time_area, d2_a.index(max(d2_a)))

            if time_area < time_edge:
                d2_a = _side_d2(o, d, a, time_area)
                return _flip_event(tri, time_area, d2_a.index(max(d2_a)))

            d2_e = _side_d2(o, d, a, time_edge)
            abs_zeros = [near_zero(v) for v in d2_e]
            ct = abs_zeros.count(True)
            if ct == 3:
                return _edge_event(tri, time_edge, (0, 1, 2))
            if ct == 1:
                return _edge_event(tri, time_edge, (abs_zeros.index(True),))
            raise ValueError("can this happen?")

        if time_edge is not None:
            d2_e = _side_d2(o, d, a, time_edge)
            abs_zeros = [near_zero(v) for v in d2_e]
            ct = abs_zeros.count(True)
            if ct == 3:
                return _edge_event(tri, time_edge, (0, 1, 2))
            if ct == 1:
                return _edge_event(tri, time_edge, (abs_zeros.index(True),))
            raise ValueError("0 triangle with 2 or 0 side collapse while edge collapse time computed?")

        assert time_area is not None
        d2_a = _side_d2(o, d, a, time_area)
        return _flip_event(tri, time_area, d2_a.index(max(d2_a)))

    def _finite_1(self, tri: KineticTriangle, now: float) -> Optional[Event]:
        o, d, a = _finite_vertices(tri)
        assert tri.neighbours.count(None) == 1
        wavefront_side = tri.neighbours.index(None)

        # Wavefront edge is opposite wavefront_side
        ow = tri.vertices[ccw(wavefront_side)]
        dw = tri.vertices[cw(wavefront_side)]
        aw = tri.vertices[wavefront_side]
        assert isinstance(ow, KineticVertex) and isinstance(dw, KineticVertex) and isinstance(aw, KineticVertex)

        t_crash = vertex_crash_time(ow, dw, aw, now)
        if t_crash is not None and near_zero(abs(t_crash - now)):
            d2_now = _side_d2(o, d, a, now)
            zero_len_sides = tuple(i for i, v in enumerate(d2_now) if near_zero(math.sqrt(v)))
            if len(zero_len_sides) == 1:
                return _edge_event(tri, now, (zero_len_sides[0],))

            lens_now = _side_lengths(o, d, a, now)
            longest = lens_now.index(max(lens_now))
            return (_split_event(tri, now, longest) if longest == wavefront_side else _flip_event(tri, now, longest))

        # Candidate event times (absolute):
            # - vertex: wavefront apex hits wavefront edge (1-triangle specific)
            # - area: signed area of (o,d,a) goes to zero (flip/split candidate)
            # - edge: wavefront edge (ow,dw) collapses (edge event on wavefront side)
        time_vertex = self.sieve([t_crash] if t_crash is not None else [], now)
        times_area = area_collapse_times(o, d, a, now)
        time_area = self.sieve(times_area, now)
        time_edge = self.sieve([t for t in [collapse_time_edge(ow, dw, now)] if t is not None], now)

        if time_edge is None and time_vertex is None:
            if time_area is None:
                return None
            if near_zero(time_area - now):
                return _split_event(tri, now, wavefront_side)

            d2 = _side_d2(o, d, a, time_area)
            # ignore wavefront side in "longest side" selection by forcing -1
            d2_masked = [
                d2[0] if tri.neighbours[0] is not None else -1.0,
                d2[1] if tri.neighbours[1] is not None else -1.0,
                d2[2] if tri.neighbours[2] is not None else -1.0,
            ]
            side = int(d2_masked.index(max(d2_masked)))
            return _flip_event(tri, time_area, side)

        if time_edge is None and time_vertex is not None:
            time = time_vertex
            if time_area is not None and time_area < time_vertex:
                time = time_area

            lens = _side_lengths(o, d, a, time)
            mx = max(lens)
            longest_sides = [i for i, v in enumerate(lens) if near_zero(v - mx)]
            if wavefront_side in longest_sides and len(longest_sides) == 1:
                return _split_event(tri, time_vertex, wavefront_side)

            zeros = [near_zero(v) for v in lens]
            if zeros.count(True) == 1:
                return _edge_event(tri, time, (lens.index(min(lens)),))
            return _flip_event(tri, time, lens.index(max(lens)))

        if time_edge is not None and time_vertex is None:
            return _edge_event(tri, time_edge, (wavefront_side,))

        assert time_edge is not None and time_vertex is not None
        if time_edge <= time_vertex:
            d2 = _side_d2(o, d, a, time_edge)
            return _edge_event(tri, time_edge, (d2.index(min(d2)),))

        lens = _side_lengths(o, d, a, time_vertex)
        zeros = [near_zero(v) for v in lens]
        if True in zeros and zeros.count(True) == 1:
            return _edge_event(tri, time_vertex, (zeros.index(True),))
        if True in zeros and zeros.count(True) == 3:
            return _edge_event(tri, time_vertex, (0, 1, 2))

        max_side = lens.index(max(lens))
        return (_split_event(tri, time_vertex, max_side) if tri.neighbours[max_side] is None else _flip_event(tri, time_vertex, max_side))

    def _finite_2(self, tri: KineticTriangle, now: float) -> Optional[Event]:
        o, d, a = _finite_vertices(tri)
        assert tri.neighbours.count(None) == 2

        times: list[Optional[float]] = []
        for side in (0, 1, 2):
            if tri.neighbours[side] is None:
                p, q = _edge_pair_for_side(o, d, a, side)
                times.append(collapse_time_edge(p, q, now))

        uniq = get_unique_times(times)
        time = self.sieve(uniq, now)
        if time is None:
            time = self.sieve(area_collapse_times(o, d, a, now), now)
        if time is None:
            return None

        lens = _side_lengths(o, d, a, time)
        m = min(lens)
        rel = [v - m for v in lens]
        zeros = [near_zero(v) for v in rel]
        ct = zeros.count(True)
        if ct == 3:
            return _edge_event(tri, time, (0, 1, 2))
        if ct == 2:
            raise ValueError(f"This is not possible with this type of triangle [{tri.info}]")
        if ct == 1:
            return _edge_event(tri, time, (rel.index(min(rel)),))
        return None

    def _finite_3(self, tri: KineticTriangle, now: float) -> Optional[Event]:
        o, d, a = _finite_vertices(tri)
        assert tri.neighbours.count(None) == 3

        # edge times in side order: 0:(d,a), 1:(a,o), 2:(o,d)
        t_e: list[Optional[float]] = []
        for side in (0, 1, 2):
            p, q = _edge_pair_for_side(o, d, a, side)
            t_e.append(collapse_time_edge(p, q, now))

        dists: list[float] = []
        for side, t in enumerate(t_e):
            p, q = _edge_pair_for_side(o, d, a, side)
            dists.append(math.inf if t is None else p.distance2_at(q, t))

        indices = [i for i, val in enumerate(map(math.sqrt, dists)) if near_zero(val)]

        time_edge = self.sieve([t for t in t_e if t is not None], now)
        time_area = self.sieve(area_collapse_times(o, d, a, now), now)

        if time_edge is not None:
            sides = tuple(indices) if indices else (0, 1, 2)
            if len(sides) in (2, 0):
                sides = (0, 1, 2)
            return _edge_event(tri, time_edge, sides)
        if time_area is not None:
            return _edge_event(tri, time_area, (0, 1, 2))
        return None

    # Infinite triangles: only the finite edge (o,d) can collapse; otherwise we
    # use area-collapse as a proxy and classify as edge/flip accordingly.
    def _infinite(self, tri: KineticTriangle, now: float) -> Optional[Event]:
        inf_idx = None
        for idx, v in enumerate(tri.vertices):
            if isinstance(v, InfiniteVertex):
                inf_idx = idx
                break
        assert inf_idx is not None

        side = inf_idx
        o = tri.vertices[cw(side)]
        d = tri.vertices[ccw(side)]
        a = tri.vertices[side]
        assert isinstance(o, KineticVertex) and isinstance(d, KineticVertex) and isinstance(a, InfiniteVertex)

        if tri.neighbours[side] is None:
            assert tri.type == 1
            time = find_gt([collapse_time_edge(o, d, now)], now)
            if time is not None and near_zero(o.distance2_at(d, time)):
                return _edge_event(tri, time, (side,))
            return None

        time = self.sieve(area_collapse_times(o, d, a, now), now)
        if time is None:
            return None
        if near_zero(o.distance2_at(d, time)):
            return _edge_event(tri, time, (side,))

        # flip fallback
        dists2: list[float] = []
        for func in (cw, ccw):
            s_idx = func(side)
            start = tri.vertices[ccw(s_idx)]
            end = tri.vertices[cw(s_idx)]
            dists2.append(start.distance2_at(end, time))
        idx = dists2.index(min(dists2))
        min_side = (cw, ccw)[idx](side)
        return _flip_event(tri, time, min_side)


def compute_collapse_time(tri: KineticTriangle, now: float = 0.0, sieve: Sieve = find_gte) -> Optional[Event]:
    return CollapseEventComputer(sieve=sieve).compute(tri, now)


def compute_new_edge_collapse_event(tri: KineticTriangle, time: float) -> Event:
    o, d, a = _finite_vertices(tri)
    lens = _side_lengths(o, d, a, time)
    m = min(lens)
    zeros = [near_zero(v - m) for v in lens]
    sides = tuple(i for i, z in enumerate(zeros) if z)
    return _edge_event(tri, time, sides)
