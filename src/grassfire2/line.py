from __future__ import annotations

import logging
from dataclasses import dataclass
from math import sqrt, hypot
from typing import Optional

from .linalg import add, dot, mul, norm, sub, unit, rotate90ccw, rotate90cw, make_vector
from .tolerances import near_zero

logger = logging.getLogger(__name__)

Point = tuple[float, float]
Vec2 = tuple[float, float]


def coefficients_from_points(p: Point, q: Point) -> tuple[float, float, float]:
    (px, py) = p
    (qx, qy) = q
    if py == qy:
        a = 0.0
        if qx > px:
            b = 1.0
            c = -py
        elif qx == px:
            b = 0.0
            c = 0.0
        else:
            b = -1.0
            c = py
    elif qx == px:
        b = 0.0
        if qy > py:
            a = -1.0
            c = px
        elif qy == py:
            a = 0.0
            c = 0.0
        else:
            a = 1.0
            c = -px
    else:
        a = py - qy
        b = qx - px
        c = -px * a - py * b
    return a, b, c


def coefficients_perpendicular_through_point(la: float, lb: float, px: float, py: float) -> tuple[float, float, float]:
    a = -lb
    b = la
    c = lb * px - la * py
    return a, b, c


def coefficients_bisector_of_lines(
    pa: float, pb: float, pc: float,
    qa: float, qb: float, qc: float
) -> tuple[float, float, float]:
    # NOTE: expects general (not necessarily normalized) line coefficients.
    n1 = sqrt(pa * pa + pb * pb)
    n2 = sqrt(qa * qa + qb * qb)
    a = n2 * pa + n1 * qa
    b = n2 * pb + n1 * qb
    c = n2 * pc + n1 * qc
    if a == 0 and b == 0:
        a = n2 * pa - n1 * qa
        b = n2 * pb - n1 * qb
        c = n2 * pc - n1 * qc
    return a, b, c


def intersect_unit_normals_at_time(l1: Line2, l2: Line2, t: float) -> Point | None:
    a1, b1 = l1.w
    c1 = l1.b - t
    a2, b2 = l2.w
    c2 = l2.b - t

    denom = a1 * b2 - a2 * b1
    if near_zero(denom):
        return None
    x = (b1 * c2 - b2 * c1) / denom
    y = (a2 * c1 - a1 * c2) / denom
    return (x, y)


@dataclass(slots=True)
class Line2:
    """
    Line represented as: w · x + b = 0, where w is a *unit* normal (wx, wy).

    Invariants:
      - w is always normalized (unit length)
      - b is scaled accordingly
    """
    w: Vec2
    b: float

    def __post_init__(self) -> None:
        wx = float(self.w[0])
        wy = float(self.w[1])
        b = float(self.b)

        n = hypot(wx, wy)
        # if this ever happens, your input geometry has degenerate segments/lines
        if n == 0.0:
            raise ValueError("degenerate line with zero normal")

        inv = 1.0 / n
        self.w = (wx * inv, wy * inv)
        self.b = b * inv

    @classmethod
    def _from_normalized(cls, w: Vec2, b: float) -> "Line2":
        """
        Fast constructor when (w,b) already satisfy invariants (w is unit).
        Avoids __post_init__ normalization cost.
        """
        obj = object.__new__(cls)
        obj.w = w
        obj.b = float(b)
        return obj

    @classmethod
    def from_points(cls, start: Point, end: Point) -> "Line2":
        a, b, c = coefficients_from_points(start, end)
        return cls((a, b), c)

    def translated(self, v: Vec2) -> "Line2":
        """
        Translate line by vector v.

        For unit normal w:
          w·(x - v) + b = 0  <=>  w·x + (b - w·v) = 0

        Translation does not change w, so no renormalization.
        """
        wx, wy = self.w
        d = wx * v[0] + wy * v[1]
        return Line2._from_normalized(self.w, self.b - d)

    def at_time(self, now: float) -> "Line2":
        """
        The wavefront evolution in this code uses translating the line along +w by 'now':
          translated(w * now)

        With unit w, w·(w*now) = now, so:
          b' = b - now
        """
        if now == 0.0:
            return self  # safe: immutable-by-convention; avoids alloc
        return Line2._from_normalized(self.w, self.b - now)

    def perpendicular(self, through: Point) -> "Line2":
        a, b, c = coefficients_perpendicular_through_point(self.w[0], self.w[1], through[0], through[1])
        return Line2((a, b), c)

    def bisector(self, other: "Line2") -> "Line2":
        a, b, c = coefficients_bisector_of_lines(self.w[0], self.w[1], self.b, other.w[0], other.w[1], other.b)
        return Line2((a, b), c)

    def signed_distance(self, pt: Point) -> float:
        wx, wy = self.w
        return wx * pt[0] + wy * pt[1] + self.b

    @property
    def through(self) -> Point:
        # point closest to origin: x = -b * w
        wx, wy = self.w
        return (-self.b * wx, -self.b * wy)


@dataclass(slots=True)
class WaveFront:
    start: Point
    end: Point
    line: Line2

    def __init__(self, start: Point, end: Point, line: Optional[Line2] = None) -> None:
        if line is None:
            line = Line2.from_points(start, end)
        self.line = line
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))


class LineLineIntersectionResult:
    NO_INTERSECTION = 0
    POINT = 1
    LINE = 2


@dataclass(slots=True)
class LineLineIntersector:
    one: Line2
    other: Line2
    result: object = None

    def intersection_type(self) -> int:
        (a1, b1), c1 = self.one.w, self.one.b
        (a2, b2), c2 = self.other.w, self.other.b
        denom = a1 * b2 - a2 * b1
        if near_zero(denom):
            x1 = a1 * c2 - a2 * c1
            x2 = b1 * c2 - b2 * c1
            if near_zero(x1) and near_zero(x2):
                self.result = self.one
                return LineLineIntersectionResult.LINE
            self.result = None
            return LineLineIntersectionResult.NO_INTERSECTION

        num1 = b1 * c2 - b2 * c1
        num2 = a2 * c1 - a1 * c2
        xw = num1 / denom
        yw = num2 / denom
        self.result = (xw, yw)
        return LineLineIntersectionResult.POINT


@dataclass(slots=True)
class WaveFrontIntersector:
    left: WaveFront
    right: WaveFront

    def get_bisector(self) -> Vec2:
        """
        Compute bisector direction (velocity) between the two wavefront support lines.

        Optimized version:
        - assumes Line2 is always normalized (w is unit)
        - avoids constructing translated Line2 objects
        - avoids extra LineLineIntersector allocations in the hot path
        """
        l1 = self.left.line
        l2 = self.right.line
        (a1, b1), c1 = l1.w, l1.b
        (a2, b2), c2 = l2.w, l2.b

        denom = a1 * b2 - a2 * b1

        # Parallel or coincident
        if near_zero(denom):
            # Check if they are the same line (coincident).
            # With normalized normals, equality up to tolerance can be tested by
            # the 2x2 minors (a1*c2 - a2*c1, b1*c2 - b2*c1).
            x1 = a1 * c2 - a2 * c1
            x2 = b1 * c2 - b2 * c1
            if near_zero(x1) and near_zero(x2):
                # Same line: legacy behavior averaged normals
                return (0.5 * (a1 + a2), 0.5 * (b1 + b2))

            # Distinct parallel lines: legacy behavior summed normals
            return (a1 + a2, b1 + b2)

        # Proper intersection at t=0: solve
        #   a1*x + b1*y + c1 = 0
        #   a2*x + b2*y + c2 = 0
        x0 = (b1 * c2 - b2 * c1) / denom
        y0 = (a2 * c1 - a1 * c2) / denom

        # Intersection after translating each line by +w (i.e. at "time" 1).
        # For unit normals, at_time(1) is just c' = c - 1.
        c1p = c1 - 1.0
        c2p = c2 - 1.0
        x1p = (b1 * c2p - b2 * c1p) / denom
        y1p = (a2 * c1p - a1 * c2p) / denom

        # Direction from intersection at t=0 to intersection at t=1
        return (x1p - x0, y1p - y0)
    
    def get_intersection_at_t(self, t: float) -> Point:
        """
        Intersection of left/right wavefront support lines at time t.

        Optimized:
        - avoids constructing temporary Line2 objects via line.at_time(t)
        - directly uses coefficient update b' = b - t (valid for normalized Line2)
        """
        l1 = self.left.line
        l2 = self.right.line

        a1, b1 = l1.w
        a2, b2 = l2.w
        c1 = l1.b - t
        c2 = l2.b - t

        denom = a1 * b2 - a2 * b1
        if near_zero(denom):
            raise ValueError("parallel lines, can not compute point of intersection")

        x = (b1 * c2 - b2 * c1) / denom
        y = (a2 * c1 - a1 * c2) / denom
        return (x, y)
