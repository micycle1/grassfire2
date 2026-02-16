from __future__ import annotations

import logging
from dataclasses import dataclass
from math import sqrt
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
    return float(a), float(b), float(c)


def coefficients_perpendicular_through_point(la: float, lb: float, px: float, py: float) -> tuple[float, float, float]:
    a = -lb
    b = la
    c = lb * px - la * py
    return float(a), float(b), float(c)


def coefficients_bisector_of_lines(
    pa: float, pb: float, pc: float,
    qa: float, qb: float, qc: float
) -> tuple[float, float, float]:
    n1 = sqrt(pa * pa + pb * pb)
    n2 = sqrt(qa * qa + qb * qb)
    a = n2 * pa + n1 * qa
    b = n2 * pb + n1 * qb
    c = n2 * pc + n1 * qc
    if a == 0 and b == 0:
        a = n2 * pa - n1 * qa
        b = n2 * pb - n1 * qb
        c = n2 * pc - n1 * qc
    return float(a), float(b), float(c)


@dataclass(slots=True)
class Line2:
    w: Vec2
    b: float
    normalize: bool = True

    def __post_init__(self) -> None:
        self.w = (float(self.w[0]), float(self.w[1]))
        self.b = float(self.b)
        if self.normalize:
            self._normalize()

    def _normalize(self) -> None:
        nrm = norm(self.w)
        self.b /= nrm
        self.w = unit(self.w)

    def translated(self, v: Vec2) -> "Line2":
        d = dot(self.w, v)
        return Line2(self.w, self.b - d, normalize=(d != 0.0))

    def perpendicular(self, through: Point) -> "Line2":
        a, b, c = coefficients_perpendicular_through_point(self.w[0], self.w[1], through[0], through[1])
        return Line2((a, b), c)

    def bisector(self, other: "Line2") -> "Line2":
        a, b, c = coefficients_bisector_of_lines(self.w[0], self.w[1], self.b, other.w[0], other.w[1], other.b)
        return Line2((a, b), c)

    def signed_distance(self, pt: Point) -> float:
        return dot(self.w, pt) + self.b

    @property
    def through(self) -> Point:
        t = mul(self.w, -self.b)
        return (float(t[0]), float(t[1]))

    @classmethod
    def from_points(cls, start: Point, end: Point) -> "Line2":
        coeff = coefficients_from_points(start, end)
        ln = cls((coeff[0], coeff[1]), coeff[2], normalize=True)
        # legacy asserts: end != start
        return ln

    def at_time(self, now: float) -> "Line2":
        if now == 0.0:
            return Line2(self.w, self.b, normalize=False)
        logger.debug(" (constructing new line at t= %s)", now)
        return self.translated(mul(self.w, now))

    def ends(self) -> tuple[Point, Point]:
        ccw = rotate90ccw(self.w)
        cw = rotate90cw(self.w)
        end = add(mul(cw, 1000.0), self.through)
        start = add(mul(ccw, 1000.0), self.through)
        return (start, end)


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
        intersector = LineLineIntersector(self.left.line, self.right.line)
        res = intersector.intersection_type()
        if res == LineLineIntersectionResult.LINE:
            bi = add(mul(self.left.line.w, 0.5), mul(self.right.line.w, 0.5))
        elif res == LineLineIntersectionResult.POINT:
            left_translated = self.left.line.translated(self.left.line.w)
            right_translated = self.right.line.translated(self.right.line.w)
            intersector_inner = LineLineIntersector(left_translated, right_translated)
            inner_res = intersector_inner.intersection_type()
            assert inner_res == LineLineIntersectionResult.POINT
            bi = make_vector(end=intersector_inner.result, start=intersector.result)
        elif res == LineLineIntersectionResult.NO_INTERSECTION:
            bi = add(self.left.line.w, self.right.line.w)
        else:
            raise RuntimeError(f"Unknown intersection type: {res}")
        logger.debug("magnitude of bisector: %s", norm(bi))
        return (float(bi[0]), float(bi[1]))

    def get_intersection_at_t(self, t: float) -> Point:
        intersector = LineLineIntersector(self.left.line.at_time(t), self.right.line.at_time(t))
        if intersector.intersection_type() == LineLineIntersectionResult.POINT:
            return intersector.result  # type: ignore[return-value]
        raise ValueError("parallel lines, can not compute point of intersection")