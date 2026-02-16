from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]
Vec2 = tuple[float, float]


@dataclass(slots=True)
class Transform:
    scale: Vec2
    translate: Vec2

    def forward(self, pt: Point) -> Point:
        return (
            (pt[0] - self.translate[0]) / self.scale[0],
            (pt[1] - self.translate[1]) / self.scale[1],
        )

    def backward(self, pt: Point) -> Point:
        return (
            (pt[0] * self.scale[0]) + self.translate[0],
            (pt[1] * self.scale[1]) + self.translate[1],
        )


def get_transform(box: tuple[Point, Point]) -> Transform:
    tdx, tdy = (2.0, 2.0)
    (sxmin, symin), (sxmax, symax) = box
    scx, scy = (sxmin + sxmax) * 0.5, (symin + symax) * 0.5
    sdx, sdy = (sxmax - sxmin), (symax - symin)
    scale = max(sdx / tdx, sdy / tdy)
    return Transform((scale, scale), (scx, scy))


def get_box(pts: list[Point] | tuple[Point, ...]) -> tuple[Point, Point]:
    assert len(pts)
    it = iter(pts)
    ll = list(next(it))
    ur = list(ll[:])
    for pt in it:
        if pt[0] < ll[0]:
            ll[0] = pt[0]
        if pt[1] < ll[1]:
            ll[1] = pt[1]
        if pt[0] > ur[0]:
            ur[0] = pt[0]
        if pt[1] > ur[1]:
            ur[1] = pt[1]
    return (ll[0], ll[1]), (ur[0], ur[1])