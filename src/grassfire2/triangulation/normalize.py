from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import TypeAlias

from shapely import from_wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

Point: TypeAlias = tuple[float, float]
Ring: TypeAlias = Sequence[Sequence[Real]]
RingsInput: TypeAlias = Sequence[Ring] | Sequence[Sequence[Ring]]
GeometryInput: TypeAlias = BaseGeometry


def is_geometry(obj: object) -> bool:
    return hasattr(obj, "geom_type") and hasattr(obj, "is_empty")


def _is_point(obj: object) -> bool:
    return (
        isinstance(obj, Sequence)
        and len(obj) == 2
        and isinstance(obj[0], Real)
        and isinstance(obj[1], Real)
    )


def _to_point_list(ring: Sequence[Sequence[Real]]) -> list[Point]:
    return [(float(p[0]), float(p[1])) for p in ring]


def _polygon_from_rings(rings: Sequence[Ring]) -> Polygon:
    if not rings:
        raise TypeError("RingsInput polygon must contain at least one ring.")
    exterior = _to_point_list(rings[0])
    holes = [_to_point_list(r) for r in rings[1:]]
    return Polygon(exterior, holes=holes)


def normalize_to_geometry(geom_input: object) -> BaseGeometry:
    if is_geometry(geom_input):
        return geom_input  # type: ignore[return-value]

    if isinstance(geom_input, str):
        return from_wkt(geom_input)

    if not isinstance(geom_input, Sequence) or not geom_input:
        raise TypeError("Expected Geometry | WKT string | RingsInput.")

    first = geom_input[0]
    if _is_point(first):
        return Polygon(_to_point_list(geom_input))  # single ring

    if isinstance(first, Sequence) and first and _is_point(first[0]):
        return _polygon_from_rings(geom_input)  # polygon with holes

    if (
        isinstance(first, Sequence)
        and first
        and isinstance(first[0], Sequence)
        and first[0]
        and _is_point(first[0][0])
    ):
        polys = [_polygon_from_rings(polygon_rings) for polygon_rings in geom_input]
        return MultiPolygon(polys)

    raise TypeError("Expected Geometry | WKT string | RingsInput.")
