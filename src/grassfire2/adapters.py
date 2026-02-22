from __future__ import annotations

from typing import Any

from tri.delaunay.insert_kd import triangulate
from tri.delaunay.iter import RegionatedTriangleIterator

from .mesh import InputMesh, InputTriangle, InputVertex


def triangulate_with_tri(points, infos, segments):
    return triangulate(points, infos, segments, False)


def _vertex_xy(v) -> tuple[float, float]:
    if hasattr(v, "x") and hasattr(v, "y"):
        return float(v.x), float(v.y)
    return float(v[0]), float(v[1])


def from_tri_delaunay(dt) -> InputMesh:
    ordered_vertices = list(dt.vertices)
    seen = set(ordered_vertices)
    for t in dt.triangles:
        for v in t.vertices:
            if v is not None and v not in seen:
                seen.add(v)
                ordered_vertices.append(v)

    vertex_index = {v: idx for idx, v in enumerate(ordered_vertices)}
    vertices = []
    for v in ordered_vertices:
        x, y = _vertex_xy(v)
        vertices.append(
            InputVertex(
                x=x,
                y=y,
                is_finite=bool(getattr(v, "is_finite", True)),
                info=getattr(v, "info", None),
            )
        )

    tri_index = {t: idx for idx, t in enumerate(dt.triangles)}
    internal = set()
    for _, depth, triangle in RegionatedTriangleIterator(dt):
        if depth == 1:
            internal.add(triangle)

    triangles = []
    for t in dt.triangles:
        tv = []
        for v in t.vertices:
            if v is None:
                raise ValueError("Unexpected None vertex in input triangulation.")
            tv.append(vertex_index[v])

        tn = []
        for n in t.neighbours:
            if n is None:
                tn.append(-1)
            else:
                tn.append(tri_index.get(n, -1))

        tc = []
        for constrained in t.constrained:
            tc.append(constrained if constrained else None)

        triangles.append(
            InputTriangle(
                v=(tv[0], tv[1], tv[2]),
                n=(tn[0], tn[1], tn[2]),
                c=(tc[0], tc[1], tc[2]),
                is_internal=t in internal,
            )
        )

    return InputMesh(vertices=vertices, triangles=triangles)
