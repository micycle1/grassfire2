from __future__ import annotations

from collections import defaultdict

import shapely

from ..mesh import InputMesh, InputTriangle, InputVertex


def triangulate_with_shapely(geom):
    return shapely.constrained_delaunay_triangles(geom)


def geometry_from_points_segments(points, segments):
    lines = []
    for a, b in segments:
        lines.append(shapely.LineString([points[a], points[b]]))
    if not lines:
        return shapely.GeometryCollection()
    # Build area from planar linework; this preserves holes correctly.
    return shapely.build_area(shapely.MultiLineString(lines))


def from_shapely_constrained_delaunay(geom) -> InputMesh:
    tri_geom = triangulate_with_shapely(geom)

    if tri_geom.is_empty:
        return InputMesh(vertices=[], triangles=[])

    def _point_key(xy) -> tuple[float, float]:
        return (float(xy[0]), float(xy[1]))

    def _edge_key(a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
        return (a, b) if a <= b else (b, a)

    def _polygon_parts(g):
        for p in shapely.get_parts(g):
            if p.geom_type == "Polygon":
                yield p

    def _ring_edges(coords):
        pts = [_point_key(c) for c in coords]
        for i in range(len(pts) - 1):
            yield _edge_key(pts[i], pts[i + 1])

    constrained_edges = set()
    for poly in _polygon_parts(geom):
        constrained_edges.update(_ring_edges(poly.exterior.coords))
        for interior in poly.interiors:
            constrained_edges.update(_ring_edges(interior.coords))

    vertex_index: dict[tuple[float, float], int] = {}
    vertices: list[InputVertex] = []
    tri_vertices: list[tuple[int, int, int]] = []

    for poly in _polygon_parts(tri_geom):
        coords = [_point_key(c) for c in poly.exterior.coords[:-1]]
        if len(coords) != 3:
            raise ValueError("Shapely constrained_delaunay_triangles produced a non-triangle polygon.")
        # Match legacy orientation expected by the kinetic init logic.
        coords = (coords[0], coords[2], coords[1])

        v_idx = []
        for xy in coords:
            if xy not in vertex_index:
                vertex_index[xy] = len(vertices)
                vertices.append(InputVertex(x=xy[0], y=xy[1], is_finite=True, info=None))
            v_idx.append(vertex_index[xy])
        tri_vertices.append((v_idx[0], v_idx[1], v_idx[2]))

    edge_to_sides = defaultdict(list)
    for t_idx, tv in enumerate(tri_vertices):
        for side in range(3):
            a = vertices[tv[(side + 1) % 3]]
            b = vertices[tv[(side - 1) % 3]]
            edge_to_sides[_edge_key((a.x, a.y), (b.x, b.y))].append((t_idx, side))

    triangles: list[InputTriangle] = []
    tri_n = [[-1, -1, -1] for _ in tri_vertices]

    for sides in edge_to_sides.values():
        if len(sides) == 2:
            (t0, s0), (t1, s1) = sides
            tri_n[t0][s0] = t1
            tri_n[t1][s1] = t0

    for t_idx, tv in enumerate(tri_vertices):
        tc = [None, None, None]
        for side in range(3):
            a = vertices[tv[(side + 1) % 3]]
            b = vertices[tv[(side - 1) % 3]]
            if _edge_key((a.x, a.y), (b.x, b.y)) in constrained_edges:
                tc[side] = True

        triangles.append(
            InputTriangle(
                v=tv,
                n=(tri_n[t_idx][0], tri_n[t_idx][1], tri_n[t_idx][2]),
                c=(tc[0], tc[1], tc[2]),
                is_internal=True,
            )
        )

    return InputMesh(vertices=vertices, triangles=triangles)
