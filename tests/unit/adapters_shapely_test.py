from pytest import importorskip

from grassfire2.triangulation import from_shapely_constrained_delaunay


def test_from_shapely_constrained_delaunay_smoke():
    shapely = importorskip("shapely")
    Polygon = importorskip("shapely.geometry").Polygon

    geom = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0), (0.0, 0.0)])
    mesh = from_shapely_constrained_delaunay(geom)

    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    assert all(v.is_finite for v in mesh.vertices)
    assert all(t.is_internal for t in mesh.triangles)
    assert any(any(c is not None for c in t.c) for t in mesh.triangles)
