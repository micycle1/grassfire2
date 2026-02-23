from pytest import raises

from grassfire2 import compute_skeleton


def test_compute_skeleton_accepts_rings_and_wkt():
    rings = [[
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (10.0, 10.0),
        (10.0, 20.0),
        (0.0, 20.0),
        (0.0, 0.0),
    ]]
    sk_rings = compute_skeleton(rings, internal_only=True)
    assert len(sk_rings.segments()) == 8

    wkt = "POLYGON ((0 0, 20 0, 20 10, 10 10, 10 20, 0 20, 0 0))"
    sk_wkt = compute_skeleton(wkt, internal_only=True)
    assert len(sk_wkt.segments()) == 8


def test_compute_skeleton_rejects_legacy_conv():
    class LegacyConv:
        points = [(0.0, 0.0)]
        infos = []
        segments = []

    with raises(TypeError):
        compute_skeleton(LegacyConv())
