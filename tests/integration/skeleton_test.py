from pytest import mark, importorskip
from tri.delaunay.helpers import ToPointsAndSegments
from grassfire2 import compute_skeleton

@mark.integration
def test_internal_segments_count():
    # skip if tri is not available in test environment
    importorskip("tri")

    conv = ToPointsAndSegments()
    conv.add_polygon(
        [[
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 10.0],
            [10.0, 10.0],
            [10.0, 20.0],
            [0.0, 20.0],
            [0.0, 0.0],
        ]]
    )

    sk = compute_skeleton(conv, internal_only=True)
    segments = sk.segments()
    assert len(segments) == 8