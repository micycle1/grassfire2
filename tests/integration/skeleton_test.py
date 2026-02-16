from pytest import mark, importorskip
from tri.delaunay.helpers import ToPointsAndSegments
from grassfire2 import compute_skeleton
import csv
from pathlib import Path

def test_internal_segments_count():
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

def read_csv_polygon(path):
    rings = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            coords = [float(x) for x in row if x.strip()]
            points = []
            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i+1]))
            if points and points[0] != points[-1]:
                points.append(points[0])
            if points:
                rings.append(points)
    return rings

def segments_intersect(p1, p2, p3, p4):
    def on_segment(p, a, b):
        return (p[0] <= max(a[0], b[0]) and p[0] >= min(a[0], b[0]) and
                p[1] <= max(a[1], b[1]) and p[1] >= min(a[1], b[1]))

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0
        return 1 if val > 0 else 2

    def point_equal(a, b, tol=1e-9):
        return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol

    if (point_equal(p1, p3) or point_equal(p1, p4) or 
        point_equal(p2, p3) or point_equal(p2, p4)):
        return False

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    return False

csv_dir = Path(__file__).parent / "csv"
csv_files = sorted(list(csv_dir.glob("*.csv")))

@mark.parametrize("csv_file", csv_files, ids=[f.name for f in csv_files])
def test_skeleton_integrity(csv_file):
    importorskip("tri")
    print(f"Testing {csv_file.name}")
    rings = read_csv_polygon(csv_file)
    if not rings:
        return

    conv = ToPointsAndSegments()
    all_input_vertices = set()
    
    for ring in rings:
        for p in ring:
            all_input_vertices.add(p)
    
    conv.add_polygon(rings)

    sk = compute_skeleton(conv, internal_only=True, shrink=False)
    segments = sk.segments()

    skel_endpoints = set()
    skeleton_point_segments = []
    
    for s in segments:
        if len(s) == 2 and isinstance(s[0][0], (tuple, list)) and isinstance(s[0][1], (tuple, list)):
            points_pair = s[0]
        else:
            points_pair = s

        p1, p2 = points_pair
        
        skel_endpoints.add(p1)
        skel_endpoints.add(p2)
        skeleton_point_segments.append(points_pair)

    def find_point_in_set(pt, pt_set):
        for s_pt in pt_set:
            if pt[0]==s_pt[0] and pt[1]==s_pt[1]:
                return True
        return False
        
    for v in all_input_vertices:
        assert find_point_in_set(v, skel_endpoints), f"Input vertex {v} not found in skeleton endpoints for {csv_file.name}."

    for i in range(len(skeleton_point_segments)):
        for j in range(i + 1, len(skeleton_point_segments)):
            s1 = skeleton_point_segments[i]
            s2 = skeleton_point_segments[j]
            if segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                    assert False, f"Skeleton segments intersect in {csv_file.name}: {s1} and {s2}"