from __future__ import annotations

import logging

from tri.delaunay.iter import RegionatedTriangleIterator, StarEdgeIterator, Edge
from tri.delaunay.tds import cw, ccw, orient2d

from .model import Skeleton, SkeletonNode, InfiniteVertex, KineticTriangle, KineticVertex
from .line import WaveFront, WaveFrontIntersector

logger = logging.getLogger(__name__)


def rotate_until_not_in_candidates(t, v, direction, candidates):
    seen = set()
    while t is not None and t not in seen:
        seen.add(t)
        side = t.vertices.index(v)
        t = t.neighbours[direction(side)]
        if t not in candidates:
            return t
    return None


def make_support_line(edge: Edge):
    if edge.constrained:
        start = (edge.segment[0].x, edge.segment[0].y)
        end = (edge.segment[1].x, edge.segment[1].y)
        return WaveFront(start, end)
    return None


def split_star(v):
    around = [e for e in StarEdgeIterator(v)]
    groups = []
    group = []
    for edge in around:
        t, s = edge.triangle, edge.side
        group.append(edge)
        if Edge(t, ccw(s)).constrained:
            groups.append(group)
            group = []
    if group:
        groups.append(group)
    if len(groups) <= 1:
        return groups
    edge = groups[0][0]
    if not edge.triangle.constrained[cw(edge.side)]:
        last = groups.pop()
        last.extend(groups[0])
        groups[0] = last

    for group in groups:
        first, last = group[0], group[-1]
        assert first.triangle.constrained[cw(first.side)]
        assert last.triangle.constrained[ccw(last.side)]
        for middle in group[1:-1]:
            assert not middle.triangle.constrained[cw(middle.side)]
            assert not middle.triangle.constrained[ccw(middle.side)]
    return groups


def init_skeleton(dt) -> Skeleton:
    skel = Skeleton()
    nodes = {}

    avg_x = 0.0
    avg_y = 0.0
    for v in dt.vertices:
        if v.is_finite:
            nodes[v] = SkeletonNode(pos=(v.x, v.y), step=-1, info=v.info)
            avg_x += v.x / len(dt.vertices)
            avg_y += v.y / len(dt.vertices)

    centroid = InfiniteVertex(origin=(avg_x, avg_y))

    ktriangles: list[KineticTriangle] = []
    internal_triangles = set()
    for _, depth, triangle in RegionatedTriangleIterator(dt):
        if depth == 1:
            internal_triangles.add(triangle)

    triangle2ktriangle = {}
    for idx, t in enumerate(dt.triangles, start=1):
        k = KineticTriangle()
        k.info = idx
        k.uid = idx
        triangle2ktriangle[t] = k
        ktriangles.append(k)
        k.internal = t in internal_triangles

    link_around = []
    unwanted = []
    for t in dt.triangles:
        k = triangle2ktriangle[t]
        for i in range(3):
            edge = Edge(t, i)
            k.wavefront_support_lines[i] = make_support_line(edge)

        for j, n in enumerate(t.neighbours):
            if t.constrained[j]:
                continue
            if n is None or n.vertices[2] is None:
                unwanted.append(k)
                continue
            k.neighbours[j] = triangle2ktriangle[n]

    kvertices: list[KineticVertex] = []
    ct = 0
    for v in dt.vertices:
        assert v.is_finite, "infinite vertex found"

        groups = split_star(v)
        if len(groups) == 1:
            raise NotImplementedError("not yet dealing with PSLG in initial conversion")

        for group in groups:
            first, last = group[0], group[-1]
            tail, mid1 = Edge(last.triangle, ccw(last.side)).segment
            mid2, head = Edge(first.triangle, cw(first.side)).segment
            assert mid1 is mid2
            turn = orient2d((tail.x, tail.y), (mid1.x, mid1.y), (head.x, head.y))
            if turn < 0:
                turn_type = "RIGHT - REFLEX"
            elif turn > 0:
                turn_type = "LEFT - CONVEX"
            else:
                turn_type = "STRAIGHT"

            right = triangle2ktriangle[first.triangle].wavefront_support_lines[cw(first.side)]
            left = triangle2ktriangle[last.triangle].wavefront_support_lines[ccw(last.side)]
            assert left is not None and right is not None

            bi = WaveFrontIntersector(left, right).get_bisector()

            ur = right.line
            ul = left.line

            ct += 1
            kv = KineticVertex()
            kv.turn = turn_type
            kv.info = ct
            kv.origin = (v.x, v.y)
            kv.velocity = bi
            kv.start_node = nodes[v]
            kv.starts_at = 0.0
            kv.ul = ul
            kv.ur = ur
            kv.wfl = left
            kv.wfr = right

            for edge in group:
                ktriangle = triangle2ktriangle[edge.triangle]
                ktriangle.vertices[edge.side] = kv
                kv.internal = ktriangle.internal

            kvertices.append(kv)
            link_around.append(((last.triangle, cw(last.side)), kv, (first.triangle, ccw(first.side))))

    for left, kv, right in link_around:
        cwv = triangle2ktriangle[left[0]].vertices[left[1]]
        ccwv = triangle2ktriangle[right[0]].vertices[right[1]]
        assert isinstance(cwv, KineticVertex) and isinstance(ccwv, KineticVertex)
        kv.left = (cwv, 0.0)
        kv.right = (ccwv, 0.0)

    for _, kv, _ in link_around:
        assert kv.left is None or kv.left.wfr is kv.wfl
        assert kv.right is None or kv.wfr is kv.right.wfl
        assert kv.is_stopped is False

    infinites = {}
    for t in triangle2ktriangle:
        for v in t.vertices:
            if v is not None and not v.is_finite:
                infv = InfiniteVertex(origin=(v[0], v[1]))
                infinites[(v[0], v[1])] = infv
    assert len(infinites) == 3

    for (t, kt) in triangle2ktriangle.items():
        for i, v in enumerate(t.vertices):
            if v is not None and not v.is_finite:
                kt.vertices[i] = infinites[(v[0], v[1])]

    remove = []
    for kt in ktriangles:
        if [isinstance(v, InfiniteVertex) for v in kt.vertices].count(True) == 2:
            remove.append(kt)

    assert len(remove) == 3
    assert len(unwanted) == 3
    assert remove == unwanted

    link = []
    for kt in unwanted:
        v = kt.vertices[kt.neighbours.index(None)]
        assert isinstance(v, KineticVertex)

        neighbour_cw = rotate_until_not_in_candidates(kt, v, cw, unwanted)
        neighbour_ccw = rotate_until_not_in_candidates(kt, v, ccw, unwanted)
        side_cw = ccw(neighbour_cw.vertices.index(v))
        side_ccw = cw(neighbour_ccw.vertices.index(v))
        link.append((neighbour_cw, side_cw, neighbour_ccw))
        link.append((neighbour_ccw, side_ccw, neighbour_cw))

    for ngb, side, new_ngb in link:
        ngb.neighbours[side] = new_ngb

    for kt in unwanted:
        kt.vertices = [None, None, None]
        kt.neighbours = [None, None, None]
        ktriangles.remove(kt)

    for kt in ktriangles:
        for i, v in enumerate(kt.vertices):
            if isinstance(v, InfiniteVertex):
                kt.vertices[i] = centroid

    # stable sort similar to legacy (not required for correctness, but keeps reproducibility)
    ktriangles.sort(key=lambda t: (t.vertices[0].origin[1], t.vertices[0].origin[0]))  # type: ignore[union-attr]

    skel.sk_nodes = list(nodes.values())
    skel.triangles = ktriangles
    skel.vertices = kvertices
    return skel


def internal_only_skeleton(skel: Skeleton) -> Skeleton:
    new = Skeleton()
    new.sk_nodes = skel.sk_nodes[:]
    new.triangles = [t for t in skel.triangles if t.internal]
    new.vertices = [v for v in skel.vertices if v.internal]
    new.transform = skel.transform
    return new