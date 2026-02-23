from __future__ import annotations

import logging

from predicates import orient2d

from .line import WaveFront, WaveFrontIntersector
from .mesh import InputMesh
from .model import InfiniteVertex, KineticTriangle, KineticVertex, Skeleton, SkeletonNode
from .topology import ccw, cw

logger = logging.getLogger(__name__)

Corner = tuple[int, int]
StarGroups = list[list[list[Corner]]]


def rotate_until_not_in_candidates(t, v, direction, candidates):
    seen = set()
    while t is not None and t not in seen:
        seen.add(t)
        side = t.vertices.index(v)
        t = t.neighbours[direction(side)]
        if t not in candidates:
            return t
    return None


def _step_around_vertex(mesh: InputMesh, v_idx: int, corner: Corner, direction) -> Corner | None:
    t_idx, side = corner
    tri = mesh.triangles[t_idx]
    edge_side = ccw(side) if direction is ccw else cw(side)
    if tri.c[edge_side] is not None:
        return None
    n_idx = tri.n[edge_side]
    if n_idx < 0:
        return None
    try:
        n_side = mesh.triangles[n_idx].v.index(v_idx)
    except ValueError:
        return None
    return (n_idx, n_side)


def build_vertex_stars(mesh: InputMesh) -> StarGroups:
    stars: list[list[Corner]] = [[] for _ in mesh.vertices]
    for t_idx, tri in enumerate(mesh.triangles):
        for side, v_idx in enumerate(tri.v):
            stars[v_idx].append((t_idx, side))

    grouped: StarGroups = [[] for _ in mesh.vertices]
    for v_idx, incident in enumerate(stars):
        unvisited = set(incident)
        while unvisited:
            seed = next(iter(unvisited))
            start = seed
            seen = {seed}
            while True:
                prev_corner = _step_around_vertex(mesh, v_idx, start, ccw)
                if prev_corner is None or prev_corner == seed or prev_corner in seen:
                    break
                seen.add(prev_corner)
                start = prev_corner

            group: list[Corner] = []
            walked: set[Corner] = set()
            cur = start
            while cur is not None and cur not in walked:
                walked.add(cur)
                group.append(cur)
                unvisited.discard(cur)
                next_corner = _step_around_vertex(mesh, v_idx, cur, cw)
                if next_corner == start:
                    break
                cur = next_corner
            if group:
                group.reverse()
                grouped[v_idx].append(group)
    return grouped


def init_skeleton(mesh: InputMesh) -> Skeleton:
    skel = Skeleton()
    nodes: dict[int, SkeletonNode] = {}
    infinite_by_idx: dict[int, InfiniteVertex] = {}

    sum_x = 0.0
    sum_y = 0.0
    for idx, v in enumerate(mesh.vertices):
        if v.is_finite:
            nodes[idx] = SkeletonNode(pos=(v.x, v.y), step=-1, info=v.info)
            sum_x += v.x
            sum_y += v.y
        else:
            infinite_by_idx[idx] = InfiniteVertex(origin=(v.x, v.y))

    n = len(mesh.vertices)
    avg_x = sum_x / n if n else 0.0
    avg_y = sum_y / n if n else 0.0
    centroid = InfiniteVertex(origin=(avg_x, avg_y))

    ktriangles: list[KineticTriangle] = []
    for idx, t in enumerate(mesh.triangles, start=1):
        k = KineticTriangle()
        k.info = idx
        k.uid = idx
        k.internal = t.is_internal
        ktriangles.append(k)

    unwanted: list[KineticTriangle] = []
    for t_idx, t in enumerate(mesh.triangles):
        k = ktriangles[t_idx]

        for i, v_idx in enumerate(t.v):
            if not mesh.vertices[v_idx].is_finite:
                k.vertices[i] = infinite_by_idx[v_idx]

        for i in range(3):
            constraint = t.c[i]
            if constraint is None:
                continue
            start = mesh.vertices[t.v[ccw(i)]]
            end = mesh.vertices[t.v[cw(i)]]
            k.wavefront_support_lines[i] = WaveFront(
                (start.x, start.y),
                (end.x, end.y),
                data=constraint,
                weight=constraint.weight if constraint is not None else 1.0,
            )

        for j, n_idx in enumerate(t.n):
            if t.c[j] is not None:
                continue
            if n_idx == -1:
                unwanted.append(k)
                continue
            k.neighbours[j] = ktriangles[n_idx]

    stars = build_vertex_stars(mesh)
    kvertices: list[KineticVertex] = []
    link_around = []
    ct = 0

    for v_idx, groups in enumerate(stars):
        v = mesh.vertices[v_idx]
        if not v.is_finite:
            continue

        if not groups:
            continue

        for group in groups:
            first_t_idx, first_side = group[0]
            last_t_idx, last_side = group[-1]

            last_tri = mesh.triangles[last_t_idx]
            first_tri = mesh.triangles[first_t_idx]

            tail = mesh.vertices[last_tri.v[cw(last_side)]]
            mid1 = mesh.vertices[last_tri.v[last_side]]
            mid2 = mesh.vertices[first_tri.v[first_side]]
            head = mesh.vertices[first_tri.v[ccw(first_side)]]
            assert mid1 is mid2

            turn = orient2d((tail.x, tail.y), (mid1.x, mid1.y), (head.x, head.y))
            if turn < 0:
                turn_type = "RIGHT - REFLEX"
            elif turn > 0:
                turn_type = "LEFT - CONVEX"
            else:
                turn_type = "STRAIGHT"

            right = ktriangles[first_t_idx].wavefront_support_lines[cw(first_side)]
            left = ktriangles[last_t_idx].wavefront_support_lines[ccw(last_side)]
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
            kv.start_node = nodes[v_idx]
            kv.starts_at = 0.0
            kv.ul = ul
            kv.ur = ur
            kv.wfl = left
            kv.wfr = right

            for t_idx, side in group:
                ktriangle = ktriangles[t_idx]
                ktriangle.vertices[side] = kv
                kv.internal = ktriangle.internal

            kvertices.append(kv)
            link_around.append(((last_t_idx, cw(last_side)), kv, (first_t_idx, ccw(first_side))))

    for left, kv, right in link_around:
        cwv = ktriangles[left[0]].vertices[left[1]]
        ccwv = ktriangles[right[0]].vertices[right[1]]
        assert isinstance(cwv, KineticVertex) and isinstance(ccwv, KineticVertex)
        kv.left = (cwv, 0.0)
        kv.right = (ccwv, 0.0)

    for _, kv, _ in link_around:
        assert kv.left is None or kv.left.wfr is kv.wfl
        assert kv.right is None or kv.wfr is kv.right.wfl
        assert kv.is_stopped is False

    for kt in ktriangles:
        for i, vv in enumerate(kt.vertices):
            if isinstance(vv, InfiniteVertex):
                kt.vertices[i] = centroid

    remove = []
    for t_idx, tri in enumerate(mesh.triangles):
        count_inf = sum(1 for v_idx in tri.v if not mesh.vertices[v_idx].is_finite)
        if count_inf == 2:
            remove.append(ktriangles[t_idx])

    if remove or unwanted:
        assert len(remove) == 3
        assert len(unwanted) == 3
        assert remove == unwanted

        link = []
        for kt in unwanted:
            v = kt.vertices[kt.neighbours.index(None)]
            assert isinstance(v, KineticVertex)

            neighbour_cw = rotate_until_not_in_candidates(kt, v, cw, unwanted)
            neighbour_ccw = rotate_until_not_in_candidates(kt, v, ccw, unwanted)
            assert neighbour_cw is not None and neighbour_ccw is not None
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

    # stable sort similar to legacy (not required for correctness, but keeps reproducibility)
    ktriangles.sort(key=lambda t: (t.vertices[0].origin[1], t.vertices[0].origin[0]))

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
