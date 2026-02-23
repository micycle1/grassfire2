from __future__ import annotations

from typing import Literal, Optional

from shapely.ops import transform as geom_transform

from .events.loop import DebugHook, event_loop, init_event_list
from .init import init_skeleton, internal_only_skeleton
from .model import Skeleton
from .transform import get_box, get_transform
from .triangulation import (
    GeometryInput,
    RingsInput,
    from_shapely_constrained_delaunay,
    from_tri_delaunay,
    normalize_to_geometry,
    points_segments_infos_from_geometry,
    triangulate_with_tri,
)

AdapterName = Literal["shapely", "tri"]


def _run_skeleton(
    mesh,
    *,
    transform,
    shrink: bool,
    internal_only: bool,
    debug_hook: Optional[DebugHook],
) -> Skeleton:
    skel = init_skeleton(mesh)
    if internal_only:
        skel = internal_only_skeleton(skel)
    if shrink:
        skel.transform = transform

    queue = init_event_list(skel)
    _last = event_loop(queue, skel, debug_hook=debug_hook)
    return skel


def compute_skeleton(
    geom_input: GeometryInput | str | RingsInput,
    *,
    shrink: bool = True,
    internal_only: bool = False,
    adapter: AdapterName = "shapely",
    debug_hook: Optional[DebugHook] = None,
) -> Skeleton:
    geom = normalize_to_geometry(geom_input)

    if shrink:
        points = []
        for poly in geom.geoms if geom.geom_type == "MultiPolygon" else [geom]:
            points.extend(poly.exterior.coords[:-1])
            for interior in poly.interiors:
                points.extend(interior.coords[:-1])
        if not points:
            raise ValueError(
                "Cannot apply shrink transform: geometry has no coordinates "
                "(got empty geometry while shrink=True)."
            )
        box = get_box(points)
        transform = get_transform(box)
        geom = geom_transform(
            lambda x, y, z=None: (
                (x - transform.translate[0]) / transform.scale[0],
                (y - transform.translate[1]) / transform.scale[1],
            ),
            geom,
        )
    else:
        transform = None

    if adapter == "shapely":
        mesh = from_shapely_constrained_delaunay(geom)
    elif adapter == "tri":
        points, infos, segments = points_segments_infos_from_geometry(geom)
        dt = triangulate_with_tri(points, infos, segments)
        mesh = from_tri_delaunay(dt)
    else:
        raise ValueError(f"Unknown adapter: {adapter!r}")

    return _run_skeleton(
        mesh,
        transform=transform,
        shrink=shrink,
        internal_only=internal_only,
        debug_hook=debug_hook,
    )


def compute_segments(
    geom_input: GeometryInput | str | RingsInput,
    *,
    shrink: bool = True,
    internal_only: bool = False,
    adapter: AdapterName = "shapely",
):
    return compute_skeleton(
        geom_input,
        shrink=shrink,
        internal_only=internal_only,
        adapter=adapter,
    ).segments()


def calc_skel(
    geom_input: GeometryInput | str | RingsInput,
    *,
    shrink: bool = True,
    internal_only: bool = False,
    adapter: AdapterName = "shapely",
) -> Skeleton:
    # backwards-ish compatibility (drops pause/output flags)
    return compute_skeleton(
        geom_input,
        shrink=shrink,
        internal_only=internal_only,
        adapter=adapter,
    )
