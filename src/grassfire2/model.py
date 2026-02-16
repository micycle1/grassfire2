from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from .transform import Transform
from .linalg import dist

logger = logging.getLogger(__name__)

Point = tuple[float, float]
Vec2 = tuple[float, float]


@dataclass(slots=True)
class SkeletonNode:
    pos: Point
    step: int
    info: Optional[int] = None

    def position_at(self, time: float) -> Point:
        return self.pos


@dataclass(slots=True)
class KineticVertex:
    origin: Point | None = None
    velocity: Vec2 | None = None

    starts_at: float | None = None
    stops_at: float | None = None

    start_node: SkeletonNode | None = None
    stop_node: SkeletonNode | None = None

    left_hist: list[tuple[float, Optional[float], "VertexRef"]] = field(default_factory=list)
    right_hist: list[tuple[float, Optional[float], "VertexRef"]] = field(default_factory=list)

    # wavefront direction lines
    ul: object | None = None  # Line2
    ur: object | None = None  # Line2
    wfl: object | None = None  # WaveFront
    wfr: object | None = None  # WaveFront

    info: int = 0
    inf_fast: bool = False
    internal: bool = False
    turn: Optional[str] = None

    @property
    def is_stopped(self) -> bool:
        return self.stop_node is not None

    def position_at(self, time: float) -> Point:
        if self.inf_fast:
            assert self.start_node is not None
            return self.start_node.pos
        assert self.origin is not None and self.velocity is not None
        return (self.origin[0] + time * self.velocity[0], self.origin[1] + time * self.velocity[1])

    def visualize_at(self, time: float) -> Point:
        # kept for parity (same as position_at in core-only build)
        return self.position_at(time)

    def distance2_at(self, other: "VertexRef", time: float) -> float:
        sx, sy = self.position_at(time)
        ox, oy = other.position_at(time)  # type: ignore[attr-defined]
        return (sx - ox) ** 2 + (sy - oy) ** 2

    @property
    def left(self) -> Optional["VertexRef"]:
        return self.left_hist[-1][2] if self.left_hist else None

    @property
    def right(self) -> Optional["VertexRef"]:
        return self.right_hist[-1][2] if self.right_hist else None

    @left.setter
    def left(self, v: tuple["VertexRef", float]) -> None:
        ref, time = v
        if self.left_hist:
            s0, _, old = self.left_hist[-1]
            self.left_hist[-1] = (s0, time, old)
        self.left_hist.append((time, None, ref))

    @right.setter
    def right(self, v: tuple["VertexRef", float]) -> None:
        ref, time = v
        if self.right_hist:
            s0, _, old = self.right_hist[-1]
            self.right_hist[-1] = (s0, time, old)
        self.right_hist.append((time, None, ref))

    def left_at(self, time: float) -> Optional["VertexRef"]:
        for start, stop, ref in self.left_hist:
            if (start <= time and stop is not None and stop > time) or (start <= time and stop is None):
                return ref
        return None

    def right_at(self, time: float) -> Optional["VertexRef"]:
        for start, stop, ref in self.right_hist:
            if (start <= time and stop is not None and stop > time) or (start <= time and stop is None):
                return ref
        return None


@dataclass(slots=True)
class InfiniteVertex:
    origin: Point | None = None
    velocity: Vec2 = (0.0, 0.0)
    internal: bool = False
    info: int = 0

    def position_at(self, time: float) -> Point:
        assert self.origin is not None
        return self.origin

    def visualize_at(self, time: float) -> Point:
        return self.position_at(time)

    def distance2_at(self, other: KineticVertex, time: float) -> float:
        sx, sy = self.position_at(time)
        ox, oy = other.position_at(time)
        return (sx - ox) ** 2 + (sy - oy) ** 2


VertexRef = Union[KineticVertex, InfiniteVertex]


@dataclass(slots=True, eq=False)
class KineticTriangle:
    vertices: list[VertexRef | None] = field(default_factory=lambda: [None, None, None])
    neighbours: list[Optional["KineticTriangle"]] = field(default_factory=lambda: [None, None, None])

    wavefront_directions: list[Optional[int]] = field(default_factory=lambda: [None, None, None])
    wavefront_support_lines: list[Optional[object]] = field(default_factory=lambda: [None, None, None])  # WaveFront

    event: Optional["Event"] = None
    info: int = 0
    uid: int = 0  # deterministic tie-break
    stops_at: Optional[float] = None
    internal: bool = False

    @property
    def type(self) -> int:
        return self.neighbours.count(None)

    @property
    def is_finite(self) -> bool:
        return all(isinstance(v, KineticVertex) for v in self.vertices)

    def __eq__(self, other: object) -> bool:
        # identity-based equality keeps mutable triangles hash-consistent
        return self is other

    def __hash__(self) -> int:
        # use object identity for a stable hash
        return id(self)


@dataclass(slots=True, eq=False)
class Event:
    time: float
    tri: KineticTriangle
    side: tuple[int, ...]
    tp: Literal["edge", "flip", "split"]
    triangle_tp: int

    def __str__(self) -> str:
        finite_txt = "finite" if self.tri.is_finite else "infinite"
        return (
            f"<Event ({self.tp:5s}) at {self.time:.9g}, "
            f"{self.tri.type}-triangle: {id(self.tri)} [{self.tri.info}], "
            f"side: {self.side}, finite: {finite_txt}>"
        )


@dataclass(slots=True)
class Skeleton:
    sk_nodes: list[SkeletonNode] = field(default_factory=list)
    vertices: list[KineticVertex] = field(default_factory=list)
    triangles: list[KineticTriangle] = field(default_factory=list)
    transform: Optional[Transform] = None

    def segments(self):
        segments = []
        for v in self.vertices:
            if v.stops_at is not None:
                if v.start_node is v.stop_node:
                    continue
                assert v.start_node is not None and v.stop_node is not None
                if self.transform is not None:
                    pt = (self.transform.backward(v.start_node.pos), self.transform.backward(v.stop_node.pos))
                else:
                    pt = (v.start_node.pos, v.stop_node.pos)
                s = (pt, (v.start_node.info, v.stop_node.info))
            else:
                # legacy behavior: extend ray
                assert v.start_node is not None
                s = ((v.start_node.pos, v.position_at(1000.0)), (v.start_node.info, None))
            segments.append(s)
        return segments