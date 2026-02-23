from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class Constraint:
    weight: float = 1.0
    id: int | None = None


@dataclass(slots=True)
class InputVertex:
    x: float
    y: float
    is_finite: bool
    info: Any = None


@dataclass(slots=True)
class InputTriangle:
    v: tuple[int, int, int]
    n: tuple[int, int, int]
    c: tuple[Optional[Constraint], Optional[Constraint], Optional[Constraint]]
    is_internal: bool


@dataclass(slots=True)
class InputMesh:
    vertices: list[InputVertex]
    triangles: list[InputTriangle]
