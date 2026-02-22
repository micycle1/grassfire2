from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
    c: tuple[Optional[Any], Optional[Any], Optional[Any]]
    is_internal: bool


@dataclass(slots=True)
class InputMesh:
    vertices: list[InputVertex]
    triangles: list[InputTriangle]
