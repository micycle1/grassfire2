from __future__ import annotations

from math import hypot

Vec2 = tuple[float, float]
Point = tuple[float, float]

def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]

def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])

def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])

def mul(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)

def norm2(a: Vec2) -> float:
    return a[0] * a[0] + a[1] * a[1]

def norm(a: Vec2) -> float:
    return hypot(a[0], a[1])

def unit(a: Vec2) -> Vec2:
    inv = 1.0 / hypot(a[0], a[1])
    return (a[0] * inv, a[1] * inv)

def make_vector(end: tuple[float, ...], start: tuple[float, ...]):
    return sub(end, start)

def dist(start: tuple[float, ...], end: tuple[float, ...]) -> float:
    return norm(make_vector(end, start))

def rotate90ccw(v: Vec2) -> Vec2:
    return (-(v[1]), v[0])

def rotate90cw(v: Vec2) -> Vec2:
    return (v[1], -v[0])