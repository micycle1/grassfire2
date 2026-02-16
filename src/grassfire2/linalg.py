from __future__ import annotations

from math import fsum, sqrt
from operator import add as _add, sub as _sub, mul as _mul, truediv as _div
from typing import Iterable, Tuple

Vec2 = tuple[float, float]
Point = tuple[float, float]


def dot(v1: tuple[float, ...], v2: tuple[float, ...]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vector dimensions should be equal")
    return fsum(p * q for p, q in zip(v1, v2))


def add(a, b):
    if hasattr(b, "__iter__"):
        if len(a) != len(b):
            raise ValueError("Vector dimensions should be equal")
        return tuple(map(_add, a, b))
    return tuple(ai + b for ai in a)


def sub(a, b):
    if hasattr(b, "__iter__"):
        if len(a) != len(b):
            raise ValueError("Vector dimensions should be equal")
        return tuple(map(_sub, a, b))
    return tuple(ai - b for ai in a)


def mul(a, b):
    if hasattr(b, "__iter__"):
        if len(a) != len(b):
            raise ValueError("Vector dimensions should be equal")
        return tuple(map(_mul, a, b))
    return tuple(ai * b for ai in a)


def div(a, b):
    if hasattr(b, "__iter__"):
        if len(a) != len(b):
            raise ValueError("Vector dimensions should be equal")
        return tuple(map(_div, a, b))
    return tuple(ai / b for ai in a)


def norm2(v: tuple[float, ...]) -> float:
    return dot(v, v)


def norm(v: tuple[float, ...]) -> float:
    return sqrt(norm2(v))


def unit(v: tuple[float, ...]):
    return div(v, norm(v))


def make_vector(end: tuple[float, ...], start: tuple[float, ...]):
    return sub(end, start)


def dist(start: tuple[float, ...], end: tuple[float, ...]) -> float:
    return norm(make_vector(end, start))


def rotate90ccw(v: Vec2) -> Vec2:
    return (-(v[1]), v[0])


def rotate90cw(v: Vec2) -> Vec2:
    return (v[1], -v[0])