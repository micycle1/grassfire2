from __future__ import annotations

from typing import Iterable, List, Optional
import math


def is_close(
    a: float,
    b: float,
    rel_tol: float = 1e-9,
    abs_tol: float = 0.0,
    method: str = "weak",
) -> bool:
    if method not in ("asymmetric", "strong", "weak", "average"):
        raise ValueError('method must be one of: "asymmetric","strong","weak","average"')
    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError("error tolerances must be non-negative")
    if a == b:
        return True
    diff = abs(b - a)
    if method == "asymmetric":
        return (diff <= abs(rel_tol * b)) or (diff <= abs_tol)
    if method == "strong":
        return (((diff <= abs(rel_tol * b)) and (diff <= abs(rel_tol * a))) or (diff <= abs_tol))
    if method == "weak":
        return (((diff <= abs(rel_tol * b)) or (diff <= abs(rel_tol * a))) or (diff <= abs_tol))
    # average
    return (diff <= abs(rel_tol * (a + b) * 0.5)) or (diff <= abs_tol)


def near_zero(val: float) -> bool:
    return is_close(val, 0.0, rel_tol=1e-12, abs_tol=1e-10, method="weak")


def all_close_clusters(L: Iterable[float], abs_tol: float = 1e-7, rel_tol: float = 0.0) -> List[float]:
    L2 = list(L)
    if not L2:
        return []
    it = iter(sorted(L2))
    first = next(it)
    out = [first]
    for val in it:
        if is_close(first, val, abs_tol=abs_tol, rel_tol=rel_tol, method="average"):
            continue
        out.append(val)
        first = val
    return out


def get_unique_times(times: Iterable[Optional[float]]) -> list[float]:
    return all_close_clusters([t for t in times if t is not None])