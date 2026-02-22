from __future__ import annotations

from ...model import Event
from .lib import replace_in_queue, flip


def handle_flip_event(evt: Event, step: int, skel, queue, immediate) -> None:
    now = evt.time
    assert len(evt.side) == 1
    t, t_side = evt.tri, evt.side[0]
    n = t.neighbours[t_side]
    assert n is not None
    n_side = n.neighbours.index(t)
    flip(t, t_side, n, n_side)
    replace_in_queue(t, now, queue, immediate)
    replace_in_queue(n, now, queue, immediate)