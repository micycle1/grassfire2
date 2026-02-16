from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from ..collapse import compute_collapse_time, find_gt
from ..model import Event, Skeleton
from .queue import OrderedSequence
from .ordering import compare_event_by_time
from .handlers.edge import handle_edge_event, handle_edge_event_1side, handle_edge_event_3sides
from .handlers.flip import handle_flip_event
from .handlers.split import handle_split_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DebugState:
    step: int
    now: float
    event: Event
    queue_size: int
    immediate_size: int


DebugHook = Callable[[DebugState], None]


def choose_next_event(queue: OrderedSequence[Event]) -> Event:
    it = iter(queue)
    evt = next(it)
    queue.remove(evt)
    return evt


def init_event_list(skel: Skeleton) -> OrderedSequence[Event]:
    q: OrderedSequence[Event] = OrderedSequence(cmp=compare_event_by_time)
    for tri in skel.triangles:
        res = compute_collapse_time(tri, 0.0, find_gt)
        if res is not None:
            q.add(res)
    return q


def event_loop(queue: OrderedSequence[Event], skel: Skeleton, debug_hook: Optional[DebugHook] = None) -> float:
    NOW = 0.0
    step = 0
    immediate = deque([])

    guard = 0
    while queue or immediate:
        guard += 1
        if guard > 50_000:
            raise ValueError("loop with more than 50_000 events stopped")

        step += 1
        if immediate:
            evt = immediate.popleft()
        else:
            evt = choose_next_event(queue)
            NOW = evt.time

        if debug_hook is not None:
            debug_hook(DebugState(step=step, now=NOW, event=evt, queue_size=len(queue), immediate_size=len(immediate)))

        if evt.tri.stops_at is not None:
            logger.warning("Already stopped %s, but still queued", id(evt.tri))
            continue

        if evt.tp == "edge":
            if len(evt.side) == 3:
                handle_edge_event_3sides(evt, step, skel, queue, immediate)
            elif len(evt.side) == 1 and evt.tri.type == 3:
                handle_edge_event_1side(evt, step, skel, queue, immediate)
            elif len(evt.side) == 2:
                raise ValueError(f"Impossible configuration: triangle [{evt.tri.info}] has 2 sides collapsing")
            else:
                handle_edge_event(evt, step, skel, queue, immediate)
        elif evt.tp == "flip":
            handle_flip_event(evt, step, skel, queue, immediate)
        elif evt.tp == "split":
            handle_split_event(evt, step, skel, queue, immediate)
        else:
            raise ValueError(f"Unknown event type: {evt.tp}")

    # end-condition: internal triangles should be stopped
    not_stopped = [tri.info for tri in skel.triangles if tri.internal and tri.stops_at is None]
    if not_stopped:
        raise ValueError(f"triangles not stopped at end: {not_stopped}")

    return NOW