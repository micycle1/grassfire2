from __future__ import annotations
import heapq
from typing import Generic, TypeVar

# T is expected to be Event, but kept generic
T = TypeVar("T")

class EventQueue(Generic[T]):
    """O(log N) priority queue with O(1) lazy deletion."""
    
    def __init__(self):
        self._heap = []
        self._counter = 0  # Tie-breaker for identical events

    def add(self, item: T) -> None:
        item.valid = True
        # Order by: 
        # 1. time (ascending)
        # 2. -triangle_tp (descending)
        # 3. uid (ascending)
        # 4. _counter (prevents comparing the actual Event object if uids tie)
        sort_key = (item.time, -item.triangle_tp, item.tri.uid, self._counter, item)
        heapq.heappush(self._heap, sort_key)
        self._counter += 1

    def discard(self, item: T) -> None:
        # O(1) removal. We don't remove it from the list, just mark it invalid.
        # It will be skipped when it surfaces to the top of the heap.
        item.valid = False
        
    def remove(self, item: T) -> None:
        self.discard(item)

    def pop(self) -> T:
        """Pops the next valid event."""
        while self._heap:
            _, _, _, _, item = heapq.heappop(self._heap)
            if item.valid:
                return item
        raise KeyError("pop from empty queue")

    def __bool__(self) -> bool:
        """Returns True if there is at least one valid event left."""
        # Clean up invalid events at the top of the heap
        while self._heap and not self._heap[0][-1].valid:
            heapq.heappop(self._heap)
        return bool(self._heap)

    def __len__(self) -> int:
        """
        Slow, simple O(N) length calculation.
        Safely counts only the valid events currently sitting in the heap.
        """
        return sum(1 for *_, item in self._heap if item.valid)