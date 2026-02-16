from __future__ import annotations

from bisect import bisect_right
from functools import cmp_to_key
from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class OrderedSequence(Generic[T]):
    """Ordered sequence where duplicates are allowed (legacy behavior)."""

    def __init__(self, cmp: Callable[[T, T], int]):
        self._cmp = cmp
        self._key = cmp_to_key(cmp)
        self._items: list[T] = []
        self._keys: list[object] = []

    def add(self, item: T) -> None:
        key_item = self._key(item)
        index = bisect_right(self._keys, key_item)
        self._items.insert(index, item)
        self._keys.insert(index, key_item)

    def remove(self, item: T) -> None:
        for idx, existing in enumerate(self._items):
            if existing is item:
                del self._items[idx]
                del self._keys[idx]
                return
        raise ValueError("item not found in ordered sequence")

    def discard(self, item: T) -> None:
        try:
            self.remove(item)
        except ValueError:
            pass

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)