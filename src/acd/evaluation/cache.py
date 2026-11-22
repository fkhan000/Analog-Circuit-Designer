"""Cache of evaluation results keyed by circuit structure and values.

GAs evaluate the same circuit many times. Entries are keyed by ``full_hash``; because that hash
can (rarely) collide, a hit is confirmed with an exact isomorphism check unless ``verify=False``.
"""

from __future__ import annotations

from collections import OrderedDict

from acd.circuit.canonical import full_hash, isomorphic
from acd.circuit.graph import Circuit


class EvaluationCache[V]:
    def __init__(self, max_entries: int = 200_000, *, verify: bool = True) -> None:
        self.max_entries = max_entries
        self.verify = verify
        self._entries: OrderedDict[str, list[tuple[Circuit, V]]] = OrderedDict()
        self._size = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return self._size

    def get(self, circuit: Circuit, key: str | None = None) -> V | None:
        key = key or full_hash(circuit)
        bucket = self._entries.get(key)
        if bucket is not None:
            for stored, value in bucket:
                if not self.verify or isomorphic(stored, circuit):
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return value
        self.misses += 1
        return None

    def put(self, circuit: Circuit, value: V, key: str | None = None) -> None:
        key = key or full_hash(circuit)
        bucket = self._entries.setdefault(key, [])
        for i, (stored, _) in enumerate(bucket):
            if not self.verify or isomorphic(stored, circuit):
                bucket[i] = (circuit, value)
                self._entries.move_to_end(key)
                return
        bucket.append((circuit, value))
        self._size += 1
        self._entries.move_to_end(key)
        while self._size > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self._size -= len(evicted)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
