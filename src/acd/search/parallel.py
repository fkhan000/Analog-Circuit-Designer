"""Map a function over items in worker processes, each set up once by an initializer.

Each worker process holds its own simulator (libngspice can't be shared between threads). If a
worker dies (a crash in native code, or a worker that exits because its simulator became unusable),
the pool is rebuilt and the unfinished items are retried one at a time; an item that kills a worker
again gets ``on_crash(item)`` as its result.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any


def in_worker_process() -> bool:
    return multiprocessing.parent_process() is not None


class ParallelMap[T, R]:
    def __init__(
        self,
        fn: Callable[[T], R],
        workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[Any, ...] = (),
        on_crash: Callable[[T], R] | None = None,
    ) -> None:
        """``workers <= 1`` runs everything in this process (the initializer runs here once)."""
        self.fn = fn
        self.workers = workers
        self.initializer = initializer
        self.initargs = initargs
        self.on_crash = on_crash
        self.restarts = 0
        self._pool: ProcessPoolExecutor | None = None
        if workers <= 1 and initializer is not None:
            initializer(*initargs)

    @staticmethod
    def default_workers() -> int:
        return max(1, (os.cpu_count() or 1) - 1)

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=self.initializer,
                initargs=self.initargs,
            )
        return self._pool

    def _restart(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self._pool = None
        self.restarts += 1

    def map(self, items: Sequence[T]) -> list[R]:
        if self.workers <= 1:
            return [self.fn(x) for x in items]
        if not items:
            return []
        chunksize = max(1, len(items) // (self.workers * 4))
        try:
            return list(self._get_pool().map(self.fn, items, chunksize=chunksize))
        except BrokenProcessPool:
            self._restart()
            return [self._one_at_a_time(x) for x in items]

    def _one_at_a_time(self, item: T) -> R:
        try:
            return self._get_pool().submit(self.fn, item).result()
        except BrokenProcessPool:
            self._restart()
            if self.on_crash is None:
                raise
            return self.on_crash(item)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self) -> ParallelMap[T, R]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
