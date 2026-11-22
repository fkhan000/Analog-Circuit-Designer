"""Map a function over items in worker processes, each set up once by an initializer.

Each worker process holds its own simulator (libngspice can't be shared between threads). If a
worker dies (a crash in native code, or a worker that exits because its simulator became unusable),
results that already finished are kept, the pool is rebuilt, and the unfinished items are retried in
parallel one per task. Items still unfinished after that are run one at a time to isolate the
culprit; an item that kills a worker on its own gets ``on_crash(item)`` as its result.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

PARALLEL_RETRY_ROUNDS = 2  # rounds of parallel retries after a crash before isolating items


def _run_chunk[T, R](fn: Callable[[T], R], chunk: list[T]) -> list[R]:
    return [fn(x) for x in chunk]


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
        results: dict[int, R] = {}
        pending = list(range(len(items)))
        chunk = max(1, len(items) // (self.workers * 4))
        for _ in range(1 + PARALLEL_RETRY_ROUNDS):
            if not pending:
                break
            pool = self._get_pool()
            batches = [pending[i : i + chunk] for i in range(0, len(pending), chunk)]
            futures = [
                (b, pool.submit(_run_chunk, self.fn, [items[i] for i in b])) for b in batches
            ]
            failed: list[int] = []
            for batch, future in futures:
                try:
                    results.update(zip(batch, future.result(), strict=True))
                except BrokenProcessPool:
                    failed.extend(batch)
            if failed:
                self._restart()
            pending, chunk = failed, 1
        for i in pending:
            results[i] = self._one_at_a_time(items[i])
        return [results[i] for i in range(len(items))]

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
