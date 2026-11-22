from acd.search import ParallelMap
from helpers import crashed, square, square_or_crash


def test_serial_map() -> None:
    with ParallelMap(square, workers=1) as pm:
        assert pm.map([1, 2, 3]) == [1, 4, 9]


def test_process_pool_map() -> None:
    with ParallelMap(square, workers=2) as pm:
        assert pm.map(list(range(20))) == [x * x for x in range(20)]
        assert pm.map([]) == []


def test_worker_crash_is_recovered() -> None:
    with ParallelMap(square_or_crash, workers=2, on_crash=crashed) as pm:
        assert pm.map([1, -5, 3]) == [1, -1, 9]
        assert pm.restarts >= 1
        assert pm.map([4]) == [16]  # pool works again afterwards


def test_crash_keeps_finished_results_in_big_batches() -> None:
    items = [*range(40), -7, *range(40, 80)]
    with ParallelMap(square_or_crash, workers=3, on_crash=crashed) as pm:
        out = pm.map(items)
    assert out == [-1 if x < 0 else x * x for x in items]
