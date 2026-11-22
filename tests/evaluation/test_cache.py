from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit, Component
from acd.circuit.ports import GND, VIN, VOUT
from acd.evaluation.cache import EvaluationCache


def rc(r: float, net: str = "n1") -> Circuit:
    return Circuit(
        (
            Component("c1", Kind.R, (VIN, net), (r,)),
            Component("c2", Kind.R, (net, VOUT), (r,)),
            Component("c3", Kind.C, (VOUT, GND), (1e-9,)),
        )
    )


def test_hit_on_renamed_circuit() -> None:
    cache: EvaluationCache[float] = EvaluationCache()
    cache.put(rc(1e3), 1.5)
    assert cache.get(rc(1e3, net="mid")) == 1.5
    assert cache.hits == 1


def test_miss_on_changed_value() -> None:
    cache: EvaluationCache[float] = EvaluationCache()
    cache.put(rc(1e3), 1.5)
    assert cache.get(rc(2.2e3)) is None
    assert cache.misses == 1


def test_put_replaces_same_circuit() -> None:
    cache: EvaluationCache[float] = EvaluationCache()
    cache.put(rc(1e3), 1.0)
    cache.put(rc(1e3), 2.0)
    assert len(cache) == 1 and cache.get(rc(1e3)) == 2.0


def test_least_recently_used_is_evicted() -> None:
    cache: EvaluationCache[int] = EvaluationCache(max_entries=2)
    cache.put(rc(1e3), 1)
    cache.put(rc(2.2e3), 2)
    cache.get(rc(1e3))  # touch: now 2.2k is the oldest
    cache.put(rc(4.7e3), 3)
    assert cache.get(rc(2.2e3)) is None
    assert cache.get(rc(1e3)) == 1 and cache.get(rc(4.7e3)) == 3
