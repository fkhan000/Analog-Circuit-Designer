from hypothesis import given

from acd.circuit.canonical import full_hash, isomorphic, structural_distance, topology_hash
from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit, Component
from acd.circuit.ports import GND, VDD, VIN, VOUT
from helpers import circuits


def ce(collector: str, emitter: str, r: float = 1e3) -> Circuit:
    return Circuit(
        (
            Component("c1", Kind.NPN, (collector, VIN, emitter), (1.0,)),
            Component("c2", Kind.R, (VDD, VOUT), (r,)),
            Component("c3", Kind.R, ("n1", GND), (r,)),
        )
    )


def test_renaming_ids_and_nets_keeps_hash() -> None:
    a = ce(VOUT, "n1")
    b = Circuit(
        (
            Component("x7", Kind.R, ("e", GND), (1e3,)),
            Component("q", Kind.NPN, (VOUT, VIN, "e"), (1.0,)),
            Component("load", Kind.R, (VOUT, VDD), (1e3,)),  # pins swapped too
        )
    )
    assert full_hash(a) == full_hash(b)
    assert isomorphic(a, b)


def test_swapping_collector_and_emitter_changes_hash() -> None:
    assert topology_hash(ce(VOUT, "n1")) != topology_hash(ce("n1", VOUT))


def test_values_only_affect_full_hash() -> None:
    a, b = ce(VOUT, "n1", 1e3), ce(VOUT, "n1", 2.2e3)
    assert topology_hash(a) == topology_hash(b)
    assert full_hash(a) != full_hash(b)
    assert isomorphic(a, b, with_values=False)
    assert not isomorphic(a, b)


def test_which_port_matters() -> None:
    a = Circuit((Component("c1", Kind.R, (VIN, VOUT), (1e3,)),))
    b = Circuit((Component("c1", Kind.R, (VIN, VDD), (1e3,)),))
    assert topology_hash(a) != topology_hash(b)


def test_diode_connection_differs_from_separate_nets() -> None:
    a = Circuit((Component("c1", Kind.NPN, ("n1", "n1", GND), (1.0,)),))
    b = Circuit((Component("c1", Kind.NPN, ("n1", "n2", GND), (1.0,)),))
    assert topology_hash(a) != topology_hash(b)


@given(circuits())
def test_hash_ignores_component_order(c: Circuit) -> None:
    reordered = Circuit(tuple(reversed(c.components)), next_id=c.next_id)
    assert full_hash(c) == full_hash(reordered)


def test_structural_distance() -> None:
    a = ce(VOUT, "n1")
    assert structural_distance(a, a) == 0.0
    assert structural_distance(a, ce(VOUT, "n1", r=4.7e3)) == 0.0  # values don't matter
    renamed = Circuit(tuple(reversed(a.components)))
    assert structural_distance(a, renamed) == 0.0
    swapped = ce("n1", VOUT)
    other = Circuit((Component("c1", Kind.C, (VIN, VOUT), (1e-9,)),))
    d_near, d_far = structural_distance(a, swapped), structural_distance(a, other)
    assert 0 < d_near < d_far <= 1.0
    assert structural_distance(other, a) == d_far  # symmetric


@given(circuits(), circuits())
def test_distance_bounds(a: Circuit, b: Circuit) -> None:
    d = structural_distance(a, b)
    assert 0.0 <= d <= 1.0
    if topology_hash(a) == topology_hash(b):
        assert d == 0.0
