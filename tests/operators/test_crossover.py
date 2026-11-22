import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acd.circuit.canonical import full_hash
from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit
from acd.circuit.netlist import from_subckt, to_subckt
from acd.circuit.ports import GND, VDD, VIN, VOUT
from acd.circuit.validity import check
from acd.operators import (
    CrossoverConfig,
    MutationConfig,
    grow_region,
    initial_population,
    subgraph_swap,
)
from acd.operators.crossover import region_of
from helpers import FIXTURE_NAMES, load_fixture


def chain() -> Circuit:
    """VIN -R- a -R- b -C- VOUT, plus an R from VOUT to GND and a C from a to VDD."""
    c = Circuit()
    for kind, nets in [
        (Kind.R, (VIN, "a")),
        (Kind.R, ("a", "b")),
        (Kind.C, ("b", VOUT)),
        (Kind.R, (VOUT, GND)),
        (Kind.C, ("a", VDD)),
    ]:
        c, _ = c.add(kind, nets)
    return c


def test_region_boundary_and_interior() -> None:
    c = chain()
    r = region_of(c, {"c2", "c3"})  # a-R-b-C-VOUT
    assert r.boundary == ("a",)  # a is shared with c1 and c5 outside the region
    assert r.interior == ("b",)  # b is only used inside
    r = region_of(c, {"c1", "c2", "c3", "c5"})
    assert r.boundary == () and set(r.interior) == {"a", "b"}


def test_regions_are_connected_and_bounded() -> None:
    c = chain()
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = grow_region(c, 3, rng)
        assert 1 <= len(r.components) <= 3
        # Connected through non-rail nets: every component shares a net with another one.
        if len(r.components) > 1:
            for cid in r.components:
                nets = set(c.component(cid).nets) - {GND, VDD}
                others = {n for o in r.components if o != cid for n in c.component(o).nets}
                assert nets & others


def test_regions_do_not_grow_through_rails() -> None:
    # Two resistors joined only by GND can't form one region.
    c, _ = Circuit().add(Kind.R, (VIN, GND))
    c, _ = c.add(Kind.R, (GND, VOUT))
    rng = np.random.default_rng(0)
    assert all(len(grow_region(c, 2, rng).components) == 1 for _ in range(20))


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(FIXTURE_NAMES), st.sampled_from(FIXTURE_NAMES), st.integers(0, 2**32 - 1))
def test_children_are_valid(a_name: str, b_name: str, seed: int) -> None:
    a, b = load_fixture(a_name), load_fixture(b_name)
    result = subgraph_swap(a, b, np.random.default_rng(seed), max_components=10)
    if result.circuit is not None:
        child = result.circuit
        assert check(child) == []
        assert len(child) <= 10
        assert full_hash(from_subckt(to_subckt(child))) == full_hash(child)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_crossover_on_random_populations(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pop = initial_population(12, rng, MutationConfig())
    for i in range(len(pop) - 1):
        result = subgraph_swap(pop[i], pop[i + 1], rng, max_components=10)
        if result.circuit is not None:
            assert check(result.circuit) == []
            assert len(result.circuit) <= 10


def test_child_mixes_both_parents() -> None:
    # Parent A has only R/C, parent B only MOSFETs and one R: successful children that took a
    # region from B must contain MOSFETs while keeping some of A's capacitors.
    a, b = load_fixture("rc_lowpass"), load_fixture("ota_follower")
    mixed = 0
    for seed in range(40):
        result = subgraph_swap(a, b, np.random.default_rng(seed), max_components=10)
        if result.circuit is not None:
            kinds = {c.kind for c in result.circuit.components}
            if kinds & {Kind.NMOS, Kind.PMOS} and Kind.C in kinds:
                mixed += 1
    assert mixed > 0


def test_component_limit() -> None:
    a, b = load_fixture("ota_follower"), load_fixture("ota_follower")
    for seed in range(20):
        result = subgraph_swap(a, b, np.random.default_rng(seed), max_components=7)
        assert result.circuit is None or len(result.circuit) <= 7


def test_empty_parent() -> None:
    result = subgraph_swap(Circuit(), load_fixture("ce_amp"), np.random.default_rng(0))
    assert result.circuit is None and result.rejections["empty_parent"] == 1


def test_different_ports_rejected() -> None:
    a = load_fixture("ce_amp")
    b, _ = Circuit(ports=(VIN, VOUT, VDD, GND, "VREF")).add(Kind.R, (VIN, "VREF"))
    with pytest.raises(ValueError):
        subgraph_swap(a, b, np.random.default_rng(0))


def test_config_limits_region_size() -> None:
    a, b = load_fixture("ota_follower"), load_fixture("ota_follower")
    config = CrossoverConfig(max_region_size=1)
    for seed in range(20):
        result = subgraph_swap(a, b, np.random.default_rng(seed), config)
        if result.circuit is not None:
            assert len(result.circuit) == len(a)  # one component out, one in
