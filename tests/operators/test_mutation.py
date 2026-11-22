from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acd.circuit.canonical import full_hash
from acd.circuit.devices import Kind, spec
from acd.circuit.graph import Circuit
from acd.circuit.netlist import from_subckt, to_subckt
from acd.circuit.ports import GND, VIN, VOUT
from acd.circuit.validity import check
from acd.operators import (
    MutationConfig,
    MutationOp,
    apply,
    initial_population,
    mutate,
    random_params,
    seed_circuit,
)
from helpers import FIXTURE_NAMES, circuits, load_fixture

CONFIG = MutationConfig()


@given(circuits(max_components=10), st.sampled_from(list(MutationOp)), st.integers(0, 2**32 - 1))
def test_apply_never_raises(c: Circuit, op: MutationOp, seed: int) -> None:
    result = apply(op, c, np.random.default_rng(seed), CONFIG)
    assert result is None or isinstance(result, Circuit)


@given(st.sampled_from(FIXTURE_NAMES), st.integers(0, 2**32 - 1))
def test_mutate_returns_valid_circuits(name: str, seed: int) -> None:
    parent = load_fixture(name)
    result = mutate(parent, np.random.default_rng(seed), CONFIG)
    if result.circuit is not None:
        assert check(result.circuit) == []
        assert len(result.circuit) <= CONFIG.max_components
        assert result.ops


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_long_random_walk_keeps_rules(seed: int) -> None:
    rng = np.random.default_rng(seed)
    c = seed_circuit()
    for _ in range(60):
        result = mutate(c, rng, CONFIG)
        if result.circuit is None:
            continue
        c = result.circuit
        assert check(c) == []
        assert len(c) <= CONFIG.max_components
        assert {x.kind for x in c.components} <= set(CONFIG.kinds)
        assert set(c.ports) <= set(c.nets)
        # Every pin still has a net and the netlist round-trips.
        assert full_hash(from_subckt(to_subckt(c))) == full_hash(c)


def test_default_kinds_include_diodes_and_inductors() -> None:
    assert set(CONFIG.kinds) == set(Kind)


def test_restricted_kinds_are_respected() -> None:
    config = MutationConfig(kinds=(Kind.R, Kind.C))
    rng = np.random.default_rng(1)
    c = seed_circuit()
    for _ in range(300):
        c = mutate(c, rng, config).circuit or c
        assert {x.kind for x in c.components} <= {Kind.R, Kind.C}


def test_inductors_when_enabled() -> None:
    config = MutationConfig(kinds=(Kind.L,))
    rng = np.random.default_rng(0)
    c = apply(MutationOp.PARALLEL_INSERT, seed_circuit(), rng, config)
    assert c is not None and c.components[-1].kind == Kind.L


def test_series_insert_splits_a_net() -> None:
    rng = np.random.default_rng(3)
    c = apply(MutationOp.SERIES_INSERT, seed_circuit(), rng, MutationConfig(kinds=(Kind.C,)))
    assert c is not None and len(c) == 2
    assert len(c.internal_nets) == 1
    assert check(c) == []  # the moved pin and the new capacitor share the new net


def test_inserts_respect_component_limit() -> None:
    config = MutationConfig(max_components=1)
    rng = np.random.default_rng(0)
    assert apply(MutationOp.SERIES_INSERT, seed_circuit(), rng, config) is None
    assert apply(MutationOp.PARALLEL_INSERT, seed_circuit(), rng, config) is None


def test_delete_short_never_merges_ports() -> None:
    # The only 2-pin component spans two ports, so a short isn't possible.
    assert apply(MutationOp.DELETE_SHORT, seed_circuit(), np.random.default_rng(0), CONFIG) is None


def test_delete_short_joins_nets() -> None:
    c, _ = Circuit().add(Kind.R, (VIN, "n1"))
    c, _ = c.add(Kind.R, ("n1", VOUT))
    for seed in range(10):
        out = apply(MutationOp.DELETE_SHORT, c, np.random.default_rng(seed), CONFIG)
        assert out is not None and len(out) == 1
        assert set(out.components[0].nets) == {VIN, VOUT}


def test_change_type_maps_pins_by_role() -> None:
    c, _ = Circuit().add(Kind.NPN, ("col", "base", "emit"))
    config = MutationConfig(kinds=(Kind.NMOS,))
    out = apply(MutationOp.CHANGE_TYPE, c, np.random.default_rng(0), config)
    assert out is not None
    comp = out.components[0]
    assert comp.kind == Kind.NMOS and comp.pins == {"d": "col", "g": "base", "s": "emit"}


def test_change_value_always_changes_and_stays_snapped() -> None:
    rng = np.random.default_rng(0)
    c = seed_circuit()
    for _ in range(50):
        out = apply(MutationOp.CHANGE_VALUE, c, rng, CONFIG)
        assert out is not None
        old, new = c.components[0].params[0], out.components[0].params[0]
        p = spec(Kind.R).params[0]
        assert new != old and p.contains(new) and p.clamp_snap(new) == new
        c = out


def test_rejections_are_counted() -> None:
    # MOVE_PIN on a single resistor nearly always disconnects VIN or VOUT.
    config = MutationConfig(op_weights={MutationOp.MOVE_PIN: 1.0}, mean_extra_ops=0)
    rejections: Counter[str] = Counter()
    for seed in range(20):
        rejections += mutate(seed_circuit(), np.random.default_rng(seed), config).rejections
    assert rejections


def test_random_params_in_range() -> None:
    rng = np.random.default_rng(0)
    for kind in Kind:
        for p, v in zip(spec(kind).params, random_params(kind, rng), strict=True):
            assert p.contains(v)


def test_initial_population() -> None:
    pop = initial_population(30, np.random.default_rng(7), CONFIG)
    assert len(pop) == 30
    assert pop[0] == seed_circuit()
    assert len({full_hash(c) for c in pop}) == 30
    assert all(check(c) == [] for c in pop)
    again = initial_population(30, np.random.default_rng(7), CONFIG)
    assert [full_hash(c) for c in again] == [full_hash(c) for c in pop]  # reproducible


def test_initial_population_from_seeds() -> None:
    seeds = [load_fixture("ce_amp"), load_fixture("rc_lowpass")]
    pop = initial_population(10, np.random.default_rng(0), CONFIG, seeds=seeds)
    assert pop[:2] == seeds


def test_invalid_seed_rejected() -> None:
    bad, _ = Circuit().add(Kind.R, (VIN, GND))
    with pytest.raises(ValueError):
        initial_population(5, np.random.default_rng(0), CONFIG, seeds=[bad])
