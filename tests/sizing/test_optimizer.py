import math

import numpy as np
import pytest

from acd.circuit.devices import Kind, spec
from acd.circuit.graph import Circuit
from acd.circuit.ports import GND, VIN, VOUT
from acd.evaluation.fitness import FitnessWeights, score
from acd.evaluation.stages import evaluate
from acd.evaluation.targets import Band, MaskTarget
from acd.evaluation.testbench import TestbenchConfig
from acd.sim import Ngspice
from acd.sizing import SizingConfig, size, tunable
from helpers import load_fixture


def rlc(r: float, ind: float, c: float) -> Circuit:
    circuit, _ = Circuit().add(Kind.R, (VIN, "a"), (r,))
    circuit, _ = circuit.add(Kind.L, ("a", VOUT), (ind,))
    circuit, _ = circuit.add(Kind.C, (VOUT, GND), (c,))
    return circuit


GOAL = {"c1": 4.7e3, "c2": 2.2e-3, "c3": 3.3e-9}


def distance(circuit: Circuit) -> float:
    """Squared log-distance from the goal values (a stand-in for a simulation)."""
    return sum(math.log10(c.params[0] / GOAL[c.id]) ** 2 for c in circuit.components)


def test_tunable_lists_every_parameter() -> None:
    slots = tunable(load_fixture("ota_follower"))
    assert len(slots) == 6 * 2 + 1  # six MOSFETs (W, L) and one resistor
    assert slots[0][1] == "w"


def test_converges_on_known_optimum() -> None:
    start = rlc(10.0, 1e-8, 1e-6)  # every value decades away
    result = size(start, distance, np.random.default_rng(0), SizingConfig(budget=400))
    assert result.fitness < 0.01 < result.start_fitness
    for c in result.circuit.components:
        assert c.params[0] == pytest.approx(GOAL[c.id], rel=0.25)


def test_values_stay_snapped_and_in_range() -> None:
    result = size(
        rlc(10.0, 1e-8, 1e-6), distance, np.random.default_rng(1), SizingConfig(budget=60)
    )
    for c in result.circuit.components:
        p = spec(c.kind).params[0]
        assert p.contains(c.params[0]) and p.clamp_snap(c.params[0]) == c.params[0]


def test_never_worse_and_within_budget() -> None:
    calls = []

    def noisy(c: Circuit) -> float:
        calls.append(c)
        return distance(c) + len(calls) * 1e-3  # later calls look worse

    start = rlc(4.7e3, 2.2e-3, 3.3e-9)  # already optimal
    result = size(start, noisy, np.random.default_rng(2), SizingConfig(budget=20))
    assert result.circuit == start and result.fitness == result.start_fitness
    assert len(calls) <= 20 and result.evaluations == len(calls)


def test_start_fitness_skips_first_evaluation() -> None:
    calls = []

    def counted(c: Circuit) -> float:
        calls.append(c)
        return distance(c)

    start = rlc(10.0, 1e-8, 1e-6)
    config = SizingConfig(budget=10)
    result = size(start, counted, np.random.default_rng(3), config, start_fitness=distance(start))
    assert start not in calls
    assert result.evaluations == len(calls) + 1 <= config.budget  # the start counts, unsimulated


def test_circuit_without_values() -> None:
    result = size(Circuit(), lambda c: 1.0, np.random.default_rng(0))
    assert result.evaluations == 1 and result.fitness == 1.0


def test_bad_config() -> None:
    with pytest.raises(ValueError):
        SizingConfig(budget=0)
    with pytest.raises(ValueError):
        SizingConfig(when="sometimes")  # type: ignore[arg-type]


@pytest.mark.ngspice
def test_tunes_a_real_filter(ngspice: Ngspice) -> None:
    target = MaskTarget((Band(10, 500, -1, 0.5), Band(10e3, 100e3, max_db=-20)))
    tb = TestbenchConfig(f_start_hz=5, f_stop_hz=200e3)

    def objective(c: Circuit) -> float:
        return score(c, evaluate(c, ngspice, tb), target, FitnessWeights())[0]

    start, _ = Circuit().add(Kind.R, (VIN, VOUT), (10.0,))
    start, _ = start.add(Kind.C, (VOUT, GND), (1e-12,))  # corner at ~16 GHz: useless
    result = size(start, objective, np.random.default_rng(0), SizingConfig(budget=80))
    assert result.start_fitness > 5
    assert result.fitness < 0.5  # an RC this size can meet the mask
