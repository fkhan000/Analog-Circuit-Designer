"""The seed circuit and the initial population built from it."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from acd.circuit.canonical import full_hash
from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit
from acd.circuit.ports import VIN, VOUT
from acd.circuit.validity import PortContext, check
from acd.operators.mutation import MutationConfig, mutate


def seed_circuit() -> Circuit:
    """The simplest valid circuit: one 10 kOhm resistor from VIN to VOUT.

    The test circuit supplies the source, its DC level, the load and the supply, so this is the
    "evolvable path between input and output" that evolution starts from.
    """
    circuit, _ = Circuit().add(Kind.R, (VIN, VOUT), (10e3,))
    return circuit


def initial_population(
    size: int,
    rng: np.random.Generator,
    config: MutationConfig | None = None,
    ctx: PortContext | None = None,
    seeds: Sequence[Circuit] = (),
    max_walk: int = 5,
) -> list[Circuit]:
    """``size`` distinct valid circuits: the seeds plus random mutation walks from them.

    Each walk applies 1..``max_walk`` successive mutations to a random seed. Seeds must pass the
    graph checks and the component limit.
    """
    config = config or MutationConfig()
    bases = list(seeds) or [seed_circuit()]
    for s in bases:
        problems = check(s, ctx)
        if problems:
            raise ValueError(f"seed circuit fails graph checks: {[str(v) for v in problems]}")
        if len(s) > config.max_components:
            raise ValueError(f"seed circuit has {len(s)} components (max {config.max_components})")

    population: list[Circuit] = []
    seen: set[str] = set()

    def add(c: Circuit) -> None:
        key = full_hash(c)
        if key not in seen:
            seen.add(key)
            population.append(c)

    for s in bases:
        add(s)
    attempts = 0
    while len(population) < size and attempts < 50 * size:
        attempts += 1
        child = bases[int(rng.integers(len(bases)))]
        for _ in range(int(rng.integers(1, max_walk + 1))):
            result = mutate(child, rng, config, ctx)
            if result.circuit is None:
                break
            child = result.circuit
        add(child)
    return population[:size]
