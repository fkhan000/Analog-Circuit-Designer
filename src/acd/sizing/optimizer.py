"""Tune a circuit's component values for a fixed topology.

A good topology with badly chosen values scores badly and gets dropped before evolution can fix
the values, so each new topology gets a short value search before it is scored. The search is a
(1+lambda) evolution strategy in log10(value) space:

- each step tries ``batch`` candidates around the current best, each perturbing a random subset
  of the values (anything from one value to all of them) by Gaussian steps of size ``sigma``
  decades; moving several values together matters for resonances, where L and C must move as a
  pair
- candidates are snapped to standard values (E-series / size grids) before simulating, so the
  result is a circuit the GA could have produced itself
- an improving step grows ``sigma``, a failed one shrinks it; the search stops when the budget is
  spent or ``sigma`` falls below ``min_sigma``

The result is never worse than the starting circuit.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from acd.circuit.devices import ParamSpec, spec
from acd.circuit.graph import Circuit


@dataclass(frozen=True)
class SizingConfig:
    enabled: bool = False
    budget: int = 40  # simulations per sizing run, including the starting circuit
    batch: int = 4  # candidates per step
    sigma0: float = 0.3  # initial step size in decades (0.3 = a factor of 2)
    min_sigma: float = 0.02
    grow: float = 1.5  # step-size factor after an improving step
    shrink: float = 0.6  # step-size factor after a step with no improvement
    when: Literal["structural", "always"] = "structural"  # which new children to size

    def __post_init__(self) -> None:
        if self.budget < 1 or self.batch < 1:
            raise ValueError("budget and batch must be at least 1")
        if self.when not in ("structural", "always"):
            raise ValueError("when must be 'structural' or 'always'")


@dataclass(frozen=True)
class SizingResult:
    circuit: Circuit
    fitness: float
    start_fitness: float
    evaluations: int


Slot = tuple[str, str, ParamSpec]  # (component id, parameter name, spec)


def tunable(circuit: Circuit) -> list[Slot]:
    """Every parameter of every component, in circuit order."""
    return [(c.id, p.name, p) for c in circuit.components for p in spec(c.kind).params]


def _with_values(circuit: Circuit, slots: list[Slot], values: tuple[float, ...]) -> Circuit:
    for (cid, name, _), v in zip(slots, values, strict=True):
        if circuit.component(cid).param(name) != v:
            circuit = circuit.set_param(cid, name, v)
    return circuit


def size(
    circuit: Circuit,
    objective: Callable[[Circuit], float],
    rng: np.random.Generator,
    config: SizingConfig | None = None,
    start_fitness: float | None = None,
) -> SizingResult:
    """Search values of ``circuit`` to minimize ``objective`` within the simulation budget.

    Pass ``start_fitness`` if the starting circuit was already evaluated; it then isn't simulated
    again (but still counts towards the budget).
    """
    config = config or SizingConfig()
    slots = tunable(circuit)
    start = objective(circuit) if start_fitness is None else start_fitness
    if not slots:
        return SizingResult(circuit, start, start, 1)

    lo = np.array([math.log10(p.low) for _, _, p in slots])
    hi = np.array([math.log10(p.high) for _, _, p in slots])
    current = tuple(circuit.component(cid).param(name) for cid, name, _ in slots)
    best_values, best_f = current, start
    best_x = np.log10(np.array(current))
    seen: dict[tuple[float, ...], float] = {current: start}
    evaluations, sigma, n = 1, config.sigma0, len(slots)
    stale = 0  # consecutive candidates that snapped to values already tried

    while evaluations < config.budget and sigma >= config.min_sigma and stale < 50:
        step_best: tuple[float, tuple[float, ...]] | None = None
        for _ in range(min(config.batch, config.budget - evaluations)):
            k = int(rng.integers(1, n + 1))  # how many values this candidate moves
            idx = rng.choice(n, size=k, replace=False)
            x = best_x.copy()
            x[idx] += sigma * rng.normal(size=k)
            x = np.clip(x, lo, hi)
            values = tuple(
                p.clamp_snap(10 ** float(xi)) for (_, _, p), xi in zip(slots, x, strict=True)
            )
            if values in seen:
                stale += 1
                continue
            stale = 0
            f = objective(_with_values(circuit, slots, values))
            seen[values] = f
            evaluations += 1
            if step_best is None or f < step_best[0]:
                step_best = (f, values)
        if step_best is not None and step_best[0] < best_f:
            best_f, best_values = step_best
            best_x = np.log10(np.array(best_values))
            sigma *= config.grow
        else:
            sigma *= config.shrink

    return SizingResult(_with_values(circuit, slots, best_values), best_f, start, evaluations)
