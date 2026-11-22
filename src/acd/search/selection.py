"""Parent selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class HasFitness(Protocol):
    @property
    def fitness(self) -> float: ...


def tournament[T: HasFitness](population: Sequence[T], size: int, rng: np.random.Generator) -> T:
    """The fittest (lowest fitness) of ``size`` individuals drawn without replacement."""
    k = min(size, len(population))
    picks = rng.choice(len(population), size=k, replace=False)
    return min((population[int(i)] for i in picks), key=lambda ind: ind.fitness)
