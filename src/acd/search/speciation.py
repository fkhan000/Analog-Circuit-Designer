"""Speciation: protect new structures by letting them compete within their own species.

Without it, a new topology competes immediately against the whole population, and the population
tends to collapse onto copies of one early, mediocre circuit. NEAT-style speciation fixes that:

1. **Assign species.** Each individual (best first) joins the first species whose representative
   is within ``threshold`` of it, or founds a new species. The threshold adapts to keep roughly
   ``target`` species.
2. **Allocate offspring** to species by the rank of their best fitness (linear rank weights), so
   good species get more children but every live species gets some.
3. **Stagnation.** A species whose best hasn't improved for ``stagnation`` generations gets no
   children, unless it holds the overall best.

Parents (and crossover mates) are then chosen within each species.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acd.search.ga import Individual


@dataclass
class Species[G]:
    id: int
    representative: G
    created: int
    best_fitness: float = float("inf")
    last_improved: int = 0
    members: list[Individual[G]] = field(default_factory=list)


class Speciator[G]:
    def __init__(
        self,
        distance: Callable[[G, G], float],
        threshold: float = 0.5,
        target: int = 8,
        stagnation: int = 30,
    ) -> None:
        self.distance = distance
        self.threshold = threshold
        self.target = target
        self.stagnation = stagnation
        self.species: list[Species[G]] = []
        self._next_id = 0

    def assign(self, population: list[Individual[G]], gen: int) -> list[Species[G]]:
        """Group ``population`` (any order) into species; returns live species, best first."""
        for s in self.species:
            s.members = []
        for ind in sorted(population, key=lambda i: i.fitness):
            home = next(
                (
                    s
                    for s in self.species
                    if self.distance(ind.genome, s.representative) < self.threshold
                ),
                None,
            )
            if home is None:
                home = Species(self._next_id, ind.genome, gen, last_improved=gen)
                self._next_id += 1
                self.species.append(home)
            home.members.append(ind)

        self.species = [s for s in self.species if s.members]
        for s in self.species:
            champion = s.members[0]  # members were added best first
            s.representative = champion.genome
            if champion.fitness < s.best_fitness:
                s.best_fitness = champion.fitness
                s.last_improved = gen
        self.species.sort(key=lambda s: s.best_fitness)

        if len(self.species) > self.target:
            self.threshold = min(1.0, self.threshold * 1.15)
        elif len(self.species) < self.target:
            self.threshold = max(0.02, self.threshold / 1.15)
        return self.species

    def allocate(self, n_children: int, gen: int) -> list[tuple[Species[G], int]]:
        """Children per species: linear rank weights over species that aren't stagnant."""
        if not self.species or n_children <= 0:
            return []
        best = self.species[0]
        live = [s for s in self.species if s is best or gen - s.last_improved < self.stagnation]
        weights = [len(live) - rank for rank in range(len(live))]  # best species weighs most
        total = sum(weights)
        quotas = [n_children * w // total for w in weights]
        for i in range(n_children - sum(quotas)):  # leftovers to the best species
            quotas[i % len(live)] += 1
        return [(s, q) for s, q in zip(live, quotas, strict=True) if q > 0]
