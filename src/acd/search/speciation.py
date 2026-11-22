"""Speciation: protect new structures by letting them compete within their own species.

Without it, a new topology competes immediately against the whole population, and the population
tends to collapse onto copies of one early, mediocre circuit. NEAT-style speciation fixes that:

1. **Assign species.** Each individual (best first) joins the first species whose representative
   is within ``threshold`` of it, or founds a new species (up to ``max_species``; beyond that it
   joins the nearest one). The threshold is calibrated on the first population, then moves in
   small steps to keep roughly ``target`` species (within 25%).
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
        threshold: float | None = None,
        target: int = 8,
        stagnation: int = 30,
        step: float = 0.02,
        max_species: int | None = None,
    ) -> None:
        """``threshold=None`` calibrates the threshold on the first population assigned."""
        self.distance = distance
        self.threshold = threshold
        self.target = target
        self.stagnation = stagnation
        self.step = step
        self.max_species = max_species or 3 * target
        self.species: list[Species[G]] = []
        self._next_id = 0

    def _count_species(self, genomes: list[G], threshold: float) -> int:
        reps: list[G] = []
        for g in genomes:
            if not any(self.distance(g, r) < threshold for r in reps):
                reps.append(g)
                if len(reps) > 2 * self.target:
                    break  # clearly too many; no need to finish counting
        return len(reps)

    def calibrate(self, population: list[Individual[G]], sample: int = 100) -> float:
        """Binary-search the threshold giving about ``target`` species on a sample."""
        genomes = [i.genome for i in sorted(population, key=lambda i: i.fitness)[:sample]]
        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = (lo + hi) / 2
            if self._count_species(genomes, mid) > self.target:
                lo = mid
            else:
                hi = mid
        return hi

    def assign(self, population: list[Individual[G]], gen: int) -> list[Species[G]]:
        """Group ``population`` (any order) into species; returns live species, best first."""
        if self.threshold is None:
            self.threshold = self.calibrate(population)
        for s in self.species:
            s.members = []
        for ind in sorted(population, key=lambda i: i.fitness):
            distances = []
            home = None
            for s in self.species:
                d = self.distance(ind.genome, s.representative)
                if d < self.threshold:
                    home = s
                    break
                distances.append((d, s))
            if home is None:
                if len(self.species) < self.max_species:
                    home = Species(self._next_id, ind.genome, gen, last_improved=gen)
                    self._next_id += 1
                    self.species.append(home)
                else:  # cap the species count (and the cost): join the nearest species
                    home = min(distances, key=lambda pair: pair[0])[1]
            home.members.append(ind)

        self.species = [s for s in self.species if s.members]
        for s in self.species:
            champion = s.members[0]  # members were added best first
            s.representative = champion.genome
            if champion.fitness < s.best_fitness:
                s.best_fitness = champion.fitness
                s.last_improved = gen
        self.species.sort(key=lambda s: s.best_fitness)

        # Distances bunch together, so adapt in small additive steps, and only outside a dead band
        # around the target (otherwise the count flips back and forth every generation).
        band = max(1, self.target // 4)
        if len(self.species) > self.target + band:
            self.threshold = min(1.0, self.threshold + self.step)
        elif len(self.species) < self.target - band:
            self.threshold = max(self.step, self.threshold - self.step)
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
