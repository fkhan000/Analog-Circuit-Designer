import numpy as np

from acd.search import GAConfig, run
from acd.search.ga import Individual, Outcome
from acd.search.speciation import Speciator


def ind(x: float, fitness: float) -> Individual[float]:
    return Individual(x, fitness, f"{x}")


def dist(a: float, b: float) -> float:
    return min(1.0, abs(a - b) / 10)


def test_assign_groups_by_distance() -> None:
    sp: Speciator[float] = Speciator(dist, threshold=0.3, target=2)
    species = sp.assign([ind(0, 5), ind(1, 1), ind(50, 3), ind(52, 4)], gen=0)
    assert [sorted(i.genome for i in s.members) for s in species] == [[0, 1], [50, 52]]
    assert species[0].representative == 1  # the champion represents its species
    assert species[0].best_fitness == 1


def test_threshold_adapts_towards_target() -> None:
    def wide(a: float, b: float) -> float:
        return abs(a - b) / 100

    pop = [ind(float(x), float(x)) for x in range(0, 100, 5)]
    loose: Speciator[float] = Speciator(wide, threshold=0.02, target=3)  # starts with 20 species
    tight: Speciator[float] = Speciator(wide, threshold=0.9, target=6)  # starts with 1 species
    for gen in range(40):
        loose.assign(pop, gen)
        tight.assign(pop, gen)
    assert 2 <= len(loose.species) <= 5
    assert 4 <= len(tight.species) <= 9


def test_allocation_sums_and_favors_better_species() -> None:
    sp: Speciator[float] = Speciator(dist, threshold=0.3, target=3)
    sp.assign([ind(0, 1), ind(1, 2), ind(50, 3), ind(90, 9)], gen=0)
    alloc = sp.allocate(20, gen=1)
    assert sum(q for _, q in alloc) == 20
    quotas = [q for _, q in alloc]
    assert quotas == sorted(quotas, reverse=True) and quotas[-1] >= 1


def test_stagnant_species_stop_breeding() -> None:
    sp: Speciator[float] = Speciator(dist, threshold=0.3, target=2, stagnation=5)
    pop = [ind(0, 1), ind(50, 3)]
    sp.assign(pop, gen=0)
    sp.assign(pop, gen=10)  # neither improved since generation 0
    alloc = sp.allocate(10, gen=10)
    assert [(s.best_fitness, q) for s, q in alloc] == [(1, 10)]  # only the overall best survives


class TwoValleys:
    """Minimize a function with a wide shallow valley at x=-20 and a narrow deep one at x=30."""

    def initial_population(self, size: int, rng: np.random.Generator) -> list[float]:
        return [round(float(x), 2) for x in rng.uniform(-30, -10, size)]  # all start near -20

    def mutate(self, genome: float, rng: np.random.Generator) -> float | None:
        return round(genome + float(rng.normal(0, 3)), 2)

    def crossover(self, a: float, b: float, rng: np.random.Generator) -> float | None:
        return round((a + b) / 2, 2)

    def evaluate(self, genomes: list[float]) -> list[Outcome[float]]:
        def f(x: float) -> float:
            return min(1 + ((x + 20) / 10) ** 2, ((x - 30) / 2) ** 2)

        return [Outcome(f(g)) for g in genomes]

    def key(self, genome: float) -> str:
        return f"{genome:.2f}"

    def distance(self, a: float, b: float) -> float:
        return dist(a, b)


def test_ga_with_speciation_runs_and_keeps_size() -> None:
    config = GAConfig(population_size=40, generations=40, seed=0, speciation=True, species_target=4)
    result = run(TwoValleys(), config)
    assert len(result.population) == 40
    assert max(s.species for s in result.history) >= 2
    assert all(s.species_threshold is not None for s in result.history)
    best = [s.best_fitness for s in result.history]
    assert best == sorted(best, reverse=True)  # champions and elites keep the best


def test_speciation_keeps_population_diverse() -> None:
    def spread(speciation: bool) -> float:
        out = []
        for seed in range(5):
            config = GAConfig(
                population_size=40, generations=40, seed=seed, speciation=speciation,
                species_target=4,
            )  # fmt: skip
            out.append(float(np.std([i.genome for i in run(TwoValleys(), config).population])))
        return float(np.mean(out))

    assert spread(True) > 1.5 * spread(False)


def test_speciation_off_reports_no_species() -> None:
    result = run(TwoValleys(), GAConfig(population_size=20, generations=3))
    assert all(s.species == 0 and s.species_threshold is None for s in result.history)
