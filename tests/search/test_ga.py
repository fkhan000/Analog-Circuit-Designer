import numpy as np
import pytest

from acd.search import GAConfig, run, tournament
from acd.search.ga import Individual, Outcome


class Parabola:
    """Minimize (x - 3)^2 over floats rounded to 3 decimals."""

    def __init__(self) -> None:
        self.evaluations = 0

    def initial_population(self, size: int, rng: np.random.Generator) -> list[float]:
        return [round(float(x), 3) for x in rng.uniform(-50, 50, size)]

    def mutate(self, genome: float, rng: np.random.Generator) -> float | None:
        return round(genome + float(rng.normal(0, 1)), 3)

    def crossover(self, a: float, b: float, rng: np.random.Generator) -> float | None:
        return round((a + b) / 2, 3)

    def evaluate(self, genomes: list[float]) -> list[Outcome[float]]:
        self.evaluations += len(genomes)
        return [Outcome((g - 3) ** 2) for g in genomes]

    def key(self, genome: float) -> str:
        return f"{genome:.3f}"


def test_converges_on_toy_problem() -> None:
    result = run(Parabola(), GAConfig(population_size=30, generations=60, seed=0))
    assert abs(result.best.genome - 3) < 0.1
    assert result.stopped_because == "generations"
    assert len(result.history) == 61


def test_best_never_gets_worse() -> None:
    result = run(Parabola(), GAConfig(population_size=20, generations=30, elite_count=1, seed=4))
    best = [s.best_fitness for s in result.history]
    assert best == sorted(best, reverse=True)


def test_reproducible_with_same_seed() -> None:
    a = run(Parabola(), GAConfig(population_size=20, generations=10, seed=9))
    b = run(Parabola(), GAConfig(population_size=20, generations=10, seed=9))
    assert a.best.genome == b.best.genome
    assert [s.best_fitness for s in a.history] == [s.best_fitness for s in b.history]


def test_stops_at_target_fitness() -> None:
    result = run(Parabola(), GAConfig(population_size=30, generations=500, target_fitness=1.0))
    assert result.stopped_because == "target_fitness"
    assert result.best.fitness <= 1.0


def test_stops_when_stalled() -> None:
    class Flat(Parabola):
        def evaluate(self, genomes: list[float]) -> list[Outcome[float]]:
            return [Outcome(1.0) for _ in genomes]

    result = run(Flat(), GAConfig(population_size=10, generations=100, stall_generations=5))
    assert result.stopped_because == "stalled"
    assert len(result.history) == 6  # generation 0, then 5 generations without improvement


def test_population_size_is_kept() -> None:
    problem = Parabola()
    run(problem, GAConfig(population_size=25, generations=4, elite_count=3))
    assert problem.evaluations == 25 + 4 * 22


def test_tournament_picks_fittest_of_sample() -> None:
    pop = [Individual(i, float(i), str(i)) for i in range(10)]
    rng = np.random.default_rng(0)
    assert tournament(pop, 10, rng).fitness == 0.0
    assert tournament(pop, 1, rng) in pop


def test_bad_config() -> None:
    with pytest.raises(ValueError):
        GAConfig(population_size=1)
    with pytest.raises(ValueError):
        GAConfig(population_size=10, elite_count=10)


def test_crossover_children_are_used() -> None:
    result = run(
        Parabola(), GAConfig(population_size=30, generations=40, crossover_rate=0.5, seed=3)
    )
    assert abs(result.best.genome - 3) < 0.1
    assert sum(s.crossovers for s in result.history) > 0


def test_failed_crossover_falls_back_to_mutation() -> None:
    class NoCrossover(Parabola):
        def crossover(self, a: float, b: float, rng: np.random.Generator) -> float | None:
            return None

    problem = NoCrossover()
    result = run(problem, GAConfig(population_size=10, generations=3, crossover_rate=1.0))
    assert all(s.crossovers == 0 for s in result.history)
    assert sum(s.failed_crossovers for s in result.history) > 0
    assert problem.evaluations == 10 + 3 * 8  # every child slot still filled


def test_crossover_rate_zero_keeps_old_behavior() -> None:
    # The random stream must not change when crossover is off.
    a = run(Parabola(), GAConfig(population_size=20, generations=10, seed=5))
    b = run(Parabola(), GAConfig(population_size=20, generations=10, seed=5, crossover_rate=0.0))
    assert a.best.genome == b.best.genome


def test_bad_crossover_rate() -> None:
    with pytest.raises(ValueError):
        GAConfig(crossover_rate=1.5)


def test_replacement_genomes_are_kept() -> None:
    class Rounding(Parabola):
        """Evaluation 'tunes' each genome to the nearest integer and hands it back."""

        def evaluate(self, genomes: list[float]) -> list[Outcome[float]]:
            return [Outcome((round(g) - 3) ** 2, genome=float(round(g))) for g in genomes]

    result = run(Rounding(), GAConfig(population_size=10, generations=5, seed=1))
    assert all(ind.genome == round(ind.genome) for ind in result.population)
    assert all(ind.key == f"{ind.genome:.3f}" for ind in result.population)


def test_time_limit() -> None:
    result = run(Parabola(), GAConfig(population_size=10, generations=10**6, time_limit_s=0.2))
    assert result.stopped_because == "time_limit"
