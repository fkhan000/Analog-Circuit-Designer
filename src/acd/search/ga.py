"""A generational genetic algorithm that knows nothing about circuits.

The problem supplies the operations (initial population, mutate, crossover, evaluate, key), so the
circuit code never leaks in here and this loop could later be swapped for a multi-objective one.
Lower fitness is better. Each generation keeps ``elite_count`` best individuals and fills the rest
with children of tournament-selected parents, skipping children already in the population. A child
comes from crossover of two parents with probability ``crossover_rate`` (then is mutated with
probability ``mutate_after_crossover``); otherwise, or if crossover fails, from mutation.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from acd.search.selection import tournament


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    generations: int = 100
    elite_count: int = 2
    tournament_size: int = 3
    seed: int = 0
    stall_generations: int | None = None  # stop after this many generations without improvement
    target_fitness: float | None = None  # stop once the best fitness reaches this
    max_child_attempts: int = 10  # tries to produce a child not already in the population
    crossover_rate: float = 0.0  # probability that a child comes from crossover
    mutate_after_crossover: float = 0.5  # probability a crossover child is also mutated

    def __post_init__(self) -> None:
        for name in ("crossover_rate", "mutate_after_crossover"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 0 and population_size - 1")


@dataclass
class Individual[G]:
    genome: G
    fitness: float
    key: str
    info: Any = None
    born: int = 0  # generation the individual was created in


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_fitness: float
    median_fitness: float
    evaluated: int  # new individuals evaluated this generation
    duplicates: int  # children skipped because they were already in the population
    failed_mutations: int
    elapsed_s: float
    crossovers: int = 0  # children produced by crossover
    failed_crossovers: int = 0  # crossovers that found no valid child (fell back to mutation)


@dataclass
class GAResult[G]:
    best: Individual[G]
    population: list[Individual[G]]
    history: list[GenerationStats] = field(default_factory=list)
    stopped_because: str = ""


class Problem[G](Protocol):
    def initial_population(self, size: int, rng: np.random.Generator) -> list[G]: ...

    def mutate(self, genome: G, rng: np.random.Generator) -> G | None: ...

    def crossover(self, a: G, b: G, rng: np.random.Generator) -> G | None: ...

    def evaluate(self, genomes: list[G]) -> list[tuple[float, Any]]: ...

    def key(self, genome: G) -> str: ...


GenerationCallback = Callable[[GenerationStats, list[Individual[Any]]], None]


def _evaluate[G](
    problem: Problem[G], genomes: list[G], keys: list[str], gen: int
) -> list[Individual[G]]:
    scored = problem.evaluate(genomes)
    return [
        Individual(g, fitness, k, info, gen)
        for g, k, (fitness, info) in zip(genomes, keys, scored, strict=True)
    ]


def run[G](
    problem: Problem[G], config: GAConfig, on_generation: GenerationCallback | None = None
) -> GAResult[G]:
    rng = np.random.default_rng(config.seed)
    start = time.perf_counter()
    genomes = problem.initial_population(config.population_size, rng)
    if not genomes:
        raise ValueError("initial population is empty")
    population = _evaluate(problem, genomes, [problem.key(g) for g in genomes], 0)
    population.sort(key=lambda ind: ind.fitness)
    best = population[0]
    history: list[GenerationStats] = []

    def record(
        gen: int, evaluated: int, dups: int, failed: int, t0: float, crossed: int = 0,
        failed_x: int = 0,
    ) -> None:  # fmt: skip
        stats = GenerationStats(
            generation=gen,
            best_fitness=best.fitness,
            median_fitness=statistics.median(ind.fitness for ind in population),
            evaluated=evaluated,
            duplicates=dups,
            failed_mutations=failed,
            elapsed_s=time.perf_counter() - t0,
            crossovers=crossed,
            failed_crossovers=failed_x,
        )
        history.append(stats)
        if on_generation:
            on_generation(stats, population)

    record(0, len(population), 0, 0, start)
    stall = 0
    stopped = "generations"
    for gen in range(1, config.generations + 1):
        if config.target_fitness is not None and best.fitness <= config.target_fitness:
            stopped = "target_fitness"
            break
        if config.stall_generations is not None and stall >= config.stall_generations:
            stopped = "stalled"
            break

        t0 = time.perf_counter()
        elites = population[: config.elite_count]
        taken = {ind.key for ind in population}
        children: list[G] = []
        child_keys: list[str] = []
        dups = failed = crossed = failed_x = 0
        while len(children) < config.population_size - len(elites):
            parent = tournament(population, config.tournament_size, rng)
            child, key = parent.genome, parent.key
            for _ in range(config.max_child_attempts):
                candidate: G | None = None
                if config.crossover_rate and rng.random() < config.crossover_rate:
                    mate = tournament(population, config.tournament_size, rng)
                    candidate = problem.crossover(parent.genome, mate.genome, rng)
                    if candidate is None:
                        failed_x += 1
                    else:
                        crossed += 1
                        if rng.random() < config.mutate_after_crossover:
                            candidate = problem.mutate(candidate, rng) or candidate
                if candidate is None:
                    candidate = problem.mutate(parent.genome, rng)
                if candidate is None:
                    failed += 1
                    continue
                child, key = candidate, problem.key(candidate)
                if key not in taken:
                    break
                dups += 1
            taken.add(key)
            children.append(child)
            child_keys.append(key)

        offspring = _evaluate(problem, children, child_keys, gen)
        population = sorted([*elites, *offspring], key=lambda ind: ind.fitness)
        if population[0].fitness < best.fitness:
            best = population[0]
            stall = 0
        else:
            stall += 1
        record(gen, len(offspring), dups, failed, t0, crossed, failed_x)

    return GAResult(best, population, history, stopped)
