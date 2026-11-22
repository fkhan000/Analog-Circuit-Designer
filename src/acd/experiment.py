"""Connect the generic GA to circuits: config loading, parallel evaluation and run output.

A run writes ``runs/<timestamp>_<name>/`` containing:

- ``config.toml``: copy of the config file; ``target.json``: the target as evaluated (sine
  targets written out as curves), so ``acd plot`` can draw it
- ``meta.json``: seed, git commit, versions, start time
- ``log.jsonl``: one JSON object per generation (fitness, objectives of the best circuit, why
  candidates failed evaluation, why mutations were rejected, cache hit rate)
- ``best.cir``: the best circuit as a ``.subckt``; ``best_testbench.cir``: a complete netlist that
  ngspice can run directly (it uses ngspice ``.save`` syntax, so LTspice needs edits);
  ``best_response.csv``: its frequency response
- ``summary.json``: final metrics; ``top/rank<N>.cir``: the best few distinct circuits
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import os
import platform
import shutil
import subprocess
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import acd
from acd.circuit.canonical import full_hash, topology_hash
from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit
from acd.circuit.netlist import from_subckt, to_subckt
from acd.evaluation.cache import EvaluationCache
from acd.evaluation.fitness import SIM_FAILED_FITNESS, FitnessWeights, Objectives, score
from acd.evaluation.metrics import BiasRules, summarize
from acd.evaluation.stages import Evaluation, Stage, evaluate
from acd.evaluation.targets import Target, load_target, target_from_dict, target_to_dict
from acd.evaluation.testbench import TestbenchConfig, build_testbench
from acd.operators.crossover import CrossoverConfig, subgraph_swap
from acd.operators.mutation import DEFAULT_OP_WEIGHTS, MutationConfig, MutationOp, mutate
from acd.operators.seed import initial_population
from acd.search.ga import GAConfig, GAResult, GenerationStats, Individual, Outcome, run
from acd.search.parallel import ParallelMap, in_worker_process
from acd.sim.backend import SimStatus
from acd.sim.ngspice import Ngspice, find_library
from acd.sizing.optimizer import SizingConfig, size

# ---------------------------------------------------------------------------- config


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    target: Target
    testbench: TestbenchConfig = field(default_factory=TestbenchConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    crossover: CrossoverConfig = field(default_factory=CrossoverConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    bias: BiasRules = field(default_factory=BiasRules)
    workers: int = 0  # 0 = one per CPU core minus one; 1 = evaluate in this process
    seed_circuits: tuple[Circuit, ...] = ()
    cache_size: int = 200_000
    source_path: Path | None = None  # config file, copied into the run directory


_TOP_LEVEL_KEYS = {
    "name", "target", "workers", "seed_circuits", "cache_size",
    "ga", "mutation", "crossover", "sizing", "testbench", "weights", "bias",
}  # fmt: skip


def _build[C](cls: type[C], table: Mapping[str, Any], section: str) -> C:
    assert dataclasses.is_dataclass(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(table) - names
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
    return cls(**table)


def config_from_dict(data: Mapping[str, Any], base_dir: Path) -> ExperimentConfig:
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")

    raw_target = data.get("target")
    if isinstance(raw_target, str):
        path = Path(raw_target)
        target = load_target(path if path.is_absolute() else base_dir / path)
    elif isinstance(raw_target, Mapping):
        target = target_from_dict(raw_target, base_dir)
    else:
        raise ValueError("config needs 'target': a target file path or a [target] table")

    mut = dict(data.get("mutation", {}))
    if "kinds" in mut:
        mut["kinds"] = tuple(Kind(str(k).upper()) for k in mut["kinds"])
    if "op_weights" in mut:
        user = {MutationOp(k): float(v) for k, v in mut["op_weights"].items()}
        mut["op_weights"] = {**DEFAULT_OP_WEIGHTS, **user}

    tb = dict(data.get("testbench", {}))
    lo, hi = target.frequency_range()
    tb.setdefault("f_start_hz", lo / 2)
    tb.setdefault("f_stop_hz", hi * 2)

    seeds = tuple(
        from_subckt((base_dir / p if not Path(p).is_absolute() else Path(p)).read_text())
        for p in data.get("seed_circuits", ())
    )
    return ExperimentConfig(
        name=str(data.get("name", target.name)),
        target=target,
        testbench=_build(TestbenchConfig, tb, "testbench"),
        mutation=_build(MutationConfig, mut, "mutation"),
        crossover=_build(CrossoverConfig, data.get("crossover", {}), "crossover"),
        sizing=_build(SizingConfig, data.get("sizing", {}), "sizing"),
        ga=_build(GAConfig, data.get("ga", {}), "ga"),
        weights=_build(FitnessWeights, data.get("weights", {}), "weights"),
        bias=_build(BiasRules, data.get("bias", {}), "bias"),
        workers=int(data.get("workers", 0)),
        seed_circuits=seeds,
        cache_size=int(data.get("cache_size", 200_000)),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("rb") as fh:
        config = config_from_dict(tomllib.load(fh), path.parent)
    return dataclasses.replace(config, source_path=path)


# ------------------------------------------------------------------ evaluation


@dataclass(frozen=True)
class EvalSetup:
    """Everything a worker needs to score circuits (sent to each worker once)."""

    testbench: TestbenchConfig
    target: Target
    weights: FitnessWeights
    bias: BiasRules = field(default_factory=BiasRules)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    seed: int = 0  # mixed into each sizing run's random seed


@dataclass(frozen=True)
class Scored:
    fitness: float
    objectives: Objectives | None
    evaluation: Evaluation  # without the response arrays, to keep results small
    circuit: Circuit | None = None  # the circuit with tuned values, if sizing changed it
    simulations: int = 1
    start_fitness: float | None = None  # fitness before sizing (None if not sized)


_SETUP: EvalSetup | None = None


def _init_worker(setup: EvalSetup) -> None:
    global _SETUP
    _SETUP = setup
    Ngspice.get()


Task = tuple[Circuit, bool]  # (circuit, whether to size it)


def _score_once(circuit: Circuit, sim: Ngspice) -> tuple[float, Objectives | None, Evaluation]:
    assert _SETUP is not None, "worker not initialized"
    evaluation = evaluate(circuit, sim, _SETUP.testbench, _SETUP.bias)
    fitness, objs = score(circuit, evaluation, _SETUP.target, _SETUP.weights)
    return fitness, objs, evaluation


def _score(task: Task) -> Scored:
    assert _SETUP is not None, "worker not initialized"
    circuit, tune = task
    sim = Ngspice.get()
    fitness, objs, evaluation = _score_once(circuit, sim)
    result = Scored(fitness, objs, evaluation.slim())
    if tune and _SETUP.sizing.enabled and evaluation.failed_stage != Stage.GRAPH:
        tried: dict[Circuit, tuple[float, Objectives | None, Evaluation]] = {
            circuit: (fitness, objs, evaluation)
        }

        def objective(c: Circuit) -> float:
            if c not in tried:
                tried[c] = _score_once(c, sim)
            return tried[c][0]

        # Seeded from the circuit itself, so results don't depend on which worker runs it.
        rng = np.random.default_rng([int(full_hash(circuit)[:15], 16), _SETUP.seed])
        sized = size(circuit, objective, rng, _SETUP.sizing, start_fitness=fitness)
        best_f, best_objs, best_eval = tried[sized.circuit]
        result = Scored(
            best_f,
            best_objs,
            best_eval.slim(),
            circuit=sized.circuit if sized.circuit != circuit else None,
            simulations=len(tried),
            start_fitness=fitness,
        )
    if not sim.usable:
        if in_worker_process():
            os._exit(70)  # the pool replaces this worker with one that has a fresh ngspice
        raise RuntimeError("ngspice became unusable; use workers >= 2 to recover automatically")
    return result


def _crashed(task: Task) -> Scored:
    failed = Evaluation(False, Stage.SIMULATION, "worker_crashed", sim_status=SimStatus.FATAL)
    return Scored(SIM_FAILED_FITNESS, None, failed)


class Evaluator:
    """Scores circuits through the cache, then the worker pool (sizing new topologies there)."""

    def __init__(self, setup: EvalSetup, workers: int, cache_size: int) -> None:
        self.cache: EvaluationCache[Scored] = EvaluationCache(cache_size)
        self.pool: ParallelMap[Task, Scored] = ParallelMap(
            _score, workers, _init_worker, (setup,), on_crash=_crashed
        )
        self.simulated = 0
        self.last_outcomes: Counter[str] = Counter()  # "ok" or failure reason, last batch
        self.last_sized = 0  # circuits sized in the last batch
        self.last_sizing_gain = 0.0  # mean fitness improvement from sizing, last batch

    def evaluate(
        self, circuits: Sequence[Circuit], tune: Sequence[bool] | None = None
    ) -> list[Scored]:
        tune = tune if tune is not None else [False] * len(circuits)
        keys = [full_hash(c) for c in circuits]
        results: list[Scored | None] = [
            self.cache.get(c, k) for c, k in zip(circuits, keys, strict=True)
        ]
        missing = [i for i, r in enumerate(results) if r is None]
        fresh = self.pool.map([(circuits[i], tune[i]) for i in missing])
        self.last_outcomes = Counter()
        gains = []
        for i, scored in zip(missing, fresh, strict=True):
            self.cache.put(circuits[i], scored, keys[i])
            if scored.circuit is not None:  # the tuned circuit is now known too
                self.cache.put(scored.circuit, dataclasses.replace(scored, circuit=None))
            results[i] = scored
            self.last_outcomes[scored.evaluation.reason or "ok"] += 1
            self.simulated += scored.simulations
            if scored.start_fitness is not None and scored.start_fitness < SIM_FAILED_FITNESS:
                gains.append(scored.start_fitness - scored.fitness)
        self.last_sized = sum(1 for i in missing if tune[i])
        self.last_sizing_gain = sum(gains) / len(gains) if gains else 0.0
        done = [r for r in results if r is not None]
        assert len(done) == len(circuits)
        return done

    def close(self) -> None:
        self.pool.close()


class CircuitProblem:
    """The GA ``Problem`` for circuits."""

    def __init__(self, config: ExperimentConfig, evaluator: Evaluator) -> None:
        self.config = config
        self.evaluator = evaluator
        self.ctx = config.testbench.port_context()
        self.rejections: Counter[str] = Counter()
        self.ops: Counter[str] = Counter()
        self.crossover_rejections: Counter[str] = Counter()
        self._to_size: set[str] = set()  # keys of new children whose values should be tuned

    def _mark(self, child: Circuit, parent: Circuit | None) -> None:
        sizing = self.config.sizing
        if not sizing.enabled:
            return
        if (
            sizing.when == "always"
            or parent is None
            or topology_hash(child) != topology_hash(parent)
        ):
            self._to_size.add(full_hash(child))

    def initial_population(self, size: int, rng: np.random.Generator) -> list[Circuit]:
        population = initial_population(
            size, rng, self.config.mutation, self.ctx, self.config.seed_circuits
        )
        for c in population:
            self._mark(c, None)
        return population

    def mutate(self, genome: Circuit, rng: np.random.Generator) -> Circuit | None:
        result = mutate(genome, rng, self.config.mutation, self.ctx)
        self.rejections.update(result.rejections)
        self.ops.update(str(op) for op in result.ops)
        if result.circuit is not None:
            self._mark(result.circuit, genome)
        return result.circuit

    def crossover(self, a: Circuit, b: Circuit, rng: np.random.Generator) -> Circuit | None:
        result = subgraph_swap(
            a, b, rng, self.config.crossover, self.config.mutation.max_components, self.ctx
        )
        self.crossover_rejections.update(result.rejections)
        if result.circuit is not None:
            self._mark(result.circuit, None)
        return result.circuit

    def evaluate(self, genomes: list[Circuit]) -> list[Outcome[Circuit]]:
        tune = [full_hash(g) in self._to_size for g in genomes]
        self._to_size.clear()
        return [Outcome(s.fitness, s, s.circuit) for s in self.evaluator.evaluate(genomes, tune)]

    def key(self, genome: Circuit) -> str:
        return full_hash(genome)


# ---------------------------------------------------------------------- running


@dataclass
class RunOutcome:
    run_dir: Path
    result: GAResult[Circuit]
    best_evaluation: Evaluation  # full evaluation of the best circuit, with its response


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _make_run_dir(out_root: Path, name: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    run_dir = out_root / f"{stamp}_{safe}"
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = out_root / f"{stamp}_{safe}_{n}"
    run_dir.mkdir(parents=True)
    return run_dir


def _objectives_dict(info: Any) -> dict[str, Any] | None:
    if isinstance(info, Scored) and info.objectives is not None:
        return asdict(info.objectives)
    return None


def run_experiment(
    config: ExperimentConfig,
    out_root: Path = Path("runs"),
    log: Callable[[str], None] | None = print,
) -> RunOutcome:
    run_dir = _make_run_dir(out_root, config.name)
    if config.source_path is not None:
        shutil.copy(config.source_path, run_dir / "config.toml")
    workers = config.workers if config.workers > 0 else ParallelMap.default_workers()
    meta = {
        "name": config.name,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": config.ga.seed,
        "git_commit": _git_commit(),
        "acd_version": acd.__version__,
        "python": platform.python_version(),
        "ngspice_library": find_library(),
        "workers": workers,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "target.json").write_text(json.dumps(target_to_dict(config.target)) + "\n")

    setup = EvalSetup(
        config.testbench, config.target, config.weights, config.bias, config.sizing, config.ga.seed
    )
    evaluator = Evaluator(setup, workers, config.cache_size)
    problem = CircuitProblem(config, evaluator)
    if log:
        log(f"run directory: {run_dir}  (workers: {workers})")

    with (run_dir / "log.jsonl").open("w") as log_file:

        def on_generation(stats: GenerationStats, population: list[Individual[Any]]) -> None:
            best = population[0]
            record = {
                **asdict(stats),
                "best_components": len(best.genome),
                "best_objectives": _objectives_dict(best.info),
                "evaluation_outcomes": dict(evaluator.last_outcomes),
                "mutation_rejections": dict(problem.rejections),
                "mutation_ops": dict(problem.ops),
                "crossover_rejections": dict(problem.crossover_rejections),
                "cache_hit_rate": evaluator.cache.hit_rate,
                "simulated_total": evaluator.simulated,
                "sized": evaluator.last_sized,
                "sizing_gain": evaluator.last_sizing_gain,
            }
            problem.rejections.clear()
            problem.ops.clear()
            problem.crossover_rejections.clear()
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            if log:
                log(
                    f"gen {stats.generation:4d}  best {stats.best_fitness:10.4g}  "
                    f"median {stats.median_fitness:10.4g}  parts {len(best.genome):2d}  "
                    f"new {stats.evaluated:4d}  sized {evaluator.last_sized:4d}  "
                    f"sims {evaluator.simulated:8d}  "
                    f"{stats.elapsed_s:6.2f}s"
                )

        try:
            result = run(problem, config.ga, on_generation)
        finally:
            evaluator.close()

    best_eval = _write_results(run_dir, result, config, evaluator)
    if log:
        log(f"stopped: {result.stopped_because}; best fitness {result.best.fitness:.4g}")
        log(f"best circuit ({run_dir / 'best.cir'}):\n{result.best.genome.describe()}")
    return RunOutcome(run_dir, result, best_eval)


def _circuit_file(circuit: Circuit, header: Sequence[str]) -> str:
    return "".join(f"* {line}\n" for line in header) + to_subckt(circuit)


def _write_results(
    run_dir: Path, result: GAResult[Circuit], config: ExperimentConfig, evaluator: Evaluator
) -> Evaluation:
    best = result.best
    evaluation = evaluate(best.genome, Ngspice.get(), config.testbench, config.bias)
    fitness, objs = score(best.genome, evaluation, config.target, config.weights)

    header = [f"acd best circuit for '{config.name}'", f"fitness {fitness:.6g}"]
    if objs:
        header.append(
            f"response error {objs.response_error_db:.4g} dB, power {objs.power_w * 1e3:.4g} mW, "
            f"{objs.n_components} parts, bias penalty {objs.bias_penalty:.3g}"
        )
    (run_dir / "best.cir").write_text(_circuit_file(best.genome, header))
    (run_dir / "best_testbench.cir").write_text(
        build_testbench(best.genome, config.testbench).netlist
    )

    summary: dict[str, Any] = {
        "name": config.name,
        "target": config.target.name,
        "stopped_because": result.stopped_because,
        "generations": len(result.history) - 1,
        "simulated": evaluator.simulated,
        "cache_hit_rate": evaluator.cache.hit_rate,
        "pool_restarts": evaluator.pool.restarts,
        "fitness": fitness,
        "objectives": asdict(objs) if objs else None,
        "evaluation": {"ok": evaluation.ok, "reason": evaluation.reason},
        "circuit": best.genome.describe().splitlines(),
    }
    if evaluation.response is not None:
        resp = evaluation.response
        with (run_dir / "best_response.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["freq_hz", "gain_db", "phase_deg"])
            for row in zip(resp.freq_hz, resp.gain_db, resp.phase_deg, strict=True):
                writer.writerow([f"{v:.6g}" for v in row])
        summary["response"] = asdict(summarize(resp))
    if evaluation.op is not None:
        summary["operating_point"] = {
            "power_w": evaluation.op.power_w,
            "vout_dc": evaluation.op.vout_dc,
            "bias_score": evaluation.op.bias_score,
            "devices": [
                {"id": d.component_id, "kind": str(d.kind), "region": str(d.region),
                 "current_a": d.current_a, "gm_s": d.gm_s}
                for d in evaluation.op.devices
            ],
        }  # fmt: skip
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    top = run_dir / "top"
    top.mkdir()
    for rank, ind in enumerate(result.population[:5], start=1):
        text = _circuit_file(ind.genome, [f"rank {rank}, fitness {ind.fitness:.6g}"])
        (top / f"rank{rank}.cir").write_text(text)
    return evaluation
