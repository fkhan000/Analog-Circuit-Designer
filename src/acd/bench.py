"""Benchmark suites: run one search configuration across many targets and seeds, then score it.

A *suite* (``benchmarks/suite.toml``) lists benchmarks: a target plus any test-circuit settings
the benchmark needs (such as a source resistance). A *variant* (``benchmarks/variants/*.toml``)
holds the search settings being evaluated: the ``[ga]``, ``[mutation]``, ``[crossover]``,
``[sizing]``, ``[weights]`` and ``[bias]`` sections of an experiment config. Every benchmark and
seed gets the same time limit, so variants are compared at equal cost.

Output: ``<out>/<suite>_<variant>_<timestamp>/`` with one run directory per benchmark and seed,
``results.jsonl`` (one line per run), ``summary.md`` (scoreboard) and ``<benchmark>.png`` (the best
run's response against the target).
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import math
import shutil
import statistics
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acd.evaluation.targets import MaskTarget
from acd.experiment import ExperimentConfig, config_from_dict, run_experiment

# A mask counts as met when the response stays within it to this RMS error (dB).
MASK_MET_DB = 0.05


@dataclass(frozen=True)
class Benchmark:
    name: str
    target: Path
    testbench: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class Suite:
    name: str
    benchmarks: tuple[Benchmark, ...]
    seeds: tuple[int, ...] = (1, 2, 3)
    time_limit_s: float = 60.0
    testbench: dict[str, Any] = field(default_factory=dict)  # defaults for every benchmark


@dataclass(frozen=True)
class BenchResult:
    benchmark: str
    seed: int
    error_db: float
    fitness: float
    parts: int
    power_w: float
    met: bool | None  # None for curve targets, which have no pass/fail
    generations: int
    simulations: int
    run_dir: str


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    unknown = set(data) - {"name", "seeds", "time_limit_s", "testbench", "benchmark"}
    if unknown:
        raise ValueError(f"unknown keys in suite {path}: {sorted(unknown)}")
    benchmarks = []
    for b in data.get("benchmark", ()):
        extra = set(b) - {"name", "target", "testbench", "note"}
        if extra:
            raise ValueError(f"unknown keys in benchmark {b.get('name')!r}: {sorted(extra)}")
        target = Path(b["target"])
        benchmarks.append(
            Benchmark(
                name=str(b["name"]),
                target=target if target.is_absolute() else (path.parent / target).resolve(),
                testbench=dict(b.get("testbench", {})),
                note=str(b.get("note", "")),
            )
        )
    if not benchmarks:
        raise ValueError(f"suite {path} has no [[benchmark]] entries")
    names = [b.name for b in benchmarks]
    if len(set(names)) != len(names):
        raise ValueError("benchmark names must be unique")
    return Suite(
        name=str(data.get("name", path.stem)),
        benchmarks=tuple(benchmarks),
        seeds=tuple(int(s) for s in data.get("seeds", (1, 2, 3))),
        time_limit_s=float(data.get("time_limit_s", 60.0)),
        testbench=dict(data.get("testbench", {})),
    )


def load_variant(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    forbidden = set(data) & {"target", "name", "seed_circuits"}
    if forbidden:
        raise ValueError(f"variant {path} must not set {sorted(forbidden)} (benchmarks do)")
    return data


def benchmark_config(
    suite: Suite, bench: Benchmark, variant: dict[str, Any], seed: int, time_limit_s: float
) -> ExperimentConfig:
    """The experiment config for one benchmark, variant and seed.

    Test-circuit settings come from the variant, then the suite, then the benchmark (later wins).
    """
    data = copy.deepcopy(variant)
    data["name"] = bench.name
    data["target"] = str(bench.target)
    data["testbench"] = {**data.get("testbench", {}), **suite.testbench, **bench.testbench}
    config = config_from_dict(data, bench.target.parent)
    ga = dataclasses.replace(
        config.ga, seed=seed, time_limit_s=time_limit_s, generations=10**7, stall_generations=None
    )
    return dataclasses.replace(config, ga=ga)


def run_suite(
    suite: Suite,
    variant_path: str | Path,
    out_root: Path = Path("results/bench"),
    seeds: Sequence[int] | None = None,
    time_limit_s: float | None = None,
    workers: int | None = None,
    only: Sequence[str] | None = None,
    log: Callable[[str], None] | None = print,
) -> tuple[Path, list[BenchResult]]:
    variant_path = Path(variant_path)
    variant = load_variant(variant_path)
    if workers is not None:
        variant["workers"] = workers
    seeds = tuple(seeds or suite.seeds)
    limit = time_limit_s or suite.time_limit_s
    benchmarks = [b for b in suite.benchmarks if not only or b.name in only]
    if only and len(benchmarks) != len(set(only)):
        known = {b.name for b in suite.benchmarks}
        raise ValueError(f"unknown benchmarks: {sorted(set(only) - known)}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / f"{suite.name}_{variant_path.stem}_{stamp}"
    out_dir.mkdir(parents=True)
    shutil.copy(variant_path, out_dir / "variant.toml")
    if log:
        total = len(benchmarks) * len(seeds) * limit / 60
        log(f"benchmark output: {out_dir}  (about {total:.0f} min)")

    results: list[BenchResult] = []
    with (out_dir / "results.jsonl").open("w") as fh:
        for bench in benchmarks:
            for seed in seeds:
                config = benchmark_config(suite, bench, variant, seed, limit)
                outcome = run_experiment(config, out_root=out_dir / bench.name, log=None)
                result = _result(bench.name, seed, config, outcome.run_dir)
                results.append(result)
                fh.write(json.dumps(dataclasses.asdict(result)) + "\n")
                fh.flush()
                if log:
                    met = "" if result.met is None else ("  met" if result.met else "  not met")
                    log(
                        f"{bench.name:<16} seed {seed}: {result.error_db:7.3f} dB  "
                        f"{result.parts:2d} parts{met}"
                    )
    _write_summary(out_dir, suite, variant_path.stem, results, limit)
    return out_dir, results


def _result(name: str, seed: int, config: ExperimentConfig, run_dir: Path) -> BenchResult:
    s = json.loads((run_dir / "summary.json").read_text())
    objs = s.get("objectives") or {}
    error = float(objs.get("response_error_db", math.inf))
    met = error <= MASK_MET_DB if isinstance(config.target, MaskTarget) else None
    return BenchResult(
        benchmark=name,
        seed=seed,
        error_db=error,
        fitness=float(s["fitness"]),
        parts=int(objs.get("n_components", 0)),
        power_w=float(objs.get("power_w", math.nan)),
        met=met,
        generations=int(s["generations"]),
        simulations=int(s["simulated"]),
        run_dir=str(run_dir),
    )


def _write_summary(
    out_dir: Path, suite: Suite, variant: str, results: list[BenchResult], limit: float
) -> None:
    lines = [
        f"# {suite.name} suite, variant `{variant}`",
        "",
        f"{limit:g} s per run. Error is the RMS distance from the target in dB (0 = mask met).",
        "",
        "| Benchmark | Mean error | Best error | Met (masks) | Parts (best) | Plot |",
        "|---|---|---|---|---|---|",
    ]
    for name in dict.fromkeys(r.benchmark for r in results):
        runs = [r for r in results if r.benchmark == name]
        best = min(runs, key=lambda r: r.error_db)
        mets = [r.met for r in runs if r.met is not None]
        met = f"{sum(mets)}/{len(mets)}" if mets else "n/a"
        plot = Path(best.run_dir) / "response.png"
        link = ""
        if plot.exists():
            shutil.copy(plot, out_dir / f"{name}.png")
            link = f"[{name}.png]({name}.png)"
        mean = statistics.mean(r.error_db for r in runs)
        lines.append(
            f"| {name} | {mean:.3f} | {best.error_db:.3f} | {met} | {best.parts} | {link} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def load_results(out_dir: str | Path) -> list[BenchResult]:
    path = Path(out_dir) / "results.jsonl"
    return [BenchResult(**json.loads(line)) for line in path.read_text().splitlines()]


def compare(out_dirs: Sequence[str | Path]) -> str:
    """Scoreboard of mean error per benchmark across several benchmark output directories."""
    columns = []
    for d in map(Path, out_dirs):
        label = d.name.split("_")[1] if d.name.count("_") >= 2 else d.name
        columns.append((label, load_results(d)))
    names = list(dict.fromkeys(r.benchmark for _, rs in columns for r in rs))
    rows = [
        "| Benchmark | " + " | ".join(label for label, _ in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name in names:
        cells = []
        for _, rs in columns:
            errs = [r.error_db for r in rs if r.benchmark == name]
            mets = [r.met for r in rs if r.benchmark == name and r.met is not None]
            cell = f"{statistics.mean(errs):.2f}" if errs else "-"
            if mets:
                cell += f" ({sum(mets)}/{len(mets)} met)"
            cells.append(cell)
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"
