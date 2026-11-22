"""Command-line entry point: ``acd run``, ``ui``, ``bench``, ``eval``, ``plot``, ``testbench``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from acd.circuit.netlist import from_subckt
from acd.evaluation.fitness import FitnessWeights, score
from acd.evaluation.metrics import BiasRules, summarize
from acd.evaluation.stages import evaluate
from acd.evaluation.targets import load_target
from acd.evaluation.testbench import TestbenchConfig, build_testbench
from acd.experiment import load_config, run_experiment
from acd.sim.ngspice import Ngspice


def _testbench_config(
    config_path: Path | None,
) -> tuple[TestbenchConfig, FitnessWeights, BiasRules]:
    if config_path is None:
        return TestbenchConfig(), FitnessWeights(), BiasRules()
    config = load_config(config_path)
    return config.testbench, config.weights, config.bias


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ga_changes = {
        k: v
        for k, v in {
            "generations": args.generations,
            "population_size": args.population,
            "seed": args.seed,
        }.items()
        if v is not None
    }
    config = dataclasses.replace(config, ga=dataclasses.replace(config.ga, **ga_changes))
    if args.workers is not None:
        config = dataclasses.replace(config, workers=args.workers)
    run_experiment(config, out_root=args.out, log=None if args.quiet else print)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    circuit = from_subckt(args.circuit.read_text())
    tb, weights, rules = _testbench_config(args.config)
    evaluation = evaluate(circuit, Ngspice.get(), tb, rules)
    print(circuit.describe())
    print()
    if not evaluation.ok:
        print(f"FAILED at {evaluation.failed_stage}: {evaluation.reason}")
        for v in evaluation.violations:
            print(f"  {v}")
        return 1
    assert evaluation.op is not None and evaluation.response is not None
    op, s = evaluation.op, summarize(evaluation.response)
    print(f"power        {op.power_w * 1e3:.4g} mW")
    print(f"VOUT (DC)    {op.vout_dc:.4g} V")
    print(f"bias score   {op.bias_score:.2f}")
    for d in op.devices:
        print(f"  {d.component_id:>6} {d.kind:<5} {d.region:<8} {d.current_a * 1e6:9.3g} uA")
    print(f"LF gain      {s.lf_gain_db:.3f} dB")
    print(f"peak gain    {s.peak_gain_db:.3f} dB at {s.f_peak_hz:.4g} Hz")
    print(f"-3 dB        {s.f_3db_hz:.4g} Hz" if s.f_3db_hz else "-3 dB        (none in range)")
    print(f"unity gain   {s.f_unity_hz:.4g} Hz" if s.f_unity_hz else "unity gain   (none)")
    if args.target:
        target = load_target(args.target)
        fitness, objs = score(circuit, evaluation, target, weights)
        assert objs is not None
        print(f"target error {objs.response_error_db:.4g} dB  ({target.name})")
        print(f"fitness      {fitness:.4g}")
    return 0


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from acd.bench import load_suite, run_suite

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    only = args.only.split(",") if args.only else None
    out_dir, _ = run_suite(
        load_suite(args.suite), args.variant, args.out, seeds, args.time_limit, args.workers, only
    )
    print((out_dir / "summary.md").read_text())
    return 0


def _cmd_bench_compare(args: argparse.Namespace) -> int:
    from acd.bench import compare

    print(compare(args.dirs), end="")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from acd.ui import serve

    serve(args.port, args.host, args.workers, args.out, args.open)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from acd.plot import plot_run

    print(plot_run(args.run_dir, args.output))
    return 0


def _cmd_testbench(args: argparse.Namespace) -> int:
    circuit = from_subckt(args.circuit.read_text())
    tb, _, _ = _testbench_config(args.config)
    sys.stdout.write(build_testbench(circuit, tb).netlist)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acd", description="Evolve analog circuit topologies with a genetic algorithm."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a GA experiment from a TOML config")
    p.add_argument("config", type=Path)
    p.add_argument("--generations", type=int)
    p.add_argument("--population", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, help="0 = auto, 1 = no worker processes")
    p.add_argument("--out", type=Path, default=Path("runs"), help="output root (default: runs)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_run)

    bench = sub.add_parser("bench", help="run or compare benchmark suites").add_subparsers(
        dest="bench_command", required=True
    )
    p = bench.add_parser("run", help="run a search variant over every benchmark and seed")
    p.add_argument("suite", type=Path)
    p.add_argument("--variant", type=Path, required=True, help="variant TOML (search settings)")
    p.add_argument("--seeds", help="comma-separated, e.g. 1,2,3 (default: the suite's)")
    p.add_argument("--time-limit", type=float, help="seconds per run (default: the suite's)")
    p.add_argument("--workers", type=int)
    p.add_argument("--only", help="comma-separated benchmark names")
    p.add_argument("--out", type=Path, default=Path("results/bench"))
    p.set_defaults(func=_cmd_bench_run)
    p = bench.add_parser("compare", help="scoreboard across benchmark output directories")
    p.add_argument("dirs", nargs="+", type=Path)
    p.set_defaults(func=_cmd_bench_compare)

    p = sub.add_parser("ui", help="web UI: draw a target response and search for a circuit")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="default: only this machine")
    p.add_argument("--workers", type=int, default=0, help="0 = auto")
    p.add_argument("--out", type=Path, default=Path("runs/ui"), help="where runs are saved")
    p.add_argument("--open", action="store_true", help="open the page in a browser")
    p.set_defaults(func=_cmd_ui)

    p = sub.add_parser("eval", help="simulate one circuit (.subckt file) and print its metrics")
    p.add_argument("circuit", type=Path)
    p.add_argument("--target", type=Path, help="target TOML to score against")
    p.add_argument("--config", type=Path, help="experiment config for test-circuit settings")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("plot", help="plot a run's best response against its target (PNG)")
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--output", type=Path, help="default: <run_dir>/response.png")
    p.set_defaults(func=_cmd_plot)

    p = sub.add_parser("testbench", help="print the complete test-circuit netlist for a circuit")
    p.add_argument("circuit", type=Path)
    p.add_argument("--config", type=Path, help="experiment config for test-circuit settings")
    p.set_defaults(func=_cmd_testbench)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
