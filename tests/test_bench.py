import json
from pathlib import Path

import pytest

from acd.bench import (
    BenchResult,
    benchmark_config,
    compare,
    load_suite,
    load_variant,
    run_suite,
)
from acd.cli import main

ROOT = Path(__file__).parent.parent


def test_standard_suite_loads() -> None:
    suite = load_suite(ROOT / "benchmarks" / "suite.toml")
    assert len(suite.benchmarks) == 10
    assert all(b.target.exists() for b in suite.benchmarks)
    assert suite.testbench == {"points_per_decade": 50}
    for variant in (ROOT / "benchmarks" / "variants").glob("*.toml"):
        v = load_variant(variant)
        for b in suite.benchmarks:
            benchmark_config(suite, b, v, seed=1, time_limit_s=60)


def write_suite(tmp: Path, body: str) -> Path:
    p = tmp / "suite.toml"
    p.write_text(body)
    return p


LOWPASS = str(ROOT / "targets" / "lowpass_1k.toml")
CURVE = str(ROOT / "targets" / "pink_noise.toml")


def test_settings_merge_order(tmp_path: Path) -> None:
    suite = load_suite(
        write_suite(
            tmp_path,
            f'name = "t"\n[testbench]\nvdd = 5.0\nr_load = 1e4\n'
            f'[[benchmark]]\nname = "a"\ntarget = "{LOWPASS}"\n'
            "[benchmark.testbench]\nr_load = 1e3\n",
        )
    )
    variant = {"testbench": {"vdd": 1.8, "c_load": 2e-12}, "ga": {"population_size": 10}}
    config = benchmark_config(suite, suite.benchmarks[0], variant, seed=7, time_limit_s=5)
    tb = config.testbench
    assert (tb.vdd, tb.r_load, tb.c_load) == (5.0, 1e3, 2e-12)  # variant < suite < benchmark
    assert config.ga.seed == 7 and config.ga.time_limit_s == 5
    assert config.ga.stall_generations is None and config.ga.population_size == 10
    assert config.name == "a"


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('name = "t"\n', "no \\[\\[benchmark\\]\\]"),
        (f'typo = 1\n[[benchmark]]\nname = "a"\ntarget = "{LOWPASS}"\n', "typo"),
        (
            f'[[benchmark]]\nname = "a"\ntarget = "{LOWPASS}"\n'
            f'[[benchmark]]\nname = "a"\ntarget = "{LOWPASS}"\n',
            "unique",
        ),
    ],
)
def test_bad_suites(tmp_path: Path, body: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_suite(write_suite(tmp_path, body))


def test_variant_cannot_set_target(tmp_path: Path) -> None:
    p = tmp_path / "v.toml"
    p.write_text('target = "x.toml"\n')
    with pytest.raises(ValueError, match="target"):
        load_variant(p)


@pytest.mark.ngspice
def test_tiny_suite_run(tmp_path: Path) -> None:
    suite = load_suite(
        write_suite(
            tmp_path,
            f'name = "tiny"\nseeds = [1]\ntime_limit_s = 1.0\n'
            f'[[benchmark]]\nname = "lp"\ntarget = "{LOWPASS}"\n'
            f'[[benchmark]]\nname = "pink"\ntarget = "{CURVE}"\n',
        )
    )
    variant = tmp_path / "small.toml"
    variant.write_text('workers = 1\n[ga]\npopulation_size = 8\n[mutation]\nkinds = ["R", "C"]\n')
    out_dir, results = run_suite(suite, variant, tmp_path / "out", log=None)
    assert [r.benchmark for r in results] == ["lp", "pink"]
    assert (
        isinstance(results[0].met, bool) and results[1].met is None
    )  # masks pass/fail; curves don't
    assert len((out_dir / "results.jsonl").read_text().splitlines()) == 2
    summary = (out_dir / "summary.md").read_text()
    assert "| lp |" in summary and "| pink |" in summary
    assert (out_dir / "variant.toml").exists()
    pytest.importorskip("matplotlib")
    assert (out_dir / "lp.png").exists()


def fake_results(d: Path, rows: list[tuple[str, float, bool | None]]) -> Path:
    d.mkdir(parents=True)
    with (d / "results.jsonl").open("w") as fh:
        for i, (name, err, met) in enumerate(rows):
            r = BenchResult(name, i, err, err, 3, 1e-3, met, 10, 100, str(d))
            fh.write(json.dumps(r.__dict__) + "\n")
    return d


def test_compare_scoreboard(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = fake_results(tmp_path / "standard_baseline_1", [("lp", 2.0, False), ("lp", 0.0, True)])
    b = fake_results(tmp_path / "standard_sizing_1", [("lp", 0.0, True), ("lp", 0.0, True)])
    table = compare([a, b])
    assert "| Benchmark | baseline | sizing |" in table
    assert "| lp | 1.00 (1/2 met) | 0.00 (2/2 met) |" in table
    assert main(["bench", "compare", str(a), str(b)]) == 0
    assert "1/2 met" in capsys.readouterr().out
