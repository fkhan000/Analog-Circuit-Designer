import json
from pathlib import Path

import pytest

from acd.circuit.devices import Kind
from acd.cli import main
from acd.experiment import config_from_dict, load_config, run_experiment
from acd.operators.mutation import DEFAULT_OP_WEIGHTS, MutationOp
from helpers import FIXTURES

ROOT = Path(__file__).parent.parent
MASK = {"type": "mask", "band": [{"f_lo": 10, "f_hi": 500, "min_db": -1, "max_db": 0.5}]}


def test_example_configs_load() -> None:
    for path in (ROOT / "configs").glob("*.toml"):
        config = load_config(path)
        assert config.source_path == path
        assert config.target.frequency_range()[0] > 0


def test_config_defaults_and_conversions(tmp_path: Path) -> None:
    config = config_from_dict(
        {
            "target": MASK,
            "mutation": {"kinds": ["r", "C"], "op_weights": {"move_pin": 0}},
            "seed_circuits": [str(FIXTURES / "rc_lowpass.cir")],
        },
        tmp_path,
    )
    assert config.mutation.kinds == (Kind.R, Kind.C)
    assert config.mutation.op_weights[MutationOp.MOVE_PIN] == 0
    assert (
        config.mutation.op_weights[MutationOp.CHANGE_VALUE]
        == (DEFAULT_OP_WEIGHTS[MutationOp.CHANGE_VALUE])
    )
    assert (config.testbench.f_start_hz, config.testbench.f_stop_hz) == (5, 1000)
    assert len(config.seed_circuits) == 1
    assert config.bias.min_current_a == 100e-9


def test_bias_section(tmp_path: Path) -> None:
    config = config_from_dict({"target": MASK, "bias": {"min_current_a": 1e-6}}, tmp_path)
    assert config.bias.min_current_a == 1e-6


def test_unknown_keys_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="generatoins"):
        config_from_dict({"target": MASK, "ga": {"generatoins": 5}}, tmp_path)
    with pytest.raises(ValueError):
        config_from_dict({"target": MASK, "typo": 1}, tmp_path)
    with pytest.raises(ValueError):
        config_from_dict({}, tmp_path)


@pytest.mark.ngspice
@pytest.mark.parametrize("workers", [1, 2])
def test_small_run_writes_outputs(tmp_path: Path, workers: int) -> None:
    config = config_from_dict(
        {
            "name": "tiny run",
            "target": MASK,
            "workers": workers,
            "ga": {"population_size": 12, "generations": 3, "seed": 2, "crossover_rate": 0.5},
            "crossover": {"max_region_size": 3},
            "mutation": {"kinds": ["R", "C"]},
        },
        tmp_path,
    )
    outcome = run_experiment(config, out_root=tmp_path / "runs", log=None)
    d = outcome.run_dir
    assert d.name.endswith("_tiny-run")
    for name in ("meta.json", "log.jsonl", "best.cir", "best_testbench.cir", "summary.json"):
        assert (d / name).exists(), name
    lines = (d / "log.jsonl").read_text().splitlines()
    assert len(lines) == 4
    first, last = json.loads(lines[0]), json.loads(lines[-1])
    assert last["best_fitness"] <= first["best_fitness"]
    assert "crossover_rejections" in last and "crossovers" in last
    summary = json.loads((d / "summary.json").read_text())
    assert summary["fitness"] == pytest.approx(outcome.result.best.fitness)
    assert len(list((d / "top").glob("rank*.cir"))) == 5
    assert json.loads((d / "target.json").read_text())["type"] == "mask"
    pytest.importorskip("matplotlib")
    png = d / "response.png"  # written automatically at the end of the run
    assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
    if workers == 1:
        png.unlink()
        assert main(["plot", str(d)]) == 0  # and on demand
        assert png.exists()


@pytest.mark.ngspice
def test_cli_eval_and_testbench(capsys: pytest.CaptureFixture[str]) -> None:
    target = ROOT / "targets" / "lowpass_1k.toml"
    assert main(["eval", str(FIXTURES / "rc_lowpass.cir"), "--target", str(target)]) == 0
    out = capsys.readouterr().out
    assert "-3 dB" in out and "target error" in out
    assert main(["testbench", str(FIXTURES / "ce_amp.cir")]) == 0
    assert "xdut vin vout vdd dut" in capsys.readouterr().out


@pytest.mark.ngspice
def test_cli_eval_reports_invalid_circuit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.cir"
    bad.write_text(".subckt dut VIN VOUT VDD\nR1 VIN 0 1k\n.ends\n")
    assert main(["eval", str(bad)]) == 1
    assert "port_unconnected" in capsys.readouterr().out


@pytest.mark.ngspice
def test_sizing_and_speciation_are_reproducible(tmp_path: Path) -> None:
    def run_with(workers: int) -> tuple[float, int, list[int]]:
        config = config_from_dict(
            {
                "target": MASK,
                "workers": workers,
                "mutation": {"kinds": ["R", "C"]},
                "sizing": {"enabled": True, "budget": 12},
                "ga": {"population_size": 8, "generations": 2, "seed": 4, "speciation": True},
            },
            tmp_path,
        )
        out = run_experiment(config, out_root=tmp_path / f"runs{workers}", log=None)
        log = [json.loads(x) for x in (out.run_dir / "log.jsonl").read_text().splitlines()]
        sims = json.loads((out.run_dir / "summary.json").read_text())["simulated"]
        assert all(rec["species"] >= 1 for rec in log)
        return out.result.best.fitness, sims, [rec["sized"] for rec in log]

    serial, parallel = run_with(1), run_with(2)
    assert serial == parallel  # sizing is seeded from each circuit, not from the worker
    _, sims, sized = serial
    assert sized[0] == 8  # the whole initial population is sized
    assert sims > 8 + 2 * 7  # far more simulations than individuals
