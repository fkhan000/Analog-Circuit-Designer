import math
from pathlib import Path

import numpy as np
import pytest

from acd.circuit.validity import Violation, ViolationKind
from acd.evaluation.fitness import (
    GRAPH_INVALID_FITNESS,
    SIM_FAILED_FITNESS,
    FitnessWeights,
    Objectives,
    scalar_fitness,
)
from acd.evaluation.metrics import OperatingPoint, Response, summarize
from acd.evaluation.stages import Evaluation, Stage
from acd.evaluation.targets import (
    Band,
    CurveTarget,
    MaskTarget,
    load_target,
    sine_curve,
    target_from_dict,
    target_to_dict,
)


def first_order_lowpass(fc: float, gain_db: float = 0.0) -> Response:
    f = np.geomspace(1, 1e6, 241)
    h = 10 ** (gain_db / 20) / (1 + 1j * f / fc)
    return Response(f, h.astype(np.complex128))


def test_summary_of_known_response() -> None:
    s = summarize(first_order_lowpass(1e3, gain_db=20))
    assert s.lf_gain_db == pytest.approx(20, abs=0.01)
    assert s.f_3db_hz == pytest.approx(1e3, rel=0.01)
    assert s.f_unity_hz == pytest.approx(1e4, rel=0.02)  # 20 dB/decade rolloff


def test_mask_met_gives_zero_error() -> None:
    mask = MaskTarget((Band(1, 100, -1, 1), Band(1e5, 1e6, max_db=-35)))
    assert mask.error_db(first_order_lowpass(1e3)) == pytest.approx(0, abs=1e-9)


def test_mask_violation_is_positive_and_grows() -> None:
    mask = MaskTarget((Band(1e5, 1e6, max_db=-60),))
    near = mask.error_db(first_order_lowpass(1e3))
    far = mask.error_db(first_order_lowpass(1e4))
    assert 0 < near < far


def test_curve_error_is_rms_offset() -> None:
    resp = first_order_lowpass(1e3)
    f = (10.0, 100.0, 1e4)
    exact = tuple(float(g) for g in resp.gain_db_at(np.array(f)))
    shifted = CurveTarget(f, tuple(g + 3 for g in exact), (1.0, 1.0, 1.0))
    assert shifted.error_db(resp) == pytest.approx(3.0, abs=1e-6)


def test_load_mask_target(tmp_path: Path) -> None:
    p = tmp_path / "t.toml"
    p.write_text(
        'name = "lp"\ntype = "mask"\n'
        "[[band]]\nf_lo = 10\nf_hi = 1e3\nmin_db = -1\nmax_db = 1\n"
        "[[band]]\nf_lo = 1e4\nf_hi = 1e5\nmax_db = -20\nweight = 2\n"
    )
    t = load_target(p)
    assert isinstance(t, MaskTarget) and t.name == "lp"
    assert t.bands[1] == Band(1e4, 1e5, -math.inf, -20, 2.0)
    assert t.frequency_range() == (10, 1e5)


def test_load_curve_target_from_csv(tmp_path: Path) -> None:
    (tmp_path / "c.csv").write_text("freq_hz,gain_db\n100,20\n1000,20\n10000,17\n")
    p = tmp_path / "t.toml"
    p.write_text('type = "curve"\nfile = "c.csv"\n')
    t = load_target(p)
    assert isinstance(t, CurveTarget)
    assert t.gain_db == (20.0, 20.0, 17.0) and t.weight == (1.0, 1.0, 1.0)


def test_bad_target_type(tmp_path: Path) -> None:
    p = tmp_path / "t.toml"
    p.write_text('type = "wish"\n')
    with pytest.raises(ValueError):
        load_target(p)


def test_fitness_tiers() -> None:
    w = FitnessWeights()
    invalid = Evaluation(False, Stage.GRAPH, "x", (Violation(ViolationKind.DANGLING_NET, ("n1",)),))
    failed = Evaluation(False, Stage.SIMULATION, "no_convergence")
    ok = Evaluation(True, op=OperatingPoint(1e-3, 1.0))
    terrible = Objectives(response_error_db=500, power_w=0.1, n_components=10, bias_penalty=1)
    assert scalar_fitness(invalid, None, w) >= GRAPH_INVALID_FITNESS
    assert scalar_fitness(failed, None, w) == SIM_FAILED_FITNESS
    assert scalar_fitness(ok, terrible, w) < SIM_FAILED_FITNESS


def test_fitness_weights() -> None:
    ok = Evaluation(True)
    objs = Objectives(response_error_db=2.0, power_w=1e-3, n_components=4, bias_penalty=0.5)
    w = FitnessWeights(response=1, power_per_mw=0.1, per_component=0.05, bias=2)
    assert scalar_fitness(ok, objs, w) == pytest.approx(2.0 + 0.1 + 0.2 + 1.0)


def test_sine_target_on_log_axis() -> None:
    t = load_target(Path(__file__).parents[2] / "targets" / "sine_log.toml")
    assert isinstance(t, CurveTarget) and len(t.freq_hz) == 81
    f, g = np.array(t.freq_hz), np.array(t.gain_db)
    assert f[0] == pytest.approx(100) and f[-1] == pytest.approx(1e6)
    assert g.max() == pytest.approx(-2, abs=0.05) and g.min() == pytest.approx(-18, abs=0.05)
    # Two periods over four decades: peaks at 10^2.5 and 10^4.5 Hz.
    peaks = f[(g > np.roll(g, 1)) & (g > np.roll(g, -1))]
    assert peaks == pytest.approx([10**2.5, 10**4.5], rel=0.05)


def test_sine_target_on_linear_axis() -> None:
    t = sine_curve(1e3, 2e3, offset_db=0, amplitude_db=1, periods=1, axis="linear", points=5)
    assert t.freq_hz == pytest.approx((1000, 1250, 1500, 1750, 2000))
    assert t.gain_db == pytest.approx((0, 1, 0, -1, 0), abs=1e-12)


def test_sine_target_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sine_curve(1e3, 1e2, 0, 1, 1)
    with pytest.raises(ValueError):
        sine_curve(1e2, 1e3, 0, 1, 1, axis="sqrt")
    p = tmp_path / "t.toml"
    p.write_text('type = "sine"\nf_lo = 1\nf_hi = 10\namplitude_db = 1\nperiod = 2\n')
    with pytest.raises(ValueError, match="period"):
        load_target(p)


def test_target_dict_round_trip() -> None:
    mask = MaskTarget((Band(10, 100, max_db=-3), Band(1e3, 1e4, -1, 1, 2.0)), "m")
    d = target_to_dict(mask)
    assert "min_db" not in d["band"][0]  # unbounded limits are left out (JSON has no inf)
    assert target_from_dict(d) == mask
    curve = sine_curve(1e2, 1e4, -3, 2, 1.5, points=7, name="s")
    assert target_from_dict(target_to_dict(curve)) == curve
