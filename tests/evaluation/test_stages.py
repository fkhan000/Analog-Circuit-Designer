import math
import time

import numpy as np
import pytest

from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit
from acd.circuit.ports import GND, VIN, VOUT
from acd.evaluation.metrics import BiasRules, Region, summarize
from acd.evaluation.stages import Stage, evaluate
from acd.evaluation.testbench import TestbenchConfig, build_testbench
from acd.sim import Ngspice, SimResult, SimStatus
from helpers import load_fixture

CONFIG = TestbenchConfig()


@pytest.mark.ngspice
def test_rc_lowpass(ngspice: Ngspice) -> None:
    e = evaluate(load_fixture("rc_lowpass"), ngspice, CONFIG)
    assert e.ok, e.reason
    assert e.response is not None and e.op is not None
    s = summarize(e.response)
    assert s.lf_gain_db == pytest.approx(0, abs=0.05)
    assert s.f_3db_hz == pytest.approx(1 / (2 * math.pi * 1.6e3 * 100e-9), rel=0.02)
    assert e.op.devices == () and e.op.bias_score == 1.0


@pytest.mark.ngspice
def test_ce_amp(ngspice: Ngspice) -> None:
    e = evaluate(load_fixture("ce_amp"), ngspice, CONFIG)
    assert e.ok and e.op is not None and e.response is not None
    (q,) = e.op.devices
    assert q.region == Region.ACTIVE
    assert q.current_a == pytest.approx(0.93e-3, rel=0.05)
    assert e.op.vout_dc == pytest.approx(2.2, abs=0.1)
    assert e.op.power_w == pytest.approx(3.3 * 0.93e-3, rel=0.05)
    assert summarize(e.response).peak_gain_db == pytest.approx(19.5, abs=1.0)


@pytest.mark.ngspice
def test_ota_follower_biases_every_device(ngspice: Ngspice) -> None:
    e = evaluate(load_fixture("ota_follower"), ngspice, CONFIG)
    assert e.ok and e.op is not None and e.response is not None
    assert {d.kind for d in e.op.devices} == {Kind.NMOS, Kind.PMOS}
    assert all(d.region == Region.ACTIVE for d in e.op.devices)
    assert e.op.vout_dc == pytest.approx(CONFIG.vin_bias, abs=0.05)  # follows the input
    assert summarize(e.response).lf_gain_db == pytest.approx(0, abs=0.3)


@pytest.mark.ngspice
def test_lc_bandpass_resonance(ngspice: Ngspice) -> None:
    config = TestbenchConfig(points_per_decade=200, f_start_hz=100, f_stop_hz=1e6)
    e = evaluate(load_fixture("lc_bandpass"), ngspice, config)
    assert e.ok and e.response is not None
    s = summarize(e.response)
    assert s.f_peak_hz == pytest.approx(1 / (2 * math.pi * math.sqrt(10e-3 * 100e-9)), rel=0.02)
    assert s.peak_gain_db == pytest.approx(0, abs=0.1)  # tank impedance is infinite at f0
    assert e.response.gain_db_at(np.array([100.0]))[0] < -25  # inductor shorts low frequencies


@pytest.mark.ngspice
def test_diode_small_signal_resistance(ngspice: Ngspice) -> None:
    e = evaluate(load_fixture("diode_divider"), ngspice, CONFIG)
    assert e.ok and e.op is not None and e.response is not None
    vout = e.op.vout_dc
    assert 0.55 < vout < 0.75  # one diode drop
    # Diode current from KCL at VOUT, then rd = Vt / Id (+ the model's 1 ohm RS).
    i_d = (CONFIG.vin_bias - vout) / 10e3 + (CONFIG.vdd - vout) / 10e3 - vout / 1e6
    rd = 0.025852 / i_d + 1.0
    shunt = 1 / (1 / rd + 1 / 10e3)
    expected_db = 20 * math.log10(shunt / (10e3 + shunt))
    assert summarize(e.response).lf_gain_db == pytest.approx(expected_db, abs=0.3)


@pytest.mark.ngspice
def test_picoamp_device_is_starved(ngspice: Ngspice) -> None:
    e = evaluate(load_fixture("starved_amp"), ngspice, CONFIG)
    assert e.ok and e.op is not None
    states = {d.component_id: d for d in e.op.devices}
    assert states["4"].region == Region.ACTIVE and states["4"].current_a > 1e-7
    assert states["5"].region == Region.STARVED and states["5"].current_a < 1e-9
    assert e.op.bias_score == 0.5


@pytest.mark.ngspice
def test_min_current_is_configurable(ngspice: Ngspice) -> None:
    e = evaluate(load_fixture("starved_amp"), ngspice, CONFIG, BiasRules(min_current_a=0.0))
    assert e.op is not None and e.op.bias_score == 1.0  # the old behavior


@pytest.mark.ngspice
@pytest.mark.parametrize("name", ["supply_junction", "garbage_op"])
def test_junctions_across_sources_simulate_sanely(ngspice: Ngspice, name: str) -> None:
    # Both once broke ngspice: one hung the operating point, one returned -8e41 W "successfully".
    # BJT series resistances in the generic models bound the junction currents.
    start = time.perf_counter()
    e = evaluate(load_fixture(name), ngspice, CONFIG)
    assert time.perf_counter() - start < 0.5
    assert e.ok and e.op is not None and e.op.power_w >= 0


def fake_result(power_w: float, gain: float) -> SimResult:
    """Results for the RC low-pass fixture (no transistors) with chosen power and gain."""
    freq = np.array([10.0, 100.0, 1000.0])
    supply_current = -power_w / CONFIG.vdd  # current into VDD's + terminal
    op = {"vout": np.array([1.0]), "vdd#branch": np.array([supply_current]),
          "vin#branch": np.array([0.0])}  # fmt: skip
    ac = {"frequency": freq.astype(np.complex128), "vout": np.full(3, gain, np.complex128),
          "vin": np.ones(3, np.complex128)}  # fmt: skip
    return SimResult(SimStatus.OK, {"op1": op, "ac1": ac})


class Canned:
    def __init__(self, result: SimResult) -> None:
        self.result = result

    def run(self, netlist: str, timeout_s: float | None = None) -> SimResult:
        return self.result


@pytest.mark.parametrize(
    ("power_w", "gain", "reason"),
    [
        (-8e41, 1.0, "implausible_power"),  # what ngspice once returned without an error
        (1e-3, 1e7, "implausible_gain"),
        (1e-3, float("inf"), "non_finite"),
    ],
)
def test_implausible_results_are_rejected(power_w: float, gain: float, reason: str) -> None:
    e = evaluate(load_fixture("rc_lowpass"), Canned(fake_result(power_w, gain)), CONFIG)
    assert not e.ok and e.failed_stage == Stage.METRICS and e.reason == reason


def test_plausible_fake_result_passes() -> None:
    e = evaluate(load_fixture("rc_lowpass"), Canned(fake_result(1e-3, 0.5)), CONFIG)
    assert e.ok and e.op is not None and e.op.power_w == pytest.approx(1e-3)


@pytest.mark.ngspice
def test_source_resistance_loads_the_input(ngspice: Ngspice) -> None:
    # 10k in front of the RC low-pass moves its corner to 1 / (2 pi (10k + 1.6k) 100n) = 137 Hz.
    config = TestbenchConfig(r_source=10e3)
    e = evaluate(load_fixture("rc_lowpass"), ngspice, config)
    assert e.ok and e.response is not None
    assert summarize(e.response).f_3db_hz == pytest.approx(
        1 / (2 * math.pi * 11.6e3 * 100e-9), rel=0.03
    )
    netlist = build_testbench(load_fixture("rc_lowpass"), config).netlist
    assert "RSOURCE vsrc vin 10000" in netlist and "VIN vsrc 0 DC 1.65 AC 1" in netlist


@pytest.mark.ngspice
def test_transistor_regions(ngspice: Ngspice) -> None:
    # Collector and emitter both grounded: vce = 0, so the NPN can't be in its active region.
    c, _ = Circuit().add(Kind.NPN, (GND, VIN, GND))
    c, _ = c.add(Kind.R, (VIN, VOUT))
    e = evaluate(c, ngspice, CONFIG)
    assert e.ok and e.op is not None
    assert e.op.devices[0].region != Region.ACTIVE
    assert e.op.bias_score == 0.0


def test_graph_failure_skips_simulation() -> None:
    class Unreachable:
        def run(self, netlist: str, timeout_s: float | None = None) -> SimResult:
            raise AssertionError("should not simulate")

    c, _ = Circuit().add(Kind.R, (VIN, GND))  # VOUT unconnected
    e = evaluate(c, Unreachable(), CONFIG)
    assert not e.ok and e.failed_stage == Stage.GRAPH
    assert "port_unconnected" in e.reason


def test_simulation_failure_is_reported() -> None:
    class Failing:
        def run(self, netlist: str, timeout_s: float | None = None) -> SimResult:
            return SimResult(SimStatus.NO_CONVERGENCE, errors=["no convergence"])

    e = evaluate(load_fixture("ce_amp"), Failing(), CONFIG)
    assert not e.ok and e.failed_stage == Stage.SIMULATION
    assert e.sim_status == SimStatus.NO_CONVERGENCE


def test_testbench_netlist() -> None:
    tb = build_testbench(load_fixture("ota_follower"), CONFIG)
    assert ".subckt dut VIN VOUT VDD" in tb.netlist
    assert "xdut vin vout vdd dut" in tb.netlist
    assert "VIN vin 0 DC 1.65 AC 1" in tb.netlist
    assert "@m.xdut.m1[von]" in tb.netlist
    assert tb.netlist.rstrip().endswith(".end")
    probe = next(d for d in tb.devices if d.component.id == "1")
    assert probe.terminals == {"d": "xdut.n1", "g": "vin", "s": "xdut.tail"}


def test_port_context_follows_load() -> None:
    assert TestbenchConfig().port_context().resistive_loads == ("VOUT",)
    assert TestbenchConfig(r_load=None).port_context().resistive_loads == ()
    with_source = TestbenchConfig(r_source=10e3).port_context()
    assert with_source.voltage_sources == ("VDD",)
    assert with_source.resistive_loads == ("VOUT", "VIN")
