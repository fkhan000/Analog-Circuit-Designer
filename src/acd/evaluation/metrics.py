"""Turn simulation results into named metrics. Nothing here depends on the target.

Operating regions are computed from node voltages, not from the voltages ngspice reports for each
device, because ngspice's sign conventions differ between quantities for PMOS devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from acd.circuit.devices import Kind
from acd.evaluation.testbench import DeviceProbe, Testbench
from acd.sim.backend import SimResult

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class BiasRules:
    """Thresholds for classifying transistor operating regions."""

    bjt_on_vbe: float = 0.45  # base-emitter junction considered on (V)
    bjt_min_vce: float = 0.2  # below this the BJT is saturated (V)
    mos_sat_margin: float = 0.05  # required margin of vds above vdsat (V)
    # A transistor in its amplifying region but carrying less than this is STARVED. Without it,
    # level-1 MOSFETs (which have no subthreshold current) can sit just above threshold at
    # picoamps and act as huge resistors, a bias trick that wouldn't survive real leakage.
    min_current_a: float = 100e-9


class Region(StrEnum):
    """Operating region, named so BJTs and MOSFETs share one vocabulary.

    ACTIVE is the amplifying region: forward-active for a BJT, saturation for a MOSFET.
    STARVED is that region at a current below ``BiasRules.min_current_a``.
    LINEAR is BJT saturation or MOSFET triode.
    """

    ACTIVE = "active"
    STARVED = "starved"
    LINEAR = "linear"
    CUTOFF = "cutoff"
    REVERSE = "reverse"


@dataclass(frozen=True)
class DeviceState:
    component_id: str
    kind: Kind
    region: Region
    current_a: float  # magnitude of collector / drain current
    gm_s: float
    voltages: dict[str, float]  # e.g. {"vbe": .., "vce": ..} or {"vgs": .., "vds": .., "vov": ..}


@dataclass(frozen=True)
class OperatingPoint:
    power_w: float  # total DC power delivered by the test circuit's sources
    vout_dc: float
    devices: tuple[DeviceState, ...] = ()

    @property
    def bias_score(self) -> float:
        """Fraction of transistors in their amplifying region (1.0 with no transistors)."""
        if not self.devices:
            return 1.0
        return sum(d.region == Region.ACTIVE for d in self.devices) / len(self.devices)


@dataclass(frozen=True)
class Response:
    """Small-signal transfer function VOUT/VIN."""

    freq_hz: FloatArray
    h: ComplexArray
    _gain_db: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        gain = 20 * np.log10(np.maximum(np.abs(self.h), 1e-15))
        object.__setattr__(self, "_gain_db", gain)

    @property
    def gain_db(self) -> FloatArray:
        return self._gain_db

    @property
    def phase_deg(self) -> FloatArray:
        return np.asarray(np.degrees(np.unwrap(np.angle(self.h))), dtype=np.float64)

    def gain_db_at(self, freq_hz: FloatArray) -> FloatArray:
        """Gain interpolated on a log-frequency axis (clamped outside the simulated range)."""
        return np.asarray(
            np.interp(np.log10(freq_hz), np.log10(self.freq_hz), self.gain_db), dtype=np.float64
        )


@dataclass(frozen=True)
class ResponseSummary:
    lf_gain_db: float  # gain at the lowest simulated frequency
    peak_gain_db: float
    f_peak_hz: float
    f_3db_hz: float | None  # first drop of 3 dB below the low-frequency gain
    f_unity_hz: float | None  # first downward crossing of 0 dB


def _first_crossing(freq: FloatArray, y: FloatArray, level: float) -> float | None:
    """First frequency where ``y`` falls through ``level`` (interpolated in log frequency)."""
    below = np.nonzero(y < level)[0]
    if below.size == 0 or below[0] == 0:
        return None
    i = int(below[0])
    x0, x1 = np.log10(freq[i - 1]), np.log10(freq[i])
    y0, y1 = y[i - 1], y[i]
    t = (y0 - level) / (y0 - y1)
    return float(10 ** (x0 + t * (x1 - x0)))


def summarize(response: Response) -> ResponseSummary:
    g, f = response.gain_db, response.freq_hz
    peak = int(np.argmax(g))
    lf = float(g[0])
    return ResponseSummary(
        lf_gain_db=lf,
        peak_gain_db=float(g[peak]),
        f_peak_hz=float(f[peak]),
        f_3db_hz=_first_crossing(f, g, lf - 3.0103),
        f_unity_hz=_first_crossing(f, g, 0.0) if lf > 0 else None,
    )


def _voltage(result: SimResult, vector: str) -> float:
    return 0.0 if vector == "0" else result.op(vector)


def _device_state(result: SimResult, probe: DeviceProbe, rules: BiasRules) -> DeviceState:
    comp = probe.component
    v = {pin: _voltage(result, vec) for pin, vec in probe.terminals.items()}
    current = abs(result.op(probe.vectors["i"]))
    gm = abs(result.op(probe.vectors["gm"]))
    sign = -1.0 if comp.kind in (Kind.PNP, Kind.PMOS) else 1.0

    if comp.kind in (Kind.NPN, Kind.PNP):
        vbe = sign * (v["b"] - v["e"])
        vbc = sign * (v["b"] - v["c"])
        vce = vbe - vbc
        if vbe >= rules.bjt_on_vbe:
            region = Region.ACTIVE if vce >= rules.bjt_min_vce else Region.LINEAR
        else:
            region = Region.REVERSE if vbc >= rules.bjt_on_vbe else Region.CUTOFF
        voltages = {"vbe": vbe, "vce": vce}
    else:
        vgs = sign * (v["g"] - v["s"])
        vds = sign * (v["d"] - v["s"])
        vth = abs(result.op(probe.vectors["von"]))
        vov = vgs - vth
        if vov <= 0:
            region = Region.CUTOFF
        elif vds >= vov + rules.mos_sat_margin:
            region = Region.ACTIVE
        else:
            region = Region.LINEAR  # includes drain and source swapped (vds < 0)
        voltages = {"vgs": vgs, "vds": vds, "vov": vov}

    if region == Region.ACTIVE and current < rules.min_current_a:
        region = Region.STARVED
    return DeviceState(comp.id, comp.kind, region, current, gm, voltages)


def operating_point(
    result: SimResult, tb: Testbench, rules: BiasRules | None = None
) -> OperatingPoint:
    rules = rules or BiasRules()
    power = -sum(result.op(branch) * volts for branch, volts in tb.supply_currents)
    return OperatingPoint(
        power_w=float(power),
        vout_dc=result.op(tb.output),
        devices=tuple(_device_state(result, d, rules) for d in tb.devices),
    )


def response(result: SimResult, tb: Testbench) -> Response:
    freq = np.real(result.vector("ac", "frequency")).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):  # non-finite values are checked later
        h = np.asarray(
            result.vector("ac", tb.output) / result.vector("ac", tb.input), np.complex128
        )
    return Response(freq, h)
