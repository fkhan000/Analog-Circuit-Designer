"""Component types and everything specific to them.

Every other module asks this one for pin lists, DC paths, parameter ranges and model names, so
adding a new component type means adding one entry to ``SPECS``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from acd.circuit.ports import GND, VDD


class Kind(StrEnum):
    R = "R"
    C = "C"
    L = "L"
    NPN = "NPN"
    PNP = "PNP"
    NMOS = "NMOS"
    PMOS = "PMOS"


class PinRole(StrEnum):
    """What a pin does electrically. Used to map pins when a component changes type."""

    PASSIVE = "passive"  # either end of an R, C or L
    CONTROL = "control"  # gate / base
    OUTPUT = "output"  # drain / collector
    COMMON = "common"  # source / emitter


@dataclass(frozen=True)
class PinSpec:
    name: str
    role: PinRole


# Standard resistor/capacitor value series (one decade of mantissas).
_E24 = (
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
)  # fmt: skip
E_SERIES: dict[int, tuple[float, ...]] = {24: _E24, 12: _E24[::2], 6: _E24[::4]}


def clean_float(x: float) -> float:
    """Round to 12 significant digits so snapped values have one exact representation."""
    return float(f"{x:.12g}")


@dataclass(frozen=True)
class ESeries:
    """Snap to standard E-series values (E6, E12 or E24)."""

    n: int

    def snap(self, x: float) -> float:
        mantissas = E_SERIES[self.n]
        exponent = math.floor(math.log10(x))
        m = x / 10**exponent
        # Nearest mantissa in log distance, including the next decade's 1.0.
        candidates = (*mantissas, 10.0)
        best = min(candidates, key=lambda c: abs(math.log(c / m)))
        return clean_float(best * 10**exponent)

    def neighbors(self, x: float) -> tuple[float, float]:
        """The series values just below and just above the series value nearest ``x``."""
        exponent = math.floor(math.log10(x))
        values = sorted(
            clean_float(m * 10**e)
            for e in (exponent - 1, exponent, exponent + 1)
            for m in E_SERIES[self.n]
        )
        i = min(range(len(values)), key=lambda k: abs(math.log(values[k] / x)))
        return values[max(i - 1, 0)], values[min(i + 1, len(values) - 1)]


@dataclass(frozen=True)
class Grid:
    """Snap to integer multiples of ``step``."""

    step: float

    def snap(self, x: float) -> float:
        return clean_float(max(1, round(x / self.step)) * self.step)

    def neighbors(self, x: float) -> tuple[float, float]:
        n = round(x / self.step)
        return clean_float(max(1, n - 1) * self.step), clean_float((n + 1) * self.step)


Snap = ESeries | Grid


@dataclass(frozen=True)
class ParamSpec:
    name: str
    unit: str
    low: float
    high: float
    default: float
    snap: Snap

    def clamp_snap(self, x: float) -> float:
        """Snap ``x`` to the allowed values, staying inside ``[low, high]``."""
        x = min(max(x, self.low), self.high)
        snapped = self.snap.snap(x)
        if snapped > self.high:
            snapped = self.snap.neighbors(snapped)[0]
        if snapped < self.low:
            snapped = self.snap.neighbors(snapped)[1]
        return snapped

    def contains(self, x: float) -> bool:
        return self.low * (1 - 1e-9) <= x <= self.high * (1 + 1e-9)


@dataclass(frozen=True)
class DeviceSpec:
    kind: Kind
    prefix: str  # SPICE element letter
    pins: tuple[PinSpec, ...]  # in SPICE node order (MOSFET bulk excluded)
    symmetric: bool  # pins are interchangeable (R, C, L)
    dc_paths: tuple[tuple[str, str], ...]  # pin pairs that conduct at DC
    params: tuple[ParamSpec, ...]
    model: str | None = None  # .model name in acd/models/generic.lib
    bulk_net: str | None = None  # port net the MOSFET bulk is tied to

    @property
    def pin_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.pins)

    def pin_index(self, name: str) -> int:
        return self.pin_names.index(name)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def param_index(self, name: str) -> int:
        return self.param_names.index(name)

    @property
    def is_transistor(self) -> bool:
        return len(self.pins) == 3


_PASSIVE_PINS = (PinSpec("p", PinRole.PASSIVE), PinSpec("n", PinRole.PASSIVE))
_BJT_PINS = (
    PinSpec("c", PinRole.OUTPUT),
    PinSpec("b", PinRole.CONTROL),
    PinSpec("e", PinRole.COMMON),
)
_MOS_PINS = (
    PinSpec("d", PinRole.OUTPUT),
    PinSpec("g", PinRole.CONTROL),
    PinSpec("s", PinRole.COMMON),
)
_BJT_DC = (("c", "b"), ("b", "e"), ("c", "e"))
_MOS_DC = (("d", "s"),)

_BJT_PARAMS = (ParamSpec("m", "", 1, 8, 1, Grid(1)),)
_MOS_PARAMS = (
    ParamSpec("w", "m", 0.5e-6, 100e-6, 2e-6, Grid(10e-9)),
    ParamSpec("l", "m", 0.18e-6, 10e-6, 0.5e-6, Grid(10e-9)),
)

SPECS: dict[Kind, DeviceSpec] = {
    Kind.R: DeviceSpec(
        Kind.R, "R", _PASSIVE_PINS, True, (("p", "n"),),
        (ParamSpec("r", "ohm", 10.0, 10e6, 10e3, ESeries(24)),),
    ),
    Kind.C: DeviceSpec(
        Kind.C, "C", _PASSIVE_PINS, True, (),
        (ParamSpec("c", "F", 1e-12, 1e-6, 1e-9, ESeries(12)),),
    ),
    Kind.L: DeviceSpec(
        Kind.L, "L", _PASSIVE_PINS, True, (("p", "n"),),
        (ParamSpec("l", "H", 1e-9, 10e-3, 1e-6, ESeries(12)),),
    ),
    Kind.NPN: DeviceSpec(Kind.NPN, "Q", _BJT_PINS, False, _BJT_DC, _BJT_PARAMS, model="NPN_GEN"),
    Kind.PNP: DeviceSpec(Kind.PNP, "Q", _BJT_PINS, False, _BJT_DC, _BJT_PARAMS, model="PNP_GEN"),
    Kind.NMOS: DeviceSpec(
        Kind.NMOS, "M", _MOS_PINS, False, _MOS_DC, _MOS_PARAMS, model="NMOS_GEN", bulk_net=GND
    ),
    Kind.PMOS: DeviceSpec(
        Kind.PMOS, "M", _MOS_PINS, False, _MOS_DC, _MOS_PARAMS, model="PMOS_GEN", bulk_net=VDD
    ),
}  # fmt: skip

MODEL_TO_KIND: dict[str, Kind] = {s.model.upper(): k for k, s in SPECS.items() if s.model}


def spec(kind: Kind) -> DeviceSpec:
    return SPECS[kind]


def swap_targets(kind: Kind) -> tuple[Kind, ...]:
    """Kinds this one can change into without changing its pin count."""
    n = len(SPECS[kind].pins)
    return tuple(k for k, s in SPECS.items() if k != kind and len(s.pins) == n)


def pin_mapping(src: Kind, dst: Kind) -> dict[str, str]:
    """Map pin names of ``src`` onto pin names of ``dst`` by role (or by order for passives)."""
    a, b = SPECS[src], SPECS[dst]
    if len(a.pins) != len(b.pins):
        raise ValueError(f"cannot map {src} pins onto {dst}: pin counts differ")
    if a.symmetric or b.symmetric:
        return dict(zip(a.pin_names, b.pin_names, strict=True))
    by_role = {p.role: p.name for p in b.pins}
    return {p.name: by_role[p.role] for p in a.pins}


def default_params(kind: Kind) -> tuple[float, ...]:
    return tuple(p.default for p in SPECS[kind].params)
