"""Shared test helpers: fixture loading and Hypothesis strategies."""

from __future__ import annotations

import math
from pathlib import Path

from hypothesis import strategies as st

from acd.circuit.devices import Kind, spec
from acd.circuit.graph import Circuit, Component
from acd.circuit.netlist import from_subckt
from acd.circuit.ports import DEFAULT_PORTS

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = (
    "rc_lowpass",
    "ce_amp",
    "ota_follower",
    "lc_bandpass",
    "diode_divider",
    "starved_amp",
)


def load_fixture(name: str) -> Circuit:
    return from_subckt((FIXTURES / f"{name}.cir").read_text())


@st.composite
def param_values(draw: st.DrawFn, kind: Kind) -> tuple[float, ...]:
    values = []
    for p in spec(kind).params:
        x = math.exp(draw(st.floats(math.log(p.low), math.log(p.high))))
        values.append(p.clamp_snap(x))
    return tuple(values)


@st.composite
def circuits(
    draw: st.DrawFn, max_components: int = 8, kinds: tuple[Kind, ...] = tuple(Kind)
) -> Circuit:
    """Random circuits that satisfy the structural rules (not necessarily valid electrically)."""
    pool = (*DEFAULT_PORTS, "n1", "n2", "n3", "n4")
    n = draw(st.integers(0, max_components))
    comps = []
    for i in range(n):
        kind = draw(st.sampled_from(kinds))
        nets = tuple(draw(st.sampled_from(pool)) for _ in spec(kind).pins)
        comps.append(Component(f"c{i + 1}", kind, nets, draw(param_values(kind))))
    return Circuit(tuple(comps), next_id=n + 5)


# Top-level functions for ParallelMap tests (worker processes import them by name).


def square(x: int) -> int:
    return x * x


def square_or_crash(x: int) -> int:
    """Kills the worker process for negative inputs."""
    import os

    if x < 0:
        os._exit(1)
    return x * x


def crashed(x: int) -> int:
    return -1
