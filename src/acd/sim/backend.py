"""The simulator interface: netlist text in, results out.

A bad circuit never raises an exception. Non-convergence, timeouts and parse errors come back as a
``SimResult`` with a status, so the GA loop doesn't fill up with try/except. Exceptions are for
real bugs and for a simulator that can't be loaded at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SimStatus(StrEnum):
    OK = "ok"
    PARSE_ERROR = "parse_error"  # the netlist didn't load
    NO_CONVERGENCE = "no_convergence"  # operating point or time step failed
    TIMEOUT = "timeout"
    ERROR = "error"  # anything else, e.g. an analysis that didn't run
    FATAL = "fatal"  # the simulator instance is unusable; restart its process


Vector = NDArray[np.float64] | NDArray[np.complex128]


@dataclass
class SimResult:
    status: SimStatus
    plots: dict[str, dict[str, Vector]] = field(default_factory=dict)  # "op1" -> "v(out)" -> data
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == SimStatus.OK

    def plot(self, analysis: str) -> dict[str, Vector]:
        """First plot of an analysis type (``"op"``, ``"ac"``, ``"tran"``...)."""
        for name, vectors in self.plots.items():
            if name.rstrip("0123456789") == analysis:
                return vectors
        raise KeyError(f"no {analysis} plot in result (plots: {list(self.plots)})")

    def op(self, vector: str) -> float:
        """A scalar from the operating point, e.g. ``op("vout")`` or ``op("v(vout)")``."""
        return float(np.real(self.vector("op", vector)[0]))

    def vector(self, analysis: str, vector: str) -> Vector:
        vectors = self.plot(analysis)
        name = vector.lower()
        if name not in vectors and name.startswith("v(") and name.endswith(")"):
            name = name[2:-1]  # ngspice stores node voltages under the bare node name
        return vectors[name]


class Simulator(Protocol):
    def run(self, netlist: str, timeout_s: float | None = None) -> SimResult: ...
