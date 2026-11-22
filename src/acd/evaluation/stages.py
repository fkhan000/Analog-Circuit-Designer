"""Staged evaluation of one circuit: graph checks, then simulation, then metrics.

Each stage can stop the evaluation. The result records how far the circuit got and why it
stopped, which is what the GA logs to show where candidates die.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from acd.circuit.graph import Circuit
from acd.circuit.validity import Violation, check
from acd.evaluation.metrics import OperatingPoint, Response, operating_point, response
from acd.evaluation.testbench import TestbenchConfig, build_testbench
from acd.sim.backend import SimStatus, Simulator


class Stage(StrEnum):
    GRAPH = "graph"  # cheap graph checks
    SIMULATION = "simulation"  # ngspice run (.op + .ac)
    METRICS = "metrics"  # results contain unusable values


@dataclass(frozen=True)
class Evaluation:
    ok: bool
    failed_stage: Stage | None = None
    reason: str = ""  # short failure reason, used as a log/histogram key
    violations: tuple[Violation, ...] = ()
    sim_status: SimStatus | None = None
    op: OperatingPoint | None = None
    response: Response | None = None
    sim_time_s: float = 0.0

    def slim(self) -> Evaluation:
        """Copy without the frequency response arrays (for caches)."""
        return replace(self, response=None)


def evaluate(circuit: Circuit, simulator: Simulator, config: TestbenchConfig) -> Evaluation:
    violations = check(circuit, config.port_context())
    if violations:
        reason = "+".join(sorted({str(v.kind) for v in violations}))
        return Evaluation(False, Stage.GRAPH, reason, tuple(violations))

    tb = build_testbench(circuit, config)
    start = time.perf_counter()
    result = simulator.run(tb.netlist, config.timeout_s)
    elapsed = time.perf_counter() - start
    if not result.ok:
        return Evaluation(
            False, Stage.SIMULATION, str(result.status), sim_status=result.status,
            sim_time_s=elapsed,
        )  # fmt: skip

    try:
        op = operating_point(result, tb)
        resp = response(result, tb)
    except KeyError as e:
        return Evaluation(
            False, Stage.METRICS, f"missing_vector:{e}", sim_status=result.status,
            sim_time_s=elapsed,
        )  # fmt: skip
    if not (math.isfinite(op.power_w) and np.all(np.isfinite(resp.h))):
        return Evaluation(
            False, Stage.METRICS, "non_finite", sim_status=result.status, op=op,
            sim_time_s=elapsed,
        )  # fmt: skip
    return Evaluation(True, sim_status=result.status, op=op, response=resp, sim_time_s=elapsed)
