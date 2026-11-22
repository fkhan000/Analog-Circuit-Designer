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
from acd.evaluation.metrics import (
    BiasRules,
    OperatingPoint,
    Response,
    operating_point,
    response,
)
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


def evaluate(
    circuit: Circuit,
    simulator: Simulator,
    config: TestbenchConfig,
    rules: BiasRules | None = None,
) -> Evaluation:
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
        op = operating_point(result, tb, rules)
        resp = response(result, tb)
    except KeyError as e:
        return Evaluation(
            False, Stage.METRICS, f"missing_vector:{e}", sim_status=result.status,
            sim_time_s=elapsed,
        )  # fmt: skip
    problem = implausibility(op, resp)
    if problem:
        return Evaluation(
            False, Stage.METRICS, problem, sim_status=result.status, op=op, sim_time_s=elapsed
        )
    return Evaluation(True, sim_status=result.status, op=op, response=resp, sim_time_s=elapsed)


# A circuit of passive parts and transistors can't deliver net power back to the sources, so total
# source power below this is a numerically broken operating point, not physics. ngspice can return
# one without reporting an error (seen after its gmin and source stepping both struggled).
MIN_PLAUSIBLE_POWER_W = -1e-6
MAX_PLAUSIBLE_GAIN = 1e6  # 120 dB; beyond this the response comes from ideal-part artifacts


def implausibility(op: OperatingPoint, resp: Response) -> str:
    """Why these results can't be trusted, or "" if they look physically possible."""
    if not (math.isfinite(op.power_w) and math.isfinite(op.vout_dc)):
        return "non_finite"
    if not np.all(np.isfinite(resp.h)):
        return "non_finite"
    if op.power_w < MIN_PLAUSIBLE_POWER_W:
        return "implausible_power"
    if float(np.max(np.abs(resp.h))) > MAX_PLAUSIBLE_GAIN:
        return "implausible_gain"
    return ""
