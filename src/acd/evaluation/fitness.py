"""Compare metrics with the target and produce objective values. Lower fitness is better.

``objectives`` keeps the separate objectives (for a later multi-objective search such as NSGA-II);
``scalar_fitness`` combines them with weights for the single-objective GA in v1.

Failed circuits get large fitness values in tiers, so any circuit that simulates beats any that
doesn't, and any circuit that passes the graph checks beats any that doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass

from acd.circuit.graph import Circuit
from acd.evaluation.stages import Evaluation, Stage
from acd.evaluation.targets import Target

GRAPH_INVALID_FITNESS = 1e9
SIM_FAILED_FITNESS = 1e8


@dataclass(frozen=True)
class FitnessWeights:
    response: float = 1.0  # per dB of RMS response error
    power_per_mw: float = 0.1  # per mW of DC power
    per_component: float = 0.05  # per component (parsimony pressure)
    bias: float = 2.0  # times the fraction of transistors outside their amplifying region


@dataclass(frozen=True)
class Objectives:
    response_error_db: float
    power_w: float
    n_components: int
    bias_penalty: float  # 1 - bias score: fraction of transistors not in their active region


def objectives(circuit: Circuit, evaluation: Evaluation, target: Target) -> Objectives | None:
    if not evaluation.ok or evaluation.op is None or evaluation.response is None:
        return None
    return Objectives(
        response_error_db=target.error_db(evaluation.response),
        power_w=evaluation.op.power_w,
        n_components=len(circuit),
        bias_penalty=1.0 - evaluation.op.bias_score,
    )


def scalar_fitness(
    evaluation: Evaluation, objs: Objectives | None, weights: FitnessWeights
) -> float:
    if objs is None:
        if evaluation.failed_stage == Stage.GRAPH:
            return GRAPH_INVALID_FITNESS + 1e6 * len(evaluation.violations)
        return SIM_FAILED_FITNESS
    return (
        weights.response * objs.response_error_db
        + weights.power_per_mw * objs.power_w * 1e3
        + weights.per_component * objs.n_components
        + weights.bias * objs.bias_penalty
    )


def score(
    circuit: Circuit, evaluation: Evaluation, target: Target, weights: FitnessWeights
) -> tuple[float, Objectives | None]:
    objs = objectives(circuit, evaluation, target)
    return scalar_fitness(evaluation, objs, weights), objs
