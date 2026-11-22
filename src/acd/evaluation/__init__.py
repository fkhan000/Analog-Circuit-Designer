"""Measuring circuits: the only package where circuits and the simulator meet.

``stages.evaluate`` runs a circuit through graph checks and a test-circuit simulation and returns
target-independent metrics. ``fitness.score`` compares those with a target.
"""

from acd.evaluation.cache import EvaluationCache
from acd.evaluation.fitness import FitnessWeights, Objectives, score
from acd.evaluation.metrics import OperatingPoint, Region, Response, summarize
from acd.evaluation.stages import Evaluation, Stage, evaluate
from acd.evaluation.targets import Band, CurveTarget, MaskTarget, Target, load_target
from acd.evaluation.testbench import TestbenchConfig, build_testbench

__all__ = [
    "Band",
    "CurveTarget",
    "Evaluation",
    "EvaluationCache",
    "FitnessWeights",
    "MaskTarget",
    "Objectives",
    "OperatingPoint",
    "Region",
    "Response",
    "Stage",
    "Target",
    "TestbenchConfig",
    "build_testbench",
    "evaluate",
    "load_target",
    "score",
    "summarize",
]
