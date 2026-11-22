"""Mutation operators: which edits happen. ``Circuit`` decides how an edit happens.

Each child gets ``1 + Poisson(mean_extra_ops)`` random operations. Only the final result has to
pass the graph checks, so a mutation can pass through an invalid intermediate circuit (e.g. move a
pin, then add the component that reconnects it). If the result fails, the whole sequence is
retried, up to ``max_attempts`` times, and the rejection reasons are counted.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from acd.circuit.devices import Kind, pin_mapping, spec, swap_targets
from acd.circuit.graph import Circuit, Component
from acd.circuit.validity import PortContext, check


class MutationOp(StrEnum):
    SERIES_INSERT = "series_insert"  # split a net and bridge it with a new component
    PARALLEL_INSERT = "parallel_insert"  # new component between existing nets
    DELETE_OPEN = "delete_open"  # remove a component, leaving its nets
    DELETE_SHORT = "delete_short"  # remove a 2-pin component and join its nets (a wire)
    CHANGE_TYPE = "change_type"  # R <-> C, NPN <-> NMOS, ... keeping the pin count
    MOVE_PIN = "move_pin"  # reconnect one pin to another net
    CHANGE_VALUE = "change_value"  # log-normal step on one parameter


DEFAULT_OP_WEIGHTS: dict[MutationOp, float] = {
    MutationOp.CHANGE_VALUE: 3.0,
    MutationOp.SERIES_INSERT: 1.0,
    MutationOp.PARALLEL_INSERT: 1.0,
    MutationOp.DELETE_OPEN: 0.5,
    MutationOp.DELETE_SHORT: 0.5,
    MutationOp.CHANGE_TYPE: 1.0,
    MutationOp.MOVE_PIN: 1.0,
}

# Every component type. Restrict per experiment with [mutation] kinds (e.g. no inductors for
# on-chip designs, where they're unrealistic at most frequencies).
DEFAULT_KINDS = tuple(Kind)


@dataclass(frozen=True)
class MutationConfig:
    max_components: int = 10
    kinds: tuple[Kind, ...] = DEFAULT_KINDS  # kinds that inserts and type changes may use
    op_weights: Mapping[MutationOp, float] = field(default_factory=lambda: dict(DEFAULT_OP_WEIGHTS))
    value_sigma: float = 0.7  # std dev of the log-normal value step (natural log units)
    mean_extra_ops: float = 0.5
    max_attempts: int = 20


@dataclass
class MutationResult:
    circuit: Circuit | None  # None if every attempt failed the graph checks
    ops: tuple[MutationOp, ...] = ()
    attempts: int = 0
    rejections: Counter[str] = field(default_factory=Counter)


def random_params(kind: Kind, rng: np.random.Generator) -> tuple[float, ...]:
    """Log-uniform random values over each parameter's range, snapped."""
    out = []
    for p in spec(kind).params:
        x = math.exp(rng.uniform(math.log(p.low), math.log(p.high)))
        out.append(p.clamp_snap(x))
    return tuple(out)


def _pick[T](rng: np.random.Generator, items: tuple[T, ...] | list[T]) -> T:
    return items[int(rng.integers(len(items)))]


def _random_pin(circuit: Circuit, rng: np.random.Generator) -> tuple[Component, str]:
    comp = _pick(rng, circuit.components)
    return comp, _pick(rng, spec(comp.kind).pin_names)


def _nets_for_new(
    kind: Kind, a: str, b: str, circuit: Circuit, rng: np.random.Generator
) -> tuple[str, ...]:
    """Connect a new component across ``a`` and ``b`` (a transistor's third pin goes anywhere)."""
    if not spec(kind).is_transistor:
        return (a, b)
    nets = [a, b, _pick(rng, circuit.nets)]
    order = rng.permutation(3)
    return tuple(nets[i] for i in order)


def apply(
    op: MutationOp, circuit: Circuit, rng: np.random.Generator, config: MutationConfig
) -> Circuit | None:
    """Apply one operation. Returns None when the operation doesn't apply to this circuit."""
    n = len(circuit)
    match op:
        case MutationOp.SERIES_INSERT:
            if n == 0 or n >= config.max_components:
                return None
            comp, pin = _random_pin(circuit, rng)
            old_net = comp.net(pin)
            circuit, new_net = circuit.new_net()
            circuit = circuit.connect(comp.id, pin, new_net)
            kind = _pick(rng, config.kinds)
            circuit, _ = circuit.add(
                kind, _nets_for_new(kind, old_net, new_net, circuit, rng), random_params(kind, rng)
            )
            return circuit

        case MutationOp.PARALLEL_INSERT:
            if n >= config.max_components:
                return None
            i, j = rng.choice(len(circuit.nets), size=2, replace=False)
            a, b = circuit.nets[int(i)], circuit.nets[int(j)]
            kind = _pick(rng, config.kinds)
            circuit, _ = circuit.add(
                kind, _nets_for_new(kind, a, b, circuit, rng), random_params(kind, rng)
            )
            return circuit

        case MutationOp.DELETE_OPEN:
            if n == 0:
                return None
            return circuit.remove(_pick(rng, circuit.components).id)

        case MutationOp.DELETE_SHORT:
            candidates = [
                c
                for c in circuit.components
                if not spec(c.kind).is_transistor
                and c.nets[0] != c.nets[1]
                and not (circuit.is_port(c.nets[0]) and circuit.is_port(c.nets[1]))
            ]
            if not candidates:
                return None
            comp = _pick(rng, candidates)
            return circuit.remove(comp.id).merge_nets(comp.nets[0], comp.nets[1])

        case MutationOp.CHANGE_TYPE:
            if n == 0:
                return None
            comp = _pick(rng, circuit.components)
            targets = tuple(k for k in swap_targets(comp.kind) if k in config.kinds)
            if not targets:
                return None
            kind = _pick(rng, targets)
            mapping = pin_mapping(comp.kind, kind)  # old pin -> new pin
            new_pins = {new: comp.net(old) for old, new in mapping.items()}
            nets = tuple(new_pins[p] for p in spec(kind).pin_names)
            return circuit.put(Component(comp.id, kind, nets, random_params(kind, rng)))

        case MutationOp.MOVE_PIN:
            if n == 0:
                return None
            comp, pin = _random_pin(circuit, rng)
            others = [x for x in circuit.nets if x != comp.net(pin)]
            return circuit.connect(comp.id, pin, _pick(rng, others))

        case MutationOp.CHANGE_VALUE:
            if n == 0:
                return None
            comp = _pick(rng, circuit.components)
            param = _pick(rng, spec(comp.kind).params)
            old_value = comp.param(param.name)
            step = rng.normal(0.0, config.value_sigma)
            value = param.clamp_snap(old_value * math.exp(step))
            if value == old_value:
                lower, upper = param.snap.neighbors(old_value)
                value = param.clamp_snap(upper if step >= 0 else lower)
            if value == old_value:
                return None  # pinned at a range limit
            return circuit.set_param(comp.id, param.name, value)

    raise ValueError(f"unknown mutation {op}")


def mutate(
    circuit: Circuit,
    rng: np.random.Generator,
    config: MutationConfig | None = None,
    ctx: PortContext | None = None,
) -> MutationResult:
    """A mutated copy of ``circuit`` that passes the graph checks, or None after max_attempts."""
    config = config or MutationConfig()
    ops = [op for op, w in config.op_weights.items() if w > 0]
    weights = np.array([config.op_weights[op] for op in ops], dtype=float)
    weights /= weights.sum()
    rejections: Counter[str] = Counter()

    for attempt in range(1, config.max_attempts + 1):
        child = circuit
        applied: list[MutationOp] = []
        for _ in range(1 + int(rng.poisson(config.mean_extra_ops))):
            op = ops[int(rng.choice(len(ops), p=weights))]
            result = apply(op, child, rng, config)
            if result is not None:
                child = result
                applied.append(op)
        if not applied:
            rejections["no_applicable_op"] += 1
            continue
        if len(child) > config.max_components:
            rejections["too_many_components"] += 1
            continue
        violations = check(child, ctx)
        if violations:
            rejections.update(str(v.kind) for v in violations)
            continue
        return MutationResult(child, tuple(applied), attempt, rejections)
    return MutationResult(None, (), config.max_attempts, rejections)
