"""Electrical checks that cost nothing to run.

These rules *can* be broken (mutations often produce such circuits), so they are reported as a
list of violations instead of being enforced by ``Circuit``. Logging which violations reject
candidates is the main tool for tuning the mutation operators.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from acd.circuit.devices import Kind, spec
from acd.circuit.graph import Circuit
from acd.circuit.ports import GND, VDD, VIN, VOUT


class ViolationKind(StrEnum):
    PORT_UNCONNECTED = "port_unconnected"  # a required port has no connections
    DANGLING_NET = "dangling_net"  # an internal net with a single connection
    USELESS_COMPONENT = "useless_component"  # all pins of a component on one net
    NO_DC_PATH = "no_dc_path"  # net with no DC path to ground (matrix would be singular)
    INDUCTOR_SOURCE_LOOP = "inductor_source_loop"  # loop of inductors and voltage sources
    NO_SIGNAL_PATH = "no_signal_path"  # output can't depend on the input


@dataclass(frozen=True)
class Violation:
    kind: ViolationKind
    items: tuple[str, ...] = ()  # component ids or nets involved

    def __str__(self) -> str:
        return f"{self.kind}({', '.join(self.items)})" if self.items else str(self.kind)


@dataclass(frozen=True)
class PortContext:
    """What the test circuit attaches to the ports."""

    voltage_sources: tuple[str, ...] = (VDD, VIN)  # held relative to GND by ideal sources
    resistive_loads: tuple[str, ...] = (VOUT,)  # tied to GND through a resistor
    signal_in: str = VIN
    signal_out: str = VOUT


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        """Join two sets. Returns False if they were already joined (a loop)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def check(circuit: Circuit, ctx: PortContext | None = None) -> list[Violation]:
    ctx = ctx or PortContext()
    out: list[Violation] = []

    unconnected = [
        p
        for p in (ctx.signal_in, ctx.signal_out)
        if p in circuit.ports and not circuit.connections(p)
    ]
    if unconnected:
        out.append(Violation(ViolationKind.PORT_UNCONNECTED, tuple(unconnected)))

    dangling = [n for n in circuit.internal_nets if len(circuit.connections(n)) == 1]
    if dangling:
        out.append(Violation(ViolationKind.DANGLING_NET, tuple(dangling)))

    useless = [c.id for c in circuit.components if len(set(c.nets)) == 1]
    if useless:
        out.append(Violation(ViolationKind.USELESS_COMPONENT, tuple(useless)))

    floating = _nets_without_dc_path(circuit, ctx)
    if floating:
        out.append(Violation(ViolationKind.NO_DC_PATH, tuple(floating)))

    loops = _inductor_source_loops(circuit, ctx)
    if loops:
        out.append(Violation(ViolationKind.INDUCTOR_SOURCE_LOOP, tuple(loops)))

    if not unconnected and not _signal_path_exists(circuit, ctx):
        out.append(Violation(ViolationKind.NO_SIGNAL_PATH, (ctx.signal_in, ctx.signal_out)))

    return out


def is_valid(circuit: Circuit, ctx: PortContext | None = None) -> bool:
    return not check(circuit, ctx)


def _nets_without_dc_path(circuit: Circuit, ctx: PortContext) -> list[str]:
    uf = _UnionFind()
    uf.find(GND)
    for net in (*ctx.voltage_sources, *ctx.resistive_loads):
        if net in circuit.ports:
            uf.union(net, GND)
    for c in circuit.components:
        pins = c.pins
        for a, b in spec(c.kind).dc_paths:
            uf.union(pins[a], pins[b])
    ground = uf.find(GND)
    return [n for n in circuit.nets if circuit.connections(n) and uf.find(n) != ground]


def _inductor_source_loops(circuit: Circuit, ctx: PortContext) -> list[str]:
    """SPICE treats inductors as 0 V sources at DC, so inductors and sources must form no loop."""
    uf = _UnionFind()
    for net in ctx.voltage_sources:
        if net in circuit.ports:
            uf.union(net, GND)
    return [
        c.id for c in circuit.components if c.kind == Kind.L and not uf.union(c.nets[0], c.nets[1])
    ]


def _signal_path_exists(circuit: Circuit, ctx: PortContext) -> bool:
    """Is the output reachable from the input without passing through an AC-grounded net?"""
    ac_ground = {GND, *(n for n in ctx.voltage_sources if n != ctx.signal_in)}
    neighbors: dict[str, set[str]] = {}
    for c in circuit.components:
        nets = set(c.nets)
        for n in nets:
            neighbors.setdefault(n, set()).update(nets - {n})
    seen = {ctx.signal_in}
    queue = deque([ctx.signal_in])
    while queue:
        n = queue.popleft()
        if n == ctx.signal_out:
            return True
        for m in neighbors.get(n, ()):
            if m not in seen and m not in ac_ground:
                seen.add(m)
                queue.append(m)
    return False
