"""The circuit: a bipartite graph of components and nets.

Each component stores one net per pin (in the pin order from ``devices.py``). That tuple *is* the
bipartite graph's edge list, so two structural rules hold by construction: every pin is on exactly
one net, and the pin count matches the component type. Nets are not stored separately; a net exists
when some pin references it, or when it is a port.

Circuits never change after they are built. Every edit returns a new ``Circuit``, which keeps
parents intact, keeps cache entries correct and makes circuits safe to send to worker processes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property

from acd.circuit.devices import Kind, default_params, spec
from acd.circuit.ports import DEFAULT_PORTS, GND

_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class CircuitError(ValueError):
    """An edit or construction that would break a structural rule."""


def natural_key(name: str) -> tuple[str, int, str]:
    """Sort key that puts ``c2`` before ``c10``."""
    m = re.match(r"^([A-Za-z_]*)(\d+)(.*)$", name)
    if m:
        return (m.group(1).lower(), int(m.group(2)), m.group(3).lower())
    return (name.lower(), -1, "")


@dataclass(frozen=True)
class Component:
    id: str
    kind: Kind
    nets: tuple[str, ...]  # one net per pin, in spec pin order
    params: tuple[float, ...]  # one value per parameter, in spec order

    def __post_init__(self) -> None:
        s = spec(self.kind)
        if not _NAME_RE.match(self.id):
            raise CircuitError(f"invalid component id {self.id!r}")
        if len(self.nets) != len(s.pins):
            raise CircuitError(
                f"{self.kind} {self.id} needs {len(s.pins)} nets, got {len(self.nets)}"
            )
        if len(self.params) != len(s.params):
            raise CircuitError(
                f"{self.kind} {self.id} needs {len(s.params)} params, got {len(self.params)}"
            )
        for p, v in zip(s.params, self.params, strict=True):
            if not p.contains(v):
                raise CircuitError(
                    f"{self.kind} {self.id}: {p.name}={v:g} outside [{p.low:g}, {p.high:g}]"
                )

    @property
    def pins(self) -> dict[str, str]:
        """Pin name to net."""
        return dict(zip(spec(self.kind).pin_names, self.nets, strict=True))

    def net(self, pin: str) -> str:
        return self.nets[spec(self.kind).pin_index(pin)]

    def param(self, name: str) -> float:
        return self.params[spec(self.kind).param_index(name)]

    def with_net(self, pin: str, net: str) -> Component:
        nets = list(self.nets)
        nets[spec(self.kind).pin_index(pin)] = net
        return replace(self, nets=tuple(nets))

    def with_param(self, name: str, value: float) -> Component:
        params = list(self.params)
        params[spec(self.kind).param_index(name)] = value
        return replace(self, params=tuple(params))

    def describe(self) -> str:
        s = spec(self.kind)
        pins = " ".join(f"{p}={n}" for p, n in zip(s.pin_names, self.nets, strict=True))
        params = " ".join(f"{p}={v:g}" for p, v in zip(s.param_names, self.params, strict=True))
        return f"{self.id}: {self.kind} {pins} {params}".rstrip()


@dataclass(frozen=True)
class Circuit:
    components: tuple[Component, ...] = ()
    ports: tuple[str, ...] = DEFAULT_PORTS
    next_id: int = 1  # counter for new component ids and net names

    def __post_init__(self) -> None:
        if GND not in self.ports:
            raise CircuitError("ports must include GND")
        seen_ids: set[str] = set()
        for c in self.components:
            if c.id.lower() in seen_ids:
                raise CircuitError(f"duplicate component id {c.id!r}")
            seen_ids.add(c.id.lower())
        # ngspice ignores letter case, so net names must differ by more than case.
        seen_nets: dict[str, str] = {}
        for net in (*self.ports, *(n for c in self.components for n in c.nets)):
            if not _NAME_RE.match(net) or net == "0":
                raise CircuitError(f"invalid net name {net!r}")
            other = seen_nets.setdefault(net.lower(), net)
            if other != net:
                raise CircuitError(f"net names {other!r} and {net!r} differ only by case")
        if len(set(self.ports)) != len(self.ports):
            raise CircuitError("duplicate port names")

    # ----------------------------------------------------------------- queries

    def __len__(self) -> int:
        return len(self.components)

    @cached_property
    def by_id(self) -> dict[str, Component]:
        return {c.id: c for c in self.components}

    @cached_property
    def net_index(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Net to its ``(component id, pin)`` connections. Ports appear even when unconnected."""
        index: dict[str, list[tuple[str, str]]] = {p: [] for p in self.ports}
        for c in self.components:
            for pin, net in zip(spec(c.kind).pin_names, c.nets, strict=True):
                index.setdefault(net, []).append((c.id, pin))
        return {n: tuple(conns) for n, conns in index.items()}

    @cached_property
    def nets(self) -> tuple[str, ...]:
        """Ports first (in port order), then internal nets in natural order."""
        internal = sorted((n for n in self.net_index if n not in self.ports), key=natural_key)
        return (*self.ports, *internal)

    @property
    def internal_nets(self) -> tuple[str, ...]:
        return tuple(n for n in self.nets if n not in self.ports)

    def is_port(self, net: str) -> bool:
        return net in self.ports

    def component(self, cid: str) -> Component:
        try:
            return self.by_id[cid]
        except KeyError:
            raise CircuitError(f"no component {cid!r}") from None

    def connections(self, net: str) -> tuple[tuple[str, str], ...]:
        return self.net_index.get(net, ())

    def describe(self) -> str:
        lines = [c.describe() for c in sorted(self.components, key=lambda c: natural_key(c.id))]
        return "\n".join(lines) if lines else "(empty circuit)"

    def __str__(self) -> str:
        return self.describe()

    # ------------------------------------------------------------------- edits

    def add(
        self,
        kind: Kind,
        nets: Sequence[str] | Mapping[str, str],
        params: Sequence[float] | None = None,
    ) -> tuple[Circuit, str]:
        """Add a component. ``nets`` is in pin order, or a pin-name mapping."""
        s = spec(kind)
        if isinstance(nets, Mapping):
            if set(nets) != set(s.pin_names):
                raise CircuitError(f"{kind} pins are {s.pin_names}, got {tuple(nets)}")
            nets = tuple(nets[p] for p in s.pin_names)
        n = self.next_id
        while f"c{n}".lower() in {c.id.lower() for c in self.components}:
            n += 1
        cid = f"c{n}"
        comp = Component(cid, kind, tuple(nets), tuple(params or default_params(kind)))
        return replace(self, components=(*self.components, comp), next_id=n + 1), cid

    def remove(self, cid: str) -> Circuit:
        self.component(cid)
        return replace(self, components=tuple(c for c in self.components if c.id != cid))

    def put(self, comp: Component) -> Circuit:
        """Replace the component with the same id (used for type changes)."""
        self.component(comp.id)
        return replace(
            self, components=tuple(comp if c.id == comp.id else c for c in self.components)
        )

    def connect(self, cid: str, pin: str, net: str) -> Circuit:
        """Move one pin onto ``net``."""
        return self.put(self.component(cid).with_net(pin, net))

    def set_param(self, cid: str, name: str, value: float) -> Circuit:
        return self.put(self.component(cid).with_param(name, value))

    def new_net(self) -> tuple[Circuit, str]:
        """Reserve a fresh net name. The net exists once a pin is connected to it."""
        n = self.next_id
        taken = {x.lower() for x in self.net_index}
        while f"n{n}" in taken:
            n += 1
        return replace(self, next_id=n + 1), f"n{n}"

    def merge_nets(self, a: str, b: str) -> Circuit:
        """Join two nets into one. A port always survives; two ports can't be merged."""
        if a == b:
            return self
        if self.is_port(a) and self.is_port(b):
            raise CircuitError(f"cannot merge port nets {a} and {b}")
        keep, drop = (b, a) if self.is_port(b) else (a, b)
        comps = tuple(
            replace(c, nets=tuple(keep if n == drop else n for n in c.nets))
            for c in self.components
        )
        return replace(self, components=comps)
