"""Convert circuits to SPICE ``.subckt`` text and back.

The output contains only the subcircuit: no sources, analyses or model includes, which belong to
the test circuit. Numbers are always written as plain or scientific notation, never with SPICE
suffixes: in SPICE ``1M`` means *milli*, and mega is ``MEG``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from acd.circuit.devices import MODEL_TO_KIND, Kind, clean_float, spec
from acd.circuit.graph import Circuit, CircuitError, Component, natural_key
from acd.circuit.ports import DEFAULT_PORTS, GND

SUBCKT_NAME = "dut"
GROUND_NODE = "0"


def element_name(comp: Component) -> str:
    """SPICE element name: the type letter plus the id (``Rc3``)."""
    return spec(comp.kind).prefix + comp.id


def node_name(net: str) -> str:
    return GROUND_NODE if net == GND else net


def format_number(x: float) -> str:
    return f"{clean_float(x):.12g}"


def subckt_ports(circuit: Circuit) -> tuple[str, ...]:
    """Ports listed on the ``.subckt`` line (GND is the global node 0, so it's left out)."""
    return tuple(p for p in circuit.ports if p != GND)


def to_subckt(circuit: Circuit, name: str = SUBCKT_NAME) -> str:
    lines = [f".subckt {name} {' '.join(subckt_ports(circuit))}"]
    for c in sorted(circuit.components, key=lambda c: natural_key(c.id)):
        lines.append(_element_line(c, circuit))
    lines.append(f".ends {name}")
    return "\n".join(lines) + "\n"


def _element_line(c: Component, circuit: Circuit) -> str:
    s = spec(c.kind)
    nodes = [node_name(n) for n in c.nets]
    name = element_name(c)
    if c.kind in (Kind.R, Kind.C, Kind.L):
        return f"{name} {nodes[0]} {nodes[1]} {format_number(c.params[0])}"
    if c.kind in (Kind.D, Kind.NPN, Kind.PNP):
        return f"{name} {' '.join(nodes)} {s.model} m={format_number(c.param('m'))}"
    if c.kind in (Kind.NMOS, Kind.PMOS):
        assert s.bulk_net is not None
        if s.bulk_net not in circuit.ports:
            raise CircuitError(f"{c.kind} bulk needs port {s.bulk_net}")
        w, length = format_number(c.param("w")), format_number(c.param("l"))
        return f"{name} {' '.join(nodes)} {node_name(s.bulk_net)} {s.model} W={w} L={length}"
    raise CircuitError(f"no netlist format for {c.kind}")


# ------------------------------------------------------------------ parsing

_NUMBER_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z]*)$")
_SCALE = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "g": 1e9, "t": 1e12}


def parse_number(token: str) -> float:
    """Parse a SPICE number, including scale suffixes (``4.7k``, ``10u``, ``1meg``)."""
    m = _NUMBER_RE.match(token.strip())
    if not m:
        raise CircuitError(f"not a number: {token!r}")
    value, suffix = float(m.group(1)), m.group(2).lower()
    if suffix.startswith("meg"):
        return value * 1e6
    if suffix.startswith("mil"):
        return value * 25.4e-6
    # Only the first letter sets the scale; the rest is a unit name ("10uF", "1kohm").
    return value * _SCALE.get(suffix[:1], 1.0)


def _logical_lines(text: str) -> list[str]:
    """Strip comments and join ``+`` continuation lines."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and out:
            out[-1] += " " + line[1:].strip()
        else:
            out.append(line)
    return out


def _key_values(tokens: list[str]) -> dict[str, float]:
    kv: dict[str, float] = {}
    for t in tokens:
        if "=" not in t:
            raise CircuitError(f"unexpected token {t!r}")
        k, v = t.split("=", 1)
        kv[k.lower()] = parse_number(v)
    return kv


def from_subckt(text: str) -> Circuit:
    """Parse the ``.subckt`` subset that ``to_subckt`` writes (and hand-written equivalents).

    Node ``0`` (or ``gnd``) becomes the GND port. Component ids are element names without their
    type letter (``Rc3`` -> ``c3``), falling back to the lowercased full name on a clash
    (``R1`` and ``C1``).
    """
    lines = _logical_lines(text)
    header = next((ln for ln in lines if ln.lower().startswith(".subckt")), None)
    if header is None:
        raise CircuitError("no .subckt line")
    standard = {p.lower(): p for p in DEFAULT_PORTS}
    port_tokens = [standard.get(t.lower(), t) for t in header.split()[2:]]
    ports = (*port_tokens, GND)
    canon: dict[str, str] = {p.lower(): p for p in ports}
    canon["0"] = canon["gnd"] = GND

    def net(token: str) -> str:
        return canon.setdefault(token.lower(), token)

    start = lines.index(header) + 1
    comps: list[Component] = []
    used_ids: set[str] = set()
    for line in lines[start:]:
        low = line.lower()
        if low.startswith(".ends"):
            break
        if low.startswith("."):
            continue
        tokens = line.split()
        name, prefix = tokens[0], tokens[0][0].upper()
        cid = name[1:]
        if not cid or cid.lower() in used_ids:
            cid = name.lower()
        used_ids.add(cid.lower())
        comps.append(_parse_element(prefix, cid, tokens, net))

    numbers = [
        int(m.group(1))
        for x in (*(c.id for c in comps), *(n for c in comps for n in c.nets))
        if (m := re.match(r"^[cn](\d+)$", x))
    ]
    return Circuit(tuple(comps), ports=ports, next_id=max(numbers, default=0) + 1)


def _parse_element(
    prefix: str, cid: str, tokens: list[str], net: Callable[[str], str]
) -> Component:
    if prefix in ("R", "C", "L"):
        if len(tokens) != 4:
            raise CircuitError(f"expected '{prefix}name n1 n2 value': {' '.join(tokens)}")
        kind = Kind(prefix)
        return Component(cid, kind, (net(tokens[1]), net(tokens[2])), (parse_number(tokens[3]),))
    if prefix == "D":
        if len(tokens) < 4 or tokens[3].upper() not in MODEL_TO_KIND:
            raise CircuitError(f"expected 'Dname a k MODEL [m=..]': {' '.join(tokens)}")
        kv = _key_values(tokens[4:])
        return Component(cid, Kind.D, (net(tokens[1]), net(tokens[2])), (kv.get("m", 1.0),))
    if prefix == "Q":
        if len(tokens) < 5 or tokens[4].upper() not in MODEL_TO_KIND:
            raise CircuitError(f"expected 'Qname c b e MODEL [m=..]': {' '.join(tokens)}")
        kind = MODEL_TO_KIND[tokens[4].upper()]
        kv = _key_values(tokens[5:])
        return Component(cid, kind, tuple(net(t) for t in tokens[1:4]), (kv.get("m", 1.0),))
    if prefix == "M":
        if len(tokens) < 6 or tokens[5].upper() not in MODEL_TO_KIND:
            raise CircuitError(f"expected 'Mname d g s b MODEL W=.. L=..': {' '.join(tokens)}")
        kind = MODEL_TO_KIND[tokens[5].upper()]
        s = spec(kind)
        if net(tokens[4]) != s.bulk_net:
            raise CircuitError(f"{tokens[0]}: {kind} bulk must be tied to {s.bulk_net}")
        kv = _key_values(tokens[6:])
        if "w" not in kv or "l" not in kv:
            raise CircuitError(f"{tokens[0]}: W= and L= are required")
        return Component(cid, kind, tuple(net(t) for t in tokens[1:4]), (kv["w"], kv["l"]))
    raise CircuitError(f"unsupported element {tokens[0]!r}")
