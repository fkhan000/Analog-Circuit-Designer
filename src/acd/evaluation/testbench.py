"""Test circuits that wrap the evolved ``.subckt`` and measure it.

The transfer test circuit drives VIN with a DC level plus a 1 V AC source, powers VDD, loads VOUT
with a resistor and capacitor, and runs ``.op`` then ``.ac``. The transfer function is VOUT over
the source voltage. With ``r_source`` set, the source drives VIN through that resistance, so the
circuit has to cope with a non-ideal source (e.g. a buffer task).

Both analyses run in one simulator call: the pair takes well under a millisecond for circuits of
this size, and ngspice skips the AC analysis itself when the operating point fails, so splitting
them into separate calls would only add parsing overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

from acd.circuit.devices import spec
from acd.circuit.graph import Circuit, Component
from acd.circuit.netlist import SUBCKT_NAME, element_name, format_number, subckt_ports, to_subckt
from acd.circuit.ports import GND, VDD, VIN, VOUT
from acd.circuit.validity import PortContext
from acd.models import generic_models

INSTANCE = "xdut"


@dataclass(frozen=True)
class TestbenchConfig:
    __test__ = False  # not a pytest test class

    vdd: float = 3.3
    vin_dc: float | None = None  # DC level on VIN; None means VDD/2
    r_load: float | None = 1e6  # VOUT to GND; None for no load resistor
    r_source: float | None = None  # source resistance in front of VIN; None for an ideal source
    c_load: float | None = 1e-12  # VOUT to GND; None for no load capacitor
    f_start_hz: float = 10.0
    f_stop_hz: float = 100e6
    points_per_decade: int = 20
    timeout_s: float = 1.0  # .op + .ac normally takes ~1 ms; longer means ngspice is stuck

    @property
    def vin_bias(self) -> float:
        return self.vdd / 2 if self.vin_dc is None else self.vin_dc

    def port_context(self) -> PortContext:
        """What this test circuit attaches to the ports, for the graph checks."""
        loads = (VOUT,) if self.r_load else ()
        if self.r_source:  # VIN is held through a resistor, not by an ideal source
            return PortContext(voltage_sources=(VDD,), resistive_loads=(*loads, VIN))
        return PortContext(voltage_sources=(VDD, VIN), resistive_loads=loads)


@dataclass(frozen=True)
class DeviceProbe:
    """Where to find one transistor's operating point in the simulation results."""

    component: Component
    terminals: dict[str, str]  # pin name -> node voltage vector
    vectors: dict[str, str]  # quantity ("i", "gm", "von") -> device vector


@dataclass(frozen=True)
class Testbench:
    __test__ = False

    netlist: str
    devices: tuple[DeviceProbe, ...]
    vdd: float
    vin_dc: float
    output: str = "vout"
    input: str = "vin"  # the source voltage the response is measured against
    supply_currents: tuple[tuple[str, float], ...] = ()  # (branch vector, source voltage)


def node_vector(net: str) -> str:
    """The ngspice vector holding a DUT net's voltage (internal nets get the instance prefix)."""
    if net == GND:
        return "0"
    if net in (VIN, VOUT, VDD):
        return net.lower()
    return f"{INSTANCE}.{net.lower()}"


def device_vector(comp: Component, quantity: str) -> str:
    """ngspice names a subcircuit device ``@<letter>.<instance>.<element>[<quantity>]``."""
    letter = spec(comp.kind).prefix.lower()
    return f"@{letter}.{INSTANCE}.{element_name(comp).lower()}[{quantity}]"


def _probe(comp: Component) -> DeviceProbe:
    s = spec(comp.kind)
    terminals = {pin: node_vector(net) for pin, net in zip(s.pin_names, comp.nets, strict=True)}
    if s.prefix == "Q":
        vectors = {"i": device_vector(comp, "ic"), "gm": device_vector(comp, "gm")}
    else:
        vectors = {
            "i": device_vector(comp, "id"),
            "gm": device_vector(comp, "gm"),
            "von": device_vector(comp, "von"),
        }
    return DeviceProbe(comp, terminals, vectors)


def build_testbench(circuit: Circuit, config: TestbenchConfig) -> Testbench:
    devices = tuple(_probe(c) for c in circuit.components if spec(c.kind).is_transistor)
    ports = " ".join(node_vector(p) for p in subckt_ports(circuit))
    lines = [
        "* acd transfer test circuit",
        generic_models().rstrip(),
        to_subckt(circuit).rstrip(),
        f"VDD vdd 0 DC {format_number(config.vdd)}",
        f"{INSTANCE} {ports} {SUBCKT_NAME}",
    ]
    source = "vin"
    if config.r_source:
        source = "vsrc"
        lines.append(f"RSOURCE vsrc vin {format_number(config.r_source)}")
    lines.append(f"VIN {source} 0 DC {format_number(config.vin_bias)} AC 1")
    if config.r_load:
        lines.append(f"RLOAD vout 0 {format_number(config.r_load)}")
    if config.c_load:
        lines.append(f"CLOAD vout 0 {format_number(config.c_load)}")
    lines.append(".save all")
    lines.extend(f".save {' '.join(d.vectors.values())}" for d in devices)
    lines.append(".op")
    lines.append(
        f".ac dec {config.points_per_decade} {format_number(config.f_start_hz)} "
        f"{format_number(config.f_stop_hz)}"
    )
    lines.append(".end")
    return Testbench(
        netlist="\n".join(lines) + "\n",
        devices=devices,
        vdd=config.vdd,
        vin_dc=config.vin_bias,
        input=source,
        supply_currents=(("vdd#branch", config.vdd), ("vin#branch", config.vin_bias)),
    )
