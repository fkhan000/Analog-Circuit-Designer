"""The circuit itself: plain data, no SPICE.

``Circuit`` is a bipartite graph of components and nets. ``validity`` reports electrical problems,
``canonical`` hashes circuits, and ``netlist`` converts them to and from SPICE ``.subckt`` text.
"""

from acd.circuit.devices import Kind, spec
from acd.circuit.graph import Circuit, CircuitError, Component
from acd.circuit.ports import DEFAULT_PORTS, GND, VDD, VIN, VOUT

__all__ = [
    "DEFAULT_PORTS",
    "GND",
    "VDD",
    "VIN",
    "VOUT",
    "Circuit",
    "CircuitError",
    "Component",
    "Kind",
    "spec",
]
