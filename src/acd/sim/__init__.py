"""Running SPICE. Knows only netlist text and results, nothing about circuits."""

from acd.sim.backend import SimResult, SimStatus, Simulator
from acd.sim.ngspice import Ngspice, NgspiceNotFound, find_library

__all__ = ["Ngspice", "NgspiceNotFound", "SimResult", "SimStatus", "Simulator", "find_library"]
