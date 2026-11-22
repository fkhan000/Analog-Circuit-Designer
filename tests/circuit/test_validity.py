import pytest

from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit
from acd.circuit.ports import GND, VDD, VIN, VOUT
from acd.circuit.validity import PortContext, ViolationKind, check, is_valid
from helpers import FIXTURE_NAMES, load_fixture


def build(*parts: tuple[Kind, tuple[str, ...]]) -> Circuit:
    c = Circuit()
    for kind, nets in parts:
        c, _ = c.add(kind, nets)
    return c


def kinds(c: Circuit, ctx: PortContext | None = None) -> set[ViolationKind]:
    return {v.kind for v in check(c, ctx)}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixtures_are_valid(name: str) -> None:
    assert check(load_fixture(name)) == []


def test_unconnected_output() -> None:
    c = build((Kind.R, (VIN, GND)))
    assert ViolationKind.PORT_UNCONNECTED in kinds(c)


def test_dangling_net() -> None:
    c = build((Kind.R, (VIN, VOUT)), (Kind.R, (VOUT, "n1")))
    assert ViolationKind.DANGLING_NET in kinds(c)


def test_useless_component() -> None:
    c = build((Kind.R, (VIN, VOUT)), (Kind.C, (VOUT, VOUT)))
    assert ViolationKind.USELESS_COMPONENT in kinds(c)


def test_capacitor_only_node_has_no_dc_path() -> None:
    c = build((Kind.C, (VIN, "n1")), (Kind.C, ("n1", VOUT)))
    assert ViolationKind.NO_DC_PATH in kinds(c)


def test_floating_gate_has_no_dc_path() -> None:
    # Gate driven only through a capacitor: nothing sets its DC voltage.
    c = build(
        (Kind.C, (VIN, "g")),
        (Kind.NMOS, (VOUT, "g", GND)),
        (Kind.R, (VDD, VOUT)),
    )
    violations = check(c)
    assert any(v.kind == ViolationKind.NO_DC_PATH and "g" in v.items for v in violations)


def test_output_without_load_needs_its_own_dc_path() -> None:
    c = build((Kind.C, (VIN, VOUT)))
    assert ViolationKind.NO_DC_PATH not in kinds(c)  # default context: VOUT has a load resistor
    no_load = PortContext(resistive_loads=())
    assert ViolationKind.NO_DC_PATH in kinds(c, no_load)


def test_inductor_across_supply() -> None:
    c = build((Kind.R, (VIN, VOUT)), (Kind.L, (VDD, GND)))
    assert ViolationKind.INDUCTOR_SOURCE_LOOP in kinds(c)


def test_parallel_inductors_form_a_loop() -> None:
    c = build((Kind.L, (VIN, VOUT)), (Kind.L, (VIN, VOUT)), (Kind.R, (VOUT, GND)))
    assert ViolationKind.INDUCTOR_SOURCE_LOOP in kinds(c)


def test_single_inductor_is_fine() -> None:
    c = build((Kind.L, (VIN, VOUT)), (Kind.R, (VOUT, GND)))
    assert is_valid(c)


def test_signal_path_through_ground_does_not_count() -> None:
    c = build((Kind.R, (VIN, GND)), (Kind.R, (GND, VOUT)))
    assert ViolationKind.NO_SIGNAL_PATH in kinds(c)


def test_signal_path_through_supply_does_not_count() -> None:
    c = build((Kind.R, (VIN, VDD)), (Kind.R, (VDD, VOUT)))
    assert ViolationKind.NO_SIGNAL_PATH in kinds(c)


def test_signal_path_through_transistor() -> None:
    c = build((Kind.NPN, (VOUT, VIN, GND)), (Kind.R, (VDD, VOUT)))
    assert is_valid(c)
