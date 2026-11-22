import pytest
from hypothesis import given

from acd.circuit.canonical import full_hash, isomorphic
from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit, CircuitError, Component
from acd.circuit.netlist import from_subckt, parse_number, to_subckt
from acd.circuit.ports import GND, VDD, VIN, VOUT
from helpers import FIXTURE_NAMES, FIXTURES, circuits, load_fixture


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("4.7k", 4.7e3),
        ("10u", 10e-6),
        ("10uF", 10e-6),
        ("1M", 1e-3),  # milli, not mega
        ("1meg", 1e6),
        ("1MEG", 1e6),
        ("2.2e-9", 2.2e-9),
        ("100", 100.0),
        ("1F", 1e-15),  # SPICE reads a leading F as femto
        (".5n", 0.5e-9),
    ],
)
def test_parse_number(token: str, value: float) -> None:
    assert parse_number(token) == pytest.approx(value)


def test_parse_number_rejects_garbage() -> None:
    with pytest.raises(CircuitError):
        parse_number("abc")


def test_subckt_output() -> None:
    c = Circuit(
        (
            Component("c1", Kind.R, (VIN, VOUT), (4.7e6,)),
            Component("c2", Kind.NPN, (VOUT, VIN, GND), (2.0,)),
            Component("c10", Kind.PMOS, (VOUT, VIN, VDD), (10e-6, 1e-6)),
            Component("c3", Kind.NMOS, (VOUT, VIN, GND), (2e-6, 0.5e-6)),
        )
    )
    assert to_subckt(c) == (
        ".subckt dut VIN VOUT VDD\n"
        "Rc1 VIN VOUT 4700000\n"  # never "4.7M", which SPICE would read as milliohms
        "Qc2 VOUT VIN 0 NPN_GEN m=2\n"
        "Mc3 VOUT VIN 0 0 NMOS_GEN W=2e-06 L=5e-07\n"  # NMOS bulk tied to GND
        "Mc10 VOUT VIN VDD VDD PMOS_GEN W=1e-05 L=1e-06\n"  # PMOS bulk tied to VDD
        ".ends dut\n"
    )


def test_diode_and_inductor_lines() -> None:
    c = Circuit(
        (
            Component("c1", Kind.D, (VOUT, GND), (2.0,)),
            Component("c2", Kind.L, (VIN, VOUT), (4.7e-3,)),
        )
    )
    text = to_subckt(c)
    assert "Dc1 VOUT 0 D_GEN m=2\n" in text  # anode first
    assert "Lc2 VIN VOUT 0.0047\n" in text
    assert full_hash(from_subckt(text)) == full_hash(c)


def test_diode_direction_matters_for_hash() -> None:
    a = Circuit((Component("c1", Kind.D, (VOUT, GND), (1.0,)),))
    b = Circuit((Component("c1", Kind.D, (GND, VOUT), (1.0,)),))
    assert full_hash(a) != full_hash(b)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_round_trip(name: str) -> None:
    c = load_fixture(name)
    again = from_subckt(to_subckt(c))
    assert full_hash(again) == full_hash(c)
    assert isomorphic(again, c)
    assert to_subckt(again) == to_subckt(c)


def test_fixture_ids_and_ports() -> None:
    c = load_fixture("ota_follower")
    assert c.ports == (VIN, VOUT, VDD, GND)
    assert {x.id for x in c.components} == {"1", "2", "3", "4", "5", "6", "7"}
    assert c.component("7").param("r") == 24e3


def test_clashing_ids_fall_back_to_full_name() -> None:
    c = from_subckt(".subckt dut VIN VOUT VDD\nR1 VIN VOUT 1k\nC1 VOUT 0 1n\n.ends\n")
    assert {x.id for x in c.components} == {"1", "c1"}
    # New ids must not collide with parsed ones.
    _, new_id = c.add(Kind.R, (VIN, VOUT))
    assert new_id not in {"1", "c1"}


def test_wrong_mosfet_bulk_rejected() -> None:
    text = ".subckt dut VIN VOUT VDD\nM1 VOUT VIN 0 VOUT NMOS_GEN W=1u L=1u\n.ends\n"
    with pytest.raises(CircuitError):
        from_subckt(text)


def test_lowercase_ports_are_normalized() -> None:
    c = from_subckt(".subckt dut vin vout vdd\nR1 vin vout 1k\n.ends\n")
    assert c.ports == (VIN, VOUT, VDD, GND)
    assert c.component("1").nets == (VIN, VOUT)


def test_continuation_lines_and_comments() -> None:
    text = (
        (FIXTURES / "rc_lowpass.cir")
        .read_text()
        .replace("C2 VOUT 0 100n", "C2 VOUT\n+ 0 100n ; inline comment")
    )
    assert full_hash(from_subckt(text)) == full_hash(load_fixture("rc_lowpass"))


@given(circuits())
def test_random_circuits_round_trip(c: Circuit) -> None:
    assert full_hash(from_subckt(to_subckt(c))) == full_hash(c)
