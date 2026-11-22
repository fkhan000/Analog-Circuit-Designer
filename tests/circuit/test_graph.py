import pytest

from acd.circuit.devices import Kind
from acd.circuit.graph import Circuit, CircuitError, Component
from acd.circuit.ports import GND, VDD, VIN, VOUT


def rc() -> Circuit:
    c, _ = Circuit().add(Kind.R, (VIN, VOUT), (1e3,))
    c, _ = c.add(Kind.C, (VOUT, GND), (1e-9,))
    return c


def test_add_assigns_fresh_ids() -> None:
    c = rc()
    assert [x.id for x in c.components] == ["c1", "c2"]
    assert c.next_id == 3


def test_add_with_pin_mapping() -> None:
    c, cid = Circuit().add(Kind.NPN, {"c": VOUT, "b": VIN, "e": GND})
    assert c.component(cid).net("b") == VIN


def test_edits_leave_original_unchanged() -> None:
    c = rc()
    c2 = c.connect("c1", "n", "n9")
    assert c.component("c1").net("n") == VOUT
    assert c2.component("c1").net("n") == "n9"
    assert c.remove("c2") is not c and len(c) == 2


def test_nets_are_derived_from_pins() -> None:
    c = rc()
    assert c.nets == (VIN, VOUT, VDD, GND)
    c, net = c.new_net()
    assert net not in c.nets  # a reserved name only exists once a pin uses it
    c = c.connect("c2", "n", net)
    assert net in c.internal_nets
    assert c.connections(net) == (("c2", "n"),)


def test_ports_listed_even_when_unconnected() -> None:
    assert Circuit().connections(VDD) == ()
    assert VDD in Circuit().nets


def test_merge_keeps_port() -> None:
    c = rc().connect("c2", "n", "n5")
    merged = c.merge_nets("n5", GND)
    assert merged.component("c2").net("n") == GND
    assert "n5" not in merged.nets


def test_merging_two_ports_is_refused() -> None:
    with pytest.raises(CircuitError):
        rc().merge_nets(VIN, GND)


def test_wrong_pin_count_rejected() -> None:
    with pytest.raises(CircuitError):
        Component("c1", Kind.R, (VIN,), (1e3,))


def test_out_of_range_param_rejected() -> None:
    with pytest.raises(CircuitError):
        Component("c1", Kind.R, (VIN, VOUT), (1.0,))


def test_case_insensitive_net_clash_rejected() -> None:
    with pytest.raises(CircuitError):
        Circuit((Component("c1", Kind.R, ("n1", "N1"), (1e3,)),))


def test_ground_name_reserved() -> None:
    with pytest.raises(CircuitError):
        Circuit((Component("c1", Kind.R, (VIN, "0"), (1e3,)),))


def test_duplicate_ids_rejected() -> None:
    r = Component("c1", Kind.R, (VIN, VOUT), (1e3,))
    with pytest.raises(CircuitError):
        Circuit((r, r))


def test_new_net_skips_taken_names() -> None:
    c = Circuit((Component("c1", Kind.R, (VIN, "n1"), (1e3,)),), next_id=1)
    _, net = c.new_net()
    assert net != "n1"


def test_circuits_compare_by_value() -> None:
    assert rc() == rc()
