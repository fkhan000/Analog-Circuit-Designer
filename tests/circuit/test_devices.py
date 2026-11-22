import pytest

from acd.circuit.devices import (
    ESeries,
    Grid,
    Kind,
    ParamSpec,
    default_params,
    pin_mapping,
    spec,
    swap_targets,
)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (4600, 4700),
        (4.7e3, 4700),
        (1.05, 1.1),
        (999.9, 1000),
        (2.25e-9, 2.2e-9),
        (2.35e-9, 2.4e-9),
        (9.6, 10.0),
    ],
)
def test_e24_snap(x: float, expected: float) -> None:
    assert ESeries(24).snap(x) == pytest.approx(expected, rel=1e-12)


def test_e12_snap_skips_e24_values() -> None:
    assert ESeries(12).snap(1.1e-9) in (1.0e-9, 1.2e-9)


def test_snapped_values_are_exact() -> None:
    # Same value whichever way you get there, so hashes and caches stay stable.
    assert ESeries(24).snap(4.69e3) == ESeries(24).snap(4.71e3) == 4700.0


def test_eseries_neighbors() -> None:
    assert ESeries(24).neighbors(4700) == (4300.0, 5100.0)
    assert ESeries(24).neighbors(1000) == (910.0, 1100.0)


def test_grid_snap_and_neighbors() -> None:
    g = Grid(10e-9)
    assert g.snap(1.234e-6) == 1.23e-6
    assert g.neighbors(1.23e-6) == (1.22e-6, 1.24e-6)
    assert g.snap(1e-12) == 10e-9  # never snaps to zero


def test_clamp_snap_stays_in_range() -> None:
    p = ParamSpec("r", "ohm", 10.0, 10e6, 10e3, ESeries(24))
    assert p.clamp_snap(1.0) == 10.0
    assert p.clamp_snap(1e9) == 10e6
    assert p.contains(p.clamp_snap(123.4))


def test_defaults_are_in_range() -> None:
    for kind in Kind:
        for p, v in zip(spec(kind).params, default_params(kind), strict=True):
            assert p.contains(v), (kind, p.name)


def test_swap_targets_keep_pin_count() -> None:
    assert set(swap_targets(Kind.R)) == {Kind.C, Kind.L}
    assert set(swap_targets(Kind.NPN)) == {Kind.PNP, Kind.NMOS, Kind.PMOS}


def test_pin_mapping_by_role() -> None:
    assert pin_mapping(Kind.NPN, Kind.NMOS) == {"c": "d", "b": "g", "e": "s"}
    assert pin_mapping(Kind.R, Kind.C) == {"p": "p", "n": "n"}
    with pytest.raises(ValueError):
        pin_mapping(Kind.R, Kind.NPN)


def test_dc_paths() -> None:
    assert spec(Kind.C).dc_paths == ()
    assert spec(Kind.NMOS).dc_paths == (("d", "s"),)  # the gate conducts nothing
    assert len(spec(Kind.NPN).dc_paths) == 3
