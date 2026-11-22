import math

import numpy as np
import pytest

from acd.models import generic_models
from acd.sim import Ngspice, SimStatus

pytestmark = pytest.mark.ngspice

RC_LOWPASS = """rc low-pass
V1 in 0 DC 0 AC 1
R1 in out 1.6k
C1 out 0 100n
.ac dec 50 10 100k
.end
"""


def test_rc_lowpass_corner(ngspice: Ngspice) -> None:
    result = ngspice.run(RC_LOWPASS)
    assert result.status == SimStatus.OK, result.errors
    f = np.real(result.vector("ac", "frequency"))
    gain_db = 20 * np.log10(np.abs(result.vector("ac", "out")))
    f3db = np.interp(-3.0103, gain_db[::-1], f[::-1])
    assert f3db == pytest.approx(1 / (2 * math.pi * 1.6e3 * 100e-9), rel=0.02)


def test_operating_point_values(ngspice: Ngspice) -> None:
    result = ngspice.run("divider\nV1 a 0 DC 3\nR1 a b 2k\nR2 b 0 1k\n.op\n.end\n")
    assert result.ok
    assert result.op("b") == pytest.approx(1.0)
    assert result.op("v(b)") == pytest.approx(1.0)
    assert result.op("v1#branch") == pytest.approx(-1e-3)  # current into the + terminal


def test_bjt_multiplier_scales_current(ngspice: Ngspice) -> None:
    def collector_current(m: int) -> float:
        net = (
            f"bjt\n{generic_models()}\nVBE b 0 DC 0.65\nVCC c 0 DC 2\n"
            f"Q1 c b 0 NPN_GEN m={m}\n.save all @q1[ic]\n.op\n.end\n"
        )
        result = ngspice.run(net)
        assert result.ok, result.errors
        return result.op("@q1[ic]")

    assert collector_current(2) == pytest.approx(2 * collector_current(1), rel=1e-6)


def test_device_operating_point_parameters(ngspice: Ngspice) -> None:
    net = (
        f"mos\n{generic_models()}\nVGS g 0 DC 1\nVDS d 0 DC 2\n"
        "M1 d g 0 0 NMOS_GEN W=10u L=1u\n"
        ".save all @m1[id] @m1[von] @m1[vdsat] @m1[vgs] @m1[vds]\n.op\n.end\n"
    )
    result = ngspice.run(net)
    assert result.ok
    assert result.op("@m1[von]") == pytest.approx(0.5)
    assert result.op("@m1[vdsat]") == pytest.approx(0.5)  # vgs - von for level 1
    # Id = KP/2 * W/L * vov^2 * (1 + lambda * vds) = 1e-3 * 0.25 * 1.1
    assert result.op("@m1[id]") == pytest.approx(0.275e-3, rel=1e-3)


def test_parse_error(ngspice: Ngspice) -> None:
    result = ngspice.run("bad\nR1 a 0 1k\nQ1 a a 0 NO_SUCH_MODEL\n.op\n.end\n")
    assert result.status == SimStatus.PARSE_ERROR
    assert result.errors


def test_operating_point_failure(ngspice: Ngspice) -> None:
    # Two ideal sources in parallel: no solution exists.
    result = ngspice.run("loop\nV1 a 0 DC 1\nV2 a 0 DC 2\nR1 a 0 1k\n.op\n.end\n")
    assert result.status == SimStatus.NO_CONVERGENCE


def test_timeout_then_recovers(ngspice: Ngspice) -> None:
    # About 2 million time steps: takes ~2 s but needs little memory. (A transient so long that
    # ngspice can't allocate it makes ngspice unusable instead: that's SimStatus.FATAL.)
    slow = "slow\nV1 a 0 SIN(0 1 1meg)\nR1 a b 1k\nC1 b 0 1n\n.tran 10n 20m 0 10n\n.end\n"
    result = ngspice.run(slow, timeout_s=0.1)
    assert result.status == SimStatus.TIMEOUT
    assert ngspice.usable
    assert ngspice.run(RC_LOWPASS).ok


def test_failures_do_not_leak_into_next_run(ngspice: Ngspice) -> None:
    ngspice.run("bad\nQ1 a a 0 NO_SUCH_MODEL\n.op\n.end\n")
    first = ngspice.run(RC_LOWPASS)
    second = ngspice.run(RC_LOWPASS)
    assert first.ok and second.ok
    assert list(first.plots) == ["ac1"]
    np.testing.assert_array_equal(first.vector("ac", "out"), second.vector("ac", "out"))


def test_single_instance() -> None:
    assert Ngspice.get() is Ngspice.get()
    with pytest.raises(RuntimeError):
        Ngspice("/nonexistent")
