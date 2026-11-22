"""ngspice in-process, through its shared library (libngspice) and ctypes.

One library instance per process: libngspice keeps global state and isn't safe to share across
threads, so use ``Ngspice.get()`` and give each worker process its own copy.

Each run loads the netlist with ``ngSpice_Circ`` and starts it with ``bg_run`` on ngspice's
background thread, so a run that hangs can be stopped with ``bg_halt``. ngspice reports success
and failure only through its printed output (a failed operating point still leaves an ``op1``
plot), so the status comes from the error lines it prints.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import threading
import time
from ctypes import CFUNCTYPE, POINTER, Structure, c_bool, c_char_p, c_double, c_int, c_void_p
from pathlib import Path
from typing import ClassVar

import numpy as np

from acd.sim.backend import SimResult, SimStatus, Vector

_LIB_CANDIDATES = (
    "/opt/homebrew/lib/libngspice.dylib",
    "/usr/local/lib/libngspice.dylib",
    "/usr/lib/x86_64-linux-gnu/libngspice.so.0",
    "/usr/lib/aarch64-linux-gnu/libngspice.so.0",
    "/usr/lib/libngspice.so.0",
    "/usr/local/lib/libngspice.so",
)

_ANALYSIS_RE = re.compile(r"^\s*\.(op|ac|dc|tran|noise|tf|pz|sens|disto)\b", re.IGNORECASE)
_CONVERGENCE_HINTS = (
    "operating point",
    "timestep too small",
    "gmin",
    "source stepping",
    "singular",
    "convergence",
    "iteration limit",
)
_IGNORED = ("can't find the initialization file", "there is no circuit loaded")


class NgspiceNotFound(RuntimeError):
    pass


def find_library() -> str:
    """Path of libngspice: ``$ACD_NGSPICE_LIB``, the system search path, then common locations."""
    env = os.environ.get("ACD_NGSPICE_LIB")
    if env:
        return env
    for path in _LIB_CANDIDATES:
        if Path(path).exists():
            return path
    found = ctypes.util.find_library("ngspice")
    if found:
        return found
    raise NgspiceNotFound(
        "libngspice not found. Install it (macOS: `brew install libngspice`, Debian/Ubuntu: "
        "`apt install libngspice0`) or set ACD_NGSPICE_LIB to the library file."
    )


class _Complex(Structure):
    _fields_ = (("real", c_double), ("imag", c_double))


class _VectorInfo(Structure):
    _fields_ = (
        ("v_name", c_char_p),
        ("v_type", c_int),
        ("v_flags", ctypes.c_short),
        ("v_realdata", POINTER(c_double)),
        ("v_compdata", POINTER(_Complex)),
        ("v_length", c_int),
    )


_SendChar = CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
_SendStat = CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
_ControlledExit = CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)
_BGThreadRunning = CFUNCTYPE(c_int, c_bool, c_int, c_void_p)


class Ngspice:
    """The process-wide ngspice instance. Implements the ``Simulator`` interface."""

    _instance: ClassVar[Ngspice | None] = None
    _create_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get(cls, lib_path: str | None = None) -> Ngspice:
        with cls._create_lock:
            if cls._instance is None:
                cls._instance = cls(lib_path or find_library())
            return cls._instance

    def __init__(self, lib_path: str) -> None:
        if type(self)._instance is not None:
            raise RuntimeError("use Ngspice.get(); libngspice can only be initialized once")
        self._lib = ctypes.CDLL(lib_path)
        self._lock = threading.Lock()
        self._output: list[str] = []
        self._finished = threading.Event()
        self._exited = False  # set when ngspice hits an error it can't recover from

        # Keep references to the callbacks, or ctypes frees them while ngspice still calls them.
        # Every slot gets a real callback: ngspice 47 crashes on some NULL ones.
        self._callbacks = (
            _SendChar(self._on_output),
            _SendStat(lambda _s, _i, _p: 0),
            _ControlledExit(self._on_exit),
            _BGThreadRunning(self._on_thread),
        )
        lib = self._lib
        lib.ngSpice_Init.argtypes = (
            _SendChar, _SendStat, _ControlledExit, c_void_p, c_void_p, _BGThreadRunning, c_void_p,
        )  # fmt: skip
        lib.ngSpice_Command.argtypes = (c_char_p,)
        lib.ngSpice_Circ.argtypes = (POINTER(c_char_p),)
        lib.ngGet_Vec_Info.argtypes = (c_char_p,)
        lib.ngGet_Vec_Info.restype = POINTER(_VectorInfo)
        lib.ngSpice_AllPlots.restype = POINTER(c_char_p)
        lib.ngSpice_AllVecs.argtypes = (c_char_p,)
        lib.ngSpice_AllVecs.restype = POINTER(c_char_p)
        lib.ngSpice_running.restype = c_bool
        send_char, send_stat, controlled_exit, bg_running = self._callbacks
        lib.ngSpice_Init(send_char, send_stat, controlled_exit, None, None, bg_running, None)
        self._finished.set()

    # ---------------------------------------------------------------- callbacks

    def _on_output(self, text: bytes, _id: int, _user: c_void_p) -> int:
        self._output.append(text.decode(errors="replace"))
        return 0

    def _on_exit(self, _status: int, _unload: bool, _quit: bool, _id: int, _user: c_void_p) -> int:
        self._exited = True
        return 0

    def _on_thread(self, not_running: bool, _id: int, _user: c_void_p) -> int:
        if not_running:
            self._finished.set()
        return 0

    # ------------------------------------------------------------------- helpers

    def _command(self, cmd: str) -> int:
        return int(self._lib.ngSpice_Command(cmd.encode()))

    def _strings(self, ptr: ctypes._Pointer[c_char_p]) -> list[str]:
        out: list[str] = []
        i = 0
        while ptr and ptr[i]:
            value = ptr[i]
            assert value is not None
            out.append(value.decode())
            i += 1
        return out

    def _read_vector(self, name: str) -> Vector | None:
        info_ptr = self._lib.ngGet_Vec_Info(name.encode())
        if not info_ptr:
            return None
        info = info_ptr.contents
        n = info.v_length
        if info.v_compdata:
            raw = np.ctypeslib.as_array(
                ctypes.cast(info.v_compdata, POINTER(c_double)), shape=(2 * n,)
            )
            return raw.view(np.complex128).copy()
        if info.v_realdata:
            return np.ctypeslib.as_array(info.v_realdata, shape=(n,)).astype(np.float64)
        return np.zeros(0)

    def _read_plots(self) -> dict[str, dict[str, Vector]]:
        plots: dict[str, dict[str, Vector]] = {}
        for plot in self._strings(self._lib.ngSpice_AllPlots()):
            if plot == "const":
                continue
            vectors: dict[str, Vector] = {}
            for vec in self._strings(self._lib.ngSpice_AllVecs(plot.encode())):
                data = self._read_vector(f"{plot}.{vec}")
                if data is not None:
                    vectors[vec.lower()] = data
            plots[plot] = vectors
        return plots

    def _take_output(self) -> tuple[list[str], list[str]]:
        """Split ngspice's printed output since the last call into errors and warnings."""
        errors, warnings = [], []
        for line in self._output:
            stream, _, text = line.partition(" ")
            text = text.strip()
            low = text.lower()
            if any(s in low for s in _IGNORED):
                continue
            if low.startswith("error") or "not parsed" in low:
                errors.append(text)
            elif stream == "stderr":
                warnings.append(text)
        self._output.clear()
        return errors, warnings

    def _reset(self) -> None:
        self._command("destroy all")
        self._command("remcirc")
        self._output.clear()

    # ------------------------------------------------------------------------ run

    def run(self, netlist: str, timeout_s: float | None = 10.0) -> SimResult:
        """Simulate a complete netlist (title line, elements, analyses, ``.end``)."""
        with self._lock:
            return self._run(netlist, timeout_s)

    def _run(self, netlist: str, timeout_s: float | None) -> SimResult:
        start = time.perf_counter()
        if self._exited:
            return SimResult(SimStatus.FATAL, errors=["ngspice exited; restart the process"])
        if not self._finished.is_set():
            # A previous run timed out and never stopped. The instance can't be trusted.
            return SimResult(SimStatus.FATAL, errors=["previous run still active"])
        self._reset()

        lines = netlist.splitlines()
        expected = sum(1 for ln in lines if _ANALYSIS_RE.match(ln))
        arr = (c_char_p * (len(lines) + 1))(*(ln.encode() for ln in lines), None)
        self._lib.ngSpice_Circ(arr)
        errors, warnings = self._take_output()
        if errors:
            return SimResult(
                SimStatus.PARSE_ERROR, errors=errors, warnings=warnings,
                elapsed_s=time.perf_counter() - start,
            )  # fmt: skip

        self._finished.clear()
        self._command("bg_run")
        if not self._finished.wait(timeout_s):
            self._command("bg_halt")
            stopped = self._finished.wait(1.0)
            _, warnings = self._take_output()
            msg = f"timed out after {timeout_s} s" + ("" if stopped else " and did not stop")
            return SimResult(
                SimStatus.TIMEOUT, errors=[msg], warnings=warnings,
                elapsed_s=time.perf_counter() - start,
            )  # fmt: skip

        errors, warnings = self._take_output()
        plots = self._read_plots()
        status = SimStatus.OK
        if errors:
            joined = " ".join(errors).lower()
            converge = any(h in joined for h in _CONVERGENCE_HINTS)
            status = SimStatus.NO_CONVERGENCE if converge else SimStatus.ERROR
        elif len(plots) < expected:
            status = SimStatus.ERROR
            errors = [f"expected {expected} analyses, got plots {list(plots)}"]
        if self._exited:
            # E.g. a failed memory allocation: ngspice waits to be reset and can't run anything.
            status = SimStatus.FATAL
            errors.append("ngspice requested exit")
        return SimResult(status, plots, errors, warnings, time.perf_counter() - start)

    @property
    def usable(self) -> bool:
        """False once ngspice has hit an unrecoverable error; restart the process."""
        return not self._exited and self._finished.is_set()
