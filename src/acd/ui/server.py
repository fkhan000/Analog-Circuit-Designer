"""Local web UI: draw a target frequency response, run the GA, inspect the resulting circuit.

``acd ui`` starts a small HTTP server on 127.0.0.1 (standard library only). The page
(``static/index.html``) posts the drawn curve to ``/api/run``; the search runs in a background
thread (its evaluations in worker processes) and the page polls ``/api/jobs/<id>`` for progress,
the current best circuit and its response. One search runs at a time.
"""

from __future__ import annotations

import itertools
import json
import math
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from acd.circuit.devices import Kind, spec
from acd.circuit.graph import Circuit
from acd.circuit.netlist import to_subckt
from acd.evaluation.metrics import Response, summarize
from acd.evaluation.stages import Evaluation, evaluate
from acd.evaluation.testbench import build_testbench
from acd.experiment import ExperimentConfig, config_from_dict, run_experiment
from acd.sim.ngspice import Ngspice

STATIC = Path(__file__).parent / "static"
MIN_POINTS = 3
MAX_POINTS = 400
F_MIN, F_MAX = 1.0, 1e9


class RequestError(ValueError):
    """A run request the server refuses (sent back as HTTP 400)."""


@dataclass(frozen=True)
class RunRequest:
    points: tuple[tuple[float, float], ...]  # (frequency in Hz, gain in dB), sorted by frequency
    time_limit_s: float = 60.0
    max_components: int = 10
    kinds: tuple[Kind, ...] = tuple(Kind)
    sizing: bool = True
    seed: int = 1


def parse_request(data: Any) -> RunRequest:
    if not isinstance(data, dict):
        raise RequestError("expected a JSON object")
    raw = data.get("points")
    if not isinstance(raw, list) or not MIN_POINTS <= len(raw) <= MAX_POINTS:
        raise RequestError(f"draw a response first (need {MIN_POINTS}-{MAX_POINTS} points)")
    points: dict[float, float] = {}
    for p in raw:
        try:
            f, g = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            raise RequestError("points must be [frequency_hz, gain_db] pairs") from None
        if not (F_MIN <= f <= F_MAX and math.isfinite(g) and -200 <= g <= 200):
            raise RequestError(f"point out of range: {f:g} Hz, {g:g} dB")
        points[f] = g
    if len(points) < MIN_POINTS:
        raise RequestError(f"need at least {MIN_POINTS} distinct frequencies")

    def number(key: str, default: float, lo: float, hi: float) -> float:
        value = float(data.get(key, default))
        if not lo <= value <= hi:
            raise RequestError(f"{key} must be between {lo:g} and {hi:g}")
        return value

    kinds_raw = data.get("kinds", [k.value for k in Kind])
    try:
        kinds = tuple(Kind(str(k).upper()) for k in kinds_raw)
    except ValueError as e:
        raise RequestError(str(e)) from None
    if not kinds:
        raise RequestError("allow at least one part type")
    return RunRequest(
        points=tuple(sorted(points.items())),
        time_limit_s=number("time_limit_s", 60, 5, 3600),
        max_components=int(number("max_components", 10, 1, 10)),
        kinds=kinds,
        sizing=bool(data.get("sizing", True)),
        seed=int(number("seed", 1, 0, 2**31 - 1)),
    )


def build_config(req: RunRequest, workers: int = 0) -> ExperimentConfig:
    freqs = [f for f, _ in req.points]
    gains = [g for _, g in req.points]
    data = {
        "name": "ui",
        "target": {"type": "curve", "name": "drawn", "freq_hz": freqs, "gain_db": gains},
        "workers": workers,
        "ga": {
            "population_size": 100 if req.sizing else 300,
            "elite_count": 4,
            "tournament_size": 4,
            "seed": req.seed,
            "time_limit_s": req.time_limit_s,
            "generations": 10**7,
        },
        "mutation": {"max_components": req.max_components, "kinds": [k.value for k in req.kinds]},
        "sizing": {"enabled": req.sizing, "budget": 40},
        "testbench": {"points_per_decade": 30},
    }
    return config_from_dict(data, Path.cwd())


# ------------------------------------------------------------------ payloads


def circuit_payload(circuit: Circuit) -> dict[str, Any]:
    components = []
    for c in circuit.components:
        s = spec(c.kind)
        components.append(
            {
                "id": c.id,
                "kind": c.kind.value,
                "pins": dict(zip(s.pin_names, c.nets, strict=True)),
                "params": [
                    {"name": p.name, "value": v, "unit": p.unit}
                    for p, v in zip(s.params, c.params, strict=True)
                ],
            }
        )
    return {
        "components": components,
        "ports": list(circuit.ports),
        "nets": list(circuit.nets),
        "netlist": to_subckt(circuit),
    }


def response_payload(response: Response) -> dict[str, Any]:
    return {
        "freq_hz": [float(x) for x in response.freq_hz],
        "gain_db": [float(x) for x in response.gain_db],
        "phase_deg": [float(x) for x in response.phase_deg],
    }


def _evaluation_payload(evaluation: Evaluation) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": evaluation.ok, "reason": evaluation.reason}
    if evaluation.response is not None:
        out["response"] = response_payload(evaluation.response)
        summary = summarize(evaluation.response)
        out["summary"] = {
            "lf_gain_db": summary.lf_gain_db,
            "peak_gain_db": summary.peak_gain_db,
            "f_peak_hz": summary.f_peak_hz,
        }
    if evaluation.op is not None:
        out["power_w"] = evaluation.op.power_w
        out["devices"] = [
            {"id": d.component_id, "kind": d.kind.value, "region": d.region.value,
             "current_a": d.current_a}
            for d in evaluation.op.devices
        ]  # fmt: skip
    return out


# ----------------------------------------------------------------------- jobs


@dataclass
class Job:
    id: str
    request: RunRequest
    config: ExperimentConfig
    status: str = "running"  # running, done, stopped, failed
    started: float = field(default_factory=time.time)
    progress: list[dict[str, float]] = field(default_factory=list)
    best: dict[str, Any] | None = None  # live best circuit, response and error
    result: dict[str, Any] | None = None
    error: str = ""
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "time_limit_s": self.request.time_limit_s,
                "elapsed_s": time.time() - self.started,
                "progress": list(self.progress),
                "best": self.best,
                "result": self.result,
                "error": self.error,
            }


def run_job(job: Job, out_root: Path) -> None:
    sim = Ngspice.get()
    elapsed = 0.0
    best_fitness = math.inf

    def progress(record: dict[str, Any], best: Circuit) -> None:
        nonlocal elapsed, best_fitness
        elapsed += record["elapsed_s"]
        objs = record.get("best_objectives") or {}
        point = {
            "generation": record["generation"],
            "elapsed_s": elapsed,
            "fitness": record["best_fitness"],
            "error_db": objs.get("response_error_db", math.nan),
            "parts": record["best_components"],
            "simulations": record["simulated_total"],
        }
        live = None
        if record["best_fitness"] < best_fitness:  # new best: show its response
            best_fitness = record["best_fitness"]
            evaluation = evaluate(best, sim, job.config.testbench, job.config.bias)
            if evaluation.response is not None:
                live = {
                    "circuit": circuit_payload(best),
                    "evaluation": _evaluation_payload(evaluation),
                    "error_db": point["error_db"],
                    "generation": point["generation"],
                }
        with job.lock:
            job.progress.append(point)
            if live is not None:
                job.best = live

    try:
        outcome = run_experiment(
            job.config, out_root=out_root, log=None, progress=progress, should_stop=job.stop.is_set
        )
        summary = json.loads((outcome.run_dir / "summary.json").read_text())
        best = outcome.result.best.genome
        result = {
            "circuit": circuit_payload(best),
            "evaluation": _evaluation_payload(outcome.best_evaluation),
            "objectives": summary.get("objectives"),
            "fitness": summary["fitness"],
            "generations": summary["generations"],
            "simulations": summary["simulated"],
            "stopped_because": summary["stopped_because"],
            "testbench": build_testbench(best, job.config.testbench).netlist,
            "run_dir": str(outcome.run_dir),
        }
        with job.lock:
            job.result = result
            job.status = "stopped" if job.stop.is_set() else "done"
    except Exception as e:  # report to the page instead of killing the server
        traceback.print_exc()
        with job.lock:
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------- server


class App:
    def __init__(self, workers: int = 0, out_root: Path = Path("runs/ui")) -> None:
        self.workers = workers
        self.out_root = out_root
        self.jobs: dict[str, Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def start(self, req: RunRequest) -> Job:
        with self._lock:
            if any(j.status == "running" for j in self.jobs.values()):
                raise RuntimeError("a search is already running")
            job = Job(str(next(self._ids)), req, build_config(req, self.workers))
            self.jobs[job.id] = job
        threading.Thread(target=run_job, args=(job, self.out_root), daemon=True).start()
        return job


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
        pass

    def _send(
        self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _job(self, job_id: str) -> Job | None:
        job = self.server.app.jobs.get(job_id)
        if job is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"no job {job_id}"})
        return job

    def do_GET(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts == [""] or parts == ["index.html"]:
            body = (STATIC / "index.html").read_bytes()
            self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
        elif len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            job = self._job(parts[2])
            if job:
                self._json(HTTPStatus.OK, job.snapshot())
        elif len(parts) == 4 and parts[:2] == ["api", "jobs"]:
            self._download(parts[2], parts[3])
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _download(self, job_id: str, what: str) -> None:
        """``netlist``: the best circuit's .subckt; ``testbench``: the full netlist (when done)."""
        job = self._job(job_id)
        if job is None:
            return
        snap = job.snapshot()
        text: str | None = None
        if what == "netlist":
            source = snap["result"] or snap["best"]
            text = source["circuit"]["netlist"] if source else None
        elif what == "testbench" and snap["result"]:
            text = snap["result"]["testbench"]
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if text is None:
            self._json(HTTPStatus.CONFLICT, {"error": "no circuit yet"})
            return
        disposition = f'attachment; filename="{what}.cir"'
        self._send(
            HTTPStatus.OK, text.encode(), "text/plain; charset=utf-8",
            {"Content-Disposition": disposition},
        )  # fmt: skip

    def do_POST(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts == ["api", "run"]:
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = parse_request(json.loads(self.rfile.read(length) or b"null"))
                job = self.server.app.start(req)
            except (RequestError, json.JSONDecodeError) as e:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            except RuntimeError as e:
                self._json(HTTPStatus.CONFLICT, {"error": str(e)})
                return
            self._json(HTTPStatus.OK, {"job": job.id})
        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            target = self._job(parts[2])
            if target:
                target.stop.set()
                self._json(HTTPStatus.OK, {"stopping": True})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: App) -> None:
        super().__init__(address, Handler)
        self.app = app


def serve(
    port: int = 8765,
    host: str = "127.0.0.1",
    workers: int = 0,
    out_root: Path = Path("runs/ui"),
    open_browser: bool = False,
) -> None:
    Ngspice.get()  # fail fast if ngspice is missing
    server = Server((host, port), App(workers, out_root))
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"acd UI at {url}  (Ctrl+C to quit; runs are saved under {out_root}/)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
