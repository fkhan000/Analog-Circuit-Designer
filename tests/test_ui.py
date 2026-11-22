import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from acd.circuit.devices import Kind
from acd.evaluation.targets import CurveTarget
from acd.ui.server import App, RequestError, Server, build_config, circuit_payload, parse_request
from helpers import load_fixture

POINTS = [[100.0, 0.0], [1e3, -3.0], [1e4, -20.0], [1e5, -40.0]]


def test_parse_valid_request() -> None:
    req = parse_request({"points": [*POINTS, [1e3, -4.0]], "kinds": ["r", "C"], "seed": 3})
    assert req.points == ((100.0, 0.0), (1e3, -4.0), (1e4, -20.0), (1e5, -40.0))  # sorted, deduped
    assert req.kinds == (Kind.R, Kind.C) and req.seed == 3 and req.sizing


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"points": POINTS[:2]}, "draw a response first"),
        ({"points": [[0.1, 0], *POINTS]}, "out of range"),
        ({"points": POINTS, "kinds": ["Q"]}, "not a valid Kind"),
        ({"points": POINTS, "kinds": []}, "at least one part type"),
        ({"points": POINTS, "time_limit_s": 1}, "time_limit_s"),
        ({"points": POINTS, "max_components": 11}, "max_components"),
        ({"points": [["x", 1]] * 4}, "pairs"),
        ([], "JSON object"),
    ],
)
def test_parse_rejects_bad_requests(data: Any, match: str) -> None:
    with pytest.raises(RequestError, match=match):
        parse_request(data)


def test_build_config() -> None:
    config = build_config(parse_request({"points": POINTS, "max_components": 4, "sizing": False}))
    assert isinstance(config.target, CurveTarget)
    assert config.target.freq_hz == (100.0, 1e3, 1e4, 1e5)
    assert config.mutation.max_components == 4 and not config.sizing.enabled
    assert config.ga.population_size == 300  # larger population without sizing
    assert config.testbench.f_start_hz == 50 and config.testbench.f_stop_hz == 2e5


def test_circuit_payload() -> None:
    payload = circuit_payload(load_fixture("ce_amp"))
    q = next(c for c in payload["components"] if c["kind"] == "NPN")
    assert q["pins"] == {"c": "VOUT", "b": "VIN", "e": "e1"}
    r = next(c for c in payload["components"] if c["id"] == "2")
    assert r["params"] == [{"name": "r", "value": 1200.0, "unit": "ohm"}]
    assert payload["netlist"].startswith(".subckt dut")


@pytest.fixture
def server() -> Iterator[str]:
    srv = Server(("127.0.0.1", 0), App(workers=1))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    for job in srv.app.jobs.values():
        job.stop.set()
    srv.shutdown()
    srv.server_close()


def call(url: str, body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read().decode()
            return r.status, json.loads(raw) if r.headers["Content-Type"].startswith(
                "application/json"
            ) else raw
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_done(base: str, job: str, timeout: float = 90) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, s = call(f"{base}/api/jobs/{job}")
        if s["status"] != "running":
            return dict(s)
        time.sleep(0.3)
    raise AssertionError("search didn't finish")


@pytest.mark.ngspice
def test_page_and_errors(server: str) -> None:
    status, page = call(server + "/")
    assert status == 200 and "<title>Circuit Designer</title>" in page
    assert call(server + "/api/run", {"points": POINTS[:1]})[0] == 400
    assert call(server + "/api/jobs/99")[0] == 404
    assert call(server + "/nope")[0] == 404


@pytest.mark.ngspice
def test_search_end_to_end(server: str) -> None:
    body = {"points": POINTS, "time_limit_s": 5, "max_components": 3, "kinds": ["R", "C"]}
    status, started = call(server + "/api/run", body)
    assert status == 200
    job = started["job"]
    assert call(server + "/api/run", body)[0] == 409  # one search at a time
    s = wait_done(server, job)
    assert s["status"] == "done" and s["progress"]
    result = s["result"]
    assert 1 <= len(result["circuit"]["components"]) <= 3
    assert {c["kind"] for c in result["circuit"]["components"]} <= {"R", "C"}
    assert len(result["evaluation"]["response"]["freq_hz"]) > 10
    assert result["objectives"]["response_error_db"] < 10
    assert call(f"{server}/api/jobs/{job}/netlist")[1].startswith(".subckt")
    assert ".ac dec" in call(f"{server}/api/jobs/{job}/testbench")[1]


@pytest.mark.ngspice
def test_stop(server: str) -> None:
    _, started = call(server + "/api/run", {"points": POINTS, "time_limit_s": 600, "sizing": False})
    time.sleep(1.5)
    assert call(f"{server}/api/jobs/{started['job']}/stop", {})[0] == 200
    s = wait_done(server, started["job"], timeout=30)
    assert s["status"] == "stopped" and s["result"]["stopped_because"] == "stopped"
