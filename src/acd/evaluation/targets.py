"""Target frequency responses: an exact curve to match, or a spec mask of allowed bands.

Target files are TOML::

    name = "lowpass 1 kHz"
    type = "mask"

    [[band]]               # passband: within +-1 dB
    f_lo = 10
    f_hi = 1000
    min_db = -1
    max_db = 1

    [[band]]               # stopband: below -20 dB (a missing bound is unbounded)
    f_lo = 10000
    f_hi = 100000
    max_db = -20

or a curve, given inline or as a CSV file with columns ``freq_hz,gain_db[,weight]``::

    name = "flat 20 dB"
    type = "curve"
    freq_hz = [100, 1000, 10000]
    gain_db = [20, 20, 20]
    # file = "my_curve.csv"   (path relative to the TOML file)

Frequencies are plain numbers (``1e3``, not ``"1k"``), which avoids SPICE's ``M`` = milli trap.
"""

from __future__ import annotations

import csv
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from acd.evaluation.metrics import FloatArray, Response

_MASK_POINTS_PER_BAND = 25


@dataclass(frozen=True)
class CurveTarget:
    freq_hz: tuple[float, ...]
    gain_db: tuple[float, ...]
    weight: tuple[float, ...]
    name: str = "curve"

    def __post_init__(self) -> None:
        n = len(self.freq_hz)
        if n == 0 or len(self.gain_db) != n or len(self.weight) != n:
            raise ValueError("curve target needs matching, non-empty freq_hz/gain_db/weight")
        if any(f <= 0 for f in self.freq_hz):
            raise ValueError("frequencies must be positive")

    def error_db(self, response: Response) -> float:
        """Weighted RMS difference between the response and the curve, in dB."""
        f = np.asarray(self.freq_hz)
        diff = response.gain_db_at(f) - np.asarray(self.gain_db)
        w = np.asarray(self.weight)
        return float(np.sqrt(np.sum(w * diff**2) / np.sum(w)))

    def frequency_range(self) -> tuple[float, float]:
        return min(self.freq_hz), max(self.freq_hz)


@dataclass(frozen=True)
class Band:
    f_lo: float
    f_hi: float
    min_db: float = -math.inf
    max_db: float = math.inf
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.f_lo <= self.f_hi:
            raise ValueError(f"band needs 0 < f_lo <= f_hi, got {self.f_lo}, {self.f_hi}")
        if self.min_db > self.max_db:
            raise ValueError("band min_db is above max_db")

    def sample_freqs(self) -> FloatArray:
        return np.geomspace(self.f_lo, self.f_hi, _MASK_POINTS_PER_BAND)

    def excess_db(self, gain_db: FloatArray) -> FloatArray:
        """How far each gain value lies outside the band's limits (0 inside)."""
        return np.maximum(self.min_db - gain_db, 0) + np.maximum(gain_db - self.max_db, 0)


@dataclass(frozen=True)
class MaskTarget:
    bands: tuple[Band, ...]
    name: str = "mask"

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("mask target needs at least one band")

    def error_db(self, response: Response) -> float:
        """Weighted RMS of how far the response lies outside the mask, in dB (0 = spec met)."""
        total, weights = 0.0, 0.0
        for band in self.bands:
            excess = band.excess_db(response.gain_db_at(band.sample_freqs()))
            total += band.weight * float(np.sum(excess**2))
            weights += band.weight * excess.size
        return math.sqrt(total / weights)

    def frequency_range(self) -> tuple[float, float]:
        return min(b.f_lo for b in self.bands), max(b.f_hi for b in self.bands)


Target = CurveTarget | MaskTarget


def target_from_dict(data: Mapping[str, Any], base_dir: Path | None = None) -> Target:
    kind = data.get("type")
    name = str(data.get("name", kind or "target"))
    if kind == "mask":
        bands = tuple(
            Band(
                f_lo=float(b["f_lo"]),
                f_hi=float(b["f_hi"]),
                min_db=float(b.get("min_db", -math.inf)),
                max_db=float(b.get("max_db", math.inf)),
                weight=float(b.get("weight", 1.0)),
            )
            for b in data.get("band", ())
        )
        return MaskTarget(bands, name)
    if kind == "curve":
        if "file" in data:
            path = Path(data["file"])
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            return _curve_from_csv(path, name)
        freqs = tuple(float(x) for x in data["freq_hz"])
        gains = tuple(float(x) for x in data["gain_db"])
        weights = tuple(float(x) for x in data.get("weight", [1.0] * len(freqs)))
        return CurveTarget(freqs, gains, weights, name)
    raise ValueError(f"target type must be 'mask' or 'curve', got {kind!r}")


def _curve_from_csv(path: Path, name: str) -> CurveTarget:
    freqs, gains, weights = [], [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            freqs.append(float(row["freq_hz"]))
            gains.append(float(row["gain_db"]))
            weights.append(float(row.get("weight") or 1.0))
    return CurveTarget(tuple(freqs), tuple(gains), tuple(weights), name)


def load_target(path: str | Path) -> Target:
    path = Path(path)
    with path.open("rb") as fh:
        return target_from_dict(tomllib.load(fh), path.parent)
