"""Plot a run: the best circuit's response against the target. Needs the ``plot`` extra.

``uv sync --extra plot`` installs matplotlib.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from acd.evaluation.targets import CurveTarget, MaskTarget, target_from_dict

# Reference palette (dataviz skill): first two categorical slots, light surface, text inks.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
RESPONSE = "#2a78d6"  # series 1
TARGET = "#eb6834"  # series 2


def _read_response(path: Path) -> tuple[np.ndarray[Any, Any], ...]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = ("freq_hz", "gain_db", "phase_deg")
    return tuple(np.array([float(r[c]) for r in rows]) for c in cols)


def _is_linear(freqs: tuple[float, ...]) -> bool:
    steps = np.diff(freqs)
    return len(steps) > 2 and bool(np.allclose(steps, steps[0], rtol=1e-6))


def plot_run(run_dir: str | Path, out: str | Path | None = None) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit("plotting needs matplotlib: run `uv sync --extra plot`") from e

    run_dir = Path(run_dir)
    freq, gain, phase = _read_response(run_dir / "best_response.csv")
    target = target_from_dict(json.loads((run_dir / "target.json").read_text()))
    summary = json.loads((run_dir / "summary.json").read_text())

    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
        }
    )
    fig, (ax_gain, ax_phase) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6.5), height_ratios=(3, 1), facecolor=SURFACE
    )
    for ax in (ax_gain, ax_phase):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    linear = isinstance(target, CurveTarget) and _is_linear(target.freq_hz)
    if isinstance(target, CurveTarget):
        ax_gain.plot(
            target.freq_hz, target.gain_db, color=TARGET, linewidth=2, linestyle=(0, (5, 3)),
            label="target",
        )  # fmt: skip
        lo, hi = target.frequency_range()
    else:
        assert isinstance(target, MaskTarget)
        ymin = min(
            float(np.min(gain)), *(b.min_db for b in target.bands if math.isfinite(b.min_db))
        )
        ymax = max(
            float(np.max(gain)), *(b.max_db for b in target.bands if math.isfinite(b.max_db))
        )
        pad = 0.1 * (ymax - ymin + 1)
        for i, b in enumerate(target.bands):
            lower = b.min_db if math.isfinite(b.min_db) else ymin - pad
            upper = b.max_db if math.isfinite(b.max_db) else ymax + pad
            ax_gain.fill_between(
                [b.f_lo, b.f_hi], lower, upper, color=TARGET, alpha=0.18, linewidth=0,
                label="target (allowed region)" if i == 0 else None,
            )  # fmt: skip
        lo, hi = target.frequency_range()

    ax_gain.plot(freq, gain, color=RESPONSE, linewidth=2, label="best circuit")
    ax_phase.plot(freq, phase, color=RESPONSE, linewidth=2)
    if not linear:
        ax_gain.set_xscale("log")
    ax_gain.set_xlim(lo / 2 if not linear else 0, hi * 2 if not linear else hi * 1.02)
    ax_gain.set_ylabel("gain (dB)")
    ax_phase.set_ylabel("phase (deg)")
    ax_phase.set_xlabel("frequency (Hz)")
    ax_gain.legend(frameon=False, loc="best")

    objectives = summary.get("objectives") or {}
    error = objectives.get("response_error_db")
    parts = objectives.get("n_components")
    subtitle = f"response error {error:.3g} dB RMS, {parts} parts" if error is not None else ""
    ax_gain.set_title(
        f"{summary.get('name', run_dir.name)}  ({target.name})\n{subtitle}",
        loc="left", fontsize=11, color=TEXT_PRIMARY,
    )  # fmt: skip
    fig.tight_layout()
    out_path = Path(out) if out else run_dir / "response.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
