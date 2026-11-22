# Analog-Circuit-Designer

Evolves analog circuit topologies with a genetic algorithm. Each candidate circuit is a bipartite
graph of components and nets, simulated with ngspice (`.op` + `.ac`), and scored against a target
frequency response plus power and size objectives. See [DESIGN.md](DESIGN.md) for the design.

## Setup

Requires macOS or Linux, Python 3.12+, [uv](https://docs.astral.sh/uv/) and the ngspice shared
library.

```sh
brew install libngspice uv     # Linux: apt install libngspice0 (plus uv)
uv sync                        # creates .venv with the package and dev tools
```

If the library is installed somewhere unusual, point `ACD_NGSPICE_LIB` at the `.dylib`/`.so` file.

## Development

```sh
uv run pytest                  # tests (ngspice tests are skipped if the library is missing)
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Usage

```sh
uv run acd run configs/lowpass_1k.toml          # passive target, ~3 s
uv run acd run configs/amp_20db.toml            # needs transistors, ~12 s on 12 cores
uv run acd run configs/amp_20db.toml --generations 300 --seed 7 --workers 4
uv run acd run configs/sine_log.toml            # a sine on a Bode plot, with sizing (~2 min)
uv run acd plot runs/<run-dir>                  # redraws response.png (runs also write it)

uv run acd eval tests/fixtures/ce_amp.cir --target targets/amp_20db.toml
uv run acd testbench tests/fixtures/ce_amp.cir  # full netlist; runs as-is in ngspice
```

Each run writes `runs/<timestamp>_<name>/`:

| File | Contents |
|---|---|
| `best.cir` | best circuit as a `.subckt` |
| `best_testbench.cir` | complete netlist for the best circuit, ready to simulate |
| `best_response.csv` | its frequency response (`freq_hz, gain_db, phase_deg`) |
| `summary.json` | final objectives, response summary, operating point per transistor |
| `log.jsonl` | one line per generation: fitness, why candidates failed, why mutations were rejected |
| `top/rank<N>.cir` | the five best circuits in the final population |
| `target.json` | the target as evaluated (used by `acd plot`) |
| `response.png` | the best circuit's frequency response against the target |

Configs are TOML: see `configs/` for every section (`[ga]`, `[mutation]`, `[crossover]`,
`[testbench]`, `[weights]`) and `targets/` for spec-mask, curve and sine targets. Unknown keys are errors, so typos
don't get silently ignored. Circuits for `seed_circuits` or `acd eval` are `.subckt` files like
those in `tests/fixtures/`.

Calling `acd.experiment.run_experiment` from your own script with more than one worker needs the
usual `if __name__ == "__main__":` guard: worker processes start fresh and re-import the script.
