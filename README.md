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
