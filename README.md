# TP IA ML IPPSI 2026

Data exploration project using scikit-learn datasets.<br>
Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python main.py
```

## Tasks

| Command | Description |
|---|---|
| `make test` | Run tests with coverage |
| `make lint` | Lint with ruff |
| `make format` | Format with ruff |
| `make ci` | Lint + test |

## Structure

```
src/tp_ia/data/loader.py   — dataset loading
tests/                     — pytest suite
```
