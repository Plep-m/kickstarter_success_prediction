.PHONY: install lint format

install:
	uv sync --all-extras

lint:
	uv run ruff check src

format:
	uv run ruff format src
