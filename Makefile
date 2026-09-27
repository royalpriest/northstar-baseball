.PHONY: install test lint baseline

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

baseline:
	python scripts/build_baseline.py --start 2025-03-18 --end 2025-09-28
