.PHONY: install install-dev lint format typecheck test test-cov reproduce clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,docs]"
	pre-commit install

lint:
	ruff check wfcrc tests
	black --check wfcrc tests

format:
	ruff check --fix wfcrc tests
	black wfcrc tests

typecheck:
	mypy wfcrc

test:
	pytest --no-cov

test-cov:
	pytest

# MS1 stub: later milestones wire this to re-run a reference experiment and
# diff its metrics against a stored golden file (Implementation Blueprint §17).
reproduce:
	@echo "make reproduce: no experiments implemented yet (MS1 scope is infrastructure only)."

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
