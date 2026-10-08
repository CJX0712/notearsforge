# notearsforge -- Makefile (one-command dev loop)

VENV ?= .venv
PY ?= python3
PIP := $(VENV)/bin/pip
ifeq ($(OS),Windows_NT)
	PIP := $(VENV)/Scripts/pip.exe
	PYRUN := $(VENV)/Scripts/python.exe
else
	PYRUN := $(VENV)/bin/python
endif

.PHONY: help venv deps check test fmt demo all clean

help:
	@echo "Targets: venv | deps | check (ruff) | test | demo | all | clean"

venv:
	$(PY) -m venv $(VENV)

deps: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.lock.txt
	$(PIP) install "pytest>=8" "ruff==0.16.10" pytest-cov

check:
	$(PYRUN) -m ruff check .
	$(PYRUN) -m ruff format --check .

test:
	$(PYRUN) -m pytest -q -W ignore::UserWarning --cov=. --cov-report=term

fmt:
	$(PYRUN) -m ruff check --fix .
	$(PYRUN) -m ruff format .

demo:
	$(PYRUN) examples/run_demo.py

# Full pre-push gate (mirrors CI): lint + format check + tests + demo determinism
all: check test demo
	@echo "[notearsforge] all gates green"

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
