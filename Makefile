# FlowPilot — Makefile
# Intended to run on the Jetson Orin Nano (Linux/ARM64).
# On Windows, use WSL or run commands manually.

PYTHON   := python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
RUFF     := $(VENV)/bin/ruff
BLACK    := $(VENV)/bin/black
RESULTS  := benchmarks/results

.PHONY: setup lint fmt test bench bench-compare clean

# ── Environment ──────────────────────────────────────────────────────────────

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Venv ready. For CUDA extras run: $(PIP) install -e '.[cuda]'"
	@echo "For Jetson PyTorch: see scripts/setup_jetson.sh"

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	$(RUFF) check flowpilot tests
	$(BLACK) --check flowpilot tests

fmt:
	$(RUFF) check --fix flowpilot tests
	$(BLACK) flowpilot tests

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(PYTEST) -m "not cuda and not jetson"

test-cuda:
	$(PYTEST) -m cuda

test-all:
	$(PYTEST)

# ── Benchmarks ───────────────────────────────────────────────────────────────

$(RESULTS):
	mkdir -p $(RESULTS)

bench: $(RESULTS)
	$(PYTHON) -m flowpilot.benchmarks.hardware --out $(RESULTS)/hardware_$(shell date +%Y%m%d_%H%M%S).json

bench-compare:
	$(PYTHON) -m flowpilot.benchmarks.compare $(RESULTS)

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) __pycache__ .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
