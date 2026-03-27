SHELL := /bin/bash

VENV ?= .venv
PYTHON ?= python3
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV_PYTHON) -m pip
VENV_PYTEST := $(VENV)/bin/pytest
VENV_RUFF := $(VENV)/bin/ruff
INSTALL_STAMP := $(VENV)/.stamp-install
DEV_INSTALL_STAMP := $(VENV)/.stamp-install-dev

.PHONY: help venv install install-dev install-all test lint smoke clean

help:
	@echo "Targets:"
	@echo "  make venv          Create venv and upgrade pip"
	@echo "  make install       Install package only"
	@echo "  make install-dev   Install package + dev dependencies"
	@echo "  make install-all   Alias for install-dev"
	@echo "  make test          Run pytest"
	@echo "  make lint          Run ruff checks"
	@echo "  make smoke         Run random rollout sanity check"
	@echo "  make clean         Remove caches/build artifacts"

venv: $(VENV_PYTHON)

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip

$(INSTALL_STAMP): pyproject.toml | venv
	@if [ -f "$(DEV_INSTALL_STAMP)" ]; then \
		touch "$(INSTALL_STAMP)"; \
	else \
		$(VENV_PIP) install -e .; \
		touch "$(INSTALL_STAMP)"; \
	fi

install: $(INSTALL_STAMP)

$(DEV_INSTALL_STAMP): pyproject.toml | venv
	$(VENV_PIP) install -e '.[dev]'
	touch "$(DEV_INSTALL_STAMP)"
	touch "$(INSTALL_STAMP)"

install-dev: $(DEV_INSTALL_STAMP)

install-all: install-dev

test: install-dev
	$(VENV_PYTEST) -q

lint: install-dev
	$(VENV_RUFF) check src tests scripts

smoke: install
	$(VENV_PYTHON) scripts/random_rollout.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf build dist *.egg-info
	rm -f "$(INSTALL_STAMP)" "$(DEV_INSTALL_STAMP)"
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
