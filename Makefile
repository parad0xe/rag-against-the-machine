MAKEFLAGS=--no-print-directory

# structure
ARGS ?= 


VENV := .venv
VENV_STATE_PROD := $(VENV)/.install
VENV_STATE_DEV := $(VENV)/.install-dev

UV_LOCK := uv.lock
PYPROJECT_TOML := pyproject.toml

# cache
CACHE_DIRS := __pycache__ .mypy_cache .pytest_cache
CACHE_EXCLUDE = -name "$(VENV)" -prune -o
CACHE_SEARCH = $(foreach cache,$(CACHE_DIRS),-name "$(cache)" -o)
FIND_CACHES = find . \
	$(CACHE_EXCLUDE) \
	-type d \( $(CACHE_SEARCH) -false \) -print

# tools
UV := uv
PYTHON := $(VENV)/bin/python3
FLAKE8 := $(PYTHON) -m flake8 --exclude $(VENV),libs,.git,vllm*,
MYPY := $(PYTHON) -m mypy --exclude $(VENV) --exclude libs --exclude .git --exclude vllm*

# rules
install: uv_check $(UV_LOCK) $(VENV_STATE_PROD)

install-dev: uv_check $(UV_LOCK) $(VENV_STATE_DEV)

run: install
	@echo "$(UV) run python -m src $(ARGS)"
	@$(UV) run python -m src $(ARGS)

$(UV_LOCK): $(PYPROJECT_TOML)
	@$(UV) lock
	@touch $(UV_LOCK)


$(VENV_STATE_PROD): $(PYPROJECT_TOML)
	@$(UV) sync --no-dev --inexact
	@touch $(VENV_STATE_PROD)

$(VENV_STATE_DEV): $(PYPROJECT_TOML)
	@$(UV) sync
	@touch $(VENV_STATE_PROD) $(VENV_STATE_DEV)

cache-clean:
	$(FIND_CACHES) -exec rm -rf {} + 1>/dev/null

clean: cache-clean
	rm -rf $(VENV)

debug: install-dev
	$(UV) run python -m pdb -m src $(ARGS)

lint: install-dev
	@$(FLAKE8)
	@$(MYPY) . --check-untyped-defs \
	--warn-unused-ignores --ignore-missing-imports \
	--warn-return-any --disallow-untyped-defs

lint-strict: install-dev
	@$(FLAKE8)
	@$(MYPY) . --strict
	
uv_check:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "Error: The '$(UV)' tool is not installed or not in the PATH." >&2; \
		echo "To install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; \
		exit 1; \
	}

.PHONY: install install-dev run cache-clean clean debug lint lint-strict uv_check
