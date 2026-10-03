.DEFAULT_GOAL := help
.PHONY: help install lint format test eval demo red compare record mcp hooks scan \
        lint-docs check-learning-numbers docker-build docker-run clean

IMAGE  ?= ailab-agents:local
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
RUN_A  ?= results/run_tools.json
RUN_B  ?= results/run_notools.json

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

install: ## Create .venv from uv.lock (with dev tools)
	uv sync --locked

lint: ## Ruff lint + format check + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite with coverage (fails under 90%)
	uv run pytest --cov --cov-report=term-missing

eval: ## Run the eval and enforce the regression gate
	uv run ailab-eval

demo: ## Show the agent stages on a couple of tasks, then run the eval
	uv run ailab-demo

red: ## Proof-of-gate: each sabotage must exit non-zero and name the failing metric
	@bash scripts/prove_gate.sh

compare: ## Diff two saved runs that change ONE variable (RUN_A vs RUN_B)
	uv run python -m ailab_agents.compare $(RUN_A) $(RUN_B)

record: ## Record the agent cassette from a live model (needs OLLAMA_HOST; opt-in)
	uv run python scripts/record_cassette.py

mcp: ## Smoke the MCP server: list tools over a real stdio round-trip
	printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | uv run python -m ailab_agents.mcp_server

hooks: ## Enable the repo's git hooks (required once per clone)
	git config core.hooksPath .githooks
	@echo "hooks enabled: .githooks/pre-push will run gitleaks + denylist scan"

scan: ## Secret scan: gitleaks over full history + denylist scan
	gitleaks git --no-banner --redact --config .gitleaks.toml .
	scripts/denylist_scan.sh

lint-docs: ## Fail if any em dash (U+2014) appears in docs/ or *.md
	bash scripts/no_emdash.sh

check-learning-numbers: ## Assert tagged docs/LEARNING.md figures match the committed results
	uv run python scripts/check_learning_numbers.py

docker-build: ## Build the runtime image
	docker build --build-arg GIT_COMMIT=$(COMMIT) -t $(IMAGE) .

docker-run: ## Run the image (demo + eval)
	docker run --rm $(IMAGE)

clean: ## Remove caches and generated results
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
