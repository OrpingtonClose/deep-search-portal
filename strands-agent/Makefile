# Strands Agent — development targets
# Usage: make [target]

.PHONY: test test-integ lint format install help

# Run unit evals (no API keys required)
test:
	python -m pytest evals/ -v --tb=short -m "not integ"

# Run integration evals (requires VENICE_API_KEY)
test-integ:
	python -m pytest evals/ -v --tb=short -m "integ"

# Run all evals
test-all:
	python -m pytest evals/ -v --tb=short

# Lint with ruff
lint:
	python -m ruff check .
	python -m ruff format --check .

# Auto-format with ruff
format:
	python -m ruff format .
	python -m ruff check --fix .

# Install dev dependencies
install:
	pip install -e ".[test]"
	pip install ruff

# Show available targets
help:
	@echo "Available targets:"
	@echo "  test       - Run unit evals (no API keys)"
	@echo "  test-integ - Run integration evals (needs VENICE_API_KEY)"
	@echo "  test-all   - Run all evals"
	@echo "  lint       - Check code style with ruff"
	@echo "  format     - Auto-format with ruff"
	@echo "  install    - Install dev dependencies"
