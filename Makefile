.PHONY: install install-dev build test lint fmt clean help setup-hooks

help:
	@echo "pyrobovision development tasks:"
	@echo "  make install         Install pre-commit hooks"
	@echo "  make test            Run all tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make lint            Run linters (black, isort, ruff, mypy)"
	@echo "  make fmt             Format code"
	@echo "  make fmt-check       Check format without changing"
	@echo "  make clean           Remove build artifacts"

install: setup-hooks
	@echo "✓ Development environment ready"

setup-hooks:
	@command -v pre-commit >/dev/null 2>&1 || pip install pre-commit
	pre-commit install

test:
	pytest -v

test-cov:
	pytest -v --cov=pyrobovision --cov-report=term-missing

lint:
	black --check .
	isort --check-only .
	ruff check .
	mypy pyrobovision

fmt:
	black .
	isort .
	ruff check . --fix

fmt-check:
	black --check .
	isort --check-only .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache
