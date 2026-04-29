# AGENTS.md

## Project Overview

Machine Learning Testground (MLTG) - A Python-based virtual agent project using JAX, Flax, and Optax for machine learning.

## Setup Commands

- Install dependencies: `uv sync`
- Install dev dependencies: `uv sync --all-extras`
- Run tests: `pytest`
- Run linting: `ruff check`

## Code Style

- Python 3.13+
- Type hints encouraged
- Follows Ruff linting rules (see pyproject.toml)
- Single quotes, no semicolons
- Trailing commas for multi-line collections

## Testing Instructions

- Run all tests: `pytest`
- Run linting: `ruff check`
- Run type checking: `pyright` (if configured)
- Fix any test or lint errors before committing

## Development Environment

- Uses uv for dependency management
- Virtual environment automatically managed by uv
- Python version: 3.13 (see .python-version)

## Project Structure

- `mltg/` - Main package directory
- `notebooks/` - Jupyter notebooks for experimentation
- `pyproject.toml` - Project configuration and dependencies
- `.github/` - GitHub workflows and configuration
