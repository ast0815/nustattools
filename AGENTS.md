# Build/Lint/Test:

- Run all tests: `nox -s tests`
- Run single test: `nox -s tests -- tests/test_file.py::test_name`
- Lint: `nox -s lint`
- PyLint: `nox -s pylint`
- Typecheck: `nox -s mypy` (via pre-commit)

# Code Style:

- Use `ruff` for formatting/linting (configured in pyproject.toml)
- Imports: Follow `isort` rules with `from __future__ import annotations`
- Types: All public APIs must be typed (strict mypy)
- Naming: snake_case for variables/functions, CamelCase for classes
- Error handling: Use specific exceptions, avoid bare except
- Docstrings: Google style for public APIs
- Testing: Use pytest, include coverage, follow existing patterns

# Additional Guidelines:

- All code must pass mypy type checking with strict mode enabled
- Use f-strings for string formatting
- Prefer list comprehensions over explicit loops where applicable
- Use type hints for all function parameters and return values
- Follow PEP 8 style guide with ruff enforcement
- Avoid magic numbers; use constants instead
- Write docstrings for all public functions and classes
- Use meaningful variable names that describe their purpose
- Keep functions small and focused on a single responsibility
- Use `pytest` fixtures for test setup and teardown
- All tests must have 100% coverage

# Testing Best Practices:

- Test edge cases and error conditions
- Use parametrize for multiple test inputs
- Mock external dependencies in unit tests
- Test both positive and negative cases
- Run tests with --cov to verify coverage
