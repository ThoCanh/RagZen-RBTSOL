# Contributing to RagZen

Thank you for contributing to RagZen!

## Development Workflow

1. Clone the repository and set up a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -e ".[dev]"
   ```

2. Run formatting, linting, type and security checks:
   ```bash
   python -m ruff format --check src tests examples
   python -m ruff check src tests
   python -m mypy src
   python -m bandit -r src -q
   python -m pip_audit
   ```

3. Run the test suite:
   ```bash
   python -m pytest --cov=ragzen --cov-branch --cov-report=term-missing
   ```

4. Submitting Pull Requests:
   - Ensure all tests pass.
   - Include unit/integration tests for any new features or bug fixes.
   - Follow semantic versioning and update CHANGELOG.md.
