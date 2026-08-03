# Contributing to RagZen

Thank you for contributing to RagZen!

## Development Workflow

1. Clone the repository and set up a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -e ".[dev]"
   ```

2. Run code formatting and linting:
   ```bash
   python -m ruff check src/
   ```

3. Run the test suite:
   ```bash
   python -m pytest tests/ -v
   ```

4. Submitting Pull Requests:
   - Ensure all tests pass.
   - Include unit/integration tests for any new features or bug fixes.
   - Follow semantic versioning and update CHANGELOG.md.
