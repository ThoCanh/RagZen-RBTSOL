# Release Process

This document describes the release workflow for RagZen.

## Release Steps

1. Install release tooling: `pip install -e ".[dev,server]"`.
2. Run Ruff lint/format checks, Mypy, Bandit, and `pip-audit`.
3. Run `pytest --cov=ragzen --cov-branch --cov-report=term-missing`.
4. Build the documentation with `mkdocs build --strict`.
5. Update versions in `pyproject.toml` and `src/ragzen/__init__.py` and update the changelog.
6. Remove prior artifacts, then build with `python -m build` and run `twine check dist/*`.
7. Install the wheel in a clean virtual environment and run a persistence smoke test.
8. Build the production Docker image and validate the authenticated Compose deployment.
9. Upload to TestPyPI, verify installation, then publish the same artifacts to PyPI.
10. Create the signed git tag and GitHub release with checksums and release notes.
