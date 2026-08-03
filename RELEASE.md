# Release Process

This document describes the release workflow for RagZen.

## Release Steps

1. Verify all tests pass: `pytest tests/`
2. Verify code quality: `ruff check src/`
3. Update version in `pyproject.toml` and `src/ragzen/__init__.py`.
4. Update `CHANGELOG.md` with release notes.
5. Build wheel and sdist: `python -m build`
6. Verify wheel installation in clean environment.
7. Create git tag and GitHub release.
