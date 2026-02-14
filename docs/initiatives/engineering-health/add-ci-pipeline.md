# Add CI pipeline for PRs

Status: **done**
PR: https://github.com/keithbro/rtty/pull/3

## Description
Add a GitHub Actions workflow that runs linting and tests on every pull request, so issues are caught before merge.

## Notes
- Use a simple workflow that installs dependencies, runs `ruff check`, and runs `pytest`.
- Target Python 3.9+ to match the current development environment.
- Trigger on `pull_request` against `master`.
