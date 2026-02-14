# Add linting

Status: **done**
PR: https://github.com/keithbro/rtty/pull/2

## Description
Add a Python linter (ruff) to catch code quality issues like builtin shadowing, unused imports, and style violations. Configure it with sensible defaults for the project.

## Notes
- Ruff is fast and covers flake8, pyflakes, pycodestyle, and more in a single tool.
- The `A` ruleset (flake8-builtins) catches builtin shadowing like the `bin` issue we fixed.
- Add `ruff` to `requirements.txt` and a `ruff.toml` or `[tool.ruff]` section in `pyproject.toml`.
