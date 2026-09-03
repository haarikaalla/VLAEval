# Contributing to VLA-Eval

Thanks for your interest in contributing! This document describes the workflow
for proposing changes.

## Development Setup

```bash
git clone https://github.com/your-org/vla-eval.git
cd vla-eval
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
make install-dev
```

## Workflow

1. Create a feature branch from `main`: `git checkout -b feat/my-change`.
2. Make your changes with tests and documentation updates.
3. Run local checks before pushing:
   ```bash
   make lint
   make type-check
   make test
   ```
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/)
   (e.g. `feat(training): add cosine LR scheduler`).
5. Open a pull request against `main`. CI must pass before merge.

## Code Style

- Python: formatted with `black`, linted with `ruff`, type-checked with `mypy`.
- TypeScript/React: formatted with `prettier`, linted with `eslint`.
- All new modules should include type hints and docstrings.
- Prefer small, focused pull requests.

## Testing

- Unit tests live in `tests/unit/`, integration tests in `tests/integration/`.
- New features require accompanying tests. Aim to keep coverage stable or improved.
- Frontend tests use `vitest` under `frontend/src/**/__tests__`.

## Reporting Issues

Use GitHub Issues with the appropriate template (bug report / feature request).
For security issues, see [SECURITY.md](SECURITY.md) instead of opening a public issue.
