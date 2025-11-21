# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/ml_blueprint/` with entries in scripts.
- Notebooks: `notebooks/` (e.g., `one_stop_notebook.ipynb`).
- Config: `pyproject.toml` (project/deps) and `uv.lock` (resolved versions).
- Add new modules under `src/ml_blueprint/`; export minimal APIs in `__init__.py` when useful.
- Recommended (if missing): `tests/`, `scripts/`, `data/`, `artifacts/` (large files untracked).

## Build, Test, and Development Commands
- `uv sync`: install dependencies (Python >= 3.11).
- `uv run <script>`: run a script.
- `uv add <pkg>` / `uv remove <pkg>`: manage dependencies.
- `uv build`: build wheel/sdist using `uv_build` backend.

## Coding Style & Naming Conventions
- Follow PEP 8; 4‑space indents; include type hints for new/modified code.
- Names: modules `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Keep scripts thin; implement logic in `src/ml_blueprint/*` with small, testable functions.

## Commit & Pull Request Guidelines
- Commits: small, atomic; imperative subject (e.g., "add", "fix").
- Prefer Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- PRs: clear purpose, summary of changes, how to run, linked issues; screenshots for UI/notebook outputs when relevant.

## Security & Data Tips
- Do not commit secrets or large datasets. Keep local data under `data/` and artifacts under `artifacts/`; add to `.gitignore`.
- Keep notebooks reproducible: record config and minimize stored outputs.

