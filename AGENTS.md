# ocviapy

## Project Overview

ocviapy is a Python library that wraps the OpenShift `oc` CLI, providing utilities for running
commands, querying resources, and waiting for resources to reach a ready state. Works with both
OpenShift and plain Kubernetes clusters. Published to PyPI as `ocviapy`.

## Dependencies

**Runtime:** `sh` (>= 1.13.1), `wait_for`. Three additional packages are declared in `setup.cfg`
(`anytree`, `kubernetes`, `pyyaml`) but are never imported in the current code.

**Dev/test:** `ruff` (via pre-commit), `flake8`, `pytest`, `pytest-mock` (declared as test extras
in `setup.cfg`).

## Development Commands

See [Development][readme-dev] in the README for the full setup and command reference.

```sh
# Install in editable mode with test dependencies
pip install -e '.[test]'

# Lint and format (active tool — enforced in CI)
pre-commit run --all-files

# Build
python -m build
python -m twine check dist/*
```

CI runs ruff (via pre-commit), a smoke import check (`python -c "from ocviapy import ..."`), and
package build verification. There are no unit tests.

## Architecture

Single-module library: all code lives in `ocviapy/__init__.py`. The `oc()` function is the central
entry point — it wraps `sh.oc` with logging, retries, and error handling. `Resource`,
`ResourceWatcher`, and `ResourceWaiter` form a three-tier resource monitoring system for tracking
readiness state. See the [architecture documentation][architecture] for design decisions, dependency
analysis, and tradeoffs.

## Code Style

- **Linter/formatter:** ruff (enforced via pre-commit and CI). A legacy flake8 config exists in
  `setup.cfg` but is not used in CI — ruff is authoritative.
- **Line length:** 100 (from `setup.cfg [flake8]`; ruff uses its default).
- **Python version:** >= 3.6.
- **Shell execution:** uses the `sh` library, not `subprocess`.

## Common Mistakes

1. **Using `subprocess` instead of `sh`.** All shell commands must go through `oc()` / `_exec_oc()`,
   which use the `sh` library. This ensures consistent logging, retry logic, and error handling.
   Never use `subprocess` directly.

2. **Assuming `get_api_resources()` returns fresh data.** This function is decorated with
   `@lru_cache` and returns the same result for the entire process lifetime. The same applies to
   `_can_list_resource()`, `available_checkable_resources()`, and `on_k8s()`. Do not expect these
   to reflect cluster changes during execution.

3. **Referencing flake8 as the active linter.** The `setup.cfg [flake8]` section is legacy
   configuration. The active linter/formatter is ruff, configured in `.pre-commit-config.yaml` and
   enforced in CI. Do not add flake8 rules or assume flake8 is running.

4. **Treating `anytree`, `kubernetes`, or `pyyaml` as available imports.** These are declared as
   dependencies in `setup.cfg` but are never imported in the codebase. They are phantom dependencies
   and should not be relied upon in new code without an explicit decision to use them.

5. **Adding a new resource type without updating `_CHECKABLE_RESOURCES`.** To support status
   checking for a new resource type, add it to `_MANDATORY_CHECKABLE_RESOURCES` or
   `_OPTIONAL_CHECKABLE_RESOURCES` and add a corresponding branch in
   `_check_status_for_restype()`. The ordering in the tuple matters — it follows the ownership
   hierarchy (pod → replicaset → deployment).

## Deployment

Releases are published to PyPI via GitHub Actions. Pushing a tag triggers the `release.yml`
workflow, which builds the package and publishes it to PyPI using an API token
(`pypa/gh-action-pypi-publish`). Version is derived from git tags via `setuptools_scm`.

[readme-dev]: README.md#development
[architecture]: ARCHITECTURE.md
