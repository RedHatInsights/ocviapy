# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ocviapy is a Python library that wraps the `oc` (OpenShift CLI) shell command, providing Pythonic utilities for interacting with OpenShift/Kubernetes clusters. Published to PyPI as `ocviapy`.

## Development Commands

```bash
# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Lint
flake8

# Verify package builds correctly
python setup.py sdist bdist_wheel
python -m twine check dist/*
```

There are no unit tests in this repository currently. CI runs flake8 linting, a build/import check, and package verification.

## Architecture

The entire library is a single module: `ocviapy/__init__.py`. There are no submodules.

**Core pattern**: All OpenShift interactions go through the `oc()` function, which wraps `sh.oc` with logging, error handling, and automatic retries for conflicts and I/O errors. The internal `_exec_oc()` runs commands in the background with `sh` and uses callback-based stdout/stderr streaming.

**Key classes**:
- `Resource` — wraps a single k8s/OpenShift resource with properties for status checking (`ready`, `image_pull_error`, etc.)
- `ResourceWatcher` — daemon thread that continuously polls all checkable resource types in a namespace
- `ResourceWaiter` — monitors a specific resource (and optionally its owned resources) until ready or timeout, using `wait_for`

**Status checking**: `_check_status_for_restype()` contains resource-type-specific readiness logic. Supported types are listed in `_CHECKABLE_RESOURCES`. Adding a new resource type requires adding it to that tuple and adding a status check branch.

**API resource discovery**: `get_api_resources()` is `lru_cache`d and parses the column-based output of `oc api-resources` to resolve resource type names and shortcuts.

## Code Style

- Max line length: 100 (configured in setup.cfg `[flake8]`)
- Python 3.6+ compatibility
- Uses `sh` library (not `subprocess`) for shell command execution
