# ocviapy

Python utilities that wrap the OpenShift `oc` CLI. Provides a high-level interface for running `oc`
commands, querying resources, and waiting for resources to reach a ready state. Works with both
OpenShift and plain Kubernetes clusters.

## Installation

```sh
pip install ocviapy
```

**Prerequisite:** The `oc` CLI must be installed and available on your `PATH`, and you must be
logged in to a cluster (`oc login`).

## Usage

### Running `oc` commands

The `oc()` function is the core entry point. It accepts the same arguments you would pass to `oc` on
the command line:

```python
from ocviapy import oc

# Run any oc command
oc("get", "pods", "-n", "my-namespace")

# Ignore errors instead of raising
oc("delete", "pod", "old-pod", _ignore_errors=True)

# Print output directly to stdout/stderr
oc("logs", "my-pod", _print=True)

# Suppress logging
output = oc("get", "nodes", _silent=True)
```

### Querying resources as JSON

```python
from ocviapy import get_json

# Get a single resource
deployment = get_json("deployment", "my-app", namespace="my-namespace")

# Get all resources of a type
all_pods = get_json("pod", namespace="my-namespace")

# Filter by label
labeled = get_json("pod", label="app=my-app", namespace="my-namespace")
```

### Waiting for resources to be ready

```python
from ocviapy import wait_for_ready, ResourceWatcher, ResourceWaiter

# Wait for a deployment to be ready (watches owned pods/replicasets too)
success = wait_for_ready("my-namespace", "deployment", "my-app", timeout=300)

# Wait for multiple resources in parallel
watcher = ResourceWatcher("my-namespace")
watcher.start()
try:
    waiters = [
        ResourceWaiter("my-namespace", "deployment", "app-1", watcher=watcher),
        ResourceWaiter("my-namespace", "deployment", "app-2", watcher=watcher),
    ]
    from ocviapy import wait_for_ready_threaded
    success = wait_for_ready_threaded(waiters, timeout=600)
finally:
    watcher.stop()
```

### Inspecting resource status

```python
from ocviapy import Resource

resource = Resource(restype="deployment", name="my-app", namespace="my-namespace")
print(resource.ready)             # True/False
print(resource.status_conditions) # List of condition strings
print(resource.details_str)       # Human-readable status summary
```

### Other utilities

```python
from ocviapy import (
    apply_config,
    export,
    get_routes,
    copy_namespace_secrets,
    process_template,
    scale_down_up,
    on_k8s,
    get_current_namespace,
    set_current_namespace,
)

# Apply a k8s List resource
apply_config("my-namespace", list_resource)

# Export a resource (like deprecated oc --export)
data = export("deployment", "my-app", namespace="my-namespace")

# Get route hostnames
routes = get_routes("my-namespace")  # {"route-name": "hostname"}

# Copy secrets between namespaces
copy_namespace_secrets("src-ns", "dst-ns", ["secret-1", "secret-2"], "bonfire.ignore")

# Process an OpenShift Template
result = process_template(template_data, {"PARAM": "value"})

# Scale a deployment down to 0 and back up
scale_down_up("my-namespace", "deployment", "my-app", timeout=300)

# Detect cluster type
if on_k8s():
    print("Running on Kubernetes")
else:
    print("Running on OpenShift")
```

## API Overview

| Function / Class | Description |
| --- | --- |
| `oc(*args, **kwargs)` | Run any `oc` CLI command with logging, retries, and error handling |
| `get_json(restype, name, label, namespace)` | Get resource data as parsed JSON |
| `apply_config(namespace, list_resource)` | Apply a Kubernetes List resource |
| `export(restype, name, label, namespace)` | Get resource data with cluster-specific info stripped |
| `get_routes(namespace)` | Get dict of route names to hostnames |
| `wait_for_ready(namespace, restype, name, ...)` | Wait for a resource (and owned resources) to be ready |
| `wait_for_ready_threaded(waiters, ...)` | Wait for multiple resources in parallel |
| `copy_namespace_secrets(src, dst, names, ...)` | Copy secrets between namespaces |
| `process_template(template_data, params)` | Process an OpenShift Template |
| `scale_down_up(namespace, restype, name, ...)` | Scale a deployment down to 0 and back up |
| `on_k8s()` | Detect Kubernetes vs OpenShift cluster |
| `get_all_namespaces(label)` | List all namespaces/projects |
| `get_current_namespace()` | Get the current namespace/project |
| `set_current_namespace(namespace)` | Set the current namespace/project |
| `Resource` | Lazy-loading wrapper around a single Kubernetes resource |
| `ResourceWatcher` | Daemon thread that continuously polls resource state |
| `ResourceWaiter` | Orchestrates waiting for resources to reach ready state |

For internal design details, see the [architecture documentation][architecture].

## Development

### Setup

```sh
# Clone the repository
git clone https://github.com/bsquizz/ocviapy.git
cd ocviapy

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode with test dependencies
pip install -e '.[test]'

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Linting

The project uses [ruff][ruff] for linting and formatting, enforced via
pre-commit hooks:

```sh
# Run all pre-commit hooks
pre-commit run --all-files

# Run ruff directly
ruff check .
ruff format .
```

### Building

Version is derived from git tags via `setuptools_scm`:

```sh
python -m build
python -m twine check dist/*
```

### Tests

There are currently no unit tests in the repository. The CI pipeline verifies linting (ruff via
pre-commit), a smoke import check, and package build integrity.

## License

[MIT][license]

[architecture]: ARCHITECTURE.md
[ruff]: https://docs.astral.sh/ruff/
[license]: LICENSE
