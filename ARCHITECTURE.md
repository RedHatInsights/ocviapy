# Architecture

Internal architecture of ocviapy: design decisions, dependency points, and key tradeoffs.

## Module Structure

The entire library is a **single-file module**. `ocviapy/__init__.py` contains all functions,
classes, constants, and exceptions. There are no submodules or internal packages. Everything is
importable directly from `ocviapy`.

This is a deliberate design choice — the library's scope is narrow enough that a single file keeps
the code navigable without the overhead of cross-module imports.

## Core Design: Two-Layer Command Execution

All interaction with the `oc` CLI flows through a two-layer wrapper:

### `_exec_oc()` — Internal Execution Engine

Handles the mechanics of running `oc` commands:

- Forces `_bg=True` and `_bg_exc=False` on every `sh.oc()` call, then immediately blocks with
  `cmd.wait()`. This pattern gives the library control over stdout/stderr via line-by-line streaming
  callbacks while the process runs.
- Defines `_out_line_handler` and `_err_line_handler` callbacks that capture output into lists and
  optionally log or print each line in real time.
- Implements a **retry loop** (3 retries, linear backoff at 3s/6s/9s) for three error categories:
  - **Immutable field errors** — silently ignored when `_ignore_immutable=True` (default).
  - **Conflict errors** ("error from server (conflict)") — retried when `_retry_conflicts=True`
    (default).
  - **I/O timeout errors** — retried when `_retry_io_errors=True` (default).
- Re-initializes `ErrorReturnCode` exceptions with captured stdout/stderr to work around a `sh`
  library bug where buffers may not be flushed before the exception is created.

### `oc()` — Public Entry Point

Adds error-suppression policy on top of `_exec_oc()`:

- Pops `_ignore_errors` (default `False`).
- Delegates to `_exec_oc()`.
- If `_ignore_errors=True`, catches `ErrorReturnCode` and logs a warning instead of raising.

This separation keeps retry/error-classification logic independent from error-suppression policy.

## Resource Monitoring System

Three classes form a tiered resource monitoring system:

### `Resource`

A lazy-loading wrapper around a single Kubernetes/OpenShift resource. Can be initialized with
either `restype` + `name` (data fetched lazily on first access) or a pre-fetched `data` dict.

Key behaviors:

- `ready` property delegates to `_check_status_for_restype()`, which contains type-specific
  readiness logic for 17+ resource types (pods, deployments, CRDs like ClowdApp, Kafka, CAPI
  resources).
- `image_pull_error` inspects container statuses for `ImagePullBackOff`, `ErrImagePull`, and
  `ErrImageNeverPull`.

### `ResourceWatcher`

A daemon thread that continuously polls (every 5 seconds) all "checkable" resource types in a
namespace. Maintains a `resources` dict keyed by `resource.key` (e.g., `"deployment/myapp"`).
Automatically prunes disappeared resources on each update cycle.

### `ResourceWaiter`

Orchestrates waiting for a specific resource (and optionally its owned resources) to reach "ready"
state. Can read from a `ResourceWatcher`'s cache (0.1s poll interval) or make direct API calls (5s
poll interval). Uses `ownerReferences` metadata to track resource ownership chains (e.g., deployment
→ replicaset → pod).

**Relationship:** `ResourceWatcher` continuously populates `Resource` objects. `ResourceWaiter`
reads from the watcher's cache (if provided) or creates its own `Resource` instances. Both
ultimately call `oc()` for cluster communication.

## Dependency Points

| Dependency | Used For |
| ---------- | -------- |
| `sh` | Core dependency. Provides `sh.oc()` for shell execution, `ErrorReturnCode` and `TimeoutException` for error handling. Chosen over `subprocess` for streaming line-by-line callbacks and readable command syntax. |
| `wait_for` | Polling loops in `ResourceWaiter.wait_for_ready()` and `_scale_down_up_using_match_labels()`. Provides `wait_for()` function and `TimedOutError` exception. |
| `anytree` | Declared in `setup.cfg` but **never imported**. Vestigial dependency from a removed feature. |
| `kubernetes` | Declared in `setup.cfg` but **never imported**. Unused. |
| `pyyaml` | Declared in `setup.cfg` but **never imported**. Unused. |

Three of five declared runtime dependencies are phantom dependencies — installed but not used by
current code.

## Key Design Decisions

### `sh` Over `subprocess`

The `sh` library provides streaming line-by-line callbacks for stdout/stderr (`_out` and `_err`
parameters). This lets ocviapy log command output in real time as the process runs, rather than
buffering everything until completion. The tradeoff is an external dependency with its own quirks,
such as the stdout/stderr buffer flushing bug that required a workaround in `_exec_oc()`.

### Caching with `lru_cache`

Several functions use `functools.lru_cache` based on the assumption that cluster state is stable
during execution:

- `get_api_resources()` — API resources do not change mid-run.
- `_can_list_resource()` — RBAC permissions are stable.
- `available_checkable_resources()` — derived from the above two.
- `on_k8s()` — cluster type (OpenShift vs. Kubernetes) is fixed.

Tradeoff: if the cluster state changes during a long-running process, cached results become stale.
This is acceptable for the intended use case of short-lived CI/CD operations.

### Mandatory vs. Optional Resource Types

Resource types are split into mandatory (standard Kubernetes: pod, deployment, etc.) and optional
(CRDs: ClowdApp, Kafka, CAPI resources). Optional types are silently skipped if the user lacks
`list` permission, checked via `oc auth can-i list <kind>`. This lets the library work across
clusters with different CRD installations without failing.

The `_CHECKABLE_RESOURCES` tuple is ordered from "lowest" to "highest" in the ownership hierarchy
(pod → replicaset → deployment), which matters for ownership chain traversal in `ResourceWaiter`.

### Dual-Mode Kubernetes/OpenShift Support

`on_k8s()` detects the cluster type by checking for the `project` API resource (OpenShift-specific).
Several functions branch on this result: namespace listing uses "project" vs. "namespace",
current-namespace detection uses `oc project -q` vs. `oc config current-context`, and namespace
switching uses `oc project` vs. `oc config set-context`.

### Tabular Parsing of `oc api-resources`

The `get_api_resources()` function parses fixed-width column output by measuring header column widths
with regex. This is fragile — it depends on the exact output format of `oc api-resources`. The
`name` field is stripped of a trailing "s" as a rough singularization heuristic. The alternative
(`-o json`) is not available for `oc api-resources`.
