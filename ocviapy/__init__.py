import functools
import json
import logging
import re
import shlex
import sys
import threading
import time

import sh
from sh import ErrorReturnCode, TimeoutException
from wait_for import TimedOutError, wait_for

log = logging.getLogger(__name__)
logging.getLogger("sh").setLevel(logging.CRITICAL)


# assume that the result of this will not change during execution of our app
@functools.lru_cache(maxsize=None, typed=False)
def get_api_resources():
    output = oc("api-resources", verbs="list", _silent=True).strip()
    if not output:
        log.info("oc api-resources came back empty")
        return []

    lines = output.split("\n")
    # lines[0] is the table header, use it to figure out length of each column
    groups = re.findall(r"(\w+\s+)", lines[0])

    name_start = 0
    name_end = len(groups[0])
    shortnames_start = name_end
    shortnames_end = name_end + len(groups[1])
    apigroup_start = shortnames_end
    apigroup_end = shortnames_end + len(groups[2])
    namespaced_start = apigroup_end
    namespaced_end = apigroup_end + len(groups[3])
    kind_start = namespaced_end

    resources = []
    for line in lines[1:]:
        shortnames = line[shortnames_start:shortnames_end].strip()
        resource = {
            "name": line[name_start:name_end].strip().rstrip("s") or None,
            "shortnames": shortnames.split(",") if shortnames else [],
            "apigroup": line[apigroup_start:apigroup_end].strip() or "",
            "namespaced": line[namespaced_start:namespaced_end].strip() == "true",
            "kind": line[kind_start:].strip() or None,
        }
        resources.append(resource)
    return resources


def traverse_keys(d, keys, default=None):
    """
    Allows you to look up a 'path' of keys in nested dicts without knowing whether each key exists
    """
    key = keys.pop(0)
    item = d.get(key, default)
    if len(keys) == 0:
        return item
    if not item:
        return default
    return traverse_keys(item, keys, default)


def parse_restype(string):
    """
    Given a resource type or its shortcut, return the full resource type name.
    """
    s = string.lower()
    for r in get_api_resources():
        if s in r["shortnames"] or s == r["name"]:
            return r["name"]

    raise ValueError("Unknown resource type: {}".format(string))


def _only_immutable_errors(err_lines):
    if not err_lines:
        # this check is needed since all([]) returns 'True'
        return False
    return all("field is immutable after creation" in line.lower() for line in err_lines)


def _conflicts_found(err_lines):
    return any("error from server (conflict)" in line.lower() for line in err_lines)


def _io_error_found(err_lines):
    return any("i/o timeout" in line.lower() for line in err_lines)


def _get_logging_args(args, kwargs):
    # Format the cmd args/kwargs for log printing before the command is run
    cmd_args = " ".join([str(arg) for arg in args if arg is not None])

    cmd_kwargs = []
    for key, val in kwargs.items():
        if key.startswith("_"):
            continue
        if len(key) > 1:
            cmd_kwargs.append("--{} {}".format(key, val))
        else:
            cmd_kwargs.append("-{} {}".format(key, val))
    cmd_kwargs = " ".join(cmd_kwargs)

    return cmd_args, cmd_kwargs


def _exec_oc(*args, **kwargs):
    _print = kwargs.pop("_print", False)
    _silent = kwargs.pop("_silent", False)
    _ignore_immutable = kwargs.pop("_ignore_immutable", True)
    _retry_conflicts = kwargs.pop("_retry_conflicts", True)
    _retry_io_errors = kwargs.pop("_retry_io_errors", True)
    _stdout_log_prefix = kwargs.pop("_stdout_log_prefix", " |stdout| ")
    _stderr_log_prefix = kwargs.pop("_stderr_log_prefix", " |stderr| ")
    kwargs["_bg"] = True
    kwargs["_bg_exc"] = False

    # define stdout/stderr callback funcs
    err_lines = []
    out_lines = []

    def _err_line_handler(line, _, process):
        threading.current_thread().name = f"pid-{process.pid}"
        if _print:
            print(line.rstrip(), file=sys.stderr)
        if not _silent:
            log.info("%s%s", _stderr_log_prefix, line.rstrip())
        err_lines.append(line)

    def _out_line_handler(line, _, process):
        threading.current_thread().name = f"pid-{process.pid}"
        if _print:
            print(line.rstrip())
        if not _silent:
            log.info("%s%s", _stdout_log_prefix, line.rstrip())
        out_lines.append(line)

    retries = 3
    backoff = 3
    last_err = None

    for count in range(1, retries + 1):
        cmd = sh.oc(*args, **kwargs, _tee=True, _out=_out_line_handler, _err=_err_line_handler)
        if not _silent:
            cmd_args, cmd_kwargs = _get_logging_args(args, kwargs)
            log.info("running (pid %d): oc %s %s", cmd.pid, cmd_args, cmd_kwargs)
        try:
            return cmd.wait()
        except ErrorReturnCode as err:
            # Sometimes stdout/stderr is empty in the exception even though we appended
            # data in the callback. Perhaps buffers are not being flushed ... so just
            # set the out lines/err lines we captured on the Exception before re-raising it by
            # re-init'ing the err and causing it to rebuild its message template.
            #
            # see https://github.com/amoffat/sh/blob/master/sh.py#L381
            err.__init__(
                full_cmd=err.full_cmd,
                stdout="\n".join(out_lines).encode(),
                stderr="\n".join(err_lines).encode(),
                truncate=err.truncate,
            )

            # Make these plain strings for easier exception handling
            err.stdout = "\n".join(out_lines)
            err.stderr = "\n".join(err_lines)

            last_err = err
            # Ignore warnings that are printed to stderr in our error analysis
            err_lines = [line for line in err_lines if not line.lstrip().startswith("Warning:")]

            # Check if these are errors we should handle
            if _ignore_immutable and _only_immutable_errors(err_lines):
                log.warning("Ignoring immutable field errors")
                break
            elif _retry_conflicts and _conflicts_found(err_lines):
                sleep_time = count * backoff
                log.warning(
                    "Hit resource conflict, retrying in %d sec (attempt %d/%d)",
                    sleep_time,
                    count,
                    retries,
                )
                time.sleep(sleep_time)
                continue
            elif _retry_io_errors and _io_error_found(err_lines):
                sleep_time = count * backoff
                log.warning(
                    "Hit i/o error, retrying in %d sec (attempt %d/%d)",
                    sleep_time,
                    count,
                    retries,
                )
                time.sleep(sleep_time)
                continue

            # Bail if not
            raise last_err
    else:
        log.error("Retried %d times, giving up", retries)
        raise last_err


def oc(*args, **kwargs):
    """
    Run 'sh.oc' and print the command, show output, catch errors, etc.

    Optional kwargs:
        _ignore_errors: if ErrorReturnCode is hit, don't re-raise it (default False)
        _silent: don't log command or resulting output (default False)
        _print: print stdout/stderr output directly to stdout/stderr (default False)
        _ignore_immutable: ignore errors related to immutable objects (default True)
        _retry_conflicts: retry commands if a conflict error is hit
        _retry_io_errors: retry commands if i/o error is hit
        _stdout_log_prefix: prefix this string to stdout log output (default " |stdout| ")
        _stderr_log_prefix: prefix this string to stderr log output (default " |stderr| ")

    Returns:
        None if cmd fails and _exit_on_err is False
        command output (str) if command succeeds
    """
    _ignore_errors = kwargs.pop("_ignore_errors", False)
    # The _silent/_ignore_immutable/_retry_* kwargs are passed on so don't pop them yet

    try:
        return _exec_oc(*args, **kwargs)
    except ErrorReturnCode:
        if not _ignore_errors:
            raise
        else:
            if not kwargs.get("_silent"):
                log.warning("Non-zero return code ignored")


def apply_config(namespace, list_resource):
    """
    Apply a k8s List of items
    """
    if namespace is None:
        oc("apply", "-f", "-", _in=json.dumps(list_resource))
    else:
        oc("apply", "-f", "-", "-n", namespace, _in=json.dumps(list_resource))


def get_json(restype, name=None, label=None, namespace=None):
    """
    Run 'oc get' for a given resource type/name/label and return the json output.

    If name is None all resources of this type are returned

    If label is not provided, then "oc get" will not be filtered on label
    """
    restype = parse_restype(restype)

    args = ["get", restype]
    if name:
        args.append(name)
    if label:
        args.extend(["-l", label])
    if namespace:
        args.extend(["-n", namespace])
    try:
        output = oc(*args, o="json", _silent=True)
    except ErrorReturnCode as err:
        if "NotFound" in err.stderr:
            return {}
        raise

    try:
        parsed_json = json.loads(str(output))
    except ValueError:
        return {}

    return parsed_json


def remove_cluster_specific_info(resource):
    """Remove cluster-specific attributes from a resource."""
    if "metadata" in resource:
        metadata = resource["metadata"]

        last_applied_key = "kubectl.kubernetes.io/last-applied-configuration"
        last_applied = metadata.get("annotations", {}).get(last_applied_key)

        if last_applied:
            del metadata["annotations"][last_applied_key]
        for key in ["namespace", "resourceVersion", "uid", "selfLink"]:
            if key in metadata:
                del metadata[key]
        metadata["creationTimestamp"] = None

    if resource.get("kind", "").lower() == "list":
        for item in resource.get("items", []):
            remove_cluster_specific_info(item)

    return resource


def export(restype, name=None, label=None, namespace=None):
    """Get data for resource but strip cluster-specific identifiers.

    Replacement for oc --export
    """
    return remove_cluster_specific_info(get_json(restype, name, label, namespace))


def get_routes(namespace):
    """
    Get all routes in the project.

    Return dict with key of service name, value of http route
    """
    data = get_json("route", namespace=namespace)
    ret = {}
    for route in data.get("items", []):
        ret[route["metadata"]["name"]] = route["spec"]["host"]
    return ret


class StatusError(Exception):
    pass


# Resources we are able to parse the status of
#
# In order for the 'ResourceWatcher' to collect resource info for underlying "owned" resources
# correct, these types should be ordered according to where they fall on the "ownership chain."
# For example, 'deployments' own 'replicasets' which own 'pods' -- so we want to list the resource
# types from "lowest" to "highest" on the hierarchy here -- pod, replicaset, deployment
_CHECKABLE_RESOURCES = (
    "pod",
    "replicaset",
    "replicationcontroller",
    "deploymentconfig",
    "deployment",
    "statefulset",
    "daemonset",
    "clowdapp",
    "clowdjobinvocation",
    "clowdenvironment",
    "kafka",
    "kafkaconnect",
    "cyndipipeline",
    "xjoinpipeline",
)


def _is_checkable(kind):
    return kind.lower() in _CHECKABLE_RESOURCES


def available_checkable_resources(namespaced=False):
    """Returns resources we are able to parse status of that are present on the cluster."""
    checkable_resources = []
    api_resources = get_api_resources()
    for checkable_kind in _CHECKABLE_RESOURCES:
        for api_resource in api_resources:
            kind = api_resource["kind"].lower()
            if kind == checkable_kind:
                if not namespaced or (namespaced and api_resource["namespaced"]):
                    checkable_resources.append(kind)

    return checkable_resources


def _get_name_for_kind(kind):
    for r in get_api_resources():
        if r["kind"].lower() == kind.lower():
            return r["name"]
    raise ValueError(f"unable to find resource name for kind '{kind}'")


def _check_status_condition(status, expected_type, expected_value):
    conditions = status.get("conditions", [])
    expected_type = str(expected_type).lower()
    expected_value = str(expected_value).lower()

    for c in conditions:
        status_value = str(c.get("status")).lower()
        status_type = str(c.get("type")).lower()
        if status_value == expected_value and status_type == expected_type:
            return True
    return False


def _check_status_for_restype(restype, json_data):
    """
    Depending on the resource type, check that it is "ready" or "complete"

    Uses the status json from an 'oc get'

    Returns True if ready, False if not.
    """
    restype = parse_restype(restype)

    if restype != "pod" and restype not in _CHECKABLE_RESOURCES:
        raise ValueError(f"Checking status for resource type {restype} currently not supported")

    try:
        status = json_data["status"]
    except KeyError:
        status = None

    if not status:
        return False

    generation = json_data["metadata"].get("generation")
    status_generation = status.get("observedGeneration") or status.get("generation")
    if generation and status_generation and generation != status_generation:
        return False

    if restype in ("deploymentconfig", "deployment"):
        spec_replicas = json_data["spec"]["replicas"]
        available_replicas = status.get("availableReplicas", 0)
        updated_replicas = status.get("updatedReplicas", 0)
        if available_replicas == spec_replicas and updated_replicas == spec_replicas:
            return True

    elif restype in ("replicaset", "replicationcontroller"):
        spec_replicas = json_data["spec"]["replicas"]
        available_replicas = status.get("availableReplicas", 0)
        if available_replicas == spec_replicas:
            return True

    elif restype == "statefulset":
        spec_replicas = json_data["spec"]["replicas"]
        ready_replicas = status.get("readyReplicas", 0)
        return ready_replicas == spec_replicas

    elif restype == "daemonset":
        desired = status.get("desiredNumberScheduled", 1)
        available = status.get("numberAvailable")
        return desired == available

    elif restype == "pod":
        if status.get("phase").lower() == "running":
            return True

    elif restype in ("clowdenvironment", "clowdapp"):
        return _check_status_condition(
            status, "DeploymentsReady", "true"
        ) and _check_status_condition(status, "ReconciliationSuccessful", "true")

    elif restype == "clowdjobinvocation":
        return _check_status_condition(
            status, "JobInvocationComplete", "true"
        ) and _check_status_condition(status, "ReconciliationSuccessful", "true")

    elif restype in ("kafka", "kafkaconnect"):
        return _check_status_condition(status, "ready", "true")

    elif restype == "cyndipipeline":
        return (
            _check_status_condition(status, "valid", "true")
            and status.get("activeTableName") is not None
        )

    elif restype == "xjoinpipeline":
        return (
            _check_status_condition(status, "valid", "true")
            and status.get("activeIndexName") is not None
        )


class Resource:
    def __init__(self, restype=None, name=None, namespace=None, data=None):
        if not data and not (restype and name):
            raise ValueError("Resource must be instantiated with restype/name or data")

        self._restype = restype
        self._name = name
        self._namespace = namespace
        self._data = data

    def get_json(self):
        self._data = get_json(self._restype, name=self._name, namespace=self._namespace)
        return self._data

    @property
    def data(self):
        if not self._data:
            self.get_json()
        return self._data

    @property
    def kind(self):
        return self.data["kind"].lower()

    @property
    def restype(self):
        return _get_name_for_kind(self.kind)

    @property
    def name(self):
        return self.data["metadata"]["name"]

    @property
    def namespace(self):
        return self.data["metadata"]["namespace"]

    @property
    def key(self):
        if self._restype and self._name:
            return f"{self._restype}/{self._name}"
        else:
            return f"{self.restype}/{self.name}"

    @property
    def uid(self):
        return self.data["metadata"]["uid"]

    @property
    def ready(self):
        return _check_status_for_restype(self.restype, self.data)

    @property
    def status_conditions(self):
        status_conditions = []
        conditions = self.data.get("status", {}).get("conditions", [])
        for c in conditions:
            status_value = c.get("status")
            status_type = c.get("type")
            txt = f"{status_type}: {status_value}"

            status_msg = c.get("message")
            status_reason = c.get("reason")
            msg = status_msg or status_reason
            if msg:
                txt += f" ({msg})"

            status_conditions.append(txt)
        return status_conditions

    @property
    def details_str(self):
        detail_msg = f"{self.key} {'not' if not self.ready else ''} ready"
        if self.status_conditions:
            detail_msg += ", status conditions:\n{}".format(
                "\n".join([f"  - {s}" for s in self.status_conditions])
            )
        return detail_msg


class ResourceWatcher(threading.Thread):
    def __init__(self, namespace, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = True
        self.namespace = namespace
        self.resources = {}
        self._stopped = threading.Event()

    def update_resources(self):
        found_keys = []
        for restype in available_checkable_resources():
            response = get_json(restype, namespace=self.namespace)
            for item in response.get("items", []):
                r = Resource(data=item)
                self.resources[r.key] = r
                found_keys.append(r.key)
        for key in list(self.resources.keys()):
            if key not in found_keys:
                del self.resources[key]

    def run(self):
        log.debug("starting resource watcher for namespace '%s'", self.namespace)
        while not self._stopped.is_set():
            self.update_resources()
            time.sleep(5)
        log.debug("resource watcher stopped for namespace '%s'", self.namespace)

    def stop(self):
        self._stopped.set()


class ResourceWaiter:
    def __init__(self, namespace, restype, name, watch_owned=False, watcher=None):
        self.namespace = namespace
        self.restype = parse_restype(restype)
        self.name = name.lower()
        self.watch_owned = watch_owned
        self.watcher = watcher
        self.observed_resources = dict()
        self.key = f"{self.restype}/{self.name}"
        self.resource = None
        self.timed_out = False
        self._log_msg_interval = 60
        self._time_last_logged = 0
        self._time_remaining = 0

        if self.watch_owned and not self.watcher:
            raise ValueError("watcher must be specified if using watch_owned=True")

        if self.restype not in available_checkable_resources():
            raise ValueError(
                f"unable to check status of '{self.restype}' resources on this cluster"
            )

    def _check_owned_resources(self, resource):
        for owner_ref in resource.data["metadata"].get("ownerReferences", []):
            restype_matches = owner_ref["kind"].lower() == self.restype
            owner_uid_matches = owner_ref["uid"] == self.resource.uid
            if restype_matches and owner_uid_matches:
                # this resource is owned by "self"
                previously_observed = False
                previously_ready = False
                if resource.key in self.observed_resources:
                    # so we don't keep logging on every loop
                    previously_observed = True
                    previous_resource = self.observed_resources[resource.key]
                    previously_ready = True if previous_resource.ready else False

                # update our records for this resource
                self.observed_resources[resource.key] = resource

                if not previously_observed and not resource.ready:
                    log.info(
                        "[%s] found owned resource %s, not yet ready",
                        self.key,
                        resource.key,
                    )

                # check if ready state has transitioned for this resource
                if not previously_ready and resource.ready:
                    log.info("[%s] owned resource %s is ready!", self.key, resource.key)

    @property
    def _all_resources_ready(self):
        resources_ready = [r.ready is True for _, r in self.observed_resources.items()]
        return len(resources_ready) > 0 and all(resources_ready)

    def _observe(self, resource):
        key = resource.key

        # update our records for this resource
        self.observed_resources[key] = resource

        if self.watch_owned:
            # use .copy() in case dict changes during iteration
            for _, r in self.watcher.resources.copy().items():
                self._check_owned_resources(r)

            # check to see if any of the owned resources we were previously watching are now no
            # longer present in the ResourceWatcher
            disappeared_resources = {
                key for key in self.observed_resources if key not in self.watcher.resources
            }
            for key in disappeared_resources:
                log.info("[%s] resource has disappeared, no longer monitoring it", key)
                del self.observed_resources[key]

        if self._all_resources_ready:
            log.info("[%s] resource is ready!", key)

    def check_ready(self):
        if self.watcher:
            self.resource = self.watcher.resources.get(self.key)
            data = self.resource.data if self.resource else None
        else:
            self.resource = Resource(self.restype, self.name, self.namespace)
            data = self.resource.get_json()

        if data:
            self._observe(self.resource)
            return self._all_resources_ready
        return False

    def _check_with_periodic_log(self):
        current_time = int(time.time())

        if self.check_ready():
            return True

        # print a log message every "log_msg_interval" sec while wait_for loop is running
        time_to_print_next_log_msg = self._time_last_logged + self._log_msg_interval
        if current_time >= time_to_print_next_log_msg:
            self._time_remaining -= self._log_msg_interval
            if self._time_remaining > 0:
                log.info("[%s] waiting %dsec longer", self.key, self._time_remaining)
                self._time_last_logged = current_time
        return False

    def wait_for_ready(self, timeout, reraise=False):
        self.timed_out = False
        self._time_last_logged = int(time.time())
        self._time_remaining = timeout

        # we can loop with a much smaller delay if using a ResourceWatcher thread
        delay = 0.1 if self.watcher else 5

        try:
            # check for ready initially, only wait_for if we need to
            log.debug("[%s] checking if 'ready'", self.key)
            if not self.check_ready():
                log.info("[%s] waiting up to %dsec for resource to be 'ready'", self.key, timeout)
                wait_for(
                    self._check_with_periodic_log,
                    message=f"wait for {self.key} to be 'ready'",
                    delay=delay,
                    timeout=timeout,
                )
            return True
        except (StatusError, ErrorReturnCode) as err:
            log.error("[%s] hit error waiting for resource to be ready: %s", self.key, str(err))
            if reraise:
                raise
        except (TimeoutException, TimedOutError):
            # check one last time and error out if its still not ready
            if not self.check_ready():
                self.timed_out = True
                # log a "bulleted list" of the not ready resources and their status conditions
                msg = f"[{self.key}] timed out waiting for resource to be ready"
                details = [
                    f"  {r.details_str}" for _, r in self.observed_resources.items() if not r.ready
                ]
                if details:
                    msg += ", details: {}\n".format("\n".join(details))
                log.error(msg)

            if reraise:
                raise
        return False


def wait_for_ready(namespace, restype, name, timeout=600, watch_owned=True):
    if watch_owned:
        watcher = ResourceWatcher(namespace)
        watcher.start()
    else:
        watcher = None

    try:
        waiter = ResourceWaiter(namespace, restype, name, watch_owned=watch_owned, watcher=watcher)
        return waiter.wait_for_ready(timeout)
    finally:
        if watcher:
            watcher.stop()


def wait_for_ready_threaded(waiters, timeout=600):
    threads = [
        threading.Thread(target=waiter.wait_for_ready, daemon=True, args=(timeout,))
        for waiter in waiters
    ]
    for thread in threads:
        thread.name = thread.name.lower()
        thread.start()
    for thread in threads:
        thread.join()

    timed_out_resources = [w.key for w in waiters if w.timed_out]

    if timed_out_resources:
        log.info("some resources failed to become ready: %s", ", ".join(timed_out_resources))
        return False

    log.info("all resources being monitored reached 'ready' state")
    return True


def copy_namespace_secrets(src_namespace, dst_namespace, secret_names, ignore_annotation_key):
    for secret_name in secret_names:
        secret_data = export("secret", secret_name, namespace=src_namespace)
        ignore = secret_data["metadata"].get("annotations", {}).get(ignore_annotation_key)
        if str(ignore).lower() == "true":
            log.debug(
                "secret '%s' in namespace '%s' has bonfire.ignore==true, skipping",
                secret_name,
                src_namespace,
            )
            continue

        log.info(
            "copying secret '%s' from namespace '%s' to namespace '%s'",
            secret_name,
            src_namespace,
            dst_namespace,
        )
        oc(
            "apply",
            f="-",
            n=dst_namespace,
            _in=json.dumps(secret_data),
            _silent=True,
        )


def process_template(template_data, params, local=True):
    api_version = template_data.get("apiVersion")
    kind = template_data.get("kind")

    if not api_version:
        raise ValueError("template data has no 'apiVersion' defined")
    if not kind:
        raise ValueError("template data has no 'kind' defined")
    if str(kind).lower() != "template":
        raise ValueError("template data 'kind' is not 'Template'")
    if str(api_version).lower() == "v1":
        # convert apiVersion since non-groupified resources are no longer supported
        # in newer versions of the oc client (e.g. 4.17)
        log.warning("converted template's deprecated apiVersion 'v1' to 'template.openshift.io/v1'")
        template_data["apiVersion"] = "template.openshift.io/v1"

    valid_pnames = set(p["name"] for p in template_data.get("parameters", []))

    param_strs = []
    for key, val in params.items():
        if key not in valid_pnames:
            continue
        # prevent python bools from getting passed to templates as "True"/"False"
        if isinstance(val, bool):
            val = str(val).lower()
        param_strs.append(f"-p {key}='{val}'")

    param_str = " ".join(param_strs)
    local_str = str(local).lower()

    args = f"process --local={local_str} --ignore-unknown-parameters -o json -f - {param_str}"

    output = oc(shlex.split(args), _silent=True, _in=json.dumps(template_data))

    return json.loads(str(output))


# assume that the result of this will not change during execution of a single 'bonfire' command
@functools.lru_cache(maxsize=None, typed=False)
def on_k8s():
    """Detect whether this is a k8s or openshift cluster based on existence of projects."""
    project_resource = [r for r in get_api_resources() if r["name"] == "project"]

    if project_resource:
        return False
    return True


def get_all_namespaces(label=None):
    if not on_k8s():
        all_namespaces = get_json("project", label=label)["items"]
    else:
        all_namespaces = get_json("namespace", label=label)["items"]

    return all_namespaces


def get_current_namespace():
    """Get current namespace/project"""
    if not on_k8s():
        namespace = oc("project", "-q", _ignore_errors=True)
        if not namespace:
            return None
        return namespace.strip()

    context_name = oc("config", "current-context").strip()
    if not context_name:
        return None

    context = oc("config", "get-contexts", context_name)
    if not context:
        return None

    context_lines = context.splitlines()
    if len(context_lines) < 2:
        return None

    headers = context_lines[0].split()
    try:
        namespace_idx = headers.index("NAMESPACE")
    except ValueError:
        return None

    try:
        return context_lines[1].split()[namespace_idx]
    except IndexError:
        return None


def set_current_namespace(namespace):
    """Sets a namespace on current context"""
    if not on_k8s():
        oc("project", namespace)
    else:
        # set namespace on current context
        oc("config", "set-context", "--current", "--namespace", namespace)


def any_pods_running(namespace, label):
    """
    Return true if any pods are running associated with provided label
    """
    pod_data = get_json("pod", label=label, namespace=namespace)
    if not pod_data or not len(pod_data.get("items", [])):
        log.info("No pods found for label '%s'", label)
        return False
    for pod in pod_data["items"]:
        if _check_status_for_restype("pod", pod):
            return True
    return False


def all_pods_running(namespace, label):
    """
    Return true if all pods are running associated with provided label
    """
    pod_data = get_json("pod", label=label, namespace=namespace)
    if not pod_data or not len(pod_data.get("items", [])):
        log.info("No pods found for label '%s'", label)
        return False
    statuses = []
    for pod in pod_data["items"]:
        statuses.append(_check_status_for_restype("pod", pod))
    return len(statuses) > 0 and all(statuses)


def no_pods_running(namespace, label):
    """
    Return true if there are no pods running associated with provided label
    """
    return not any_pods_running(namespace, label)


def _get_associated_pods_using_match_labels(namespace, restype, name):
    data = get_json(restype, name, namespace=namespace)
    if not data:
        raise ValueError(f"resource {restype}/{name} not found")

    match_labels = traverse_keys(data, ["spec", "selector", "matchLabels"])
    if match_labels is None:
        raise ValueError(f"resource {restype}/{name} has no 'matchLabels' selector specified")

    label_str = ",".join([f"{key}={val}" for key, val in match_labels.items()])

    return get_json("pod", label=label_str, namespace=namespace)


def get_associated_pods(namespace, restype, name):
    """
    Get all pods associated with specified resource
    """
    restype = parse_restype(restype)
    if restype == "deployment":
        return _get_associated_pods_using_match_labels(namespace, restype, name)
    raise ValueError(f"unsupported restype: {restype}")


def _scale_down_up_using_match_labels(namespace, restype, name, timeout):
    data = get_json(restype, name, namespace=namespace)
    if not data:
        raise ValueError(f"resource {restype}/{name} not found")

    orig_replicas = traverse_keys(data, ["spec", "replicas"])
    if orig_replicas is None:
        raise ValueError(f"resource {restype}/{name} has no 'replicas' in 'spec'")
    if orig_replicas == 0:
        raise ValueError(f"resource {restype}/{name} has 'replicas' set to 0 in spec")

    match_labels = traverse_keys(data, ["spec", "selector", "matchLabels"])
    if match_labels is None:
        raise ValueError(f"resource {restype}/{name} has no 'matchLabels' selector specified")

    label_str = ",".join([f"{key}={val}" for key, val in match_labels.items()])

    oc("scale", restype, name, namespace=namespace, replicas=0)
    wait_for(
        no_pods_running,
        func_args=(
            namespace,
            label_str,
        ),
        message=f"wait for {restype}/{name} to have no pods running",
        timeout=timeout,
        delay=5,
        log_on_loop=True,
    )

    oc("scale", restype, name, namespace=namespace, replicas=orig_replicas)
    wait_for_ready(namespace, restype, name, timeout)


def scale_down_up(namespace, restype, name, timeout=300):
    """
    Scale specified resource down to 0 and back up to original replica count
    """
    restype = parse_restype(restype)
    if restype == "deployment":
        return _scale_down_up_using_match_labels(namespace, restype, name, timeout)
    raise ValueError(f"unsupported restype for scaling down/up: {restype}")
