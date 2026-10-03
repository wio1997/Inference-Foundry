"""Durable HOST execution of one exact task-owned native SDK client.

This module never signals inference servers, NPU workers or a guessed process
group. Actual SDK acknowledgements remain caller-validated raw log evidence.
"""
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from phase_runner import atomic_json, process_identity, same_process, utc


class NativeClientExecutionError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"native client {record['status']}; exit={record['exit_code']}")


def _argv(pid):
    try:
        return [x.decode() for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
    except FileNotFoundError:
        return []


def _matching_container_clients(expected, container):
    raw = subprocess.check_output(
        ["docker", "top", container, "-eo", "pid,ppid,stat,comm,args"], text=True, timeout=15
    )
    found = []
    for line in raw.splitlines()[1:]:
        row = line.split(None, 4)
        if len(row) == 5 and row[2][0] not in ("Z", "X"):
            pid = int(row[0])
            if _argv(pid) == expected:
                found.append(process_identity(pid))
    if len(found) > 1:
        raise RuntimeError("ambiguous exact native client; no native signals")
    return found[0] if found else None


def run_native_client(argv, *, record_path, stdout_path, stderr_path,
                      expected_native_argv, timeout_s, guard=lambda: None,
                      container="glm52-single", host_only=False, grace_s=120):
    """Stream to exclusive files; record every exit and exact cancellation.

    host_only is for explicit CPU contracts. Production uses docker exec and
    requires a unique absolute GLM Run client script, separately from its model.
    """
    if not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("nonempty argv required")
    if not 0 < timeout_s < float("inf") or not 0 < grace_s < float("inf"):
        raise ValueError("finite positive execution and cancellation bounds required")
    if not host_only:
        if argv[:3] != ["docker", "exec", container]:
            raise ValueError("only foreground task container exec supported")
        if ("script" not in expected_native_argv
                or not any(re.fullmatch(r"/.+/GLM-RUN-[0-9]{4}/[^/]*client\.py", x)
                           for x in expected_native_argv)
                or "serve" in expected_native_argv):
            raise ValueError("exact task client script required; model operations forbidden")
        if _matching_container_clients(expected_native_argv, container):
            raise RuntimeError("client already exists; refusing replay")
    paths = [Path(record_path), Path(stdout_path), Path(stderr_path)]
    if any(x.exists() for x in paths):
        raise FileExistsError("exclusive execution artifacts already exist; refusing replay")
    record_path, stdout_path, stderr_path = paths
    rec = dict(schema_version=1, kind="durable_native_client_execution",
               status="starting", started_at=utc(), finished_at=None,
               argv=argv, expected_native_argv=expected_native_argv,
               host_only=host_only, timeout_s=timeout_s, grace_s=grace_s,
               exit_code=None, timed_out=False, signal_attempts=[], native_client=None,
               docker_exec_child=None, SDK_acknowledgements="caller must verify raw logs")
    atomic_json(record_path, rec)
    started = time.monotonic()
    process = None
    native = None
    guard()
    def persist():
        atomic_json(record_path, rec)
    def stop_native(sig):
        if native and same_process(native):
            guard()
            if _argv(native["pid"]) != expected_native_argv:
                raise RuntimeError("native client argv changed; no signal")
            if not host_only:
                fresh = _matching_container_clients(expected_native_argv, container)
                if not fresh or any(fresh[k] != native[k] for k in ("pid", "boot_id", "start_ticks")):
                    raise RuntimeError("container ownership changed; no signal")
            rec["signal_attempts"].append(dict(at=utc(),pid=native["pid"],
                identity=native,signal=sig.name,scope="exact_pinned_task_client_only"))
            persist()
            try:
                os.kill(native["pid"], sig)
            except ProcessLookupError:
                rec["signal_attempts"][-1]["raced_exit"] = True
            persist()
    try:
        with stdout_path.open("xb", buffering=0) as out, stderr_path.open("xb", buffering=0) as err:
            process = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            rec.update(status="running", docker_exec_child=process_identity(process.pid))
            if host_only:
                native = process_identity(process.pid)
                if _argv(process.pid) != expected_native_argv:
                    # exec may be in progress; resolve once below.
                    native = None
            persist()
            while process.poll() is None:
                guard()
                if native is None:
                    native = (process_identity(process.pid) if host_only
                              else _matching_container_clients(expected_native_argv, container))
                    if native and _argv(native["pid"]) != expected_native_argv:
                        native = None
                    if native:
                        rec["native_client"] = native
                        persist()
                elapsed = time.monotonic() - started
                if native is None and elapsed > min(120, timeout_s):
                    raise RuntimeError("native client capture unavailable; no guessed signals")
                if elapsed >= timeout_s:
                    rec["timed_out"] = True
                    persist()
                    if native is None:
                        raise RuntimeError("deadline without captured client; no guessed signals")
                    stop_native(signal.SIGTERM)
                    end = time.monotonic() + grace_s
                    while process.poll() is None and time.monotonic() < end:
                        guard()
                        time.sleep(.1)
                    if process.poll() is None:
                        stop_native(signal.SIGKILL)
                        # The outer child is our exact Popen process, never a process group.
                        if process.poll() is None and same_process(rec["docker_exec_child"]):
                            process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if same_process(rec["docker_exec_child"]):
                                process.kill()
                            process.wait(timeout=10)
                    break
                time.sleep(.2)
            rec["exit_code"] = process.wait()
            for stream in (out, err):
                os.fsync(stream.fileno())
            rec["status"] = "timed_out" if rec["timed_out"] else "succeeded" if rec["exit_code"] == 0 else "failed"
    except BaseException as error:
        rec.update(status="error", error_type=type(error).__name__, error=str(error))
        # Ambiguous/missing native identity is deliberately not repaired by guessing.
        if process is not None and process.poll() is None:
            if native is not None:
                try:
                    stop_native(signal.SIGTERM)
                    process.wait(timeout=grace_s)
                except Exception as cleanup_error:
                    rec["cleanup_unknown"] = str(cleanup_error)
            if process.poll() is None and same_process(rec["docker_exec_child"]):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if same_process(rec["docker_exec_child"]):
                        process.kill()
                    process.wait(timeout=10)
        if process is not None:
            rec["exit_code"] = process.poll()
    finally:
        rec["native_client_alive"] = bool(native and same_process(native))
        rec["finished_at"] = utc()
        rec["elapsed_s"] = time.monotonic() - started
        for key, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if path.exists():
                data = path.read_bytes()
                rec[key] = dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        persist()
    if rec["status"] != "succeeded":
        raise NativeClientExecutionError(rec)
    return rec
