"""Record real child exit status without a shell pipeline. No service operations."""
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return {"pid": pid, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": fields[19], "state": fields[0], "pgid": int(fields[2])}
    except FileNotFoundError:
        if Path("/proc").is_dir():
            return None
        p = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
        if not p.stdout.strip():
            return None
        return {"pid": pid, "start_ticks": p.stdout.strip(), "boot_id": "unavailable", "state": "unknown"}


def same_process(identity):
    if not identity:
        return False
    current = process_identity(identity["pid"])
    return bool(current and current["state"] != "Z" and all(
        current[k] == identity[k] for k in ("pid", "boot_id", "start_ticks")))


class PhaseExecutionError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"phase {record['phase']} {record['status']}; observed exit={record['exit_code']}")


def run_phase(argv, *, phase, record_path, log_path, cwd=None, timeout_s=None,
              cancel_requested=lambda: False, heartbeat=lambda: None, echo=True, run_id=None):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("argv must be a nonempty list; shell strings are forbidden")
    if timeout_s is not None and (timeout_s <= 0 or not __import__('math').isfinite(timeout_s)):
        raise ValueError("timeout must be finite and positive or null")
    record_path, log_path = Path(record_path), Path(log_path)
    if record_path.exists() or log_path.exists():
        raise FileExistsError("phase artifacts already exist; refusing overwrite/replay")
    record = {"schema_version": 1, "run_id": run_id, "phase": phase, "argv": argv,
              "cwd": str(Path(cwd or os.getcwd()).resolve()), "status": "starting", "started_at": utc(),
              "finished_at": None, "exit_code": None, "signal": None, "timed_out": False,
              "cancelled": False, "child": None, "timeout_s": timeout_s,
              "request_counts": None, "artifact_acceptance": "unverified", "error_type": None}
    atomic_json(record_path, record)
    started = time.monotonic()
    process = None
    interrupted = []
    old_handlers = {}
    selector = selectors.DefaultSelector()
    digest, log_bytes = hashlib.sha256(), 0

    def on_signal(signum, _frame):
        interrupted.append(signum)

    def stop_child():
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    try:
        # A log-open failure must be recorded before any child can start.
        with log_path.open("xb") as log:
            process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            record.update(status="running", child=process_identity(process.pid))
            atomic_json(record_path, record)
            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    old_handlers[signum] = signal.signal(signum, on_signal)
            except ValueError:  # Only main-thread callers may install handlers.
                pass
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            exited_at = None
            while selector.get_map() or process.poll() is None:
                heartbeat()
                timed_out = timeout_s is not None and time.monotonic() - started >= timeout_s
                if process.poll() is None and (timed_out or interrupted or cancel_requested()):
                    record.update(timed_out=timed_out, cancelled=not timed_out,
                                  received_signal=interrupted[-1] if interrupted else None)
                    stop_child()
                for key, _ in selector.select(0.05):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        log.write(block)
                        digest.update(block)
                        log_bytes += len(block)
                        if echo:
                            # Broken consumer output must not obscure child execution evidence.
                            try:
                                os.write(1, block)
                            except BrokenPipeError:
                                echo = False
                if process.poll() is not None:
                    exited_at = exited_at or time.monotonic()
                    if selector.get_map() and time.monotonic() - exited_at > 1:
                        raise RuntimeError("child exited but descendants retain stdout; reconciliation required")
            record["exit_code"] = process.wait()
            log.flush()
            os.fsync(log.fileno())
            record["status"] = ("timed_out" if record["timed_out"] else "cancelled" if record["cancelled"]
                                else "succeeded" if record["exit_code"] == 0 else "failed")
    except BaseException as error:
        stop_child()
        record["error_type"] = type(error).__name__
        if process:
            record["exit_code"] = process.returncode
        record["status"] = "error"
    finally:
        selector.close()
        if process and process.stdout:
            process.stdout.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        record["signal"] = -record["exit_code"] if record["exit_code"] is not None and record["exit_code"] < 0 else None
        record.update(finished_at=utc(), elapsed_s=time.monotonic() - started,
                      log={"path": str(log_path), "bytes": log_bytes, "sha256": digest.hexdigest()})
        atomic_json(record_path, record)
    if record["status"] != "succeeded":
        raise PhaseExecutionError(record)
    return record
