"""DRAFT_PENDING_PRIMARY_REVIEW: one-shot detached wrapper for L17 formal v2.

Execution is refused until a primary executor authorization freezes the
wrapper, coordinator target, runner, contract, controller, settings and review
SHA values.  The authorization is a separate file so the settings/review pair
does not need a circular self-pin.  On
success it creates one new run directory with an exclusive lock, starts one
durable local coordinator process with DEVNULL stdin and file stdout/stderr,
records the PID and source SHAs, and exits without waiting.  It never polls,
retries, or claims completion.
"""
import ast
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20261019T-L17-formal-v2-detached-executor-v1"
RUN = ROOT / "runs" / RUN_ID
WRAPPER = "scripts/launch_C32_L17_formal_v2_detached_once.py"
TARGET = "scripts/run_C32_L17_formal_v2_coordinator.py"
RUNNER = "scripts/run_C32_L17_native_formal_v2.py"
CONTRACT = "scripts/glm53_C32_L17_pipeline_contract_v3.py"
CONTROLLER = "scripts/glm53_C32_L17_formal_resource_controller_v2.py"
SETTINGS = "config/C32-L17-native-runtime-v1/formal-C32-native-v2.json"
REVIEW = "results/C32-L17-native-formal-actual-review-v2.json"
AUTHORIZATION = "results/C32-L17-formal-v2-detached-executor-authorization-v1.json"
EXPECTED_RUN_ID = "20261019T-L17-sglang-formal-C32-native-v2"
EXPECTED_NAMESPACE = "glm53-formal-20261019T-L17-sglang-formal-C32-native-v2-C32"
ORIGINAL_CONTROL_SOCKET = "/Users/wio/work/sglang-glm53-pd/ssh-control"
FRESH_CONTROL_SOCKET = "/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(relative):
    path = ROOT / relative
    assert path.is_file() and not path.is_symlink(), "missing or symlinked source: " + relative
    return path


def has_main_guard(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        value = test.comparators[0]
        if not isinstance(value, ast.Constant) or value.value != "__main__":
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if not isinstance(inner, ast.Call):
                    continue
                function = inner.func
                if isinstance(function, ast.Name) and function.id == "main":
                    return True
    return False


def coordinator_runs_actual_main(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "run_path":
            continue
        if not isinstance(function.value, ast.Name) or function.value.id != "runpy":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        run_name = keywords.get("run_name")
        if isinstance(run_name, ast.Constant) and run_name.value == "__main__":
            return True
    return False


def frozen_authorization_guard():
    wrapper_path = source_path(WRAPPER)
    target_path = source_path(TARGET)
    runner_path = source_path(RUNNER)
    contract_path = source_path(CONTRACT)
    controller_path = source_path(CONTROLLER)
    settings_path = source_path(SETTINGS)
    review_path = source_path(REVIEW)
    authorization_path = source_path(AUTHORIZATION)

    assert ROOT.name == "glm53", "unexpected project root: " + str(ROOT)
    assert (ROOT / "AGENTS.md").is_file() and (ROOT / "EXECUTION_CONTRACT.md").is_file(), "project root identity mismatch"
    assert FRESH_CONTROL_SOCKET != ORIGINAL_CONTROL_SOCKET
    assert Path(FRESH_CONTROL_SOCKET).is_socket(), "fresh SSH control socket missing: " + FRESH_CONTROL_SOCKET

    hashes = {
        "wrapper_sha256": sha256_file(wrapper_path),
        "target_sha256": sha256_file(target_path),
        "runner_sha256": sha256_file(runner_path),
        "contract_sha256": sha256_file(contract_path),
        "controller_sha256": sha256_file(controller_path),
        "settings_sha256": sha256_file(settings_path),
        "review_sha256": sha256_file(review_path),
        "authorization_sha256": sha256_file(authorization_path),
    }
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))

    assert settings["project_id"] == "glm53"
    assert settings["framework"] == "sglang" and settings["concurrency"] == 32
    assert settings["run_id"] == EXPECTED_RUN_ID and settings["namespace"] == EXPECTED_NAMESPACE
    assert settings["status"] == "FROZEN_READY_FOR_SINGLE_EXECUTION", "settings are still a draft"
    assert settings["runtime_authorized"] is True, "settings do not authorize runtime"
    assert settings["retries"] == 0 and settings["profiling"] is False
    assert settings["actual_review_path"] == REVIEW
    assert settings["actual_review_sha256"] == hashes["review_sha256"], "settings do not pin this review"
    assert settings["resource_controller_sha256"] == hashes["controller_sha256"], "settings controller SHA differs"
    assert settings["backend_HTTP_attempts_planned"] == 320
    assert settings["formal_requests"] == 128 and settings["warmup_logical_requests"] == 32
    assert settings["successful_output_tokens_if_all_pass"] == 76992

    assert review["project_id"] == "glm53"
    assert str(review.get("decision", "")).startswith("APPROVED"), "review is not approved"
    assert review["runtime_authorized"] is True, "review does not authorize runtime"
    assert review["zero_retries"] is True and review["retries"] == 0 and review["new_profiles"] == 0
    assert review["run_id"] == EXPECTED_RUN_ID and review["namespace"] == EXPECTED_NAMESPACE
    assert review["concurrency"] == 32
    assert review["warmup_logical_requests"] == 32 and review["formal_requests"] == 128
    assert review["backend_HTTP_attempts"] == 320
    assert review["successful_output_tokens_if_all_pass"] == 76992
    assert review["input_manifest_sha256"] == settings["input_manifest_sha256"]
    assert review["launcher_sha256"] == hashes["runner_sha256"], "review launcher SHA differs"
    assert review["controller_sha256"] == hashes["controller_sha256"], "review controller SHA differs"
    assert review["contract_sha256"] == hashes["contract_sha256"], "review contract SHA differs"

    assert authorization["project_id"] == "glm53"
    assert str(authorization.get("decision", "")).startswith("APPROVED"), "executor authorization is not approved"
    assert authorization["runtime_authorized"] is True, "executor authorization does not authorize runtime"
    assert authorization["zero_retries"] is True and authorization["retries"] == 0
    assert authorization["one_shot_run_id"] == RUN_ID
    for key in ["wrapper_sha256", "target_sha256", "runner_sha256", "contract_sha256",
                "controller_sha256", "settings_sha256", "review_sha256"]:
        assert isinstance(authorization.get(key), str) and len(authorization[key]) == 64, (
            "executor authorization missing: " + key
        )
        assert authorization[key] == hashes[key], "executor authorization SHA differs: " + key
    assert authorization["executor_system"] == platform.system(), "executor system differs"
    assert authorization["executor_machine"] == platform.machine(), "executor machine differs"
    assert authorization["executor_hostname"] == socket.gethostname(), "executor hostname differs"
    assert str(Path(authorization["python_executable"]).resolve()) == str(Path(sys.executable).resolve()), (
        "executor python differs"
    )

    target_text = target_path.read_text(encoding="utf-8")
    assert 'RUNNER_SHA256 = "%s"' % hashes["runner_sha256"] in target_text, "coordinator runner pin differs"
    assert 'CONTRACT_SHA256 = "%s"' % hashes["contract_sha256"] in target_text, "coordinator contract pin differs"
    assert 'CONTROLLER_SHA256 = "%s"' % hashes["controller_sha256"] in target_text, "coordinator controller pin differs"
    assert "pinned_source(RUNNER, RUNNER_SHA256)" in target_text, "coordinator does not pin the runner before execution"
    assert 'runpy.run_path(str(runner), run_name="__main__")' in target_text, "coordinator does not run the pinned runner main"
    assert has_main_guard(target_path), "coordinator main guard missing"
    assert coordinator_runs_actual_main(target_path), "coordinator can bypass the runner main"
    assert has_main_guard(runner_path), "runner main guard missing"
    return settings, review, authorization, hashes


def main():
    settings, review, authorization, hashes = frozen_authorization_guard()

    if RUN.exists():
        raise SystemExit("one-shot run directory already exists: " + str(RUN))
    RUN.mkdir(parents=True, exist_ok=False)
    lock_path = RUN / "execution-once.lock"
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(lock_fd, json.dumps({
            "project_id": "glm53",
            "run_id": RUN_ID,
            "wrapper_pid": os.getpid(),
            "started_utc": utc_now(),
            "phase": "WRAPPER_LOCKED_NOT_COMPLETION",
        }, sort_keys=True).encode())
        os.fsync(lock_fd)
    finally:
        os.close(lock_fd)

    stdout_path = RUN / "coordinator-stdout.log"
    stderr_path = RUN / "coordinator-stderr.log"
    argv = [sys.executable, "-u", str(source_path(TARGET))]
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
                cwd=str(ROOT),
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
            )
    except BaseException as error:
        failure = {
            "project_id": "glm53",
            "run_id": RUN_ID,
            "phase": "DETACHED_LAUNCH_FAILED",
            "wrapper_pid": os.getpid(),
            "argv": argv,
            "error": repr(error),
            "utc": utc_now(),
            "completion_claimed": False,
            "retries": 0,
        }
        (RUN / "launch-failed.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise

    launch = {
        "project_id": "glm53",
        "run_id": RUN_ID,
        "phase": "DETACHED_STARTED_NOT_COMPLETION",
        "wrapper": WRAPPER,
        "wrapper_sha256": hashes["wrapper_sha256"],
        "wrapper_pid": os.getpid(),
        "target": TARGET,
        "target_sha256": hashes["target_sha256"],
        "coordinator_pid": process.pid,
        "coordinator_session_detached": True,
        "runner": RUNNER,
        "runner_sha256": hashes["runner_sha256"],
        "contract": CONTRACT,
        "contract_sha256": hashes["contract_sha256"],
        "controller": CONTROLLER,
        "controller_sha256": hashes["controller_sha256"],
        "settings": SETTINGS,
        "settings_sha256": hashes["settings_sha256"],
        "settings_status": settings["status"],
        "review": REVIEW,
        "review_sha256": hashes["review_sha256"],
        "review_decision": review["decision"],
        "executor_authorization": AUTHORIZATION,
        "executor_authorization_sha256": hashes["authorization_sha256"],
        "executor_authorization_decision": authorization["decision"],
        "argv": argv,
        "stdin": "DEVNULL",
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "coordinator_exit_record": str((RUN / "coordinator-exit.json").relative_to(ROOT)),
        "execution_once_lock": str(lock_path.relative_to(ROOT)),
        "start_new_session": True,
        "close_fds": True,
        "waits_for_completion": False,
        "polling": False,
        "retries": 0,
        "completion_claimed": False,
        "started_utc": utc_now(),
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "python_executable": str(Path(sys.executable).resolve()),
            "project_root": str(ROOT),
            "fresh_control_socket": FRESH_CONTROL_SOCKET,
        },
        "source_sha256": hashes,
    }
    (RUN / "launch.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "project_id": "glm53",
        "run_id": RUN_ID,
        "phase": "DETACHED_STARTED_NOT_COMPLETION",
        "wrapper_pid": os.getpid(),
        "coordinator_pid": process.pid,
        "waits_for_completion": False,
        "source_sha256": hashes,
        "executor_authorization": AUTHORIZATION,
        "launch_receipt": str(RUN / "launch.json"),
        "coordinator_exit_record": str(RUN / "coordinator-exit.json"),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
