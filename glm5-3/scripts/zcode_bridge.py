#!/usr/bin/env python3
"""Bound Zcode CLI output and validate file-based job/result handoffs. Stdlib only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

STATUSES = {"queued", "running", "completed", "failed", "cancelled", "timed_out", "needs_decision"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, max_bytes):
    with Path(path).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    require(len(raw) <= max_bytes, "JSON exceeds handoff size limit")
    value = json.loads(raw)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_job(job):
    require(type(job.get("schema_version")) is int and job["schema_version"] == 1, "schema_version must be 1")
    require(isinstance(job.get("job_id"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["job_id"]), "invalid job_id")
    for key in ("owner", "kind", "goal"):
        require(isinstance(job.get(key), str) and job[key].strip(), key + " must be nonempty")
    for key in ("inputs", "acceptance"):
        require(isinstance(job.get(key), list), key + " must be a list")
    require(isinstance(job.get("parent"), dict) and isinstance(job.get("scope"), dict), "parent/scope required")
    execution = job.get("execution", {})
    result = job.get("result", {})
    require(isinstance(execution, dict) and isinstance(result, dict), "execution/result must be objects")
    require(execution.get("mode") in ("task", "command"), "mode must be task or command")
    require(execution.get("work_type") in ("analysis", "command", "persistent"), "work_type must be analysis, command or persistent")
    require(execution["mode"] != "command" or execution["work_type"] == "command", "command mode requires command work_type")
    require(isinstance(execution.get("cwd"), str) and Path(execution["cwd"]).is_absolute(), "cwd must be absolute")
    require(type(execution.get("timeout_s")) in (int, float) and math.isfinite(execution["timeout_s"]) and execution["timeout_s"] > 0, "finite positive timeout_s required")
    key = "command" if execution["mode"] == "command" else "prompt"
    require(isinstance(execution.get(key), str) and execution[key].strip(), key + " required")
    require(isinstance(result.get("path"), str) and Path(result["path"]).is_absolute(), "result.path must be absolute")
    require(type(result.get("max_bytes")) is int and result["max_bytes"] > 0, "positive result.max_bytes required")


def validate_result(job, result):
    require(result.get("schema_version") == 1 and type(result.get("schema_version")) is int, "result schema_version must be 1")
    require(result.get("job_id") == job["job_id"], "result job_id mismatch")
    require(isinstance(result.get("status"), str) and result["status"] in STATUSES, "invalid result status")
    require(isinstance(result.get("summary"), str) and result["summary"].strip(), "summary required")
    for key in ("findings", "evidence", "unknowns"):
        require(isinstance(result.get(key), list), key + " must be a list")
    require("decision_request" in result and "next_check_at" in result, "decision/check fields required")
    execution = result.get("execution", {})
    require(isinstance(execution, dict), "result execution required")
    require("inner_exit_code" in execution, "inner_exit_code required (null if unknown/not applicable)")
    code = execution["inner_exit_code"]
    require(code is None or type(code) is int, "inner_exit_code must be integer or null")
    require(execution.get("acceptance") in ("passed", "failed", "unverified"), "invalid acceptance status")
    require(isinstance(execution.get("processes"), list), "processes must be a list")
    if result["status"] == "completed":
        require(execution["acceptance"] == "passed", "completed requires passed acceptance")
        require(code is None or code == 0, "completed cannot report a failed inner command")
        if job["execution"]["work_type"] != "analysis":
            require(code == 0, "completed command requires observed inner_exit_code 0")
    ids = set()
    for evidence in result["evidence"]:
        require(isinstance(evidence, dict), "evidence must be an object")
        require(isinstance(evidence.get("id"), str) and evidence["id"].strip() and evidence["id"] not in ids, "unique evidence id required")
        ids.add(evidence["id"])
        require(isinstance(evidence.get("path") or evidence.get("uri"), str), "evidence path/uri required")
        require(all(k in evidence for k in ("sha256", "bytes", "locator")), "evidence identity fields required")
        digest = evidence["sha256"]
        require(digest is None or isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{64}", digest), "invalid sha256")
        require(evidence["bytes"] is None or type(evidence["bytes"]) is int and evidence["bytes"] >= 0, "invalid bytes")
    for finding in result["findings"]:
        require(isinstance(finding, dict) and finding.get("kind") in ("fact", "inference"), "finding kind required")
        require(isinstance(finding.get("text"), str) and finding["text"].strip(), "finding text required")
        require(isinstance(finding.get("scope"), dict), "finding scope/coverage required")
        refs = finding.get("evidence_ids")
        require(isinstance(refs, list) and all(isinstance(ref, str) and ref in ids for ref in refs), "finding evidence references invalid")
    if result["status"] == "running":
        require(execution["processes"], "running requires persistent process/session identity")
        for process in execution["processes"]:
            require(isinstance(process, dict), "process must be an object")
            pid, session = process.get("pid"), process.get("session_id")
            require(type(pid) is int and pid > 0 or isinstance(session, str) and session.strip(), "running requires actual pid/session")
            require(isinstance(process.get("host"), str) and process["host"].strip(), "process host required")
            require(process.get("readiness") in ("ready", "unready", "unknown"), "process readiness required")
            refs = process.get("evidence_ids")
            require(isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in ids for ref in refs), "running process evidence required")


def validate_bridge(job, result_path):
    bridge = read_json(Path(result_path).parent / "bridge.json", 65536)
    require(bridge.get("job_id") == job["job_id"], "bridge job_id mismatch")
    require(type(bridge.get("cli_exit_code")) is int and bridge["cli_exit_code"] == 0
            and bridge.get("timed_out") is False and bridge.get("cli_error") is None,
            "bridge records CLI failure/timeout or unverified exit")


def prompt_for(job):
    if job["execution"]["mode"] == "command":
        return job["execution"]["command"]
    return (
        "执行以下Job。在现场读取输入，大日志/trace留文件；遵守scope和现有controller，"
        "不要复制原始输出到回复。将紧凑JSON原子写到result.path。"
        "必需字段：schema_version=1, job_id, status(queued/running/completed/failed/cancelled/"
        "timed_out/needs_decision), summary, execution(inner_exit_code:null或真实整数,"
        "acceptance:passed/failed/unverified,processes:[]), findings:[], evidence:[], unknowns:[],"
        "decision_request:null或具体问题,next_check_at:null或检查时间。"
        "completed必须验收passed，work_type非analysis还须真实inner_exit_code=0。"
        "running必须有实际host/pid或session_id、readiness及对应evidence_ids，"
        "不要猜退出码、PID或数值。evidence每项含id,path或uri,"
        "sha256,bytes,locator，不可核验值用null。findings每项含kind:fact/inference,"
        "text,scope(窗口/过滤/实例rank覆盖),evidence_ids(引用已有evidence.id)。\n"
        + json.dumps(job, ensure_ascii=False)
    )


def run_job(job, executable):
    cwd = Path(job["execution"]["cwd"])
    result_path = Path(job["result"]["path"])
    require(cwd.is_dir(), "actual cwd does not exist")
    require(not result_path.exists(), "result already exists; use validate for updates or a new job_id for retry")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_path = result_path.parent / "bridge.json"
    require(not bridge_path.exists(), "job directory already used; preserve prior execution")
    with (result_path.parent / "job.claim").open("x", encoding="utf-8") as claim:
        claim.write(job["job_id"] + "\n")
    atomic_json(result_path.parent / "job.json", job)
    stdout_path = result_path.parent / "cli.stdout.log"
    stderr_path = result_path.parent / "cli.stderr.log"
    bridge = {"job_id": job["job_id"], "started_at": datetime.now(timezone.utc).isoformat(),
              "backend_family": "deepseek", "backend_family_source": "user_configuration",
              "backend_model_id": None, "cli_exit_code": None, "timed_out": False,
              "stdout": str(stdout_path), "stderr": str(stderr_path)}
    cli_error = None
    atomic_json(bridge_path, bridge)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen([executable, "--prompt", prompt_for(job)], cwd=cwd,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            bridge["cli_pid"] = process.pid
            atomic_json(bridge_path, bridge)
            try:
                bridge["cli_exit_code"] = process.wait(timeout=job["execution"]["timeout_s"])
            except subprocess.TimeoutExpired:
                bridge["timed_out"] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                bridge["cli_exit_code"] = process.returncode
        except OSError as error:
            cli_error = str(error)
    bridge["finished_at"] = datetime.now(timezone.utc).isoformat()
    bridge["cli_error"] = cli_error
    bridge["stdout_bytes"] = stdout_path.stat().st_size
    bridge["stderr_bytes"] = stderr_path.stat().st_size
    bridge["stdout_sha256"] = file_sha256(stdout_path)
    bridge["stderr_sha256"] = file_sha256(stderr_path)
    atomic_json(bridge_path, bridge)
    envelope = {"job_id": job["job_id"], "bridge": str(bridge_path), "result_path": str(result_path),
                "cli_exit_code": bridge["cli_exit_code"], "timed_out": bridge["timed_out"]}
    try:
        require(cli_error is None and not bridge["timed_out"] and bridge["cli_exit_code"] == 0,
                cli_error or "CLI failed/timed out; service state may need recovery")
        result = read_json(result_path, job["result"]["max_bytes"])
        validate_result(job, result)
        envelope.update({"handoff": "VALID", "result": result})
        return envelope, 0
    except (ValueError, OSError) as error:
        envelope.update({"handoff": "INVALID", "error": str(error)})
        return envelope, 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("job")
    run.add_argument("--zcode", default="zcode", help="actual CLI executable path")
    validate = commands.add_parser("validate")
    validate.add_argument("job")
    validate.add_argument("result")
    validate.add_argument("--format-only", action="store_true", help="subagent/native results without CLI evidence; not execution validation")
    args = parser.parse_args()
    try:
        job = read_json(args.job, 65536)
        validate_job(job)
        if args.action == "run":
            envelope, code = run_job(job, args.zcode)
        else:
            result = read_json(args.result, job["result"]["max_bytes"])
            validate_result(job, result)
            if not args.format_only:
                validate_bridge(job, args.result)
            envelope, code = {"handoff": "FORMAT_VALID" if args.format_only else "VALID",
                              "cli_execution_verified": not args.format_only, "result": result}, 0
    except (ValueError, OSError) as error:
        envelope, code = {"handoff": "INVALID", "error": str(error)}, 2
    print(json.dumps(envelope, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
