"""Run the pinned GSM8K cohort with a bounded C16 submission window."""

import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import time
import urllib.error
import urllib.request


MODEL = "ds41flash"
CONCURRENCY = 16
REQUESTS = 1319
REQUEST_TIMEOUT = 600
RUN_TIMEOUT = 14400
DATASET_FILE = "grade_school_math/data/test.jsonl"
GRADER_FILE = "grade_school_math/dataset.py"
DATASET_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
GRADER_SHA256 = "c22b81bccaaebfd1123be296e8b3c3dca3ddc704c46b70bf4a50603bb627e197"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def stored_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(stored_json(value))
    temporary.replace(path)


def request_body(row):
    question = row["question"]
    prompt = question + '\nSolve the problem. Show a brief calculation, then end your response with "#### <numeric answer>".'
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 2048,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }


def assets(root):
    provenance = json.loads((root / "dataset_provenance.json").read_bytes())
    for entry in provenance["files"]:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Asset SHA256 mismatch for {entry['path']}")
    dataset, grader = root / DATASET_FILE, root / GRADER_FILE
    if digest(dataset) != DATASET_SHA256 or digest(grader) != GRADER_SHA256:
        raise ValueError("Expected the pinned dataset and official grader")
    rows = [json.loads(line) for line in dataset.read_bytes().splitlines()]
    if len(rows) != REQUESTS:
        raise ValueError("Expected all 1319 held-out GSM8K questions")
    nodes = []
    for node in ast.parse(grader.read_bytes()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"ANS_RE", "INVALID_ANS"} for target in node.targets
        ):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in {"extract_answer", "is_correct"}:
            nodes.append(node)
    if len(nodes) != 4:
        raise ValueError("Could not load the exact official grading rules")
    settings = {"re": re}
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(tree, "<pinned official GSM8K grader>", "exec"), settings)
    if any(settings["extract_answer"](row["answer"]) == settings["INVALID_ANS"] for row in rows):
        raise ValueError("Unparseable official ground truth")
    return rows, provenance["commit"], settings["extract_answer"], settings["is_correct"]


def evaluate(router, output, extract, correct, index, row):
    tag = f"{index:04d}"
    body = request_body(row)
    request_path = output / f"{tag}_request.json"
    response_path = output / f"{tag}_response.json"
    save(request_path, body)
    started = time.monotonic()
    source = {
        "index": index, "expected": extract(row["answer"]), "request_started_UTC": utc(),
        "request_sha256": digest(request_path),
    }
    req = urllib.request.Request(
        router + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read()
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status {response.status}")
        response_path.write_bytes(raw)
        result = json.loads(raw)
        choice = result["choices"][0]
        usage = result.get("usage") or {}
        if result["model"] != MODEL or usage.get("completion_tokens", 0) <= 0:
            raise ValueError("Response model or server usage did not match")
        text = choice["message"].get("content") or ""
        return {
            **source, "transport_ok": True, "correct": correct(text, row), "extracted": extract(text),
            "finish_reason": choice.get("finish_reason"), "usage": usage,
            "elapsed_s": time.monotonic() - started, "response_sha256": sha256(raw).hexdigest(),
            "server_response_id": result.get("id"),
        }
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            (output / f"{tag}_HTTP_error.txt").write_bytes(exc.read())
        return {
            **source, "transport_ok": False, "correct": False,
            "error": repr(exc), "elapsed_s": time.monotonic() - started,
        }


def evaluate_bounded(rows, worker, deadline, *, request_timeout=REQUEST_TIMEOUT, concurrency=CONCURRENCY):
    """Keep no more than one in-flight request per worker and reserve a full request window."""
    next_index = 0
    pending = set()
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        while True:
            while (len(pending) < concurrency and next_index < len(rows)
                   and time.monotonic() < deadline - request_timeout):
                pending.add(pool.submit(worker, next_index, rows[next_index]))
                next_index += 1
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Whole GSM8K run exceeded its bound")
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done:
                raise TimeoutError("GSM8K requests still in flight at the whole-run deadline")
            for future in done:
                yield future.result()
        if next_index != len(rows):
            raise TimeoutError(f"Stopped at {next_index} submitted requests to preserve the full timeout")
    finally:
        # There is no queued backlog. The launcher supplies a hard deadline for in-flight I/O.
        pool.shutdown(wait=False, cancel_futures=True)


def interrupted(signum, _frame):
    raise TimeoutError(f"GSM8K worker received signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    router = os.environ["SG_GSM8K_ROUTER_URL"].rstrip("/")
    started = time.monotonic()
    deadline = started + RUN_TIMEOUT
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    for caught in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(caught, interrupted)
    signal.alarm(RUN_TIMEOUT)

    def state(phase, **details):
        value = {"phase": phase, "pid": os.getpid(), "at": utc(),
                 "elapsed_s": time.monotonic() - started, **details}
        save(output / "state.json", value)
        print(json.dumps(value), flush=True)

    results = []
    try:
        rows, dataset_commit, extract, correct = assets(args.assets.resolve(strict=True))
        manifest = {
            "dataset_sha256": DATASET_SHA256, "dataset_commit": dataset_commit, "grader_sha256": GRADER_SHA256,
            "test_rows": REQUESTS, "concurrency": CONCURRENCY, "request_timeout_s": REQUEST_TIMEOUT,
            "run_timeout_s": RUN_TIMEOUT, "max_tokens": 2048, "temperature": 0, "seed": 0,
            "thinking": False, "retries": 0, "scope": "G1317 paired Ascend 910B1 GSM8K quality validation",
        }
        with urllib.request.urlopen(router + "/v1/models", timeout=20) as response:
            models = json.load(response)
        if models["data"][0]["id"] != MODEL:
            raise ValueError("Expected the held SGLang model instance")
        save(output / "actual_models.json", models)
        save(output / "manifest.json", manifest)
        state("full_test_running", rows=REQUESTS, concurrency=CONCURRENCY)
        worker = partial(evaluate, router, output, extract, correct)
        with (output / "results.jsonl").open("x", buffering=1) as stream:
            for row in evaluate_bounded(rows, worker, deadline):
                results.append(row)
                print(json.dumps(row), file=stream)
                if len(results) % 64 == 0:
                    state("full_test_running", completed=len(results), correct=sum(r["correct"] for r in results))
        if len(results) != REQUESTS or {r["index"] for r in results} != set(range(REQUESTS)):
            raise ValueError("A complete GSM8K run must cover every original index once")
        right = sum(row["correct"] for row in results)
        summary = {
            "completed": len(results), "correct": right, "accuracy": right / REQUESTS,
            "transport_failures": sum(not row["transport_ok"] for row in results),
            "length_limited_responses": sum(row.get("finish_reason") == "length" for row in results),
            "all_raw_responses_present": all((output / f"{i:04d}_response.json").is_file() for i in range(REQUESTS)),
            "dataset_sha256": DATASET_SHA256, "grader_sha256": GRADER_SHA256,
            "results_jsonl_sha256": digest(output / "results.jsonl"),
            "quality_comparison_accepted": False, "scope": manifest["scope"],
        }
        save(output / "result.json", summary)
        state("full_test_complete_requires_parent_review", **summary)
    except BaseException as exc:
        state("failed_requires_review", completed=len(results), error=repr(exc))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
