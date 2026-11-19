"""Regrade two terminal GSM8K rounds from immutable raw requests and responses."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from run_full_gsm8k_SG_PD_g1402 import (
    CONCURRENCY, DATASET_SHA256, GRADER_SHA256, MODEL, REQUESTS,
    assets, digest, request_body, stored_json,
)


PROTOCOL = {
    "dataset_sha256": DATASET_SHA256, "grader_sha256": GRADER_SHA256, "test_rows": REQUESTS,
    "concurrency": CONCURRENCY, "max_tokens": 2048, "temperature": 0, "seed": 0,
    "thinking": False, "retries": 0,
}


def read(path):
    return json.loads(path.read_bytes())


def check_record(root, row, dataset_row, extract, official):
    index = row["index"]
    prefix = f"{index:04d}"
    posted = (root / f"{prefix}_request.json").read_bytes()
    expected = extract(dataset_row["answer"])
    if posted != stored_json(request_body(dataset_row)) or row["expected"] != expected:
        raise ValueError(f"Request bytes or reference answer differ at index {index}")
    request_sha = sha256(posted).hexdigest()
    response_path = root / f"{prefix}_response.json"
    if row["transport_ok"] is False:
        if row["correct"] is not False:
            raise ValueError(f"Failed transport was counted as correct at index {index}")
        if row.get("request_sha256", request_sha) != request_sha:
            raise ValueError(f"Failed request SHA mismatch at index {index}")
        response_sha = digest(response_path) if response_path.exists() else None
        return "transport_failure", False, request_sha, response_sha

    raw = response_path.read_bytes()
    response_sha = sha256(raw).hexdigest()
    response = json.loads(raw)
    choice = response["choices"][0]
    text = choice["message"].get("content") or ""
    usage = response.get("usage") or {}
    answer_correct = official(text, dataset_row)
    finish = choice["finish_reason"]
    if (
        row["transport_ok"] is not True or row["request_sha256"] != request_sha
        or row["response_sha256"] != response_sha or response.get("model") != MODEL
        or usage != row["usage"] or usage.get("completion_tokens", 0) <= 0
        or row["finish_reason"] != finish or finish not in {"stop", "length"}
        or row["extracted"] != extract(text) or row["correct"] is not answer_correct
    ):
        raise ValueError(f"Raw response, usage or grade mismatch at index {index}")
    return "truncated" if finish == "length" else "complete", answer_correct, request_sha, response_sha


def audit_run(root, results_pin, dataset, extract, official):
    phase = read(root / "state.json")
    if phase["phase"] != "full_test_complete_requires_parent_review" or phase["completed"] != REQUESTS:
        raise ValueError("Expected an actual terminal quality run")
    manifest = read(root / "manifest.json")
    if any(manifest.get(key) != expected for key, expected in PROTOCOL.items()):
        raise ValueError("Quality run manifest does not match the frozen cohort")
    raw = (root / "results.jsonl").read_bytes()
    if sha256(raw).hexdigest() != results_pin:
        raise ValueError("Results log SHA256 mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != REQUESTS or any(type(row["index"]) is not int for row in rows):
        raise ValueError("Unexpected request count or index type")
    indexed = {row["index"]: row for row in rows}
    if set(indexed) != set(range(REQUESTS)):
        raise ValueError("Requests missing or repeated in the terminal log")

    results = {}
    raw_catalog = sha256()
    for index, question in enumerate(dataset):
        status, correct, request_sha, response_sha = check_record(
            root, indexed[index], question, extract, official,
        )
        results[index] = (status, correct)
        raw_catalog.update(f"{index}:{request_sha}:{response_sha or '-'}\n".encode("ascii"))
    summary = read(root / "result.json")
    counts = Counter(status for status, _ in results.values())
    aggregate = {
        "completed": REQUESTS, "correct": sum(right for _, right in results.values()),
        "transport_failures": counts["transport_failure"], "length_limited_responses": counts["truncated"],
        "dataset_sha256": DATASET_SHA256, "grader_sha256": GRADER_SHA256,
        "results_jsonl_sha256": results_pin,
    }
    if any(summary[key] != value for key, value in aggregate.items()):
        raise ValueError("Official regrade does not match the worker's terminal rollup")
    return results, {**aggregate, "ordered_raw_file_catalog_SHA256": raw_catalog.hexdigest(),
                     "manifest_SHA256": digest(root / "manifest.json")}


def paired(native, candidate):
    counts = Counter()
    excluded = []
    for index, (native_status, native_correct) in native.items():
        candidate_status, candidate_correct = candidate[index]
        if native_status == candidate_status == "complete":
            counts[native_correct, candidate_correct] += 1
        else:
            excluded.append({"index": index, "native_status": native_status, "SGLang_status": candidate_status})
    return {
        "valid_untruncated_pairs": sum(counts.values()),
        "both_correct": counts[True, True],
        "native_only_correct": counts[True, False],
        "SGLang_only_correct": counts[False, True],
        "neither_correct": counts[False, False],
        "excluded_pairs": excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for directory in ("assets", "native", "candidate"):
        parser.add_argument(f"--{directory}", type=Path, required=True)
    parser.add_argument("--native-results-sha256", required=True)
    parser.add_argument("--candidate-results-sha256", required=True)
    args = parser.parse_args()
    rows, _, extract, official = assets(args.assets)
    native, baseline_rollup = audit_run(args.native, args.native_results_sha256, rows, extract, official)
    candidate, candidate_rollup = audit_run(
        args.candidate, args.candidate_results_sha256, rows, extract, official,
    )
    if baseline_rollup["transport_failures"] or baseline_rollup["length_limited_responses"]:
        raise ValueError("Expected a fully valid native reference")
    print(json.dumps({
        "vLLM": baseline_rollup, "SGLang": candidate_rollup,
        "paired_official_grades": paired(native, candidate),
        "all_frozen_request_file_bytes_verified_for_both_runs": REQUESTS,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
