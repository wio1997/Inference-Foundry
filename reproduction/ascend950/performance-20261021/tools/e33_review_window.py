#!/usr/bin/env python3
# E33 diagnostic derivative; original E27 utility remains unchanged.
"""Review native artifacts; no new HTTP/model requests are issued."""
import argparse
import collections
import csv
import io
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def metric(text, name):
    values = {}
    for line in text.splitlines():
        if line.startswith("vllm:" + name + "{"):
            match = re.search(r'engine="(\d+)"', line)
            if match:
                k = int(match.group(1))
                values[k] = values.get(k, 0) + float(line.rsplit(" ", 1)[1])
    return values


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--framework", choices=["di", "vllm"], required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--output", type=int, required=True)
    ap.add_argument("--cached", type=int, required=True)
    ap.add_argument("--root", help="Read a locally archived native artifact tree")
    ap.add_argument("--result", help="Write an independent review without replacing archived evidence")
    a = ap.parse_args()
    if a.root:
        ROOT = pathlib.Path(a.root).resolve()
    directory = ROOT / "evidence-e33" / "windows" / a.run_id
    paths = list((ROOT / "runs/e33-ais" / a.run_id).rglob("*_details.jsonl"))
    if len(paths) != 1:
        raise ValueError("Require exactly one native AIS details artifact")
    p = paths[0]
    records = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    frozen = {str(x["request_index"]): x["input_token_ids"] for x in
              (json.loads(l) for l in (ROOT / a.corpus).read_text().splitlines() if l.strip())}
    lengths = {len(ids) for ids in frozen.values()}
    if len(lengths) != 1 or len(frozen) != a.requests or a.requests % 8:
        raise ValueError("Invalid frozen cohort")
    L = lengths.pop()
    extra = [r["extra_details_data"] for r in records]
    expected_owners = {i: a.requests // 8 for i in range(8)}
    checks = dict(request_count=len(records) == a.requests,
                  exact_frozen_inputs=all(e["input_ids"] == frozen[str(r["input"])] for r, e in zip(records, extra)),
                  success=all(r["success"] for r in records),
                  length_usage=all(r["input_tokens"] == L and r["output_tokens"] == a.output for r in records),
                  actual_output_ids=all(len(e.get("native_output_token_ids", [])) == a.output for e in extra),
                  first_counted_SSE_has_ids=all(any(c.get("token_ids") for c in e.get("first_sse_payload", {}).get("choices", [])) for e in extra),
                  finish_reason=all(e.get("finish_reason") == "length" for e in extra),
                  owner_distribution=dict(collections.Counter(e["owner_rank"] for e in extra)) == expected_owners,
                  unique_native_ids=len({e.get("first_sse_payload", {}).get("id") for e in extra}) == a.requests)
    checks["server_usage_matches"] = all(e.get("server_usage", {}).get("prompt_tokens") == L and
                                          e.get("server_usage", {}).get("completion_tokens") == a.output for e in extra)
    pre = json.loads((directory / "preflight.json").read_text())
    bootstrap = pre.get("bootstrap_warm_only") is True and pre.get("formal_window") is False
    deltas = {}
    if a.framework == "vllm":
        checks["native_response_id_matches"] = all(
            e.get("first_sse_payload", {}).get("id") in
            [e["request_id"], "cmpl-" + e["request_id"]] for e in extra)
        texts = {label: (directory / (label + ".metrics")).read_text() for label in ["before", "after"]}
        for label, text in texts.items():
            if bootstrap and label == "before":
                continue
            for name in ["num_requests_running", "num_requests_waiting"]:
                checks[label + "_" + name + "_8idle"] = metric(text, name) == {i: 0 for i in range(8)}
        if not bootstrap:
            for name, expected in [("prefix_cache_queries_total", L), ("prefix_cache_hits_total", a.cached)]:
                before, after = [metric(texts[label], name) for label in ["before", "after"]]
                deltas[name] = {i: after[i] - before[i] for i in after if i in before}
                checks[name + "_exact"] = deltas[name] == {i: expected * a.requests // 8 for i in range(8)}
    else:
        for label in ["before", "after"]:
            d = json.loads((directory / (label + ".status")).read_text())
            checks[label + "_8idle"] = d["ranks_ready"] == 8 and d["server_idle"] is True and not d.get("fatal") and all(d[k] == 0 for k in ["inflight", "running", "waiting"])
            checks[label + "_8192_16"] = len(d["ranks"]) == 8 and all(r["resolved"]["max_num_batched_tokens"] == 8192 and r["resolved"]["max_num_seqs"] == 16 for r in d["ranks"])
        native = json.loads((directory / "after.diag_requests").read_text())["requests"]
        byid = {x["request_id"]: x for x in native}
        checks["native_prompt_work"] = all(e["request_id"] in byid and sum(s["num_batched_tokens"] for s in byid[e["request_id"]]["prompt_steps"]) == L - a.cached for e in extra)
        checks["cached_K_each"] = all(e.get("server_usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") == a.cached for e in extra)
        checks["actual_owner_each"] = all(e.get("first_sse_payload", {}).get("assigned_dp_rank") == e["owner_rank"] for e in extra)
        if a.run_id.endswith("HOT-R2"):
            jit = json.loads((directory / "native-jit.json").read_text())
            checks["no_new_DG_JIT_in_fixed_hot_R2"] = jit["changed_metadata_count"] == 0
    abbr = p.name.removesuffix("_details.jsonl")
    common = json.loads((p.parent / (abbr + ".json")).read_text())
    latency = list(csv.DictReader(io.StringIO((p.parent / (abbr + ".csv")).read_text())))
    actual_peak = common["Max Concurrency"]["total"]
    result = dict(run_id=a.run_id, framework=a.framework, input_tokens=L, output_tokens=a.output,
                  requests=a.requests, target_concurrency=a.concurrency, observed_peak_concurrency=actual_peak,
                  target_peak_reached=actual_peak >= a.concurrency, passed=all(checks.values()), checks=checks,
                  common_metrics=common, latency_metrics=latency, native_counter_deltas=deltas,
                  bootstrap_warm=bootstrap, raw_details=str(p),
                  capacity_claim="Use raw server samples to establish maintained load; peak alone is not sustained capacity")
    target = pathlib.Path(a.result) if a.result else directory / "review.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["run_id", "passed", "target_concurrency", "observed_peak_concurrency", "target_peak_reached"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
