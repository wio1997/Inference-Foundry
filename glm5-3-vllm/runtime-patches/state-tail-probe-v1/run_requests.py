#!/usr/bin/env python3
"""Two frozen cold native PD requests. Run only in a separately authorized window."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import uuid

MODEL = "glm5"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute-functional", action="store_true", help="explicit execution gate; does not grant a device window")
    p.add_argument("--p-url", required=True)
    p.add_argument("--d-url", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--p-probe-dir", type=Path, required=True,
                   help="fresh P run_dir/state-tail-probe, visible to the node32 host client")
    a = p.parse_args()
    if not a.execute_functional:
        p.error("functional HTTP execution is disabled; a new authorized model window is required")
    for role, url, host in (("P", a.p_url, "172.16.12.32"), ("D", a.d_url, "172.16.12.33")):
        parsed = urllib.parse.urlsplit(url)
        assert parsed.scheme == "http" and parsed.hostname == host and parsed.port and 1024 <= parsed.port <= 65535
        assert not (parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip("/"))
    assert not a.output.exists(), "fresh output directory required"
    assert str(a.p_probe_dir).startswith("/data/wio/glm5-3-vllm-execution/20261018-state-tail/")
    assert ".." not in a.p_probe_dir.parts
    a.output.mkdir(parents=True)
    report = {"scope": "Two C1 cold native PD functionality/trace requests; no query adoption, quality suite or throughput measurement",
              "completed": False, "rows": []}

    def post(url, payload, req_id):
        request = urllib.request.Request(url + "/v1/completions", json.dumps(payload).encode(),
            {"Content-Type": "application/json", "X-Request-Id": req_id})
        return urllib.request.urlopen(request, timeout=90)

    try:
        for case in CASES:
            ids = case["prompt_token_ids"]
            assert len(ids) == case["input_tokens"]
            assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == case["prompt_sha256"]
            req_id = "state-tail-" + uuid.uuid4().hex
            payload = {"model": MODEL, "prompt": ids, "max_tokens": 2, "min_tokens": 1,
                       "temperature": 0, "top_p": 1, "seed": 1024, "ignore_eos": True,
                       "return_token_ids": True, "cache_salt": req_id, "stream": True,
                       "stream_options": {"include_usage": True}}
            prefill = {k: v for k, v in payload.items() if k != "stream_options"}
            prefill.update(max_tokens=1, stream=False,
                kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False,
                    "remote_engine_id": None, "remote_block_ids": None,
                    "remote_host": None, "remote_port": None})
            row = {"input_tokens": case["input_tokens"], "prompt_sha256": case["prompt_sha256"],
                   "request_id": req_id, "P_request": prefill, "D_request": None,
                   "P_send_ns": time.monotonic_ns()}
            report["rows"].append(row)
            with post(a.p_url, prefill, req_id) as response:
                row["P_response"] = json.load(response)
            row["P_received_ns"] = time.monotonic_ns()
            producer = row["P_response"]
            actual = producer["choices"][0]["prompt_token_ids"]
            assert actual == ids and producer["usage"]["completion_tokens"] == 1
            params = producer["kv_transfer_params"]
            assert params["remote_num_prompt_tokens"] == len(ids) and params["do_remote_prefill"] is True
            assert producer["choices"][0]["token_ids"] == [params["last_token_id"]]
            records = [json.loads(line) for path in a.p_probe_dir.glob("*.jsonl")
                       for line in path.read_text().splitlines() if line.strip()]
            tails = [r for r in records if r.get("kind") == "P_tail" and r.get("req_id") == params["remote_request_id"]]
            row["P_tail_precheck"] = [{k: r.get(k) for k in ("tp_rank", "status", "mtp_calls",
                "hidden_tail_matches", "event_ready_after_native_wait", "runnable_hook_bound")} for r in tails]
            assert len(tails) == 16 and {r["tp_rank"] for r in tails} == set(range(16)), "P per-rank trace incomplete"
            assert all(r.get("status") == "observed" and r.get("mtp_calls") == 1 and
                       r.get("hidden_tail_matches") is True and r.get("event_ready_after_native_wait") is True
                       and r.get("runnable_hook_bound") is True
                       for r in tails), "P MTP call/provenance not established; stop before D"
            payload["kv_transfer_params"] = params
            row.update(D_request=payload, D_send_ns=time.monotonic_ns(), D_chunks=[], D_token_ids=[])
            usage = None
            with post(a.d_url, payload, req_id) as response:
                assert response.status == 200
                for line in response:
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    chunk = json.loads(line[6:])
                    assert not chunk.get("error"), chunk
                    row["D_chunks"].append(chunk)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        row["D_token_ids"].extend(choice.get("token_ids") or [])
            row["D_received_ns"] = time.monotonic_ns()
            assert len(row["D_token_ids"]) == 2 and row["D_token_ids"][0] == params["last_token_id"]
            assert usage and usage["completion_tokens"] == 2
            row.update(functional_pass=True, D_usage=usage)
            (a.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        report["completed"] = True
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    (a.output / "requests.json").write_text(json.dumps(report, indent=2) + "\n")
    print("GLM53_RESULT_JSON_BEGIN\n" + json.dumps(report, indent=2) + "\nGLM53_RESULT_JSON_END")
    return 0 if report["completed"] else 2


CASES = [{'input_tokens': 256, 'prompt_sha256': 'e55a62b8e0046a49b7750c8bcc1925aa5e6cd2effcc61c949561dfaff0c7593b', 'prompt_token_ids': [8855, 21465, 19, 64, 102573, 65, 100899, 6066, 19, 66, 118285, 127020, 7628, 102807, 22, 100441, 16, 98503, 6066, 18699, 12, 116265, 13, 4457, 1493, 10514, 8387, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 198, 14572, 25, 3555, 374, 220, 17, 488, 220, 18, 5267, 16127, 25, 220]}, {'input_tokens': 257, 'prompt_sha256': '01e6c332da1bcc7fec4e7f2e3da0131baada26bff210c3904dc848cd8ad4da83', 'prompt_token_ids': [8855, 21465, 19, 64, 102573, 65, 100899, 6066, 19, 66, 118285, 127020, 7628, 102807, 22, 100441, 16, 98503, 6066, 18699, 12, 116265, 13, 4457, 1493, 10514, 8387, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 198, 14572, 25, 3555, 374, 220, 17, 488, 220, 18, 5267, 16127, 25, 220, 220]}]

if __name__ == "__main__":
    raise SystemExit(main())
