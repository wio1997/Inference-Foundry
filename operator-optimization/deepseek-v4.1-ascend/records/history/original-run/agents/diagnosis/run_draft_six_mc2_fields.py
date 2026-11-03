#!/usr/bin/env python3
"""Dispatch fixed CPU-only query to the dedicated existing container."""
from pathlib import Path
import csv, hashlib, json, shlex, subprocess
T = Path(__file__).resolve().parents[2]
diag = T / "agents/diagnosis"
join_path = diag / "draft-six-alltoall-mc2-local-join.json"
decomp_path = diag / "draft-target-operator-decomposition.json"
strict_path = T / "draft-mc2-analysis/strict_draft_calls.csv"
query_path = diag / "query_draft_six_mc2_fields.py"
out = diag / "draft-six-mc2-exported-fields.json"
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
expected = {
    join_path: "1dee9a2dcebaa852c83fdfb8ec03f1d328c54cea4e59c57e21cdc42cc01c6e63",
    decomp_path: "7384031bcfa37219d347f535ae5cf8cf7a099d4f89565867af7288de9d795197",
    strict_path: "8c8b6539870fce5fd21a089c1eb3556fd70e2d458f37f87fb16ea24342b1c75f",
}
if any(sha(p) != v for p, v in expected.items()) or out.exists():
    raise RuntimeError("source changed or output exists")
join = json.loads(join_path.read_text())
decomp = json.loads(decomp_path.read_text())
with strict_path.open(newline="") as f:
    strict = list(csv.DictReader(f))
ranks = {}
for x in join["samples"]:
    hits = [r for r in strict if r["rank"] == x["rank"]
            and int(r["mc2_start_ns"]) == x["mc2_start_ns"]
            and int(r["mc2_end_ns"]) == x["mc2_end_ns"]]
    if len(hits) != 1:
        raise RuntimeError("non-unique strict summary")
    item = dict(x, expected_aic_num=hits[0]["mc2_aic_num"],
                expected_aic_den=hits[0]["mc2_aic_den"])
    ranks.setdefault(x["rank"], []).append(item)
if sorted(map(len, ranks.values())) != [3, 3]:
    raise RuntimeError("unexpected selection")
payload = {"ranks": ranks, "csv_sha256_by_rank": {
    rank: decomp["csv_sha256_by_rank"][rank] for rank in ranks}}
code = query_path.read_text().replace("FROZEN_INPUT", repr(json.dumps(payload, separators=(",", ":"))))
ssh = "/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh"
container = "wio-vllm-v41-repro169-stage-20261028"
trace = "/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces"
r = subprocess.run([ssh, shlex.join(["docker", "exec", container, "python3", "-c", code,
                                     "--trace-root", trace])],
                   capture_output=True, text=True, timeout=180)
if r.returncode:
    raise RuntimeError("query failed: " + r.stderr[-2000:])
result = json.loads(r.stdout)
result["input_sha256"] = {str(p.relative_to(T)): h for p, h in expected.items()}
result["query_sha256"] = sha(query_path)
result["wrapper_sha256"] = sha(Path(__file__))
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "QUERY_COMPLETED", "output": str(out), "sha256": sha(out)}))
