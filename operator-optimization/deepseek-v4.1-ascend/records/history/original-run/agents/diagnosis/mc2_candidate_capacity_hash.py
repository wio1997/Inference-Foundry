#!/usr/bin/env python3
"""Recheck two saved, redacted MC2 attr_1 values, no device access."""
from pathlib import Path
import hashlib, json
R = Path(__file__).resolve().parents[2]
p = R / "agents/diagnosis/mc2-group-attr.json"
s = R / "sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp"
h = lambda b: hashlib.sha256(b).hexdigest()
x = json.loads(p.read_text())
t = s.read_text()
needle = "constexpr uint32_t ATTR_MAX_OUTPUT_SIZE_INDEX = 1;"
assert needle in t
candidate = h(b"65536")
samples = []
for item in x["samples"]:
    e = [a for a in item["entries"] if a["label"] == "attr_1"]
    assert len(e) == 1
    samples.append({"rank_ep": item["rank"].split("_ep", 1)[1].split("_", 1)[0], "attr1_sha256": e[0]["value_sha256"], "candidate_65536_hash_equal": e[0]["value_sha256"] == candidate, "value_length": e[0]["value_length"]})
assert len(samples) == 2
print(json.dumps({"saved_input_sha256":h(p.read_bytes()), "tiling_source_sha256":h(s.read_bytes()), "utf8_candidate_sha256":candidate, "samples":samples, "limitations":["same hash is evidence subject to hash collision assumptions", "attr_N to ABI index correspondence unproven", "historical binary identity and tiling bytes unbound", "two old selected tasks only; no inference to other layers or captures", "no expert counts or max-output capacity asserted"]},ensure_ascii=False,indent=2))
