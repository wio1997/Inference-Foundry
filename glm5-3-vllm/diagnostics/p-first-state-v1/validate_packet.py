#!/usr/bin/env python3
"""Mac-only plan integrity checks. Never launches or accepts device state evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def token_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate(packet, repo):
    def need(condition, message):
        if not condition:
            raise ValueError(message)
    need(packet["execution_ready"] is False and packet["device_result"] is None,
         "packet cannot claim execution readiness or a device result")
    need(packet["nodes"] == {"32": 16, "33": 16}, "wrong node/die scope")
    need([c["input_tokens"] for c in packet["cases"]] == [256, 257], "missing or extra cases")
    for case in packet["cases"]:
        n, ids = case["input_tokens"], case["prompt_token_ids"]
        need(len(ids) == n and all(type(i) is int and i >= 0 for i in ids), "empty or invalid prompt")
        need(token_sha(ids) == case["prompt_sha256"], "prompt identity mismatch")
        positions = [n - 2, n - 1]
        need(case["sample_positions"] == positions, "misaligned/missing sample position")
    need(packet["cases"][1]["prompt_token_ids"][:256] == packet["cases"][0]["prompt_token_ids"],
         "boundary pair prefix mismatch")
    for role, node, k in (("P", 32, 1), ("D", 33, 5)):
        ref = packet["base_configs"][role]
        path = repo / ref["path"]
        need(hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"], "base config SHA mismatch")
        plan = json.loads(path.read_text())
        need(plan["image"]["id"] == packet["image_id"], "image identity mismatch")
        need(plan["devices"]["davinci"] == list(range(16)), "wrong devices")
        need(not plan.get("runtime_patches"), "requires a native base without historical overlays")
        argv = plan["serve_argv"]
        need("--async-scheduling" in argv, "async event boundary is required")
        spec = json.loads(argv[argv.index("--speculative-config") + 1])
        need(spec["method"] == "deepseek_mtp" and spec["num_speculative_tokens"] == k, "wrong MTP configuration")
        need(spec["enforce_eager"] is True and not spec.get("parallel_drafting", False), "wrong drafter mode")
        kv = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        need(kv["kv_connector"] == "MooncakeConnectorV2" and kv["kv_role"] == ("kv_producer" if role == "P" else "kv_consumer"), "wrong KV role")
        need(kv["kv_connector_extra_config"]["use_ascend_direct"] is True, "direct path required")
        need(plan["env"]["HCCL_INTRA_ROCE_ENABLE"] == "0" and plan["env"]["HCCL_INTER_HCCS_DISABLE"] == "FALSE", "SDMA path controls changed")
        need(plan["reservation"]["node_name"].endswith("-" + str(node)), "wrong base node")
    need(packet["sampling"] == {"temperature": 0, "top_p": 1, "seed": 1024,
         "ignore_eos": True, "min_tokens_P": 1, "min_tokens_D": 1,
         "max_tokens_P": 1, "max_tokens_D": 2}, "sampling/budget mismatch")
    return {"packet_integrity": True, "execution_ready": False, "device_run": False,
            "scope": "Archived config and intended input/position consistency only; no KV validity or sampling equivalence acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(__file__).with_name("review-plan.json"))
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.packet.read_text()), Path(__file__).resolve().parents[3]), indent=2))


if __name__ == "__main__":
    main()
