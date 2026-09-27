#!/usr/bin/env python3
"""Scope-preserving Bound ledger update after Run437/440 and HCCL Test.

The generator deliberately cannot emit a finite endpoint. Numeric promotion
needs an independently checked work/capacity or necessary-path payload.
"""
import argparse
import gzip
import math
import json
import re
import statistics
from pathlib import Path

from extreme_bound_calibration_v3_15 import validate_dag, obligation

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run435/bound_calibration_v3_15.json"
VALIDATION = "evidence/20260927_loop079_identity/run437/validation.json"
REVIEW = "evidence/20260927_loop079_identity/run440/astra_run437_review.md"
HCCL = "evidence/20260927_loop079_identity/run444/findings.md"
HCCL_SOURCE = "evidence/20260927_loop079_identity/run444/installed_allgather_rootinfo_test.cc.gz"
HCCL_MANIFEST = "evidence/20260927_loop079_identity/run444/command_manifest.json"
HCCL_LOGS = [f"evidence/20260927_loop079_identity/run444/allgather_bfp16_24821760_rep{i}.log"
             for i in (1, 2, 3)]
DECISION = "evidence/20260927_loop079_identity/run438/astra_bound_gap_review.md"
DESIGN = "evidence/20260927_loop079_identity/run439/scheduling_slice_design.md"
NUMERIC_ENDPOINT_KEYS = {"numeric_tps", "numeric_s", "latency_floor_s",
                         "finite_tps_upper_bound", "maximum_capacity_bound",
                         "unresolved_gap_tps"}


def require_null_endpoints(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in NUMERIC_ENDPOINT_KEYS and value is not None:
                raise ValueError(f"uncertified numeric endpoint inherited at {path}")
            require_null_endpoints(value, path)
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            require_null_endpoints(value, f"{prefix}[{index}]")


def parse_hccl_samples():
    source_bytes = gzip.decompress((ROOT / HCCL_SOURCE).read_bytes())
    source = source_bytes.decode()
    if "data->count = (data->count + rank_size - 1) / rank_size;" not in source:
        raise ValueError("HCCL Test AllGather byte convention is unproved")
    manifest = json.loads((ROOT / HCCL_MANIFEST).read_text())
    if not (manifest["host"] == "S900K3-49"
            and manifest["container"] == "vllm-ascend26-dsv4f-w4a8"
            and manifest["tool_sha256"] == "b276a969e778ab2d627643225fc9a29b45878b016999079de9ae705a73997b9d"
            and manifest["mpi_ranks"] == manifest["npus_per_node"] == 8
            and manifest["dtype"] == "bfp16"
            and manifest["cli_minbytes"] == manifest["cli_maxbytes"] == 24821760
            and manifest["rank_input_bytes_from_source"] == 3102720
            and manifest["rank_output_bytes_from_source"] == 24821760
            and manifest["warmup_iters"] == 10 and manifest["timed_iters"] == 30
            and manifest["check"] == 1 and manifest["onlydevicetime"] == 0
            and manifest["independent_process_logs"] == [Path(p).name for p in HCCL_LOGS]):
        raise ValueError("HCCL Test command manifest mismatch")
    import hashlib
    if manifest["installed_source_sha256"] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("HCCL Test source hash mismatch")
    samples = []
    pattern = re.compile(r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*success\s*$", re.M)
    for name in HCCL_LOGS:
        log = (ROOT / name).read_text()
        rows = pattern.findall(log)
        if (len(rows) != 1 or "iters is 30, warmup_iters is 10" not in log
                or re.search(r"\b(fail|failed|error)\b", log, re.I)):
            raise ValueError(f"HCCL Test log is not one successful 30-iteration sample: {name}")
        size, micros, bw = rows[0]
        if int(size) != 24821760:
            raise ValueError(f"HCCL Test CLI aggregate output size mismatch: {name}")
        micros, bw = float(micros), float(bw)
        if not math.isfinite(micros) or micros <= 0.005 or not math.isfinite(bw) or bw <= 0:
            raise ValueError(f"HCCL Test metric is invalid: {name}")
        # The printed time is rounded to 0.01 us and bandwidth to 0.00001 GB/s.
        bw_low = 24821760 / (micros + 0.005) / 1000 - 0.000005
        bw_high = 24821760 / (micros - 0.005) / 1000 + 0.000005
        if not bw_low <= bw <= bw_high:
            raise ValueError(f"HCCL Test bandwidth/time mismatch: {name}")
        samples.append({"path": name, "tool_root_rank_average_us": micros,
                        "tool_algorithm_bandwidth_GB_s": bw})
    if 24821760 // 8 != 3102720:
        raise ValueError("HCCL rank input size mismatch")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((ROOT / PRIOR).read_text())
    v = json.loads((ROOT / VALIDATION).read_text())
    review = (ROOT / REVIEW).read_text()
    hccl = (ROOT / HCCL).read_text()
    decision = (ROOT / DECISION).read_text()
    design = (ROOT / DESIGN).read_text()
    if not (v["diagnostic_valid"] and not v["all43_row_identity_certificate"]
            and sorted(c["cohort"] for c in v["cohorts"]) == [1, 2, 3, 4, 5]):
        raise ValueError("Run437 scoped diagnostic gate failed")
    if not all(c["certificates"]["full_graph_input_binding_valid"]
               and c["certificates"]["cp_value_map_valid"]
               and c["certificates"]["all43_row_composition"] == "CONDITIONAL_NATIVE"
               for c in v["cohorts"]):
        raise ValueError("Run437 cohort certificate scope changed")
    for snippet, content in (("ACCEPT Run437", review),
                             ("24,821,760", hccl),
                             ("one positive", decision),
                             ("native HCCL completion", design)):
        if snippet not in content:
            raise ValueError(f"review/design anchor missing: {snippet}")
    hccl_samples = parse_hccl_samples()

    resource = model["certificate_graph"]["algorithm_resource"]
    row = resource["all43_current_graph_row_identity"]
    row["evidence"].extend([VALIDATION, REVIEW])
    row["missing"] = [
        "per-layer selected branch and actual layer-to-CP consumer binding",
        "FlashComm/DSA/embedding/residual/native row composition and graph all-slot semantics",
        "loaded native binary/tiling and GMM/scale row association",
    ]
    row["bound_effect"] = (
        "Run437 closes selected FULL explicit input-storage and CP value-prefix provenance, "
        "but all43 retained-route numerator remains conditional")
    model["certificate_graph"]["hardware_resource"]["isolated_terminal_allgather"] = {
        "status": "attained_isolated_observation", "evidence": [HCCL],
        "payload_per_rank_bytes": 3102720,
        "tool_cli_aggregate_output_bytes": 24821760,
        "tool_root_rank_average_us_three_processes": [x["tool_root_rank_average_us"] for x in hccl_samples],
        "tool_root_rank_median_us": statistics.median(x["tool_root_rank_average_us"] for x in hccl_samples),
        "scope": "CANN9.1 HCCL Test bfp16 8 ranks, 10 warmup/30 measured, correctness pass; root-rank ACL event loop span can include submission gaps; not mixed Graph or C_plus",
        "bound_effect": "no strict or Product endpoint promotion",
    }
    model["certificate_graph"]["scheduling_execution"]["terminal_logits_join"] = {
        "status": "design_only", "evidence": [DESIGN],
        "missing": ["installed native HCCL-to-consumer-stream completion contract",
                    "original-path P/J/G/C0/C1 correlated events",
                    "same-policy A0-B-A1 overhead controls"],
        "bound_effect": "no Current exposed interval or necessary-path duration floor yet",
    }

    nodes = model["proof_dag"]["nodes"]
    nodes["selected_full_graph_input_binding_sample"] = obligation(
        "observation", "sampled", evidence=[VALIDATION, REVIEW], missing=[
            "implicit contexts and all43 layer/native row composition remain open"])
    nodes["isolated_terminal_allgather_service_sample"] = obligation(
        "observation", "sampled", evidence=[HCCL], missing=[
            "matching mixed contention and original-path consumer join"])
    nodes["all43_current_graph_identity"]["evidence"].extend([VALIDATION, REVIEW])
    nodes["all43_current_graph_identity"]["missing"] = [
        "per-layer actual branch/layer-to-CP binding and native row composition",
        "exact graph all-slot semantics, GMM/scale association and loaded binary provenance",
    ]
    nodes["typed_necessary_path"]["missing"] = [
        "selected necessary dependency path and relevant cross-rank/storage edges; complete all8 graph only if claimed endpoint uses it",
    ]
    nodes["legal_resource_storage_schedule"]["missing"] = [
        "feasible resource/storage schedule with all-rank contention and overlap, independently of a necessary-path relaxation",
    ]
    resolved = validate_dag(nodes)
    if any(resolved.values()):
        raise ValueError("Run437/444 observations cannot certify a Bound proof node")
    model["proof_dag"]["certified"] = resolved
    model["proof_dag"]["scope_note"] = (
        "Run437/444 are sampled observations. The DAG verifies structure/status only; "
        "a future numeric certificate requires evidence truth, units, rank aggregation and uncertainty checks.")
    model["next_measurement"]["priority"] = (
        "For useful Scheduling interval, verify exact installed native logits HCCL join, then acquire one "
        "original-path terminal logits→AllGather→argmax slice with matched A0-B-A1 controls. "
        "For first loose strict outer ceiling, independently certify one fresh required BF16 projection "
        "in the Product window and an exact-board aggregate C_plus. Harvest Run437's first unresolved "
        "layer/native row transform only if retained-route numerator remains the selected Resource path.")
    model["next_measurement"]["specific_gate"] = (
        "First bind the installed torch_npu HCCL stream-join semantics to the actual terminal logits branch. "
        "Then collect the single selected original-path join slice with source restoration, exact60 protocol "
        "and matched A0-B-A1 controls; classify Current exposure without treating attained service as a floor.")
    model["input_paths"].extend([PRIOR, VALIDATION, REVIEW, HCCL, HCCL_SOURCE, HCCL_MANIFEST,
                                 *HCCL_LOGS, DECISION, DESIGN])
    require_null_endpoints(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "proof_nodes": len(nodes),
                      "finite_endpoints": 0, "run437_scoped": True,
                      "hccl_attained_isolated": True}))


if __name__ == "__main__":
    main()
