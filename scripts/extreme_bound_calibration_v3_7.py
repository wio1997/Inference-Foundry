#!/usr/bin/env python3
"""Promote source-checked current HCCL tensors and sparse original-path timing only."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(relative):
    return json.loads((ROOT / relative).read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (ROOT / "evidence/20260927_loop077_bound/run393/contamination.json").exists():
        raise SystemExit("Run390 is contaminated by stale Run386 client; V3.7 promotion forbidden")
    model = load("evidence/20260927_loop077_bound/run385/bound_calibration_v3_6.json")
    hccl = load("evidence/20260927_loop077_bound/run378/ledger.json")
    phases = load("evidence/20260927_loop077_bound/run390/analysis.json")
    assert (ROOT / "evidence/20260927_loop077_bound/run391/findings.md").is_file()
    assert phases["status"] == "original_path_sparse_device_event_and_acceptance_ledger"
    assert phases["source_integrity"]["source_before_after_sha_match"]
    assert phases["summary"]["rank_cycles"] == 32
    assert phases["summary"]["cohorts"] == 2 and phases["summary"]["ranks"] == 8
    assert all(x["exact_1024"] for x in phases["client_gates"].values())
    assert len(phases["paired_cycle64_65"]) == 16
    assert hccl["summary"]["tasks_per_rank_cycle"] == 265
    assert hccl["summary"]["conditional_API_input_bytes_sum"] == 115107840
    output_bytes = sum(x["conditional_API_output_bytes"] for x in hccl["ordered_tasks"])
    assert output_bytes == 145342464
    alg = model["bound_ladder"]["algorithm_resource"]
    alg["run389_selected_useful_output"] = {
        "cohort_cycle64_65": [[45,50],[39,37]],
        "source": "Run389/390 original eligible two cohorts, current-stream sparse event capture and Runtime count history",
        "scope": "Current Runtime-clipped output; separate trajectory from Run375/380 route, no same-cycle GMM join, not external token attribution or whole-contract acceptance limit",
    }
    hardware = model["bound_ladder"]["hardware_resource"]
    hardware["target_hccl_current_schedule_API_tensor_bytes_per_rank_cycle"] = {
        "input": hccl["summary"]["conditional_API_input_bytes_sum"],
        "output": output_bytes,
        "collectives": hccl["summary"]["tasks_per_rank_cycle"],
        "source": "Run378 ordered trace; Run391 independent Astra audit of installed CANN9.1 hccl.h count ABI and vLLM-Ascend actual DSA CP/MoE/tail source callsites",
        "confidence": "high for source/ABI-consistent current-schedule tensor shapes; native binary per-call correlation not independently observed",
        "scope": "Current API buffer inventory, not compulsory mathematical information flow, topology wire traffic, physical link bytes or latency floor",
    }
    hardware["target_hccl_conditional_API_input_bytes_per_rank_cycle"] = None
    hardware["missing"].append("alternative TP8 layout/consumer information-flow lower bound and real topology wire-byte measurement")
    scheduling = model["bound_ladder"]["scheduling_execution"]
    summary = phases["summary"]
    scheduling["run389_original_path_current_stream_diagnostic"] = {
        "eligible_cohorts": summary["cohorts"],
        "selected_rank_cycles": summary["rank_cycles"],
        "selected_cycles_per_cohort": summary["selected_cycles"],
        "same_rank_paired_interval_ms": summary["paired_same_rank_interval_summary_ms"],
        "same_rank_stage_ms": summary["runtime_stage_ms_across_rank_cycles"],
        "host_anchor_sync_bracket_ms_range": summary["anchor_sync_bracket_ms_range"],
        "source": "Run389 original fixed Runtime FULL Graph and Run390 device-event/acceptance validator; no Level1 profiler, selected cycle64/65 only",
        "scope": "Current-stream observed intervals under sparse marker overhead; two eligible of five possible 12-slot cohorts, not a full 60-client ledger or formal Run99 path. Internal graph and side-stream joins unproven; rank clocks not aligned; not a removable saving or attainable Scheduling floor.",
    }
    scheduling["missing"].append("same-run side-stream producer/consumer completion and calibrated all8 cross-rank arrival/join; sparse marker A/A overhead")
    product = model["bound_ladder"]["product_e2e"]
    assert product["finite_tps_upper_bound"] is None
    assert all(model["bound_ladder"][key]["latency_floor_s"] is None for key in ("algorithm_resource","hardware_resource","scheduling_execution"))
    model["next_measurement"]["priority"] = "close complete original-path side-stream/all8 dependency and compulsory resource work before finite ceiling promotion"
    model["next_measurement"]["specific_gate"] = "Run389 current-stream phase envelope plus explicit Target/DSpark side-stream, host-count-copy, actual all8 collective producer/consumer join and A/A marker control; source-pinned current HCCL tensor sizes now known, physical wire and alternative-layout compulsory bytes still open"
    model["input_paths"].extend(["evidence/20260927_loop077_bound/run389/findings.md", "evidence/20260927_loop077_bound/run390/analysis.json", "evidence/20260927_loop077_bound/run391/findings.md"])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"status":"ok", "cohorts":summary["cohorts"], "rank_cycles":summary["rank_cycles"], "current_formal_tps":model["current"]["accepted_formal_tps"], "finite_product_upper":product["finite_tps_upper_bound"]}))

if __name__ == "__main__":
    main()
