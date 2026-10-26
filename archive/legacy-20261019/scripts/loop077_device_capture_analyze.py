#!/usr/bin/env python3
"""Join cohort-end route/acceptance capture with exact client and Runtime ledgers."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop077_bound/run375"
FLOPS_PER_ROUTED = 2 * (4096 * 4096 + 2048 * 4096)
BYTES_PER_EXPERT = 4096 * 512 * 4 + 2048 * 512 * 4


def source_check():
    before = (BASE / "source_before.sha256").read_text().splitlines()
    after = (BASE / "source_after.sha256").read_text().splitlines()
    assert before == after
    install = json.loads((BASE / "install.json").read_text())
    restore = json.loads((BASE / "restore.json").read_text())
    assert install["action"] == "install" and restore["action"] == "restore"
    return {"source_before_after_sha_match": True, "files": len(before)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clients = {}
    for file, n in (("warmup48.json", 48), ("bench.json", 12)):
        obj = json.loads((BASE / file).read_text())
        assert len(obj["requests"]) == n
        assert all(r["output_tokens"] == 1024 and r["error"] is None for r in obj["requests"])
        clients[file] = {"requests": n, "exact_1024": True,
                         "summary": obj["summary"]}
    rows = []
    for cohort in range(1, 6):
        rank_rows = []
        for rank in range(8):
            file = BASE / "capture" / f"rank{rank}_cohort{cohort}.json"
            assert file.exists(), file
            obj = json.loads(file.read_text())
            runtime = json.loads((BASE / "runtime" / f"rank{rank}_cohort{cohort}.json").read_text())
            assert obj["rank"] == runtime["rank"] == rank
            assert obj["cohort"] == runtime["cohort"] == cohort
            assert runtime["pass"] and obj["cycles"] == runtime["cycles"]
            counts = obj["accepted_counts_by_cycle_slot"]
            assert len(counts) == obj["cycles"] and all(len(x) == 12 and all(0 <= v <= 8 for v in x) for x in counts)
            needed = obj["remaining"]
            assert len(needed) == 12
            generated = [0] * 12
            useful = []
            for row in counts:
                clipped = [min(value, max(0, needed[i] - generated[i])) for i, value in enumerate(row)]
                useful.append(sum(clipped))
                generated = [generated[i] + clipped[i] for i in range(12)]
            assert generated == needed
            route = obj["route_rows"]
            assert [x["cycle"] for x in route] == [64, 65]
            for snap in route:
                assert len(snap["counts"]) == 43
                assert all(len(r) == 32 and all(isinstance(v, int) and v >= 0 for v in r) for r in snap["counts"])
                active = sum(sum(v > 0 for v in layer) for layer in snap["counts"])
                routed = sum(sum(layer) for layer in snap["counts"])
                rows.append({"cohort": cohort, "rank": rank, "cycle": snap["cycle"],
                             "raw_accepted_tokens": sum(counts[snap["cycle"]]),
                             "useful_clipped_tokens": useful[snap["cycle"]],
                             "routed_tokens": routed, "active_expert_layer_pairs": active,
                             "standard_GMM_Gflop": routed * FLOPS_PER_ROUTED / 1e9,
                             "active_packed_weight_GB": active * BYTES_PER_EXPERT / 1e9,
                             "route_sha256": hashlib.sha256(json.dumps(snap["counts"]).encode()).hexdigest()})
            rank_rows.append({"rank": rank, "counts": counts, "useful": useful,
                              "snapshots": route})
        reference = rank_rows[0]
        for row in rank_rows[1:]:
            assert row["counts"] == reference["counts"]
            assert row["useful"] == reference["useful"]
        for cycle in (64, 65):
            for layer in range(43):
                total = sum(rr["snapshots"][cycle - 64]["counts"][layer][expert]
                            for rr in rank_rows for expert in range(32))
                assert total == 576, (cohort, cycle, layer, total)
    assert len(rows) == 80
    cohort_cycle = []
    for cohort in range(1, 6):
        for cycle in (64, 65):
            selected = [x for x in rows if x["cohort"] == cohort and x["cycle"] == cycle]
            assert len(selected) == 8
            useful = selected[0]["useful_clipped_tokens"]
            total_work = sum(x["standard_GMM_Gflop"] for x in selected)
            cohort_cycle.append({"cohort": cohort, "cycle": cycle,
                                 "useful_clipped_tokens": useful,
                                 "TP8_target_GMM_standard_Gflop": total_work,
                                 "TP8_target_GMM_Gflop_per_useful_token": total_work / useful,
                                 "max_rank_active_packed_weight_GB": max(x["active_packed_weight_GB"] for x in selected)})
    assert max(x["TP8_target_GMM_standard_Gflop"] for x in cohort_cycle) - min(x["TP8_target_GMM_standard_Gflop"] for x in cohort_cycle) < 1e-6
    out = {"status": "unprofiled_device_route_and_acceptance_join_diagnostic",
           "run": "run375", "source_integrity": source_check(), "client_gates": clients,
           "rows": rows, "cohort_cycle": cohort_cycle,
           "summary": {"rank_cycles": len(rows), "cohorts": 5, "ranks": 8,
                       "captured_cycles": [64, 65],
                       "TP8_target_GMM_standard_Gflop_each_cycle": cohort_cycle[0]["TP8_target_GMM_standard_Gflop"],
                       "TP8_target_GMM_Gflop_per_useful_token_range": [min(x["TP8_target_GMM_Gflop_per_useful_token"] for x in cohort_cycle), max(x["TP8_target_GMM_Gflop_per_useful_token"] for x in cohort_cycle)],
                       "GMM_standard_Gflop_per_rank_cycle_range": [min(x["standard_GMM_Gflop"] for x in rows), max(x["standard_GMM_Gflop"] for x in rows)],
                       "active_packed_weight_GB_per_rank_cycle_range": [min(x["active_packed_weight_GB"] for x in rows), max(x["active_packed_weight_GB"] for x in rows)],
                       "useful_clipped_tokens_cycle64_65_by_cohort": [
                           [next(x["useful_clipped_tokens"] for x in rows if x["cohort"] == c and x["rank"] == 0 and x["cycle"] == cycle)
                            for cycle in (64, 65)] for c in range(1, 6)]},
           "limits": ["Route stack adds one selected-cycle device operation, although no hot-path D2H/synchronize; captured cycle timing needs control A/A and cannot be promoted as unperturbed bound.",
                      "Accepted counts are Runtime staged tokens clipped to remaining output; pre-handoff published p_i and client latency DAG remain separate.",
                      "Draft route and GMM work, physical HBM, true all-rank critical path, attainable concurrent capacity and Product ceiling remain open."]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
