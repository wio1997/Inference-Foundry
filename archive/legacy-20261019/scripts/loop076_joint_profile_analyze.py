#!/usr/bin/env python3
"""Join Run367 measured-cohort routes to its own FULL Graph counter and DAG trace."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop076_bound/run367"
OLD = ROOT / "evidence/20260925_loop039_gmm/run121/counts"
GMM2 = "aclnnGroupedMatmulWeightNz_"
GMM1 = "aclnnGroupedMatmulSwigluQuantWeightNzV2_"
WEIGHT_BYTES_PER_EXPERT_GMM2 = 2048 * 512 * 4
WEIGHT_BYTES_PER_EXPERT_GMM1 = 4096 * 512 * 4
GMM_STANDARD_FLOPS_PER_ROUTED_TOKEN = 2 * (4096 * 4096 + 2048 * 4096)


def number(row, key):
    try:
        return float(row[key].strip())
    except (KeyError, ValueError, AttributeError):
        return 0.0


def source_check():
    before = (BASE / "source_before.sha256").read_text().splitlines()
    after = (BASE / "source_after.sha256").read_text().splitlines()
    assert before == after, "borrowed source did not restore"
    install = json.loads((BASE / "install.json").read_text())
    restore = json.loads((BASE / "restore.json").read_text())
    assert install["action"] == "install" and restore["action"] == "restore"
    return {"source_before_after_sha_match": True, "install": install, "restore": restore}


def pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denominator = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    assert denominator > 0
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for rank in range(8):
        folders = sorted((BASE / "profile").glob(f"rank{rank}_*/ASCEND_PROFILER_OUTPUT"))
        assert len(folders) == 5, (rank, len(folders))
        folder = folders[-1]
        events = json.loads((folder / "trace_view.json").read_text())
        kernels = list(csv.DictReader((folder / "kernel_details.csv").open(newline="")))
        scopes = sorted((e for e in events if e.get("cat") == "cpu_op" and e.get("name") == "extreme::target"), key=lambda e: float(e["ts"]))
        assert len(scopes) == 2, (rank, len(scopes))
        window = json.loads((BASE / "profile" / f"rank{rank}_window.json").read_text())
        assert window["rank"] == rank and window["first_cycle"] == 64 and window["cycle_count"] == 2
        assert window["started_ns"] <= int(float(scopes[0]["ts"]) * 1000)
        assert int((float(scopes[-1]["ts"]) + float(scopes[-1]["dur"])) * 1000) <= window["stopped_ns"]
        runtime_path = BASE / "runtime" / f"rank{rank}_cohort5.json"
        runtime = json.loads(runtime_path.read_text())
        assert runtime["rank"] == rank and runtime["cohort"] == 5 and runtime["pass"]
        dag_path = BASE / "dag" / f"rank{rank}.json"
        dag = json.loads((BASE / "dag" / f"rank{rank}.json").read_text())
        assert dag["rank"] == rank and dag["cycles"] > 65
        snap_paths = [BASE / "counts" / f"rank{rank}_cycle{cycle}.json" for cycle in (64, 65)]
        capture_times = [path.stat().st_mtime_ns for path in snap_paths]
        assert window["started_ns"] <= capture_times[0] <= capture_times[1] <= window["stopped_ns"]
        assert window["stopped_ns"] <= dag_path.stat().st_mtime_ns <= runtime_path.stat().st_mtime_ns
        for idx, cycle in enumerate((64, 65)):
            snap_path = snap_paths[idx]
            snap = json.loads(snap_path.read_text())
            assert snap["rank"] == rank and snap["cycle"] == cycle and snap["ref_count"] == 86
            assert all(snap["rows"][i]["ordinal"] == i for i in range(86))
            count = snap["rows"][64]["counts"]
            # Run121 established ordinals 0..42 are static captured refs and
            # 43..85 are live refs. Live 64 maps to layer index 21.
            static_count = snap["rows"][21]["counts"]
            begin = float(scopes[idx]["ts"])
            end = begin + float(scopes[idx]["dur"])
            selected = [r for r in kernels if begin <= number(r, "Start Time(us)") < end]
            g1 = [r for r in selected if r["Name"].startswith(GMM1)]
            g2 = [r for r in selected if r["Name"].startswith(GMM2)]
            assert len(g1) == len(g2) == 43, (rank, cycle, len(g1), len(g2))
            live = []
            for layer in range(43):
                routed = snap["rows"][43 + layer]["counts"]
                assert len(routed) == 32
                n = sum(routed)
                active_layer = sum(v > 0 for v in routed)
                live.append({
                    "layer": layer,
                    "route_tokens": n,
                    "active_experts": active_layer,
                    "standard_GMM_Gflop": n * GMM_STANDARD_FLOPS_PER_ROUTED_TOKEN / 1e9,
                    "active_packed_weight_bytes": active_layer * (WEIGHT_BYTES_PER_EXPERT_GMM1 + WEIGHT_BYTES_PER_EXPERT_GMM2),
                    "gmm1_native_AIC_read_bytes": number(g1[layer], "aic_read_main_memory_datas(KB)") * 1024,
                    "gmm2_native_AIC_read_bytes": number(g2[layer], "aic_read_main_memory_datas(KB)") * 1024,
                    "gmm1_native_AIV_read_bytes": number(g1[layer], "aiv_read_main_memory_datas(KB)") * 1024,
                    "gmm2_native_AIV_read_bytes": number(g2[layer], "aiv_read_main_memory_datas(KB)") * 1024,
                    "gmm1_native_profiled_duration_us": number(g1[layer], "Duration(us)"),
                    "gmm2_native_profiled_duration_us": number(g2[layer], "Duration(us)"),
                })
            kernel = g2[21]
            old = json.loads((OLD / f"rank{rank}_cycle{cycle}.json").read_text())
            old_count = old["rows"][64]["counts"]
            active = sum(value > 0 for value in count)
            weight_bytes = active * WEIGHT_BYTES_PER_EXPERT_GMM2
            assert weight_bytes > 0
            rows.append({
                "rank": rank, "cycle": cycle,
                "route_tokens": sum(count), "active_experts": active,
                "static_ref21_route_tokens_diagnostic": sum(static_count),
                "route_count_sha256": hashlib.sha256(json.dumps(count).encode()).hexdigest(),
                "run121_route_exact_same": count == old_count,
                "run121_route_tokens": sum(old_count),
                "active_packed_GMM2_weight_bytes": weight_bytes,
                "gmm2_ordinal21_AIC_read_bytes": number(kernel, "aic_read_main_memory_datas(KB)") * 1024,
                "gmm2_ordinal21_AIV_read_bytes": number(kernel, "aiv_read_main_memory_datas(KB)") * 1024,
                "gmm2_ordinal21_GM_to_L1_bytes": number(kernel, "aic_GM_to_L1_datas(KB)") * 1024,
                "gmm2_ordinal21_profiled_duration_us": number(kernel, "Duration(us)"),
                "gmm2_ordinal21_input_shapes": kernel.get("Input Shapes"),
                "gmm2_ordinal21_input_types": kernel.get("Input Data Types"),
                "gmm2_ordinal21_input_formats": kernel.get("Input Formats"),
                "all43_live_layer_GMM": live,
                "all43_standard_GMM_Gflop": sum(layer["standard_GMM_Gflop"] for layer in live),
                "all43_active_packed_weight_GB": sum(layer["active_packed_weight_bytes"] for layer in live) / 1e9,
                "all43_profiled_AIC_GMM_read_GB": sum(layer["gmm1_native_AIC_read_bytes"] + layer["gmm2_native_AIC_read_bytes"] for layer in live) / 1e9,
                "all43_profiled_AIC_AIV_GMM_read_GB": sum(layer["gmm1_native_AIC_read_bytes"] + layer["gmm2_native_AIC_read_bytes"] + layer["gmm1_native_AIV_read_bytes"] + layer["gmm2_native_AIV_read_bytes"] for layer in live) / 1e9,
                "instrumented_dag_stage_ms": dag["runtime_stage_ms"][cycle],
                "instrumented_dspark_stage_ms": dag["dspark_stage_ms"][cycle],
                "source_profile_folder": str(folder.relative_to(ROOT)),
                "source_profile_window_start_stop_ns": [window["started_ns"], window["stopped_ns"]],
                "source_count_mtime_ns": capture_times[idx],
                "source_dag_mtime_ns": dag_path.stat().st_mtime_ns,
                "source_runtime_cohort5_mtime_ns": runtime_path.stat().st_mtime_ns,
                "source_count": str(snap_path.relative_to(ROOT)),
            })
    ratios = [row["gmm2_ordinal21_AIC_read_bytes"] / row["active_packed_GMM2_weight_bytes"] for row in rows]
    full_gmm_ratios = [row["all43_profiled_AIC_AIV_GMM_read_GB"] / row["all43_active_packed_weight_GB"] for row in rows]
    cross_rank = {}
    for cycle in (64, 65):
        cycle_rows = [row for row in rows if row["cycle"] == cycle]
        assert len(cycle_rows) == 8
        tokens = [sum(row["all43_live_layer_GMM"][layer]["route_tokens"] for row in cycle_rows) for layer in range(43)]
        assert all(value == 576 for value in tokens), (cycle, tokens)
        cross_rank[str(cycle)] = tokens
    all_layers = [layer for row in rows for layer in row["all43_live_layer_GMM"]]
    active = [layer["active_experts"] for layer in all_layers]
    read_correlation = {
        "gmm1_AIC_read_vs_active_experts": pearson(active, [layer["gmm1_native_AIC_read_bytes"] for layer in all_layers]),
        "gmm2_AIC_read_vs_active_experts": pearson(active, [layer["gmm2_native_AIC_read_bytes"] for layer in all_layers]),
    }
    assert all(value > 0.98 for value in read_correlation.values()), read_correlation
    out = {
        "status": "same_run_same_cycle_route_counter_join_instrumented",
        "run": "run367",
        "source_integrity": source_check(),
        "rows": rows,
        "summary": {
            "samples": len(rows),
            "rank_count": len({r["rank"] for r in rows}),
            "cycles": [64, 65],
            "run121_exact_route_matches": sum(r["run121_route_exact_same"] for r in rows),
            "active_experts_range": [min(r["active_experts"] for r in rows), max(r["active_experts"] for r in rows)],
            "gmm2_AIC_read_to_active_packed_weight_ratio_range": [min(ratios), max(ratios)],
            "gmm2_AIC_read_to_active_packed_weight_ratio_median": statistics.median(ratios),
            "gmm2_profiled_duration_us_range": [min(r["gmm2_ordinal21_profiled_duration_us"] for r in rows), max(r["gmm2_ordinal21_profiled_duration_us"] for r in rows)],
            "all43_AIC_AIV_read_to_active_packed_weight_ratio_range": [min(full_gmm_ratios), max(full_gmm_ratios)],
            "all43_AIC_AIV_read_to_active_packed_weight_ratio_median": statistics.median(full_gmm_ratios),
            "all43_standard_GMM_Gflop_per_rank_cycle_range": [min(r["all43_standard_GMM_Gflop"] for r in rows), max(r["all43_standard_GMM_Gflop"] for r in rows)],
            "all43_active_packed_weight_GB_per_rank_cycle_range": [min(r["all43_active_packed_weight_GB"] for r in rows), max(r["all43_active_packed_weight_GB"] for r in rows)],
            "cross_rank_routed_tokens_per_layer": cross_rank,
            "same_ordinal_active_expert_to_AIC_read_pearson_688_pairs": read_correlation,
        },
        "limits": [
            "Run121 count snapshot patch does device synchronize and D2H copies plus JSON inside extreme::target scope before mark(target); profiler also syncs Target entry and exit. Target Host span and DAG target event themselves are perturbed, as is the next cycle predecessor, and cannot calibrate a native Scheduling floor.",
            "Run367 is one warmed diagnostic measured cohort with two profiled cycles, not Run99 accepted formal E2E.",
            "AIC main-memory counter bytes are task counters, not unique compulsory physical HBM traffic; active packed-weight bytes are a storage estimate, not a compulsory per-cycle HBM read.",
            "The per-layer GMM arithmetic is current-route standard GEMM-equivalent work, not a proof of the minimum possible implementation or the full Target/Draft arithmetic.",
            "The same-cycle route and native task are aligned by rank/cycle/ordinal, but physical L2 hit rate, weight pointers on the native task, actual resource overlap and downstream all-rank device join are still unknown.",
            "Very high active-expert versus same-ordinal AIC read correlation supports task/order mapping but does not expose native input pointers or prove compulsory bytes.",
            "Run367 does not export cycle64/65 acceptance count_history or pre-handoff published p_i; route/counter alignment is not an Algorithm/Resource useful-tokens-per-cycle ledger.",
        ],
        "bound_effect": "Same-cycle route/counter observable closes; no finite Resource/Hardware, Scheduling or Product TPS upper endpoint promoted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "summary": out["summary"]}))


if __name__ == "__main__":
    main()
