#!/usr/bin/env python3
"""Auditable V0 census: separate diagnostic work, operand footprint, and traffic.

No row in this script certifies a strict compulsory HBM byte or a Product bound.
Run566 A0 is its own diagnostic W0; it is not Run99's formal W0.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = {
    "headers": "evidence/20260927_loop079_identity/run520/astra_compulsory_census_inputs.json",
    "work": "evidence/20260927_loop079_identity/run521/conditional_work_ledger.json",
    "inventory": "evidence/20260926_loop060_resource/run242/inventory.json",
    "compressor": "evidence/20260926_loop062_nongmm/run256/census.json",
    "gmm": "evidence/20260925_loop044_target/run146/gmm_bound_audit.json",
    "counter": "evidence/20260926_loop060_resource/run247/analysis.json",
    "attribution": "evidence/20260926_loop060_resource/run248/attribution.json",
    "shapes": "evidence/20260926_loop060_resource/run249/shapes.json",
    "config": "/data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json",
    "a0_admission": "evidence/20260928_loop080_bound/run566/a0/arm_admission.json",
}
TRACE_DIR = "evidence/20260928_loop080_bound/run566/a0/trace"
RUNTIME_DIR = "evidence/20260928_loop080_bound/run566/a0/runtime"


def read(path: str):
    p = Path(path) if path.startswith("/") else ROOT / path
    return json.loads(p.read_text())


def sha(path: str) -> str:
    p = Path(path) if path.startswith("/") else ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="raise",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_work(work: dict, config: dict, headers: dict) -> tuple[list[dict], list[dict]]:
    assert config["num_hidden_layers"] == 43
    assert config["num_experts_per_tok"] == 6 and config["n_routed_experts"] == 256
    assert config["hidden_size"] == 4096
    assert headers["source_sha256"][P["config"]] == sha(P["config"])
    assert work["source"][P["headers"]] == sha(P["headers"])
    tensor_headers = headers["target_wo_a"]
    assert len(tensor_headers) == 43
    assert {h["tensor"] for h in tensor_headers} == {
        f"layers.{layer}.attn.wo_a.weight" for layer in range(43)}
    assert all(h["dtype"] == "BF16" and h["shape"] == [8192, 4096] and
               h["data_offsets"][1] - h["data_offsets"][0] == 8192 * 4096 * 2
               for h in tensor_headers)
    wa, moe = work["target_wo_a"], work["routed_moe"]
    a, b = wa["entries"], moe["entries"]
    assert len(a) == 344 and len(b) == 258
    assert {(e["layer"], e["semantic_output_group"]) for e in a} == {
        (layer, group) for layer in range(43) for group in range(8)}
    assert {(e["layer"], e["expert_incidence_ordinal"]) for e in b} == {
        (layer, expert) for layer in range(43) for expert in range(6)}
    assert {e["dtype"] for e in a} == {"BF16"}
    assert all(e["conventional_dense_ops_per_fresh_row"] == 8388608 for e in a)
    assert all(e["conventional_gemm_equivalent_ops_per_fresh_row"] == 50331648 for e in b)
    assert sum(e["conventional_dense_ops_per_fresh_row"] for e in a) == 2885681152
    assert sum(e["conventional_gemm_equivalent_ops_per_fresh_row"] for e in b) == 12985565184
    assert headers["arithmetic"]["one_group_conventional_ops"] == 8388608
    assert headers["arithmetic"]["routed_moe_ops_per_expert_row"] == 50331648
    assert wa["formal_F_by_layer_group"] is None
    assert wa["traffic_lower_bytes"] is None
    assert moe["actual_expert_ids_and_storage_reuse"] is None
    return a, b


def validate_counter_rows(counter: dict, attr: dict, shapes: dict) -> list[tuple[dict, dict, dict]]:
    def keyed(rows):
        out = {}
        for row in rows:
            key = (row["rank"], row["cycle"])
            assert key not in out
            out[key] = row
        return out
    c, a, s = (keyed(x["windows"]) for x in (counter, attr, shapes))
    assert len(c) == len(a) == len(s) == 16 and set(c) == set(a) == set(s)
    assert {rank for rank, _ in c} == set(range(8))
    assert all(len(x["groups"]) == 13 for x in a.values())
    assert all(row["capture"] == 4 for row in c.values())
    for key in c:
        cr, ar, sr = c[key], a[key], s[key]
        for group in (cr["families"], ar["groups"], sr["shapes"]):
            for item in group.values():
                assert isinstance(item["count"], int) and item["count"] >= 0
                for field in ("read_GB", "write_GB", "duration_sum_ms",
                              "read_KB", "write_KB", "kernel_sum_ms"):
                    if field in item:
                        assert math.isfinite(item[field]) and item[field] >= 0
        assert sum(item["count"] for item in cr["families"].values()) == sum(
            item["count"] for item in ar["groups"].values())
        for name in ("read", "write"):
            source = sum(item[f"{name}_KB"] * 1024 / 1e9
                         for item in cr["families"].values())
            attributed = sum(item[f"{name}_GB"] for item in ar["groups"].values())
            assert math.isclose(source, attributed, abs_tol=1e-7)
    return [(c[k], a[k], s[k]) for k in sorted(c)]


def read_a0_trace(admission: dict):
    root = ROOT / TRACE_DIR
    paths = sorted(root.glob("trace_rank*_cohort*.json"))
    assert len(paths) == 32, len(paths)
    assert admission["status"] == "diagnostic_arm_admitted" and admission["mode"] == "off"
    assert admission["cohort"] == 5
    rows, hashes, cycles, ids_by_cohort = [], {}, {}, {}
    for path in paths:
        stem = path.stem.split("_")
        rank = int(stem[1][4:]); cohort = int(stem[2][6:])
        assert 0 <= rank < 8 and 5 <= cohort <= 8
        trace = json.loads(path.read_text())
        assert trace and all(len(cycle["target_input_ids"]) == 12 and
                             all(len(slot) == 8 for slot in cycle["target_input_ids"])
                             for cycle in trace)
        assert all(len(cycle["counts"]) == 12 for cycle in trace)
        key = (rank, cohort)
        assert key not in hashes
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert admission["trace_file_sha256"][f"trace/{path.name}"] == hashes[key]
        runtime_path = ROOT / RUNTIME_DIR / f"rank{rank}_cohort{cohort}.json"
        runtime = json.loads(runtime_path.read_text())
        assert admission["runtime_file_sha256"][f"runtime/{runtime_path.name}"] == hashlib.sha256(
            runtime_path.read_bytes()).hexdigest()
        assert runtime["rank"] == rank and runtime["cohort"] == cohort and runtime["cycles"] == len(trace)
        assert len(runtime["req_ids"]) == 12 and len(set(runtime["req_ids"])) == 12
        req_ids = tuple(runtime["req_ids"])
        if cohort in ids_by_cohort:
            assert ids_by_cohort[cohort] == req_ids
        else:
            ids_by_cohort[cohort] = req_ids
        if cohort in cycles:
            assert cycles[cohort] == len(trace)
        else:
            cycles[cohort] = len(trace)
        rows.append(dict(arm="run566_a0_diagnostic", rank=rank, cohort=cohort,
                         cycles=len(trace), physical_target_rows_per_cycle=96,
                         current_graph_physical_rows=len(trace) * 96,
                         logical_required_rows="",
                         fixed_formal_W0_certified=False,
                         source=str(path.relative_to(ROOT))))
    assert set(hashes) == {(r, c) for r in range(8) for c in range(5, 9)}
    return rows, hashes, cycles, ids_by_cohort


def census(out: Path) -> dict:
    inp = {k: read(v) for k, v in P.items()}
    entries_wa, entries_moe = validate_work(inp["work"], inp["config"], inp["headers"])
    joined = validate_counter_rows(inp["counter"], inp["attribution"], inp["shapes"])
    trace_rows, trace_hashes, cycles, req_ids = read_a0_trace(inp["a0_admission"])
    work_rows = []
    for e in entries_wa:
        work_rows.append(dict(phase="target", layer=e["layer"], family="wo_a",
            work_class="BF16_conventional_dense", operand=e["source_tensor"],
            group_or_incidence=e["semantic_output_group"],
            logical_rows="", current_graph_physical_rows=96,
            ops_per_fresh_required_row=e["conventional_dense_ops_per_fresh_row"],
            formal_in_window_ops_lower="", formal_in_window_ops_upper="",
            status="conditional_per_row_formula", missing_gate="fresh_required_F_by_group"))
    for e in entries_moe:
        work_rows.append(dict(phase="target", layer=e["layer"], family="routed_moe",
            work_class="W4A8_GEMM_equivalent", operand="expert_id_unknown",
            group_or_incidence=e["expert_incidence_ordinal"],
            logical_rows="", current_graph_physical_rows=96,
            ops_per_fresh_required_row=e["conventional_gemm_equivalent_ops_per_fresh_row"],
            formal_in_window_ops_lower="", formal_in_window_ops_upper="",
            status="conditional_per_row_formula", missing_gate="fresh_required_F_and_expert_ids"))
    for width, item in sorted(inp["compressor"]["summary"].items(), key=lambda x: int(x[0])):
        work_rows.append(dict(phase="target", layer="all", family=f"compressor_{width}",
            work_class="nominal_current_96row_geometry", operand="two_projection_weights",
            group_or_incidence=item["calls_per_rank_cycle"], logical_rows="",
            current_graph_physical_rows=96, ops_per_fresh_required_row="",
            formal_in_window_ops_lower="", formal_in_window_ops_upper="",
            status=f"current_nominal_projection_ops_per_rank_cycle={item['nominal_two_projection_flops_per_cycle']}",
            missing_gate="fresh_rows_and_nonprojection_arithmetic"))
    for phase, family, missing in (
        ("target", "shared_expert_and_other_dense", "complete_source_shape_and_fresh_rows"),
        ("target", "attention_indexer_cache", "logical_arithmetic_and_KV_ranges"),
        ("target", "metadata_and_communication", "necessary_payload_and_collective_cut"),
        ("dspark", "full_proposer", "layer_shapes_and_semantic_required_evaluations"),
        ("prefill", "cached_residual_and_seed", "actual_shape_sequence_and_required_work"),
        ("serving", "Host_Runtime", "critical_path_and_fixed_W0_schedule"),
    ):
        work_rows.append(dict(phase=phase, layer="", family=family, work_class="not_censused",
            operand="", group_or_incidence="", logical_rows="", current_graph_physical_rows="",
            ops_per_fresh_required_row="", formal_in_window_ops_lower="", formal_in_window_ops_upper="",
            status="explicit_missing_row", missing_gate=missing))
    work_headers = list(work_rows[0])
    traffic_rows = []
    for _, a, _ in joined:
        for family, item in sorted(a["groups"].items()):
            traffic_rows.append(dict(rank=a["rank"], cycle=a["cycle"], family=family,
                current_kernel_count=item["count"], observed_read_GB=item["read_GB"],
                observed_write_GB=item["write_GB"], observed_kernel_sum_ms=item["duration_sum_ms"],
                operand_footprint_bytes="", compulsory_HBM_lower_bytes="",
                constructive_HBM_upper_bytes="", link_bytes_status="unavailable_in_counter_export",
                status="current_instrumented_kernel_counter_only",
                source="Run248 same window; Run247/249 independent classification"))
    operand_rows = [
        ("BF16_wo_a_distinct_checkpoint", inp["work"]["target_wo_a"]["distinct_checkpoint_storage_bytes"], "Run520/521"),
        ("W4A8_one_active_expert_layer_packed", inp["gmm"]["packed_bytes_per_expert"], "Run146"),
    ] + [(f"compressor_{width}_current_geometry_weights",
          item["unique_two_weight_tensor_footprint_bytes_per_cycle"], "Run256")
         for width, item in sorted(inp["compressor"]["summary"].items(), key=lambda x: int(x[0]))]
    for family, footprint, source in operand_rows:
        traffic_rows.append(dict(rank="", cycle="", family=family,
            current_kernel_count="", observed_read_GB="", observed_write_GB="",
            observed_kernel_sum_ms="", operand_footprint_bytes=footprint,
            compulsory_HBM_lower_bytes="", constructive_HBM_upper_bytes="",
            link_bytes_status="not_applicable", status="operand_footprint_not_HBM_lower", source=source))
    shape_rows = []
    for _, _, s in joined:
        for signature, item in sorted(s["shapes"].items()):
            shape_rows.append(dict(rank=s["rank"], cycle=s["cycle"], signature=signature,
                current_kernel_count=item["count"], observed_read_GB=item["read_GB"],
                observed_write_GB=item["write_GB"], observed_kernel_sum_ms=item["duration_sum_ms"],
                compulsory_HBM_lower_bytes="", status="current_instrumented_shape_only"))
    out.mkdir(parents=True, exist_ok=True)
    assert all(not (out / name).exists() for name in (
        "work_census.csv", "traffic_census.csv", "shape_census.csv",
        "a0_workload.csv", "coverage.json")), "census output already exists"
    write_csv(out / "work_census.csv", work_headers, work_rows)
    write_csv(out / "traffic_census.csv", list(traffic_rows[0]), traffic_rows)
    write_csv(out / "shape_census.csv", list(shape_rows[0]), shape_rows)
    write_csv(out / "a0_workload.csv", list(trace_rows[0]), trace_rows)
    assert any(r["rank"] == 0 and r["cycle"] == 0 for r in traffic_rows)
    assert all(r["compulsory_HBM_lower_bytes"] == "" for r in traffic_rows)
    coverage = dict(
        schema_version=1, status="conditional_diagnostic_census_no_strict_bound",
        input_sha256={v: sha(v) for v in P.values()},
        a0_trace_sha256={f"rank{r}_cohort{c}": h for (r, c), h in sorted(trace_hashes.items())},
        a0_request_ids_sha256_by_cohort={str(c): hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for c, ids in sorted(req_ids.items())},
        counts=dict(wo_a_groups=len(entries_wa), routed_expert_incidences=len(entries_moe),
                    counter_windows=len(joined), traffic_family_rows=len(traffic_rows),
                    shape_rows=len(shape_rows), a0_trace_files=len(trace_rows),
                    a0_cycles_by_cohort=cycles),
        conditional_per_fresh_row=dict(BF16_wo_a_ops=sum(e["conventional_dense_ops_per_fresh_row"] for e in entries_wa),
            W4A8_MoE_GEMM_equivalent_ops=sum(e["conventional_gemm_equivalent_ops_per_fresh_row"] for e in entries_moe)),
        operand_footprint_not_HBM_lower=dict(BF16_wo_a_distinct_checkpoint_bytes=inp["work"]["target_wo_a"]["distinct_checkpoint_storage_bytes"],
            W4A8_packed_bytes_per_active_expert_layer=inp["gmm"]["packed_bytes_per_expert"]),
        physical_rows_scope="96 current Target input rows/cycle; per-incidence expert M and logical required rows are unknown",
        current_traffic_sources="Run247/248/249 capture4 diagnostic windows, never joined to Run566 A0 W0; HCCL link bytes unavailable",
        missing=["formal Run99 W0 trace", "fresh required per-layer rows", "full DSpark/pre-fill/attention/shared-expert work",
                 "expert IDs and ownership", "compulsory KV/activation/metadata/HCCL traffic",
                 "capacity-constrained residency and exact-board C+/B", "all8 critical path and attainable mixed service"],
        prohibited=["counter bytes as compulsory traffic", "distinct footprint as HBM lower",
                    "BF16 and W4A8 ops mixed into one capacity", "multiply rank work by TP8 without ownership proof",
                    "sum family medians as a real window", "Run566 A0 as formal Run99 W0"],
        finite_resource_floor_s=None, finite_scheduling_floor_s=None, finite_product_tps_ceiling=None,
        current_formal_tps=571.681,
    )
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    return coverage


def self_test():
    work, cfg, headers = read(P["work"]), read(P["config"]), read(P["headers"])
    validate_work(work, cfg, headers)
    for mutation in ("dtype", "duplicate", "missing"):
        bad = copy.deepcopy(work)
        if mutation == "dtype": bad["target_wo_a"]["entries"][0]["dtype"] = "FP32"
        if mutation == "duplicate": bad["target_wo_a"]["entries"][0] = bad["target_wo_a"]["entries"][1]
        if mutation == "missing": bad["routed_moe"]["entries"].pop()
        try: validate_work(bad, cfg, headers)
        except AssertionError: pass
        else: raise AssertionError(f"accepted {mutation} corruption")
    for key in ("num_experts_per_tok", "n_routed_experts"):
        bad = copy.deepcopy(cfg); bad[key] += 1
        try: validate_work(work, bad, headers)
        except AssertionError: pass
        else: raise AssertionError(f"accepted {key} config drift")
    c, a, s = (read(P[k]) for k in ("counter", "attribution", "shapes"))
    validate_counter_rows(c, a, s)
    bad = copy.deepcopy(s); bad["windows"][0]["rank"] = 99
    try: validate_counter_rows(c, a, bad)
    except AssertionError: pass
    else: raise AssertionError("accepted wrong-rank counter join")
    for value in (-1.0, float("nan")):
        bad = copy.deepcopy(a); bad["windows"][0]["groups"]["gmm1"]["read_GB"] = value
        try: validate_counter_rows(c, bad, s)
        except AssertionError: pass
        else: raise AssertionError("accepted invalid counter")
    bad = copy.deepcopy(c); bad["windows"][0]["capture"] = 3
    try: validate_counter_rows(bad, a, s)
    except AssertionError: pass
    else: raise AssertionError("accepted wrong capture")
    admission = read(P["a0_admission"])
    _, hashes, _, _ = read_a0_trace(admission)
    assert len(hashes) == 32
    bad = copy.deepcopy(admission)
    first = next(iter(bad["trace_file_sha256"]))
    bad["trace_file_sha256"][first] = "0" * 64
    try: read_a0_trace(bad)
    except AssertionError: pass
    else: raise AssertionError("accepted unadmitted trace")
    print("Run569 V0 census self-test PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test: self_test()
    else:
        assert a.out_dir
        print(json.dumps(census(Path(a.out_dir))["counts"], sort_keys=True))
