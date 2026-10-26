#!/usr/bin/env python3
"""Rebuild eligible capacity priors without promoting attained service to C+.

All source measurements predate this script. There is no model/NPU run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GMM = "evidence/20260927_loop075_bound"
HCCL = "evidence/20260926_loop061_bound"
EVENT = "evidence/20260927_loop077_bound/run395/analysis.json"
EVENT_RAW = "evidence/20260927_loop077_bound/run394"
FULL = "evidence/20260927_loop079_identity/run487/intervals.json"


def raw(path: str) -> bytes:
    return (ROOT / path).read_bytes()


def sha(path: str) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def load(path: str):
    return json.loads(raw(path))


def finite_positive(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)


def row(*, source: str, family: str, work_join: str, metric: str,
        value: float, unit: str, sample_count: int, classification: str,
        shape: str, conditions: str, disallowed_transfer: str,
        lower=None, upper=None, source_paths=()):
    assert classification in ("attained_service", "instrumented_current")
    assert math.isfinite(value) and value > 0 and sample_count > 0
    assert lower is None or math.isfinite(lower)
    assert upper is None or math.isfinite(upper)
    return dict(source=source, family=family, run569_work_join=work_join,
        metric=metric, value=value, unit=unit, observed_min=lower, observed_max=upper,
        samples=sample_count, classification=classification, shape=shape,
        conditions=conditions, disallowed_transfer=disallowed_transfer,
        source_sha256={p: sha(p) for p in source_paths},
        strict_capacity_upper=None, fixed_formal_W0_match=False,
        product_bound_eligible=False)


def gmm_rows() -> list[dict]:
    analysis_path=f"{GMM}/run357/analysis.json"
    analysis=load(analysis_path)
    assert analysis["status"] == "valid_same_layer_bank_A_B_A2_timing_only"
    assert analysis["route_ordinal"] == 64
    output=[]
    for case in ("gmm1", "gmm2"):
        rank_records=[]; paths=[analysis_path]
        for rank in range(8):
            samples=[]
            reference=None
            for run_id, banks in ((354,1),(355,8),(356,1)):
                p=f"{GMM}/run{run_id}/rank{rank}.json";paths.append(p)
                x=load(p)
                assert x["rank"]==rank and x["route_ordinal"]==64 and x["weight_banks"]==banks
                assert x["weight_format"]==29 and x["w1_shape"]==[32,4096,512]
                assert x["w2_shape"]==[32,2048,512]
                if reference is None:reference=x["route_counts"]
                assert x["route_counts"]==reference
                y=x["cases"][case]
                assert y["graph_ops"]==16 and y["post_replay_all_captured_outputs_finite"] is True
                vals=y["device_event_us_per_op"]
                assert len(vals)==30;finite_positive(vals)
                median=statistics.median(vals)
                assert math.isclose(median,y["median_us"],abs_tol=1e-5)
                samples.append(median)
            prior=analysis["cases"][case]["ranks"][rank]
            for value,key in zip(samples,("single_bank_A_us","eight_bank_B_us","single_bank_A2_us")):
                assert math.isclose(value,prior[key],abs_tol=1e-5)
            midpoint=(samples[0]+samples[2])/2
            effect=100*(samples[1]/midpoint-1)
            assert math.isclose(effect,prior["B_vs_A_midpoint_percent"],abs_tol=1e-5)
            rank_records.append(samples)
        effects=[100*(v[1]/((v[0]+v[2])/2)-1) for v in rank_records]
        assert math.isclose(statistics.median(effects),
            analysis["cases"][case]["median_B_vs_A_midpoint_percent"],abs_tol=1e-5)
        strict=sum(v[1]>max(v[0],v[2]) for v in rank_records)
        assert strict==analysis["cases"][case]["strict_B_slower_ranks"]
        for index,label in enumerate(("bank1_A","bank8_B","bank1_A2")):
            across=[v[index] for v in rank_records]
            output.append(row(source="Run354-357", family=f"W4A8_{case}",
                work_join="routed_moe",metric=f"per_rank_graph_call_median_{label}",
                value=statistics.median(across),lower=min(across),upper=max(across),
                unit="us_per_call",sample_count=8,classification="attained_service",
                shape="ordinal64 real Run121 route-count vector; packed format29; 16 captured calls/replay",
                conditions="8 rank medians of 30 replays each; independent synthetic zero weight/input Graph timing; Host windows overlap, per-replay device overlap unproved; bank working set 1 or 8",
                disallowed_transfer="not original FULL Graph residency, nonzero-data attained service, C+, mixed Target/DSpark capacity or Product latency",
                source_paths=paths))
            if index == 1:
                output[-1]["bank8_vs_A_A2_median_percent"] = statistics.median(effects)
                output[-1]["bank8_strict_slower_ranks"] = strict
    return output


HCCL_253=(
    ("ag_12k_fp32","AllGather","FP32",12288),
    ("ag_96k_bf16","AllGather","BF16",98304),
    ("ag_384k_bf16","AllGather","BF16",393216),
    ("ag_3102720_bf16","AllGather","BF16",3102720),
    ("rs_96k_bf16","ReduceScatter","BF16",98304),
    ("a2a_96k_bf16","AllToAll","BF16",98304),
)


def parse_hccl(log: str, exit_path: str, expected: int) -> float:
    assert raw(exit_path).decode().strip()=="0"
    matches=re.findall(r"(?m)^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*success\s*$",raw(log).decode())
    assert len(matches)==1
    size,time,bw=matches[0]
    assert int(size)==expected
    time=float(time);bw=float(bw)
    assert math.isfinite(time) and time>0 and math.isfinite(bw) and bw>0
    # The tool rounds time to 0.01 us before printing, while bandwidth has
    # five decimals; compare with the corresponding rounding envelope.
    assert math.isclose(expected/(time*1000),bw,rel_tol=5e-4)
    return time


def hccl_rows() -> list[dict]:
    output=[]
    env=f"{HCCL}/environment.txt"
    assert b"26.0.rc1" in raw(env) and b"9.1.0" in raw(env)
    for stem,op,dtype,size in HCCL_253:
        log=f"{HCCL}/run253/{stem}.log";exit_path=f"{HCCL}/run253/{stem}.exit"
        value=parse_hccl(log,exit_path,size)
        output.append(row(source="Run253 CANN9.1 HCCL Test",family=f"HCCL_{op}",
            work_join="metadata_and_communication",metric="tool_normal_average",
            value=value,unit="us",sample_count=50,classification="attained_service",
            shape=f"data_size={size}B per NPU; {dtype}; TP8",
            conditions="8x910B3 driver26.0.rc1 CANN9.1.0; official HCCL Test; correctness success; 20 warmups",
            disallowed_transfer="tool data_size is not wire bytes; no Graph/interleaved-compute/rank-arrival match; not compulsory or minimum latency",
            source_paths=(env,log,exit_path,f"{HCCL}/run253/run.sh")))
    for stem,op in (("ag","AllGather"),("rs","ReduceScatter"),("a2a","AllToAll")):
        for mode in ("t0","t1"):
            log=f"{HCCL}/run254/{stem}.{mode}.log"
            exit_path=f"{HCCL}/run254/{stem}.{mode}.exit"
            value=parse_hccl(log,exit_path,98304)
            output.append(row(source="Run254 CANN9.1 HCCL Test",family=f"HCCL_{op}",
                work_join="metadata_and_communication",metric=f"tool_{mode}_average",
                value=value,unit="us",sample_count=50,classification="attained_service",
                shape="data_size=98304B per NPU; BF16; TP8",
                conditions="8x910B3 driver26.0.rc1 CANN9.1.0; HCCL_BUFFSIZE=256MB; 20 warmups; t1 device-only excludes tool Host dispatch",
                disallowed_transfer="isolated tool timing is not Graph collective lower latency, physical wire bytes or additive current-cycle cost",
                source_paths=(env,log,exit_path,f"{HCCL}/run254/run.sh")))
    return output


def current_rows() -> list[dict]:
    x=load(EVENT)
    assert x["status"]=="original_path_sparse_device_event_and_acceptance_ledger"
    assert x["source_integrity"]["source_before_after_sha_match"] is True
    assert x["client_gates"]["warmup48.json"]["exact_1024"] is True
    assert x["client_gates"]["bench.json"]["exact_1024"] is True
    assert len(x["rows"])==80 and len(x["paired_cycle64_65"])==40
    expected_rows={(cohort,rank,cycle) for cohort in range(1,6)
                   for rank in range(8) for cycle in (64,65)}
    expected_pairs={(cohort,rank) for cohort in range(1,6) for rank in range(8)}
    assert {(r["cohort"],r["rank"],r["cycle"]) for r in x["rows"]}==expected_rows
    assert {(r["cohort"],r["rank"]) for r in x["paired_cycle64_65"]}==expected_pairs
    raw_paths=[EVENT,f"{EVENT_RAW}/source_before.sha256",
               f"{EVENT_RAW}/source_after.sha256",
               f"{EVENT_RAW}/warmup48.json",f"{EVENT_RAW}/bench.json",
               "scripts/loop077_event_capture_analyze_run394.py"]
    assert raw(raw_paths[1])==raw(raw_paths[2])
    for cohort,rank in sorted(expected_pairs):
        p=f"{EVENT_RAW}/events/rank{rank}_cohort{cohort}.json"
        raw_paths.append(p)
        capture=load(p)
        assert (capture["cohort"],capture["rank"])==(cohort,rank)
        assert [e["cycle"] for e in capture["runtime_events"]]==[64,65]
        assert [e["cycle"] for e in capture["dspark_events"]]==[64,65]
        assert len(capture["accepted_counts_by_cycle_slot"])>65
        for idx,cycle in enumerate((64,65)):
            r=next(r for r in x["rows"] if (r["cohort"],r["rank"],r["cycle"])==(cohort,rank,cycle))
            m={v["label"]:v["ms_before_anchor"] for v in capture["runtime_events"][idx]["markers"]}
            assert math.isclose(r["runtime_stage_ms"]["target"],m["derived_target_metadata"]-m["target"],abs_tol=1e-7)
            assert math.isclose(r["runtime_stage_ms"]["proposer"],m["state_advance"]-m["proposer"],abs_tol=1e-7)
            assert r["raw_accepted_tokens"]==sum(capture["accepted_counts_by_cycle_slot"][cycle])
        pair=next(r for r in x["paired_cycle64_65"] if (r["cohort"],r["rank"])==(cohort,rank))
        begin=[e["markers"][0]["ms_before_anchor"] for e in capture["runtime_events"]]
        assert math.isclose(pair["cycle64_begin_to_cycle65_begin_ms"],begin[0]-begin[1],abs_tol=1e-7)
    output=[]
    for stage,join in (("target","target_all_families"),("proposer","dspark_full_proposer")):
        values=[r["runtime_stage_ms"][stage] for r in x["rows"]]
        finite_positive(values)
        published=x["summary"]["runtime_stage_ms_across_rank_cycles"][stage]
        assert math.isclose(statistics.median(values),published["median"],abs_tol=1e-7)
        output.append(row(source="Run394-395",family=f"current_{stage}",
            work_join=join,metric="selected_stage_median",value=statistics.median(values),
            lower=min(values),upper=max(values),unit="ms",sample_count=len(values),
            classification="instrumented_current",shape="original FULL Graph, c12; selected cycle64/65 across5 cohorts and8 ranks",
            conditions="sparse same-rank device event; instrumented warm48+diagnostic12; not cross-rank common-clock makespan",
            disallowed_transfer="not isolated service capacity, uninstrumented formal Current, compulsory latency or removable gap",
            source_paths=raw_paths))
    values=[r["cycle64_begin_to_cycle65_begin_ms"] for r in x["paired_cycle64_65"]]
    finite_positive(values)
    published=x["summary"]["paired_same_rank_interval_summary_ms"]["cycle64_begin_to_cycle65_begin_ms"]
    assert math.isclose(statistics.median(values),published["median"],abs_tol=1e-7)
    output.append(row(source="Run394-395",family="current_adjacent_cycle",work_join="Target+DSpark+Host_runtime",
        metric="same_rank_begin_to_begin_median",value=statistics.median(values),
        lower=min(values),upper=max(values),unit="ms",sample_count=40,
        classification="instrumented_current",shape="cycle64->65; five cohorts x eight ranks",
        conditions="same-rank selected event pair; instrumented original path",
        disallowed_transfer="not all8 makespan, passive formal cycle time or resource lower bound",
        source_paths=raw_paths))
    return output


def full_rows() -> list[dict]:
    x=load(FULL)
    assert x["rank_slices"]==40 and x["scope"].startswith("clean exact60 Target FULL frontier")
    root=x["source_root"]
    assert root=="evidence/20260927_loop079_identity/run484/b_candidate"
    captures=[];paths=[FULL]
    for rel,digest in sorted(x["input_sha256"].items()):
        p=f"{root}/{rel}"
        assert sha(p)==digest
        if rel.startswith("capture/"):
            captures.append(load(p));paths.append(p)
    assert len(captures)==40
    output=[]
    for key in ("R0_R1_ms","T_U_adjacent_ms"):
        if key=="T_U_adjacent_ms":
            vals=[sum(c["intervals"][field] for field in
                ("T_R0_ms","R0_R1_ms","R1_H_ms","H_U_ms")) for c in captures]
        else: vals=[c["intervals"][key] for c in captures]
        finite_positive(vals)
        published=x["intervals_ms"][key]
        assert math.isclose(statistics.median(vals),published["median_ms"],abs_tol=1e-5)
        output.append(row(source="Run487",family="current_Target_FULL",work_join="target_all_families",
            metric=key,value=statistics.median(vals),lower=min(vals),upper=max(vals),
            unit="ms",sample_count=40,classification="instrumented_current",
            shape="selected cycle64 FULL96 Target Graph, four hidden/aux gathers; five cohorts x eight ranks",
            conditions="same-device event intervals; replay-stream output completion conditional on exact producer/child-stream joins",
            disallowed_transfer="not passive Current, all8 makespan, necessary Target time or mixed capacity; do not splice with Run395",
            source_paths=paths))
    return output


def build(out: Path) -> dict:
    rows=gmm_rows()+hccl_rows()+current_rows()+full_rows()
    assert len(rows)==6+12+3+2
    assert all(r["strict_capacity_upper"] is None and not r["product_bound_eligible"] for r in rows)
    result=dict(schema_version=1,status="empirical_capacity_priors_not_bound",
        contract="DeepSeek V4 Flash W4A8 8x910B3 DP1TP8 DSpark7 warm48x32K->1024 c12",
        environment="server driver26.0.rc1; serving CANN9.1.0; source-specific tests listed per row",
        rows=rows,
        coverage=dict(attained_service_rows=sum(r["classification"]=="attained_service" for r in rows),
            instrumented_current_rows=sum(r["classification"]=="instrumented_current" for r in rows),
            strict_capacity_upper_rows=0,
            missing=["real-data/residency-matched concurrent all8 GMM service",
                     "mixed Target+DSpark+HCCL resource service under same W0",
                     "Graph-path collective bytes and attainable peer service",
                     "full necessary work/traffic and exact-board C+/B",
                     "complete critical-path dependency and Product schedule"]),
        numeric_bound_update=False,finite_resource_floor_s=None,
        finite_scheduling_floor_s=None,finite_product_tps_ceiling=None,
        formal_current_tps=571.681)
    out.mkdir(parents=True,exist_ok=False)
    (out/"capacity_matrix.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result["coverage"],sort_keys=True))
    return result


def self_test():
    assert math.isclose(parse_hccl(f"{HCCL}/run254/ag.t1.log",
        f"{HCCL}/run254/ag.t1.exit",98304),40.03)
    try:parse_hccl(f"{HCCL}/run254/ag.t1.log",
        f"{HCCL}/run254/ag.t1.exit",98305)
    except AssertionError:pass
    else:raise AssertionError("accepted wrong HCCL payload")
    r=row(source="x",family="x",work_join="x",metric="x",value=1,unit="ms",
          sample_count=1,classification="attained_service",shape="x",conditions="x",
          disallowed_transfer="x")
    assert r["strict_capacity_upper"] is None and not r["product_bound_eligible"]
    print("Run571 capacity scope gate PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out-dir");p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        assert a.out_dir
        build(Path(a.out_dir))
