"""Independently adjudicate the frozen Run260 bracket on resident H6 from saved raw outputs."""
from pathlib import Path
import hashlib
import json
import statistics
import sys


def main(root):
    identities = {}

    def read(name):
        raw = (root / name).read_bytes()
        identities[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(raw)

    plan=read("functional_plan.json")
    phases = ["A1", "B1", "A2", "B2"]
    complete = {p: read(p + "_complete_result.json") for p in phases}
    work = {p: read(p + "_complete_workload.json") for p in phases}
    first = complete["A1"]
    for p in phases:
        row = complete[p]
        assert row["complete"] and row["semantic_accepted"]
        assert row["finish_reason"] == "stop" and row["completion_tokens"] == 23
        assert row["prompt_tokens"] == 58
        assert row["token_ids"] == first["token_ids"] == plan["complete_golden"]["token_ids"]
        assert row["final_content"] == first["final_content"]
        assert all(work[p]["checks"].values())
        assert sum(v for k, v in row["external_KV_delta"].items()
                   if "hits_total" in k) == 58
        # D wall includes stream completion after the final token. Use raw
        # token arrivals for TPOT instead of treating DONE as the last token.
        events = read(p + "_complete_events.json")
        arrived = []
        ids = []
        for event in events:
            if event["value"] is None:  # terminal [DONE], no emitted tokens
                continue
            for choice in event["value"].get("choices", []):
                tokens = choice.get("token_ids", [])
                if tokens:
                    arrived.append(event["arrived_ns"])
                    ids.extend(tokens)
        assert ids == row["token_ids"]
        assert abs(row["TPOT_ms"] - (arrived[-1] - arrived[0]) / 1e6 / 22) < 1e-8
    signatures = {p: work[p]["signature"] for p in phases}
    matched = all(v == signatures["A1"] for v in signatures.values())
    drift = max(abs(complete["A2"]["D_wall_s"] - complete["A1"]["D_wall_s"]),
                abs(complete["B2"]["D_wall_s"] - complete["B1"]["D_wall_s"]))
    pairs = []
    for a, b in [("A1", "B1"), ("A2", "B2")]:
        x, y = complete[a], complete[b]
        saving = x["D_wall_s"] - y["D_wall_s"]
        p_saving = x["P_wall_s"] - y["P_wall_s"]
        pd_saving = x["PD_wall_s"] - y["PD_wall_s"]
        assert abs(pd_saving - saving - p_saving) < 1e-8
        pairs.append(dict(pair=a + "->" + b, D_wall_s=[x["D_wall_s"], y["D_wall_s"]],
                          TPOT_ms=[x["TPOT_ms"], y["TPOT_ms"]],
                          D_saving_s=saving, P_saving_s=p_saving, PD_saving_s=pd_saving,
                          D_wall_reduction=1 - y["D_wall_s"] / x["D_wall_s"],
                          TPOT_reduction=1 - y["TPOT_ms"] / x["TPOT_ms"],
                          PD_wall_s=[x["PD_wall_s"], y["PD_wall_s"]],
                          PD_wall_reduction=1 - y["PD_wall_s"] / x["PD_wall_s"],
                          saving_exceeds_frozen_drift=saving > drift))
    shorts = {}
    for p in phases:
        rows = [read(p + "_measure%d_result.json" % i) for i in range(2)]
        ws = [read(p + "_measure%d_workload.json" % i) for i in range(2)]
        assert all(x["token_ids"] == rows[0]["token_ids"] and
                   x["completion_tokens"] == 8 and x["finish_reason"] == "length"
                   for x in rows)
        shorts[p] = dict(TPOT_ms=[x["TPOT_ms"] for x in rows],
                         median_TPOT_ms=statistics.median(x["TPOT_ms"] for x in rows),
                         signatures=[x["signature"] for x in ws])
    witnesses = {p: read(p + "_witness.json") for p in phases}
    reference = witnesses["A1"]["H6"]["rows"]
    for p in phases:
        item=witnesses[p];w=item["H6"];expected_mode=int(p.startswith("B"))
        assert w["all_rank_witness"] and w["mode"]==1 and len(w["rows"])==16
        assert item["gather_mode"]==expected_mode and item["H6_mode"]==1
        assert [(x["pid"],x["start_ticks"]) for x in w["rows"]]==[(x["pid"],x["start_ticks"]) for x in reference]
        assert all(x["mode"]==1 and x["dispatch_cached_true"] and x["combine_cached_true"] and x["library_sha256"]==plan["native_candidate"]["sha256"] for x in w["rows"])
        role=item["same_worker_identity"]
        assert role["worker_start_ticks"]==plan["adopted_workers"]["167"]
        hw=item["H4_rows"];assert len(hw)==16 and {z["rank"] for z in hw}==set(range(16))
        assert {z["pid"] for z in hw}==set(role["worker_namespace_pids"].values())
        assert all(z["mode"]==expected_mode and z["artifact"]=='candidate:'+plan["candidate_sha256"] for z in hw)
        if expected_mode:assert all(z["actual_hidden_bitwise_equal"] and z["all_ranks_bytes_equal"] and z["input_bytes_equal"] for z in hw)
    phase, state = read("stackcompare.phase.json"), read("state.json")
    assert phase["exit_code"] == 0 and phase["status"] == "succeeded"
    assert state["status"] == "completed"
    retained=read("retained_stack.json")
    assert retained["H6"] and retained["MC2_mode"]==1 and retained["comparison_completed"]
    guard=read("guards_after.json")
    assert all(x["health"]==200 and x["idle"] and len(x["device_owners"])==16 for x in guard.values())
    assert all(x["worker_start_ticks"]==plan["adopted_workers"][h] for h,x in guard.items())
    short_matched=all(v==shorts['A1']['signatures'][0] for row in shorts.values() for v in row['signatures'])
    positive = matched and short_matched and all(x["saving_exceeds_frozen_drift"] and
                              x["D_wall_reduction"] > 0 and x["TPOT_reduction"] > 0 and x["PD_wall_reduction"] > 0
                              for x in pairs) and all(shorts[b]["median_TPOT_ms"] < shorts[a]["median_TPOT_ms"] for a,b in [("A1","B1"),("A2","B2")])
    assert retained["H4"] == positive and retained["gather_mode"]==int(positive)
    decision=read("research_decision.json");assert decision["research_positive"]==positive
    out = dict(run_id="GLM-RUN-0260", baseline_stack=["H6"], candidate_stack=["H6","H4"], frozen_rule_applied=True,
               scoped_verdict="POSITIVE" if positive else "INCONCLUSIVE",
               formal_verdict="NOT_PRODUCT_PROMOTED", Current=None, PERF_KEEP=False,
               complete_cumulative_work_matched=matched, signatures=signatures,
               D_drift_s=drift, complete_pairs=pairs, short8=shorts,
               H6_retained=True, H4_retained=positive, short_cumulative_work_matched=short_matched, phase_exit_code=0,
               limitations=["Cumulative counters do not establish ordered acceptance or target-only work.",
                            "H6 fixed mode1; same workers/common native/gather shim. This isolates H4 versus H6 baseline, not a stock full-stack comparison.",
                            "Research stack addition only if frozen gates pass; complete API/SLA/stability still pending.",
                            "Observed output timing improvement is not an additive Run249 ON-profile CPU budget."],
               input_identity=identities)
    (root / "comparison_reduced.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ["scoped_verdict", "D_drift_s", "complete_pairs"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
