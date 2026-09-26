#!/usr/bin/env python3
"""Compare adjacent frozen formal runs without confusing trajectory and wall time."""
import argparse
import json
import statistics
from pathlib import Path


def load(root: Path):
    summary=json.loads((root/"summary.json").read_text())
    rows={}
    for p in (root/"runtime").glob("rank*_cohort*.json"):
        a=json.loads(p.read_text())
        assert a["pass"] and a["target_graph_mode"] == "FULL"
        rows.setdefault(int(a["cohort"]), []).append(a)
    assert len(rows)==16 and all(len(x)==8 for x in rows.values())
    passes=[]
    for index in range(3):
        cohorts=[]
        for cohort in range(1+4*index,5+4*index):
            x=rows[cohort]
            cycle_values={r["cycles"] for r in x}
            assert len(cycle_values)==1
            cohorts.append({"cohort":cohort,"cycles":cycle_values.pop(),"latest_rank_runtime_s":max(r["wall_seconds"] for r in x),"mean_rank_runtime_s":statistics.mean(r["wall_seconds"] for r in x)})
        client=summary["runs"][index]
        cycle_sum=sum(c["cycles"] for c in cohorts)
        runtime_sum=sum(c["latest_rank_runtime_s"] for c in cohorts)
        passes.append({"pass":index+1,"client_tps":client["output_tps"],"client_duration_s":client["duration_s"],"cycles":cycle_sum,"runtime_latest_sum_s":runtime_sum,"runtime_ms_per_cycle":1000*runtime_sum/cycle_sum,"client_minus_latest_runtime_s":client["duration_s"]-runtime_sum,"cohorts":cohorts})
    return {"summary_pass":summary["pass"],"all_runtime_rows_pass":summary["all_runtime_rows_pass"],"tps_median":summary["output_tps"]["median"],"passes":passes}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--control",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    cand=load(args.candidate);ctrl=load(args.control)
    paired=[]
    for c,b in zip(cand["passes"],ctrl["passes"]):
        paired.append({"pass":c["pass"],"tps_delta":c["client_tps"]-b["client_tps"],"cycle_delta":c["cycles"]-b["cycles"],"runtime_ms_per_cycle_delta":c["runtime_ms_per_cycle"]-b["runtime_ms_per_cycle"],"client_residual_delta_s":c["client_minus_latest_runtime_s"]-b["client_minus_latest_runtime_s"]})
    out={"status":"valid" if cand["summary_pass"] and ctrl["summary_pass"] and cand["all_runtime_rows_pass"] and ctrl["all_runtime_rows_pass"] else "invalid","candidate":cand,"control":ctrl,"paired_pass":paired,"median_tps_delta":cand["tps_median"]-ctrl["tps_median"],"median_tps_delta_percent":100*(cand["tps_median"]/ctrl["tps_median"]-1),"interpretation":"Three sequential candidate then control formal passes; cohort trajectories, prefill and serving residual vary. Paired ordinal deltas are descriptive, not a randomized crossover or kernel-exposed-time proof."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:out[k] for k in ("status","paired_pass","median_tps_delta","median_tps_delta_percent")},indent=2))

if __name__=="__main__":main()
