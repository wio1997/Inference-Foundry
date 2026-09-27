#!/usr/bin/env python3
"""Validate full frozen diagnostic lifecycle before route numerator promotion."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop078_bound/run403"


def concurrency(rows):
    events = []
    for row in rows:
        events.extend([(row["start"], 1), (row["end"], -1)])
    n = maximum = 0
    for _, delta in sorted(events):
        n += delta
        maximum = max(maximum, n)
    return maximum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert (BASE/"source_before.sha256").read_text() == (BASE/"source_after.sha256").read_text()
    assert int((BASE/"server_post_count.txt").read_text()) == 60
    clients = {}
    for file, expected in (("warmup48.json",48),("bench.json",12)):
        x=json.loads((BASE/file).read_text())
        rows=x["requests"]
        assert len(rows)==expected
        assert all(r["output_tokens"]==1024 and r["error"] is None for r in rows)
        assert x["summary"]["success"]==expected and x["summary"]["fail"]==0
        max_c=concurrency(rows)
        assert max_c<=12
        clients[file]={"requests":expected,"max_observed_concurrency":max_c,
                       "diagnostic_output_tps":x["summary"]["output_tps"]}
    captures=list((BASE/"capture").glob("rank*_cohort*.json"))
    reports=list((BASE/"runtime").glob("rank*_cohort*.json"))
    assert len(captures)==len(reports)==40
    unique=set()
    cohort_cycles=[]
    for cohort in range(1,6):
        reference=None
        for rank in range(8):
            cap=json.loads((BASE/"capture"/f"rank{rank}_cohort{cohort}.json").read_text())
            rt=json.loads((BASE/"runtime"/f"rank{rank}_cohort{cohort}.json").read_text())
            assert cap["rank"]==rt["rank"]==rank and cap["cohort"]==rt["cohort"]==cohort
            assert cap["cycles"]==rt["cycles"]
            assert rt["pass"] and rt["host_mirror_exact"]
            assert rt["target_graph_requested"] and rt["target_graph_mode"]=="FULL"
            assert rt["oracle_target_calls_after_handoff"]==rt["model_runner_cycles_after_handoff"]==0
            assert rt["generated_output_counts"]==[1024]*12
            if reference is None:
                reference=(rt["req_ids"],rt["cycles"])
                cohort_cycles.append(rt["cycles"])
                unique.update(rt["req_ids"])
            else:
                assert reference==(rt["req_ids"],rt["cycles"])
    assert len(unique)==60
    stop=(BASE/"stop.log").read_text()
    tail=stop.split("逐卡 HBM 占用（MB）：")[-1]
    hbm=[int(x) for x in re.findall(r"(\d+)\s*/\s*65536",tail)]
    assert len(hbm)==8 and max(hbm)<6000
    out={"status":"clean_frozen_diagnostic_gate","client":clients,
         "server_post_count":60,"cohorts":5,"ranks":8,
         "capture_files":40,"runtime_reports":40,"unique_runtime_request_ids":60,
         "cohort_cycles":cohort_cycles,"source_sha_restored":True,
         "max_idle_hbm_MB":max(hbm),
         "scope":"48 warmup + 12 diagnostic; not repeated formal 48-request E2E"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out))


if __name__=="__main__":
    main()
