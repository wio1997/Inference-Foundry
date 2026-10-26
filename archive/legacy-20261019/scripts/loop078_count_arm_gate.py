#!/usr/bin/env python3
"""Clean 48+12 control-arm gate for Run401. Diagnostic timing is not formal."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def max_concurrency(rows):
    events = []
    for row in rows:
        events.extend(((row["start"], 1), (row["end"], -1)))
    n = peak = 0
    for _, delta in sorted(events):
        n += delta
        peak = max(peak, n)
    assert n == 0
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("A0", "B", "A1"))
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    d = a.dir
    assert (d/"source_before.sha256").read_text() == (d/"source_after.sha256").read_text()
    assert int((d/"server_post_count.txt").read_text()) == 60
    client = []
    client_max = []
    client_hash = hashlib.sha256()
    for name, expected in (("warmup48.json", 48), ("bench.json", 12)):
        raw = (d/name).read_bytes()
        client_hash.update(raw)
        data = json.loads(raw)
        assert len(data["requests"]) == expected
        assert data["summary"]["success"] == expected and data["summary"]["fail"] == 0
        assert all(x["output_tokens"] == 1024 and x["error"] is None for x in data["requests"])
        client_max.append(max_concurrency(data["requests"]))
        assert client_max[-1] <= 12
        client.append(data)
    runtime = d/"runtime"
    reports = sorted(runtime.glob("rank*_cohort*.json"))
    assert len(reports) == 40
    req_ids = set()
    cycles = []
    wall = []
    for cohort in range(1, 6):
        ref = None
        for rank in range(8):
            row = json.loads((runtime/f"rank{rank}_cohort{cohort}.json").read_text())
            assert row["rank"] == rank and row["cohort"] == cohort
            assert row["pass"] and row["host_mirror_exact"]
            assert row["target_graph_requested"] and row["target_graph_mode"] == "FULL"
            assert row["oracle_target_calls_after_handoff"] == row["model_runner_cycles_after_handoff"] == 0
            assert row["generated_output_counts"] == [1024]*12
            if ref is None:
                ref = (row["req_ids"], row["cycles"])
                req_ids.update(row["req_ids"])
                cycles.append(row["cycles"])
            else:
                assert ref == (row["req_ids"], row["cycles"])
        wall.append(max(json.loads((runtime/f"rank{rank}_cohort{cohort}.json").read_text())["wall_seconds"] for rank in range(8)))
    assert len(req_ids) == 60
    stop = (d/"stop.log").read_text()
    tail = stop.split("逐卡 HBM 占用（MB）：")[-1]
    hbm = [int(x) for x in re.findall(r"(\d+)\s*/\s*65536", tail)]
    assert len(hbm) == 8 and max(hbm) < 6000
    if a.arm == "B":
        captures = list((d/"capture").glob("rank*_cohort*.json"))
        assert len(captures) == 40
        run_id = "LOOP078-RUN410-B"
        assert all(json.loads(p.read_text())["run_id"] == run_id for p in captures)
    else:
        assert not (d/"capture").exists()
        run_id = "LOOP078-RUN413-A0" if a.arm == "A0" else "LOOP078-RUN417-A1"
    assert max(client_max) == 12
    out = dict(arm=a.arm, run_id=run_id, http_posts=60, max_concurrency=max(client_max),
        outputs_per_request=1024, cohorts=5, errors=0, full_target=True,
        post_handoff_oracle_calls=0, host_mirror_exact=True, source_restored=True,
        lifecycle_closed=True, runtime_wall_s=sum(wall), per_cohort_latest_rank_wall_s=wall,
        cohort_cycles=cycles, diagnostic12_client_wall_s=client[1]["summary"]["duration_s"],
        request_ledger_sha256=client_hash.hexdigest(),
        acceptance_sha256=None,
        acceptance_scope="Controls expose cohort cycles/window means, not per-cycle acceptance; exact A0/B/A1 trajectory parity unavailable",
        max_idle_hbm_MB=max(hbm),
        scope="Frozen 48 warmup + 12 diagnostic control; no formal E2E performance promotion")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(dict(arm=a.arm,cycles=cycles,latest_rank_wall_s=sum(wall))))


if __name__ == "__main__":
    main()
