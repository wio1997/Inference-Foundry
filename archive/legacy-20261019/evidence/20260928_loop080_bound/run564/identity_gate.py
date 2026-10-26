"""CPU-only structural negatives for Run564 arm admission and A0/B/A1 matching."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path("/data/wio/Inference_Foundry")
sys.path.insert(0, str(ROOT / "scripts"))
import loop080_pause_control_admit_run564 as admit  # noqa: E402
import loop080_pause_control_reduce_run564 as reduce  # noqa: E402


# The existing real trace arithmetic has its own checker. Keep this fixture
# tiny and test only the new identity/completeness/digest joins here.
admit.check_trace = lambda trace, runtime: None


def jwrite(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def row_token(slot):
    return [100 + slot] + [-1] * 7


def make_arm(root: Path, name: str, mode: str, measured_order):
    arm = root / name
    (arm / "runtime").mkdir(parents=True)
    (arm / "trace").mkdir()
    client_rows = []
    for phase in ("warmup", "measured"):
        for index in range(48):
            client_rows.append({"response_id": f"{name}-{phase}-{index}",
                                "phase": phase, "dataset_index": index,
                                "dataset_row_sha256": f"row-{index}",
                                "request_body_sha256": f"body-{index}"})
    jwrite(arm / "client_admission.json", {
        "status": "client_two_phase_admitted", "request_count": 96,
        "unique_response_ids": 96, "same_48_request_bodies_across_phases": True,
        "request_index": client_rows,
    })
    (arm / "server_post_count.txt").write_text("96\n")
    for rank in range(8):
        for cohort in range(1, 9):
            if cohort <= 4:
                indexes = list(range((cohort - 1) * 12, cohort * 12))
                phase = "warmup"
            else:
                block = measured_order[cohort - 5]
                indexes = list(range(block * 12, (block + 1) * 12))
                phase = "measured"
            ids = [f"{name}-{phase}-{index}-worker" for index in indexes]
            jwrite(arm / "runtime" / f"rank{rank}_cohort{cohort}.json", {
                "rank": rank, "cohort": cohort, "cycles": 144,
                "req_ids": ids, "pass": True, "host_mirror_exact": True,
                "generated_output_counts": [1024] * 12,
                "wall_seconds": 1.0,
            })
            if cohort < 5:
                continue
            trace = [{name: [] for name in admit.FIELDS} for _ in range(144)]
            for cycle in range(144):
                trace[cycle]["counts"] = [8] + ([7] * 11 if cycle < 128 else [8] * 11)
                trace[cycle]["accepted"] = [[100 + slot] * 8 for slot in range(12)]
                trace[cycle]["target_input_ids"] = [[index] * 8 for index in indexes]
                trace[cycle]["target_positions"] = [[cycle] * 8 for _ in range(12)]
            jwrite(arm / "trace" / f"trace_rank{rank}_cohort{cohort}.json", trace)
            if mode == "on" and cohort == 5:
                counts = [[8] + [7] * 11 for _ in range(128)] + [[0] + [8] * 11 for _ in range(16)]
                sampled = [[[100 + slot] * count for slot, count in enumerate(row)]
                           for row in counts]
                segment = {"generation": 5, "request_ids": ids, "slots": [0],
                           "cycle": 128, "token_ids": [[100] * 1024]}
                segment["output_digest"] = admit.digest({
                    "generation": 5, "request_ids": ids, "slots": [0],
                    "cycle": 128, "token_ids": [[100] * 1024],
                })
                trajectory = {"generation": 5, "request_ids": ids,
                              "cycles": 144, "counts": counts, "sampled": sampled}
                trajectory["sha256"] = admit.digest({
                    "generation": 5, "request_ids": ids,
                    "counts": counts, "sampled": sampled,
                })
                jwrite(arm / "trace" / f"pause_rank{rank}_cohort5.json",
                       {"segment": segment, "trajectory": trajectory,
                        "resume_pause_ms": 0.1})
    return arm


with TemporaryDirectory(prefix="run564-gate-") as temp:
    root = Path(temp)
    # B's fifth cohort is block0; the same semantic slot order appears at
    # cohort6 in A0 and cohort7 in A1.
    plans = (("a0", "off", [1, 0, 2, 3]),
             ("b", "on", [0, 1, 2, 3]),
             ("a1", "off", [1, 2, 0, 3]))
    for name, mode, order in plans:
        arm = make_arm(root, name, mode, order)
        report = admit.admit(arm, mode, 5)
        jwrite(arm / "arm_admission.json", report)
        assert report["status"] == "diagnostic_arm_admitted"
    result = reduce.reduce_run(root, 5)
    assert result["matched_cohorts"] == {"b": 5, "a0": 6, "a1": 7}
    assert result["b_observed_trace_equal_to_controls"] is True

    btrace = root / "b" / "trace" / "trace_rank3_cohort5.json"
    original = btrace.read_bytes()
    changed = json.loads(original)
    changed[0]["target_positions"][0][0] = 9
    jwrite(btrace, changed)
    try:
        reduce.reduce_run(root, 5)
        raise AssertionError("post-admission trace rewrite accepted")
    except AssertionError as exc:
        assert "changed after admission" in str(exc)
    jwrite(root / "b" / "arm_admission.json", admit.admit(root / "b", "on", 5))

    runtime_path = root / "b" / "runtime" / "rank3_cohort5.json"
    original_runtime = runtime_path.read_bytes()
    changed_runtime = json.loads(original_runtime)
    changed_runtime["wall_seconds"] = 0.001
    jwrite(runtime_path, changed_runtime)
    try:
        reduce.reduce_run(root, 5)
        raise AssertionError("post-admission Runtime rewrite accepted")
    except AssertionError as exc:
        assert "runtime_file_sha256 changed after admission" in str(exc)
    runtime_path.write_bytes(original_runtime)
    assert reduce.reduce_run(root, 5)["b_observed_trace_equal_to_controls"] is False
    btrace.write_bytes(original)
    jwrite(root / "b" / "arm_admission.json", admit.admit(root / "b", "on", 5))

    b_pause = root / "b" / "trace" / "pause_rank4_cohort5.json"
    original = b_pause.read_bytes()
    changed = json.loads(original)
    changed["trajectory"]["counts"][0][0] = 2
    jwrite(b_pause, changed)
    try:
        admit.admit(root / "b", "on", 5)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad sampled/count ledger accepted")
    b_pause.write_bytes(original)

    changed = json.loads(original)
    changed["trajectory"]["counts"][0][0] = 0
    changed["trajectory"]["sampled"][0][0] = []
    changed["trajectory"]["sha256"] = admit.digest({
        "generation": 5, "request_ids": changed["trajectory"]["request_ids"],
        "counts": changed["trajectory"]["counts"],
        "sampled": changed["trajectory"]["sampled"],
    })
    jwrite(b_pause, changed)
    try:
        admit.admit(root / "b", "on", 5)
    except AssertionError as exc:
        assert "zero count before slot completion" in str(exc)
    else:
        raise AssertionError("zero-before-completion accepted")
    b_pause.write_bytes(original)

    missing = root / "a0" / "trace" / "trace_rank7_cohort8.json"
    missing.unlink()
    try:
        admit.admit(root / "a0", "off", 5)
        raise AssertionError("missing rank/cohort trace accepted")
    except FileNotFoundError:
        pass

print("Run564 CPU identity gate PASS: cross-cohort semantic match, trace divergence, bad B ledger, missing rank")
