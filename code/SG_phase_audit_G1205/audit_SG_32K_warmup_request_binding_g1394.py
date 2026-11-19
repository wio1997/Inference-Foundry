"""Re-audit a completed 32K warmup against its original AISBench request bytes."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


RUN_ID = "sg_pd_g1317_aisbench_32k_g1324"
INPUT = 32768
OUTPUT = 1024
REQUESTS = 16


def read(path):
    return json.loads(path.read_bytes())


def argv(pid):
    try:
        return Path("/proc", str(pid), "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return []


def run(*args):
    return subprocess.check_output(args, text=True, timeout=12)


def request_binding(details, dataset):
    if len(details) != REQUESTS or len(dataset) != REQUESTS:
        raise ValueError("A complete warmup requires sixteen raw requests and dataset rows")
    ids = {row["id"] for row in details if type(row["id"]) is int}
    indices = {row["data_id"] for row in details if type(row["data_id"]) is int}
    if ids != indices or indices != set(range(REQUESTS)):
        raise ValueError("Raw request IDs or dataset indexes are missing or repeated")
    for row in details:
        source = dataset[row["data_id"]]
        if (row["success"] is not True or row["input_tokens"] != INPUT
                or row["output_tokens"] != OUTPUT or source["max_out_len"] != OUTPUT
                or row["input"] != source["question"]):
            raise ValueError(f"Unverified request at dataset index {row['data_id']}")
    return sorted(indices)


def audit(client, warmup):
    phase = f"warmup{warmup}"
    label = f"{phase}_{INPUT}"
    folder = client / "runs" / "sglang" / RUN_ID
    pipeline = client / "benchmark_pipeline_SG_g1183"
    manifest = read(folder / "run_manifest.json")
    command = read(folder / f"{label}_command.json")
    assert manifest["run_id"] == RUN_ID and manifest["lengths"] == [INPUT]
    assert manifest["warmup_passes"] == 5 and manifest["gate"]["profile_epoch"] == 1317
    assert command["dataset_phase"] == phase and command["requests"] == REQUESTS
    assert command["argv"][command["argv"].index("--num-prompts") + 1] == str(REQUESTS)
    if warmup == 1:
        runner_pid = manifest["host_runner_pid"]
    else:
        handle = read(client / "SG_AISBench_control_g1233" / RUN_ID / phase / "actual_launch_handle.json")
        assert handle["run_id"] == RUN_ID and handle["step"] == phase and handle["length"] == INPUT
        runner_pid = handle["outer_PID"]
    runner = pipeline / "run_aisbench_cold_hot_g1294.py"
    args = argv(runner_pid)
    outer_live = str(runner).encode() in args and RUN_ID.encode() in args
    container = manifest["client_container_id"]
    processes = run("docker", "top", container, "-eo", "pid,args").splitlines()[1:]
    worker_pids = [int(row.split()[0]) for row in processes
                   if "ais_bench" in row and f"/sglang_{label}.py" in row]
    base = {"run_id": RUN_ID, "epoch": 1317, "warmup": warmup, "runner_PID": runner_pid,
            "same_runner_live": outer_live, "warmup_worker_PIDs": worker_pids}
    if outer_live or worker_pids:
        return {**base, "classification": "in_flight"}

    completed = read(folder / f"{label}_audit.json")
    assert completed["fully_validated"] and completed["successful_requests"] == REQUESTS
    spec = json.loads(run("docker", "inspect", container))[0]
    assert spec["Id"] == container
    mount, = (row for row in spec["Mounts"]
              if row["Destination"] == "/task/client_ds41_g1122/datasets")
    assert not mount["RW"]
    frozen = (Path(mount["Source"]) / f"{label}.jsonl").read_bytes()
    dataset_hash = sha256(frozen).hexdigest()
    assert dataset_hash == command["dataset_sha256"] == manifest["warmup_dataset_sha256"][str(INPUT)][phase]
    config_hash = sha256((pipeline / "configs" / f"sglang_{label}.py").read_bytes()).hexdigest()
    assert config_hash == command["config_sha256"]
    phase_dir = folder / label
    details_path, = phase_dir.rglob("*_details.jsonl")
    raw = details_path.read_bytes()
    indices = request_binding([json.loads(row) for row in raw.splitlines() if row.strip()],
                              [json.loads(row) for row in frozen.splitlines() if row.strip()])
    for tag in ("before", "after"):
        snapshot = read(folder / f"{tag}_{label}_server_snapshot.json")
        assert all(snapshot[role]["metrics"]["status"] == 200 for role in ("prefill", "decode"))
    sys.path.insert(0, str(pipeline))
    from audit_aisbench import audit_phase

    measured = audit_phase(phase_dir, INPUT, requests=REQUESTS, usage_contract=manifest["usage_contract"])
    assert measured == completed and details_path.read_bytes() == raw
    return {**base, "classification": "terminal_verified", "matched_raw_data_IDs": indices,
            "dataset_SHA256": dataset_hash, "config_SHA256": config_hash,
            "standard_raw_artifact_SHA256": measured["evidence"], "duration_s": measured["duration_s"],
            "server_usage_and_report_audit_verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--warmup", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    result = audit(args.client.resolve(strict=True), args.warmup)
    print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **result}, sort_keys=True))
