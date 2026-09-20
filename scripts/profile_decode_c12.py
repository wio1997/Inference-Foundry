import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()
root = Path("/data/wio/Inference_Foundry")
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
dataset = "/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl"
base = "http://127.0.0.1:8080"


def now():
    return datetime.now(timezone.utc).isoformat()


def bench(name, limit, concurrency, max_tokens):
    command = [
        "docker", "exec", "dsv4ab", "python3", str(root / "scripts/bench.py"),
        "--dataset", dataset, "--out", str(out / f"{name}.json"),
        "--offset", "0", "--limit", str(limit), "--concurrency", str(concurrency),
        "--max-tokens", str(max_tokens),
    ]
    with (out / f"{name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


for _ in range(240):
    try:
        if requests.get(base + "/health", timeout=2).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(10)
else:
    raise RuntimeError("service readiness timeout")

marks = {"ready_utc": now()}
bench("full_warmup", 48, 12, 1)
marks["full_warmup_done_utc"] = now()
bench("warm_c12", 12, 12, 128)
marks["warm_c12_done_utc"] = now()
response = requests.post(base + "/start_profile", timeout=120)
response.raise_for_status()
marks["profile_started_utc"] = now()
try:
    bench("decode_c12", 12, 12, 512)
    marks["profiled_request_done_utc"] = now()
finally:
    marks["stop_requested_utc"] = now()
    try:
        response = requests.post(base + "/stop_profile", timeout=1200)
        marks["stop_http"] = response.status_code
        response.raise_for_status()
    finally:
        marks["profile_stopped_utc"] = now()
        (out / "phase_times.json").write_text(json.dumps(marks, indent=2) + "\n")
print(json.dumps(marks, indent=2))
