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


def bench(name, limit, concurrency, max_tokens):
    command = [
        "docker", "exec", "dsv4ab", "python3", str(root / "scripts/bench.py"),
        "--dataset", dataset, "--out", str(out / f"{name}.json"), "--offset", "0",
        "--limit", str(limit), "--concurrency", str(concurrency), "--max-tokens", str(max_tokens),
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

marks = {"ready_utc": datetime.now(timezone.utc).isoformat()}
command = ["docker", "exec", "dsv4ab", "python3", str(root / "scripts/check_functional.py"),
           "--golden", str(root / "evidence/20260920_qli_prefill_only/perf/golden4.json"),
           "--out", str(out / "functional_check.json")]
with (out / "functional.log").open("w") as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
marks["functional_done_utc"] = datetime.now(timezone.utc).isoformat()
bench("full_warmup", 48, 12, 128)
marks["warmup_done_utc"] = datetime.now(timezone.utc).isoformat()
(out / "metrics_before.txt").write_text(requests.get(base + "/metrics", timeout=30).text)
bench("decode_c12", 12, 12, 512)
marks["sample_done_utc"] = datetime.now(timezone.utc).isoformat()
(out / "metrics_after.txt").write_text(requests.get(base + "/metrics", timeout=30).text)
(out / "phase_times.json").write_text(json.dumps(marks, indent=2) + "\n")
print(json.dumps(marks, indent=2))
