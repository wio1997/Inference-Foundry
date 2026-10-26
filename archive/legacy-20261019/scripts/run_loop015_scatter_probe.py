import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

root = Path("/data/wio/Inference_Foundry")
out = root / "evidence/20260920_loop015_cold_kernels/probe"
out.mkdir(parents=True, exist_ok=True)
flag = out / "capture.flag"
base = "http://127.0.0.1:8080"
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
command = [
    "docker", "exec", "dsv4ab", "python3", str(root / "scripts/check_functional.py"),
    "--golden", str(root / "evidence/20260920_qli_prefill_only/perf/golden4.json"),
    "--out", str(out / "functional_check.json"),
]
with (out / "functional.log").open("w") as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
marks["functional_done_utc"] = datetime.now(timezone.utc).isoformat()
(out / "metrics_before.txt").write_text(requests.get(base + "/metrics", timeout=30).text)
flag.write_text("capture one cold long scatter\n")
try:
    command = [
        "docker", "exec", "dsv4ab", "python3", str(root / "scripts/bench.py"),
        "--dataset", "/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8.jsonl",
        "--out", str(out / "cold1_offset24.json"), "--offset", "24", "--limit", "1",
        "--concurrency", "1", "--max-tokens", "128",
    ]
    with (out / "cold1_offset24.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
finally:
    flag.unlink(missing_ok=True)
marks["cold_done_utc"] = datetime.now(timezone.utc).isoformat()
(out / "metrics_after.txt").write_text(requests.get(base + "/metrics", timeout=30).text)
(out / "phase_times.json").write_text(json.dumps(marks, indent=2) + "\n")
print(json.dumps(marks, indent=2))
