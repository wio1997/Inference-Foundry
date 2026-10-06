"""First real Zcode Job: inspect only; writes only its fresh Job artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

J = Path(sys.argv[1])
job = json.loads(J.read_text())
outdir = J.parent
G = Path("/data/tiankuan/wio/Inference-Foundry/glm5-3")
B = G / "records/points/GLM-OPT-0002/runs/GLM-RUN-0241/restored"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
snapshot = {}
evidence = []
findings = []


def artifact(path, ident, locator):
    raw = path.read_bytes()
    return dict(id=ident, path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), locator=locator)


def command(args, label, input=None):
    proc = subprocess.run(args, input=input, capture_output=True, timeout=90)
    path = outdir / (label + ".stdout")
    path.write_bytes(proc.stdout)
    (outdir / (label + ".stderr")).write_bytes(proc.stderr)
    evidence.append(artifact(path, label, "entire command output"))
    proc.check_returncode()
    return proc.stdout


try:
    snapshot["branch"] = command(["git", "-C", str(G.parent), "branch", "--show-current"], "branch").decode().strip()
    snapshot["HEAD"] = command(["git", "-C", str(G.parent), "rev-parse", "HEAD"], "HEAD").decode().strip()
    snapshot["zcode_path"] = command(["sh", "-c", "command -v zcode"], "cli_path").decode().strip()
    snapshot["zcode_version"] = command(["/usr/local/bin/zcode", "--version"], "cli_version").decode().strip()
    assert snapshot["branch"] == "glm5-3-autonomous-20261001"
    snapshot["controller_owner"] = json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text())
    snapshot["locks"] = command(["lslocks", "--noheadings", "--output", "PID,COMMAND,PATH"], "locks").decode()
    material = json.loads((B / "native_engines_resident.json").read_text())
    probe = (G / "records/points/GLM-OPT-0002/runs/GLM-RUN-0239/live_probe.py").read_text()
    for engine in material["engines"]:
        root = engine["roots"]["node0"]
        args = ["python3", "-c", probe]
        if root["host"] == "167":
            import shlex
            args = ["ssh", "-o", "BatchMode=yes", "root@172.16.10.167", shlex.join(args)]
        data = json.loads(command(args, "native_" + root["host"], json.dumps(dict(owner=root, NPU_count=16)).encode()))
        assert data["npu_worker_pids"] == engine["members"]["node0"]["npu_worker_pids"]
        snapshot["native_" + root["host"]] = data
    for host, port in (("127.0.0.1", 8000), ("172.16.10.166", 9081), ("172.16.10.167", 9900)):
        with HTTP.open(f"http://{host}:{port}/health", timeout=10) as response:
            snapshot[f"health_{port}"] = response.status
        with HTTP.open(f"http://{host}:{port}/metrics", timeout=10) as response:
            text = response.read().decode()
        snapshot[f"queue_{port}"] = [l for l in text.splitlines() if l.startswith("vllm:num_requests_running{") or l.startswith("vllm:num_requests_waiting{")]
    snapshot["current"] = None
    snapshot["service_operations"] = 0
    snapshot["generation_requests"] = 0
    path = outdir / "snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    evidence.append(artifact(path, "snapshot", "health/queue/native owner/NPU/controller/read-only counters"))
    findings.append(dict(kind="fact", text="两机当前native32归属核验通过，public/D0/D1 health200；只读Job未重启、发生成请求或benchmark。", scope=dict(hosts=["166", "167"], ranks=32, read_only=True), evidence_ids=["snapshot"]))
    result = dict(schema_version=1, job_id=job["job_id"], status="completed", summary="首次真实Job完成只读现场核验；不裁决性能根因或KEEP。",
        execution=dict(inner_exit_code=0, acceptance="passed", processes=[]), findings=findings, evidence=evidence,
        unknowns=["真实ready/commit/device关键路径未采集，E2E Gain unknown"], decision_request=None, next_check_at=None)
    code = 0
except Exception as error:
    result = dict(schema_version=1, job_id=job["job_id"], status="failed", summary=repr(error),
        execution=dict(inner_exit_code=1, acceptance="failed", processes=[]), findings=findings, evidence=evidence,
        unknowns=["只读检查未完成，不得推断服务异常原因"], decision_request="Sol检查失败的只读步骤", next_check_at=None)
    code = 1
temporary = outdir / "result.tmp"
temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
temporary.replace(job["result"]["path"])
raise SystemExit(code)
