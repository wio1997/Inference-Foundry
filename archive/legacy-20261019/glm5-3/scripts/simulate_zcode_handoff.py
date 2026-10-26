#!/usr/bin/env python3
"""Exercise onboarding with a deterministic fake CLI; no model or server access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MOCK_CLI = r'''import argparse, hashlib, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True)
args = parser.parse_args()
job = json.loads(args.prompt.rsplit("\n", 1)[-1])
assert job["simulation"] is True
fixture_path = Path(job["inputs"][0]["path"])
fixture = json.loads(fixture_path.read_text())
print("SIMULATED_RAW_OUTPUT_NOT_FOR_PARENT " * 8000)
evidence = {"id":"E-FIXTURE", "path":str(fixture_path),
            "sha256":hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "bytes":fixture_path.stat().st_size, "locator":"fixture-only JSON, not live observation"}
result = {
  "schema_version":1, "simulation":True, "job_id":job["job_id"], "status":"needs_decision",
  "summary":"SIMULATION：交接已完成，现场身份与Current仍unknown；需要实际环境入口。",
  "execution":{"inner_exit_code":None,"acceptance":"passed","processes":[]},
  "findings":[{"kind":"fact","text":"模拟输入中的现场连接、源码及CLI路径均未配置",
               "scope":{"source":"synthetic fixture", "observed_live_ranks":[]},
               "evidence_ids":["E-FIXTURE"]}],
  "evidence":[evidence], "unknowns":fixture["unknowns"],
  "decision_request":{"question":"请提供或在实际环境发现仓库/源码目录、服务器入口、Zcode路径及现有controller；随后先只读恢复Current。",
                      "evidence_ids":["E-FIXTURE"]}, "next_check_at":None
}
target = Path(job["result"]["path"])
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
temporary.replace(target)
print("Wrote compact result; no real model/server was accessed.")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="new directory for synthetic artifacts; never overwrite")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    bridge = Path(__file__).resolve().with_name("zcode_bridge.py")
    fixture = root / "fixture.json"
    fixture.write_text(json.dumps({"simulation":True,"real_services_touched":False,
                                  "Current":"unknown","unknowns":["actual server entry","source checkout",
                                  "effective deployment","Zcode executable/model id","controller identity"]},indent=2)+"\n")
    mock = root / "mock-zcode"
    mock.write_text("#!"+sys.executable+"\n"+MOCK_CLI)
    mock.chmod(0o755)
    job_dir = root / "jobs" / "SIM-JOB-ONBOARDING-0001"
    job_dir.mkdir(parents=True)
    job = {
      "schema_version":1,"simulation":True,"job_id":"SIM-JOB-ONBOARDING-0001",
      "parent":{"point_id":"GLM-OPT-0001","run_id":None},"owner":"simulated-main-agent",
      "controller":None,"kind":"inspect","goal":"模拟第一份只读环境恢复交接；不得访问真实服务/模型",
      "inputs":[{"path":str(fixture),"sha256":digest(fixture)}],
      "scope":{"write_paths":[str(job_dir)],"resources":[],"environment":"local synthetic fixture"},
      "acceptance":["模拟与现场事实分开","缺项明确unknown","大输出留文件","结果关联证据"],
      "execution":{"mode":"task","work_type":"analysis","cwd":str(root),
                   "prompt":"归约模拟输入，返回needs_decision；不声称有Current或运行服务", "timeout_s":10},
      "result":{"path":str(job_dir/"result.json"),"max_bytes":8192}}
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps(job,ensure_ascii=False,indent=2)+"\n")
    run = subprocess.run([sys.executable,str(bridge),"run",str(job_path),"--zcode",str(mock)],
                         capture_output=True,text=True,timeout=20)
    if run.returncode:
        raise RuntimeError(run.stdout.strip() or run.stderr.strip())
    envelope = json.loads(run.stdout)
    result = envelope["result"]
    metadata = json.loads((job_dir/"bridge.json").read_text())
    assert envelope["handoff"] == "VALID" and result["status"] == "needs_decision"
    assert result["simulation"] is True and metadata["backend_family"] == "simulated"
    assert "SIMULATED_RAW_OUTPUT_NOT_FOR_PARENT" not in run.stdout + run.stderr
    assert metadata["stdout_bytes"] > 200000 and len(run.stdout.encode()) < 8192
    assert result["evidence"][0]["sha256"] == digest(fixture)
    validation = subprocess.run([sys.executable,str(bridge),"validate",str(job_path),str(job_dir/"result.json")],
                                capture_output=True,text=True,timeout=10)
    assert validation.returncode == 0
    report = {
      "simulation":True,"job_id":job["job_id"],"point_id":"GLM-OPT-0001","run_id":None,
      "handoff":"VALID","task_status":result["status"],"Current":"unknown",
      "real_cli_invoked":False,"real_services_touched":False,"model_calls":0,
      "summary":result["summary"],"unknowns":result["unknowns"],
      "raw_cli_bytes":metadata["stdout_bytes"],"parent_return_bytes":len(run.stdout.encode()),
      "bridge_sha256":digest(bridge),"simulator_sha256":digest(__file__),
      "artifacts":{"job":str(job_path),"result":str(job_dir/"result.json"),"bridge":str(job_dir/"bridge.json")},
      "checks":{"output_isolated":True,"result_format_and_cli_exit_checked":True,
                "evidence_hash_checked":True,"simulation_not_baseline":True}}
    (root/"simulation-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
