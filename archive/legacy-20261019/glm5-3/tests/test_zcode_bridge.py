"""CLI boundary tests; fake Zcode only, no model/server/benchmark calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BRIDGE = Path(__file__).resolve().parents[1] / "scripts" / "zcode_bridge.py"


class BridgeTests(unittest.TestCase):
    def exercise(self, scenario, mode="task", cap=16384):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "job" / "result.json"
            job = {
                "schema_version": 1, "job_id": "TEST-JOB", "parent": {},
                "owner": "test", "kind": "inspect", "goal": "test CLI boundary",
                "inputs": [], "scope": {}, "acceptance": ["fixture"],
                "execution": {"mode": mode, "work_type": "command" if mode == "command" or scenario == "task_unknown_exit" else "persistent" if scenario.startswith("running") else "analysis", "cwd": str(root), "prompt": "fixture",
                              "command": "fixture command", "timeout_s": 0.05 if scenario == "timeout" else 5},
                "result": {"path": str(result_path), "max_bytes": cap}}
            job_path = root / "request.json"
            job_path.write_text(json.dumps(job))
            fake = root / "fake-zcode"
            fake.write_text("#!" + sys.executable + "\n" + '''import json, os, pathlib, sys, time
scenario = os.environ["BRIDGE_TEST_SCENARIO"]
print("RAW_LOG_ONLY_IN_FILE" * 20000)
if scenario == "timeout":
    time.sleep(10)
if scenario != "missing":
    result = {"schema_version":1,"job_id":"TEST-JOB","status":"completed",
              "summary":"compact result","execution":{"inner_exit_code":None,
              "acceptance":"passed","processes":[]},"findings":[],"evidence":[],
              "unknowns":[],"decision_request":None,"next_check_at":None}
    if scenario == "wrong_id": result["job_id"] = "OTHER-JOB"
    if scenario == "inner_failure": result["execution"]["inner_exit_code"] = 7
    if scenario in ("unknown_command", "task_unknown_exit"): result["execution"]["inner_exit_code"] = None
    if scenario == "oversized": result["summary"] = "x" * 20000
    if scenario.startswith("running"):
        result["status"] = "running"
        result["execution"]["acceptance"] = "unverified"
    if scenario == "running":
        result["evidence"] = [{"id":"E1","path":"fixture-process-state.json","sha256":None,"bytes":None,"locator":"fixture"}]
        result["execution"]["processes"] = [{"host":"fixture","session_id":"mock-persistent-session","pid":None,"readiness":"unknown","evidence_ids":["E1"]}]
    pathlib.Path(os.environ["BRIDGE_TEST_RESULT"]).write_text(json.dumps(result))
sys.exit(7 if scenario == "outer_failure" else 0)
''')
            fake.chmod(0o755)
            env = dict(os.environ, BRIDGE_TEST_SCENARIO=scenario, BRIDGE_TEST_RESULT=str(result_path))
            run = subprocess.run([sys.executable, str(BRIDGE), "run", str(job_path), "--zcode", str(fake)],
                                 capture_output=True, text=True, env=env, timeout=15)
            self.assertNotIn("RAW_LOG_ONLY_IN_FILE", run.stdout + run.stderr)
            envelope = json.loads(run.stdout)
            metadata = json.loads((result_path.parent / "bridge.json").read_text())
            if scenario == "outer_failure":
                check = subprocess.run([sys.executable, str(BRIDGE), "validate", str(job_path), str(result_path)],
                                       capture_output=True, text=True, timeout=15)
                self.assertEqual(check.returncode, 2)
                self.assertIn("bridge records CLI failure", json.loads(check.stdout)["error"])
                format_check = subprocess.run([sys.executable, str(BRIDGE), "validate", str(job_path), str(result_path), "--format-only"],
                                              capture_output=True, text=True, timeout=15)
                self.assertEqual(json.loads(format_check.stdout)["handoff"], "FORMAT_VALID")
            if scenario != "timeout":
                self.assertGreater(metadata["stdout_bytes"], 100000)
                self.assertEqual(len(metadata["stdout_sha256"]), 64)
            repeat = subprocess.run([sys.executable, str(BRIDGE), "run", str(job_path), "--zcode", str(fake)],
                                    capture_output=True, text=True, env=env, timeout=15)
            self.assertNotEqual(repeat.returncode, 0)
            return run.returncode, envelope, metadata

    def test_raw_output_stays_in_files(self):
        code, envelope, _ = self.exercise("valid")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["handoff"], "VALID")
        self.assertLess(len(json.dumps(envelope)), 2000)

    def test_failures_do_not_become_success(self):
        for scenario, mode in [("missing", "task"), ("wrong_id", "task"),
                               ("inner_failure", "task"), ("unknown_command", "command"),
                               ("task_unknown_exit", "task"), ("running_no_process", "task"),
                               ("outer_failure", "task"), ("oversized", "task"), ("timeout", "task")]:
            with self.subTest(scenario=scenario):
                code, envelope, _ = self.exercise(scenario, mode)
                self.assertEqual(code, 2)
                self.assertEqual(envelope["handoff"], "INVALID")
                expected = {"missing": "No such file", "wrong_id": "job_id mismatch",
                            "inner_failure": "failed inner command", "unknown_command": "observed inner_exit_code",
                            "task_unknown_exit": "observed inner_exit_code", "running_no_process": "process/session identity",
                            "outer_failure": "CLI failed", "oversized": "size limit", "timeout": "timed out"}
                self.assertIn(expected[scenario], envelope["error"])

    def test_running_is_not_completed(self):
        code, envelope, _ = self.exercise("running")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["result"]["status"], "running")
        self.assertEqual(envelope["result"]["execution"]["acceptance"], "unverified")


if __name__ == "__main__":
    unittest.main()
