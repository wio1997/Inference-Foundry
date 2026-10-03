from common import *
old=p/"runs/GLM-RUN-0148";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
for name,sha in [("AUDIT-RUN148-20261003T0425Z","fd55cf6b94738930e5b420d0371c2987d51fddf13aa0e7018680d4699d3059b8"),("NATIVE-FAULT-CPU3-20261003T0443Z","178ddad3d34d22b275df09b136dcb1a00617215ac5f91396af5d5d7b7b3bbb70"),("NATIVE-IDENTITY-CPU-20261003T0446Z","a09d46cbd8d17bca5f83e4d329b19359efc5acc4948727d199f1cb698de21a50")]:assert ref(p/"jobs"/name/"reduction.json")["sha256"]==sha
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare"],capture_output=True,timeout=240);(r/"prepare_epoch.stdout").write_bytes(z.stdout);(r/"prepare_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
policy=dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3)
for node in ["166","167"]:
 code="from pathlib import Path;import json,hashlib;print(json.dumps(dict(policy=json.loads(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_scheduler_v3.py').read_bytes()).hexdigest())))"
 v=json.loads(run(node,["python3","-c",code],"policy_before_"+node));assert v["policy"]==policy and v["source_sha256"]==ref(g/"runtime/issue_budget_scheduler_v3.py")["sha256"]
assert request("/healthcheck")["request_num"]==0
atomic_json(r/"prepared_state.json",dict(at=utc(),same_native32=True,policy=policy,no_native_signals=True))
