#!/usr/bin/env python3
"""R247 single launcher: verify frozen review, actual R246 gate, stage once.

``python3 launcher.py --manifest`` only prints the frozen payload SHA on the
Mac.  ``python3 dispatch.py`` is the single execution entrance; it starts this
launcher exactly once.  The gateway performs the actual R246 PASS/idle8 check
before it creates the r247 controller root.  No A node is contacted.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys

R = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(R / "controller"))

import freeze_policy  # noqa: E402

UNIT = "r247"
REVIEW_KIND = "R247_INDEPENDENT_FROZEN_SOURCE_REVIEW"
AUTH_KIND = "R247_INDEPENDENT_REVIEW_RESULT_AUTHORIZATION"
GATE_KIND = "R246_ACTUAL_PASS_IDLE8_SIGNOFF"
GATEWAY_HOST = "A3-5U-Atlas900RCKA3-1"
GATEWAY_ROOT = ("/data/wio/inference-foundry/glm53-flash/controllers/"
                "20261019-r247-b-indexer-optimized-profile")
SSH = ["ssh", "-S", "/tmp/glm53-flash-r236-control", "-p", "60011",
       "-oBatchMode=yes", "-oConnectTimeout=15", "-oServerAliveInterval=15",
       "-oServerAliveCountMax=3", "root@61.241.77.34"]
PAYLOAD_LIMIT = 16 * 1024 * 1024
PER_FILE_LIMIT = 1024 * 1024
CONTROLLER_ENTRY = "controller/controller.py"
EXACT_COMMAND = "python3 dispatch.py"
CONTROLLER_COMMAND = "python3 controller/controller.py"
R246_RECEIPT = ("/data/wio/inference-foundry/glm53-flash/runs/"
                "20261019-r246-dynamic-rowstride-tp16-score-component/host-receipt.json")
R246_CONTAINER = "if-glm53-flash-r246-rowstride-component"
SGLANG_IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"
PODS = {46: "sglang-container-access-46-20261013",
        47: "sglang-container-access-47-20261013"}
HOSTS = {46: "B8-15U-Atlas900RCKA3-46", 47: "B8-23U-Atlas900RCKA3-47"}


def sha(path):
    return freeze_policy.sha256(path)


def collect_manifest(root=R):
    freeze_path = root / "freeze-input.json"
    if not freeze_path.is_file():
        raise SystemExit("MISSING_FREEZE_INPUT")
    freeze = json.loads(freeze_path.read_text())
    immutable = freeze_policy.immutable_sha256(root)
    if freeze.get("files_sha256") != immutable:
        raise SystemExit("FREEZE_INPUT_STALE")
    files = {rel: digest for rel, digest in immutable.items()}
    composite = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return files, composite


def load_r246_gate_local(root=R, review_sha=None):
    path = root / "r246-gate.json"
    if not path.is_file():
        raise SystemExit("MISSING_R246_ACTUAL_PASS_IDLE8_SIGNOFF")
    gate = json.loads(path.read_text())
    if (gate.get("project_id"), gate.get("unit"), gate.get("kind"),
            gate.get("actual"), gate.get("passed")) != (
            "glm53-flash", UNIT, GATE_KIND, True, True):
        raise SystemExit("R246_GATE_IDENTITY_INVALID")
    if gate.get("r246_unit") != "r246":
        raise SystemExit("R246_GATE_UNIT_INVALID")
    receipt_sha = gate.get("r246_host_receipt_sha256")
    if not (isinstance(receipt_sha, str) and len(receipt_sha) == 64
            and all(c in "0123456789abcdef" for c in receipt_sha)):
        raise SystemExit("R246_GATE_RECEIPT_SHA_INVALID")
    idle = gate.get("idle8") or {}
    if idle.get("node46") is not True or idle.get("node47") is not True:
        raise SystemExit("R246_GATE_IDLE8_UNPROVEN")
    if review_sha is not None and gate.get("r247_review_sha256") != review_sha:
        raise SystemExit("R246_GATE_REVIEW_SHA_MISMATCH")
    authorization = gate.get("authorization") or {}
    if not (authorization.get("r247_dispatch_allowed") is True
            and authorization.get("execute_once") is True
            and authorization.get("exact_command") == EXACT_COMMAND):
        raise SystemExit("R246_GATE_NOT_AUTHORIZED")
    return gate


def load_review_and_authorization(root=R):
    review_path = root / "review.json"
    auth_path = root / "review-result.json"
    freeze_path = root / "freeze-input.json"
    if not review_path.is_file():
        raise SystemExit("MISSING_INDEPENDENT_FROZEN_REVIEW")
    if not auth_path.is_file():
        raise SystemExit("MISSING_INDEPENDENT_REVIEW_RESULT_AUTHORIZATION")
    if not freeze_path.is_file():
        raise SystemExit("MISSING_FREEZE_INPUT")
    review = json.loads(review_path.read_text())
    auth = json.loads(auth_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (review.get("project_id"), review.get("unit"),
            review.get("kind")) != ("glm53-flash", UNIT, REVIEW_KIND):
        raise SystemExit("REVIEW_IDENTITY_INVALID")
    immutable = freeze_policy.immutable_sha256(root)
    if review.get("files_sha256") != immutable:
        raise SystemExit("REVIEW_IMMUTABLE_SET_MISMATCH")
    if freeze.get("files_sha256") != immutable:
        raise SystemExit("FREEZE_INPUT_STALE")
    freeze_sha = sha(freeze_path)
    review_sha = sha(review_path)
    if review.get("reviewed_freeze_input_sha256") != freeze_sha:
        raise SystemExit("REVIEW_FREEZE_INPUT_SHA_MISMATCH")
    if (auth.get("project_id"), auth.get("unit"),
            auth.get("kind"), auth.get("passed")) != (
            "glm53-flash", UNIT, AUTH_KIND, True):
        raise SystemExit("REVIEW_RESULT_IDENTITY_INVALID")
    if auth.get("review_sha256") != review_sha:
        raise SystemExit("REVIEW_RESULT_REVIEW_SHA_MISMATCH")
    if auth.get("freeze_input_sha256") != freeze_sha:
        raise SystemExit("REVIEW_RESULT_FREEZE_INPUT_SHA_MISMATCH")
    authorization = auth.get("authorization") or {}
    if not (authorization.get("dispatch_allowed") is True
            and authorization.get("execute_once") is True
            and authorization.get("exact_command") == EXACT_COMMAND
            and authorization.get("controller_entry") == CONTROLLER_COMMAND):
        raise SystemExit("REVIEW_RESULT_NOT_AUTHORIZED")
    precheck = root / "sitecustomize-precheck.json"
    if not precheck.is_file():
        raise SystemExit("SITECUSTOMIZE_PRECHECK_PENDING")
    if authorization.get("sitecustomize_precheck_sha256") not in (None, sha(precheck)):
        raise SystemExit("REVIEW_RESULT_PRECHECK_SHA_MISMATCH")
    gate = load_r246_gate_local(root, review_sha=review_sha)
    return review, auth, freeze, immutable, gate


def build_payload(root=R):
    review, _auth, _freeze, immutable, gate = load_review_and_authorization(root)
    missing = ({"controller/controller.py", "controller/host_probe_sglang.py",
                "controller/launch_command.py", "controller/profile_export.py",
                "controller/profile_plan.json", "controller/freeze_policy.py",
                "stage/manifest.json", "observers/manifest-sglang.json",
                "sitecustomize-precheck.json"} - set(immutable))
    if missing:
        raise SystemExit("MISSING_CONTROLLER_CLOSURE: " + repr(sorted(missing)))
    extras = {"review.json", "review-result.json", "freeze-input.json", "r246-gate.json"}
    files = {}
    for rel in sorted(set(immutable) | extras):
        path = root / rel
        if not path.is_file():
            raise SystemExit("MISSING_FROZEN_FILE: " + rel)
        raw = path.read_bytes()
        files[rel] = {"data": base64.b64encode(raw).decode(),
                      "sha256": hashlib.sha256(raw).hexdigest()}
    decoded = sum(len(base64.b64decode(item["data"])) for item in files.values())
    encoded = sum(len(name) + len(item["data"]) for name, item in files.items())
    if decoded > PAYLOAD_LIMIT or encoded > PAYLOAD_LIMIT:
        raise SystemExit("SOURCE_PAYLOAD_OVER_LIMIT: %d/%d" % (decoded, encoded))
    for rel, item in files.items():
        if len(base64.b64decode(item["data"])) > PER_FILE_LIMIT:
            raise SystemExit("SOURCE_ITEM_OVER_1MIB: " + rel)
    staged_review = json.loads(base64.b64decode(files["review.json"]["data"]))
    if staged_review.get("files_sha256") != immutable:
        raise SystemExit("STAGED_REVIEW_SET_MISMATCH")
    return {"files": files, "decoded_bytes": decoded, "encoded_bytes": encoded,
            "immutable_count": len(immutable), "gate": gate}


def gateway_code(payload, root=GATEWAY_ROOT, host=GATEWAY_HOST):
    return GATEWAY_CODE.replace("@ROOT@", repr(root)) \
        .replace("@HOST@", repr(host)) \
        .replace("@FILES@", repr(payload["files"])) \
        .replace("@RECEIPT@", repr(R246_RECEIPT)) \
        .replace("@R246_CONTAINER@", repr(R246_CONTAINER)) \
        .replace("@IMAGE@", repr(SGLANG_IMAGE)) \
        .replace("@PODS@", repr(PODS)) \
        .replace("@HOSTS@", repr(HOSTS))


def main(root=R, run=subprocess.run):
    if "--manifest" in sys.argv:
        files, composite = collect_manifest(root)
        by_size = sorted(((len((root / rel).read_bytes()), rel) for rel in files),
                         reverse=True)[:8]
        print(json.dumps({"unit": UNIT, "manifest_sha256": composite,
                          "file_count": len(files), "largest": by_size}, indent=2))
        return 0
    if (root / "once.lock").exists():
        raise SystemExit("LAUNCHER_ONCE_LOCK_EXISTS")
    if not (root / "dispatch.once.lock").exists():
        raise SystemExit("DISPATCH_ONCE_LOCK_REQUIRED")
    payload = build_payload(root)
    os.close(os.open(root / "once.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    control = run(SSH[:-1] + ["-O", "check", SSH[-1]], capture_output=True)
    if control.returncode != 0:
        raise SystemExit("SSH_CONTROL_SOCKET_UNHEALTHY")
    result = run(SSH + ["python3", "-"], input=gateway_code(payload),
                 capture_output=True, text=True, timeout=300)
    record = {"rc": result.returncode, "stdout": result.stdout,
              "stderr": result.stderr,
              "source_payload_bytes": payload["decoded_bytes"],
              "source_payload_encoded_bytes": payload["encoded_bytes"],
              "frozen_files": payload["immutable_count"],
              "r246_gate": payload["gate"]}
    (root / "remote-submit.json").write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode != 0:
        raise SystemExit("GATEWAY_SUBMIT_FAILED: " + repr(record))
    print(result.stdout)
    return 0


GATEWAY_CODE = """import base64,hashlib,json,os,pathlib,shutil,socket,subprocess,sys
assert socket.gethostname()==@HOST@
assert pathlib.Path('/data').is_dir() and shutil.disk_usage('/data').free>=4*1024**3
files=@FILES@
decoded={}
for name,item in files.items():
 p=pathlib.PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
 b=base64.b64decode(item['data']);assert hashlib.sha256(b).hexdigest()==item['sha256'],name
 decoded[name]=b
review=json.loads(decoded['review.json']);freeze=json.loads(decoded['freeze-input.json']);gate=json.loads(decoded['r246-gate.json'])
assert (review.get('project_id'),review.get('unit'),review.get('kind'))==('glm53-flash','r247','R247_INDEPENDENT_FROZEN_SOURCE_REVIEW')
frozen=review['files_sha256'];assert freeze['files_sha256']==frozen
assert set(decoded)==set(frozen)|{'review.json','review-result.json','freeze-input.json','r246-gate.json'}
for name,h in frozen.items():assert hashlib.sha256(decoded[name]).hexdigest()==h,name
def host_code(node,code,timeout=60):
 q=subprocess.run(['kubectl','exec','-i','-n','default',@PODS@[node],'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'],input=code,capture_output=True,text=True,timeout=timeout)
 return q
q=host_code(46,'''import hashlib,json,pathlib
p=pathlib.Path(@RECEIPT@)
assert p.is_file(),'R246 receipt missing'
raw=p.read_bytes();r=json.loads(raw)
assert r.get('unit')=='r246' and r.get('passed') is True and r.get('rank_passed')==16
assert (r.get('stage_a') or {}).get('passed') is True
assert r.get('owned_container_verified') is True and r.get('bounded_wait_timeout') is not True
assert (r.get('post_inspect') or {}).get('rc')==0 and (r.get('post_inspect') or {}).get('stdout','').strip()=='false 0'
assert (r.get('post_npu') or {}).get('rc')==0 and (r.get('post_npu') or {}).get('stdout','').count('No running processes found')==8
assert isinstance((r.get('archive') or {}).get('files'),dict) and (r.get('archive') or {}).get('files')
assert r.get('container')==@R246_CONTAINER@ and r.get('image')==@IMAGE@
print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'receipt':r}))''')
assert q.returncode==0,'R246 gate read failed: '+q.stderr
row=json.loads(q.stdout);assert row['sha256']==gate['r246_host_receipt_sha256'],'R246 receipt SHA mismatch'
live={}
for node in (46,47):
 q=host_code(node,'''import json,socket,subprocess
q=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
print(json.dumps({'hostname':socket.gethostname(),'rc':q.returncode,'idle':q.stdout.count('No running processes found')}))''')
 assert q.returncode==0,'idle read failed'
 v=json.loads(q.stdout);assert v['hostname']==@HOSTS@[node] and v['idle']==8,'node not idle8: '+str(node)
 live[node]=v
root=pathlib.Path(@ROOT@)
assert not root.exists(),'controller root already exists'
root.mkdir(parents=True,exist_ok=False)
os.close(os.open(root/'submission.once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
for name,b in decoded.items():
 dst=root/pathlib.PurePosixPath(name);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
(root/'results').mkdir(exist_ok=True);(root/'artifacts').mkdir(exist_ok=True)
for name,h in frozen.items():
 p=root/name;assert p.is_file(),'MISSING_FROZEN_FILE: '+name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
assert (root/'sitecustomize-precheck.json').is_file() and (root/'controller/controller.py').is_file()
with (root/'controller.log').open('wb') as log:
 proc=subprocess.Popen([sys.executable,str(root/'controller/controller.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
result={'project_id':'glm53-flash','unit':'r247','gateway_hostname':socket.gethostname(),'controller_pid':proc.pid,'controller_root':str(root),'detached_session':True,'stdin':'DEVNULL','model_nodes':[46,47],'forbidden_nodes_not_contacted':[14,15,16,17],'gateway_NPU_model_work':False,'frozen_files_verified':len(frozen),'source_payload_bytes':sum(len(v) for v in decoded.values()),'r246_gate':{'receipt_sha256':row['sha256'],'live_idle8':live}};(root/'submission.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))"""


if __name__ == "__main__":
    raise SystemExit(main())
