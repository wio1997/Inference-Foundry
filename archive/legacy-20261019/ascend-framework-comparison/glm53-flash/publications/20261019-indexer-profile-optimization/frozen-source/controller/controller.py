#!/usr/bin/env python3
"""R247 B-only whole-model profile controller (source only).

It reuses the R238 SGLang control closure and refuses to touch any resource
until the independent R247 review, the frozen source set and the actual R246
PASS/idle8 signoff all match.  There is no A framework, no A node and no second
performance change in this unit.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pathlib
import runpy
import subprocess
import sys
import time
import traceback

import freeze_policy

R = pathlib.Path(__file__).resolve().parent.parent
STAGE = R / "stage"
CLIENT = R / "client"
ANALYSIS = R / "analysis"
UNIT = "r247"
RUN_ROOT = "/data/wio/inference-foundry/glm53-flash/runs/20261019-r247-b-indexer-optimized-profile/B"
PRIVATE_ROOT = ("/data/wio/inference-foundry/glm53-flash/controllers/"
                "20261018-r185-native-aisbench-single-case-preflight/private-C32")
TOKENIZER_ROOT = ("/data/wio/inference-foundry/glm53-flash/controllers/"
                  "20261018-r185-native-aisbench-single-case-preflight/tokenizer")
PODS = {46: "sglang-container-access-46-20261013",
        47: "sglang-container-access-47-20261013"}
HOSTS = {46: "B8-15U-Atlas900RCKA3-46", 47: "B8-23U-Atlas900RCKA3-47"}
SGLANG_IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"
CLIENT_IMAGE = "sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83"
CLIENT_BASE = pathlib.Path("/data/wio/inference-foundry/glm53-flash/controllers/"
                           "20261018-r183-native-aisbench-client-private-cache")
GLOBAL_DEADLINE = None
EVIDENCE_MOUNT = "/run-evidence"
OBSERVER_PROFILE_SUBDIR = {"sglang": "profile-B"}
CLEANUP_TERMINAL_SECONDS = 300
B_CLEANUP_BOUND_SECONDS = 180
HOST_COLLECT_SECONDS = 900
R246_RUN_ROOT = ("/data/wio/inference-foundry/glm53-flash/runs/"
                 "20261019-r246-dynamic-rowstride-tp16-score-component")
R246_RECEIPT = R246_RUN_ROOT + "/host-receipt.json"
R246_CONTAINER = "if-glm53-flash-r246-rowstride-component"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def work_stage_deadline(seconds):
    candidate = time.monotonic() + max(0.0, float(seconds))
    if GLOBAL_DEADLINE is None:
        return candidate
    return min(candidate, GLOBAL_DEADLINE)


def cleanup_deadline(seconds):
    return time.monotonic() + max(0.0, float(seconds))


def client_wait_seconds(plan, deadline=None):
    budget = float(plan["budgets"]["cohort_seconds"]) + 180.0
    if deadline is not None:
        budget = min(budget, max(1.0, deadline - time.monotonic()))
    return budget


def download_deadline(plan):
    return work_stage_deadline(plan["budgets"]["single_controller_deadline_seconds"])


def load_review():
    path = R / "review.json"
    if not path.is_file():
        raise SystemExit("NOT_FROZEN: review.json absent; no resource action taken")
    review = json.loads(path.read_text())
    if (review.get("project_id"), review.get("unit"),
            review.get("kind")) != ("glm53-flash", UNIT,
                                    "R247_INDEPENDENT_FROZEN_SOURCE_REVIEW"):
        raise SystemExit("FROZEN_REVIEW_IDENTITY_INVALID")
    freeze_path = R / "freeze-input.json"
    if not freeze_path.is_file():
        raise SystemExit("NOT_FROZEN: freeze-input.json absent")
    freeze_sha = sha(freeze_path)
    if review.get("reviewed_freeze_input_sha256") != freeze_sha:
        raise SystemExit("FROZEN_REVIEW_FREEZE_INPUT_SHA_MISMATCH")
    freeze = json.loads(freeze_path.read_text())
    expected = freeze_policy.immutable_sha256(R)
    if freeze.get("files_sha256") != expected:
        raise SystemExit("FREEZE_INPUT_STALE: regenerate make_freeze_input.py")
    if review.get("files_sha256") != expected:
        raise SystemExit("FROZEN_REVIEW_INCOMPLETE_OR_MUTABLE")
    for rel, digest in expected.items():
        target = (R / rel).resolve()
        if not target.is_file():
            raise SystemExit("FROZEN_FILE_MISSING: " + rel)
        if sha(target) != digest:
            raise SystemExit("FROZEN_SHA_MISMATCH: " + rel)
    auth_path = R / "review-result.json"
    if not auth_path.is_file():
        raise SystemExit("NOT_AUTHORIZED: independent review-result.json absent")
    auth = json.loads(auth_path.read_text())
    authorization = auth.get("authorization") or {}
    if (auth.get("project_id"), auth.get("unit"), auth.get("kind"),
            auth.get("passed")) != ("glm53-flash", UNIT,
                                    "R247_INDEPENDENT_REVIEW_RESULT_AUTHORIZATION", True):
        raise SystemExit("REVIEW_RESULT_IDENTITY_INVALID")
    if auth.get("review_sha256") != sha(path):
        raise SystemExit("REVIEW_RESULT_REVIEW_SHA_MISMATCH")
    if auth.get("freeze_input_sha256") != freeze_sha:
        raise SystemExit("REVIEW_RESULT_FREEZE_INPUT_SHA_MISMATCH")
    if not (authorization.get("dispatch_allowed") is True
            and authorization.get("execute_once") is True
            and authorization.get("controller_entry") == "python3 controller/controller.py"):
        raise SystemExit("REVIEW_RESULT_NOT_AUTHORIZED")
    return review


def preflight_sitecustomize(plan):
    path = R / "sitecustomize-precheck.json"
    if not path.is_file():
        raise SystemExit("SITECUSTOMIZE_PRECHECK_PENDING")
    data = json.loads(path.read_text())
    if (data.get("schema"), data.get("kind"), data.get("actual"),
            data.get("project_id"), data.get("unit")) != (
            "r234/sitecustomize-precheck/v1",
            "READ_ONLY_NATIVE_SITECUSTOMIZE_PRECHECK", True, "glm53-flash", "r234"):
        raise SystemExit("SITECUSTOMIZE_PRECHECK_SCHEMA_INVALID")
    row = (data.get("images") or {}).get("B") or {}
    if row.get("image") != SGLANG_IMAGE:
        raise SystemExit("SITECUSTOMIZE_PRECHECK_IMAGE_MISMATCH_B")
    if row.get("framework") not in ("B", "sglang"):
        raise SystemExit("SITECUSTOMIZE_PRECHECK_FRAMEWORK_MISMATCH_B")
    if row.get("actual") is not True or row.get("passed") is not True:
        raise SystemExit("SITECUSTOMIZE_PRECHECK_NOT_PASSED_B")
    if row.get("decision") == "inject_observer":
        if row.get("native_present") is not False:
            raise SystemExit("SITECUSTOMIZE_PRECHECK_INJECT_CONTRADICTION_B")
    elif row.get("decision") == "chain_preserved":
        if row.get("native_present") is not True or (row.get("chain_test") or {}).get("rc") != 0:
            raise SystemExit("SITECUSTOMIZE_PRECHECK_CHAIN_UNPROVEN_B")
    else:
        raise SystemExit("SITECUSTOMIZE_PRECHECK_REFUSED_B")
    if data.get("passed") is not True:
        raise SystemExit("SITECUSTOMIZE_PRECHECK_MERGED_NOT_PASSED")
    return {"sha256": sha(path), "image": row["image"],
            "decision": row["decision"], "native_present": row["native_present"]}


def verify_r246_gate_local(gate, review_sha):
    """Pure local half of the actual R246 PASS/idle8 gate."""
    if not isinstance(gate, dict):
        raise SystemExit("MISSING_R246_ACTUAL_PASS_IDLE8_SIGNOFF")
    if (gate.get("project_id"), gate.get("unit"), gate.get("kind"),
            gate.get("actual"), gate.get("passed")) != (
            "glm53-flash", UNIT, "R246_ACTUAL_PASS_IDLE8_SIGNOFF", True, True):
        raise SystemExit("R246_GATE_IDENTITY_INVALID")
    if gate.get("r246_unit") != "r246":
        raise SystemExit("R246_GATE_UNIT_INVALID")
    receipt_sha = gate.get("r246_host_receipt_sha256")
    if not (isinstance(receipt_sha, str) and len(receipt_sha) == 64
            and all(c in "0123456789abcdef" for c in receipt_sha)):
        raise SystemExit("R246_GATE_RECEIPT_SHA_INVALID")
    launch_sha = gate.get("r246_launch_result_sha256")
    if launch_sha is not None and not (isinstance(launch_sha, str) and len(launch_sha) == 64):
        raise SystemExit("R246_GATE_LAUNCH_SHA_INVALID")
    idle = gate.get("idle8") or {}
    if idle.get("node46") is not True or idle.get("node47") is not True:
        raise SystemExit("R246_GATE_IDLE8_UNPROVEN")
    if gate.get("r247_review_sha256") != review_sha:
        raise SystemExit("R246_GATE_REVIEW_SHA_MISMATCH")
    authorization = gate.get("authorization") or {}
    if not (authorization.get("r247_dispatch_allowed") is True
            and authorization.get("execute_once") is True
            and authorization.get("exact_command") == "python3 dispatch.py"):
        raise SystemExit("R246_GATE_NOT_AUTHORIZED")
    return receipt_sha


def verify_r246_receipt(receipt, gate):
    """Validate the actual node-side R246 receipt, never a source claim."""
    if not isinstance(receipt, dict):
        raise SystemExit("R246_RECEIPT_MISSING")
    if receipt.get("unit") != "r246":
        raise SystemExit("R246_RECEIPT_UNIT_INVALID")
    if receipt.get("passed") is not True:
        raise SystemExit("R246_RECEIPT_NOT_PASSED")
    if receipt.get("rank_passed") != 16:
        raise SystemExit("R246_RECEIPT_RANK_PASSED_INVALID")
    if (receipt.get("stage_a") or {}).get("passed") is not True:
        raise SystemExit("R246_RECEIPT_STAGE_A_NOT_PASSED")
    if receipt.get("owned_container_verified") is not True:
        raise SystemExit("R246_RECEIPT_OWNED_CONTAINER_UNVERIFIED")
    if receipt.get("bounded_wait_timeout") is True:
        raise SystemExit("R246_RECEIPT_BOUNDED_WAIT_TIMEOUT")
    post = (receipt.get("post_inspect") or {}).get("stdout", "").strip()
    if (receipt.get("post_inspect") or {}).get("rc") != 0:
        raise SystemExit("R246_RECEIPT_POST_INSPECT_RC")
    if post != "false 0":
        raise SystemExit("R246_RECEIPT_CONTAINER_NOT_STOPPED")
    if (receipt.get("post_npu") or {}).get("rc") != 0:
        raise SystemExit("R246_RECEIPT_POST_NPU_RC")
    if (receipt.get("post_npu") or {}).get("stdout", "").count(
            "No running processes found") != 8:
        raise SystemExit("R246_RECEIPT_IDLE8_UNPROVEN")
    archive = receipt.get("archive") or {}
    if not isinstance(archive.get("files"), dict) or not archive["files"]:
        raise SystemExit("R246_RECEIPT_ARCHIVE_MISSING")
    if receipt.get("container") != R246_CONTAINER:
        raise SystemExit("R246_RECEIPT_CONTAINER_INVALID")
    if receipt.get("image") != SGLANG_IMAGE:
        raise SystemExit("R246_RECEIPT_IMAGE_INVALID")
    return {"passed": True, "rank_passed": 16,
            "stage_a_passed": True, "idle8": True,
            "container": receipt.get("container"),
            "image": receipt.get("image")}


def verify_r246_gate():
    gate_path = R / "r246-gate.json"
    if not gate_path.is_file():
        raise SystemExit("MISSING_R246_ACTUAL_PASS_IDLE8_SIGNOFF")
    gate = json.loads(gate_path.read_text())
    expected_sha = verify_r246_gate_local(gate, sha(R / "review.json"))
    code = """
import hashlib,json,pathlib
p=pathlib.Path(%r)
assert p.is_file(),'R246 receipt missing'
raw=p.read_bytes();receipt=json.loads(raw)
print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'receipt':receipt}))
""" % R246_RECEIPT
    q = remote(46, code, 60)
    if q["rc"] != 0:
        raise SystemExit("R246_RECEIPT_READ_FAILED: " + repr(q))
    row = json.loads(q["stdout"])
    if row["sha256"] != expected_sha:
        raise SystemExit("R246_RECEIPT_SHA_MISMATCH")
    verified = verify_r246_receipt(row["receipt"], gate)
    live = {}
    for node in (46, 47):
        q = remote(node, """
import json,socket,subprocess
q=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
print(json.dumps({'hostname':socket.gethostname(),'rc':q.returncode,
                  'idle':q.stdout.count('No running processes found')}))
""", 60)
        if q["rc"] != 0:
            raise SystemExit("R246_GATE_IDLE_READ_FAILED_%d" % node)
        live[node] = json.loads(q["stdout"])
        if live[node]["hostname"] != HOSTS[node] or live[node]["idle"] != 8:
            raise SystemExit("R246_GATE_NODE_NOT_IDLE8_%d" % node)
    return {"local_gate_sha256": sha(gate_path),
            "receipt_sha256": row["sha256"],
            "receipt": verified, "live_idle8": live}


def stage_files():
    base = STAGE / "B"
    out = {}
    for path in sorted(base.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out[str(path.relative_to(base))] = path
    for path in sorted((R / "observers/common").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out["observer/common/" + str(path.relative_to(R / "observers/common"))] = path
    for path in sorted((R / "observers/sglang").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out["observer/sglang/" + str(path.relative_to(R / "observers/sglang"))] = path
    out["observer/manifest-sglang.json"] = R / "observers/manifest-sglang.json"
    payload = {}
    for rel, path in out.items():
        payload[rel] = {"data": __import__("base64").b64encode(path.read_bytes()).decode(),
                        "sha256": sha(path)}
    return payload


def remote(node, code, timeout=60):
    cmd = ["kubectl", "exec", "-i", "-n", "default", PODS[node], "-c", "hostctl",
           "--", "nsenter", "-t", "1", "-m", "--", "python3", "-"]
    q = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=timeout)
    return {"rc": q.returncode, "stdout": q.stdout, "stderr": q.stderr}


def read_json(node, path):
    q = remote(node, "import pathlib,json;p=pathlib.Path(" + repr(path)
               + ");print(p.read_text() if p.exists() else 'null')", 60)
    if q["rc"] != 0:
        raise RuntimeError(("READ_FAIL", node, path, q))
    return json.loads(q["stdout"])


def signal(node, path, obj):
    code = ("import pathlib,json;p=pathlib.Path(" + repr(path)
            + ");assert p.parent.is_dir();t=p.with_suffix('.tmp');t.write_text("
            + repr(json.dumps(obj)) + ");t.replace(p)")
    q = remote(node, code, 60)
    if q["rc"] != 0:
        raise RuntimeError(("SIGNAL_FAIL", node, path, q))


def root_for(framework):
    if framework != "sglang":
        raise ValueError("R247 is B-only")
    return RUN_ROOT


def host_path_for_container_path(path):
    mount = EVIDENCE_MOUNT.rstrip("/")
    value = "/" + str(path).lstrip("/")
    if value != mount and not value.startswith(mount + "/"):
        raise ValueError("path is outside the host-probe evidence mount: " + value)
    return RUN_ROOT + value[len(mount):]


def observer_status_path(namespace):
    profile_root = EVIDENCE_MOUNT + "/" + OBSERVER_PROFILE_SUBDIR["sglang"]
    return host_path_for_container_path(
        profile_root + "/observer-status-" + namespace + ".json")


def wait_file(node, name, deadline, containers=()):
    while time.monotonic() < deadline:
        value = read_json(node, RUN_ROOT + "/" + name)
        if value is not None:
            return value
        for container in containers:
            q = remote(node, "import subprocess;print(subprocess.run(['docker','inspect',"
                       + repr(container) + ",'--format','{{.State.Running}} {{.State.ExitCode}}'],"
                       "capture_output=True,text=True).stdout)", 30)
            if q["rc"] == 0 and q["stdout"].strip().startswith("false"):
                raise RuntimeError(("CONTAINER_EXITED_BEFORE", node, container, q["stdout"]))
        time.sleep(3)
    raise TimeoutError(("WAIT_FILE", node, name))


def wait_terminal_status(node, deadline):
    while time.monotonic() < deadline:
        value = read_json(node, RUN_ROOT + "/host-status.json")
        if value is not None:
            post = value.get("post_inspect") or {}
            if post or value.get("cleanup_error"):
                return value
        time.sleep(2)
    raise TimeoutError(("TERMINAL_STATUS", node))


def render_host_probe(node, files):
    template = (R / "controller/host_probe_sglang.py").read_text()
    launch = (R / "controller/launch_command.py").read_text()
    export = (R / "controller/profile_export.py").read_text()
    return (template.replace("__NODE__", str(node))
            .replace("__FILES__", repr(files))
            .replace("__LAUNCH_COMMAND__", launch)
            .replace("__PROFILE_EXPORT__", export))


def start_host_probe(node):
    code = render_host_probe(node, stage_files())
    q = remote(node, code, 33500)
    (R / "results" / ("host-sglang-" + str(node) + ".json")).write_text(
        json.dumps(q, indent=2) + "\n")
    if q["rc"] != 0:
        raise RuntimeError(("HOST_PROBE_EXIT", node, q))
    return json.loads(q["stdout"])


def preflight_sources():
    manifest = json.loads((STAGE / "manifest.json").read_text())["B"]
    if len(manifest) != 55:
        raise RuntimeError(("STAGE_COUNT", len(manifest)))
    for rel, row in manifest.items():
        path = STAGE / "B" / rel
        if sha(path) != row["sha256"]:
            raise RuntimeError(("STAGE_SHA_MISMATCH", rel))
        if "A" in pathlib.PurePosixPath(rel).parts:
            raise RuntimeError(("A_PART_IN_B_STAGE", rel))
    obs = json.loads((R / "observers/manifest-sglang.json").read_text())
    for rel, expected in obs["files"].items():
        if sha(R / "observers" / rel) != expected:
            raise RuntimeError(("OBSERVER_SHA_MISMATCH", rel))
    combined = json.loads((STAGE / "B/combined-manifest.json").read_text())
    if combined.get("status") != "R247_CANDIDATE_DELTA_NOT_ORIGINAL_46":
        raise RuntimeError("COMBINED_MANIFEST_STATUS")
    if len(combined.get("files", [])) != 47:
        raise RuntimeError(("COMBINED_MANIFEST_COUNT", len(combined.get("files", []))))
    return {"stage_files": len(manifest), "observer_files": len(obs["files"]),
            "overlay_files": len(combined["files"])}


def live_preflight(plan):
    out = {"kubectl": {}, "nodes": {}}
    for node in (46, 47):
        q = subprocess.run(["kubectl", "get", "pod", "-n", "default", PODS[node],
                            "-o", "jsonpath={.status.hostIP}"],
                           capture_output=True, text=True, timeout=30)
        if q.returncode != 0 or q.stdout.strip() != "172.16.12." + str(node):
            raise RuntimeError(("POD_HOST_MISMATCH", node, q.stdout, q.stderr))
        out["kubectl"][node] = q.stdout.strip()
    for node in (46, 47):
        ports = ([19030, 19031, 19032, 19024] if node == 46
                 else [19030, 19031] + list(range(19100, 19116)))
        code = """
import json,socket,pathlib,subprocess,os
node=%d
ports=%r
image=%r
out={'node':node,'hostname':socket.gethostname()}
q=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
out['npu_idle_count']=q.stdout.count('No running processes found');out['npu_rc']=q.returncode
out['ports_free']={}
for p in ports:
 try:
  s=socket.socket();s.bind(('0.0.0.0',p));s.close();out['ports_free'][str(p)]=True
 except OSError as e:
  out['ports_free'][str(p)]=False;out.setdefault('port_errors',{})[str(p)]=repr(e)
q=subprocess.run(['docker','ps','-a','--format','{{.Names}}'],capture_output=True,text=True)
out['owned_containers']=[x for x in q.stdout.splitlines() if x.startswith('if-glm53-flash-r247-')]
out['image_present']=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True).stdout.strip()==image
out['mem_available_bytes']=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
st=os.statvfs('/data/wio');out['profile_fs_free_bytes']=st.f_bavail*st.f_frsize
print(json.dumps(out))
""" % (node, ports, SGLANG_IMAGE)
        q = remote(node, code, 120)
        if q["rc"] != 0:
            raise RuntimeError(("LIVE_PREFLIGHT_EXIT", node, q))
        row = json.loads(q["stdout"])
        if row["hostname"] != HOSTS[node]:
            raise RuntimeError(("HOSTNAME_MISMATCH", node, row["hostname"]))
        if row["npu_idle_count"] != 8 or not row["image_present"]:
            raise RuntimeError(("NODE_NOT_IDLE_OR_IMAGE", node, row))
        if row["owned_containers"] or not all(row["ports_free"].values()):
            raise RuntimeError(("NODE_BUSY", node, row))
        if row["profile_fs_free_bytes"] < 16 * 1024 ** 3:
            raise RuntimeError(("PROFILE_FS_LOW", node, row["profile_fs_free_bytes"]))
        out["nodes"][node] = row
    (R / "results/preflight.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def metrics_idle(framework):
    if framework != "sglang":
        raise ValueError("R247 is B-only")
    nodes_ports = [(46, 19030), (47, 19030)]
    import urllib.request
    for node, port in nodes_ports:
        with urllib.request.urlopen("http://172.16.12.%d:%d/metrics" % (node, port),
                                    timeout=5) as response:
            raw = response.read().decode()
        for name in ("sglang:num_running_reqs", "sglang:num_queue_reqs"):
            values = [float(line.split()[-1]) for line in raw.splitlines()
                      if line and not line.startswith("#")
                      and ((name + "{") in line.split()[0] or line.startswith(name + " "))]
            if not values or any(v != 0 for v in values):
                return False, {"node": node, "port": port, "metric": name, "values": values}
        for name in ("sglang:num_prefill_bootstrap_queue_reqs",
                     "sglang:num_prefill_inflight_queue_reqs",
                     "sglang:num_decode_prealloc_queue_reqs",
                     "sglang:num_decode_transfer_queue_reqs"):
            values = [float(line.split()[-1]) for line in raw.splitlines()
                      if line and not line.startswith("#")
                      and ((name + "{") in line.split()[0] or line.startswith(name + " "))]
            if values and any(v != 0 for v in values):
                return False, {"node": node, "port": port, "metric": name,
                               "values": values}
    return True, {"endpoints": nodes_ports}


def leader_namespaces(framework):
    if framework != "sglang":
        raise ValueError("R247 is B-only")
    return ["dp0-rank0"]


def wait_observer(framework, deadline):
    namespaces = leader_namespaces(framework)
    rows = {}
    while time.monotonic() < deadline:
        rows = {}
        for namespace in namespaces:
            value = read_json(46, observer_status_path(namespace))
            if value is not None:
                rows[namespace] = value
        if len(rows) == len(namespaces) and all(
                (row.get("policy") or {}).get("complete") or row.get("error")
                for row in rows.values()):
            return rows
        time.sleep(3)
    if rows:
        return rows
    raise TimeoutError(("OBSERVER_STATUS", list(rows)))


def validate_observer(framework, rows, windows):
    out = {"framework": framework, "sampled_rank_namespaces": leader_namespaces(framework),
           "leaders": {}, "valid": True, "errors": []}
    for namespace, status in rows.items():
        row = {"rank_namespace": status.get("rank_namespace"),
               "dp_rank": status.get("dp_rank"),
               "rank": status.get("rank"),
               "installed": status.get("installed"),
               "complete": bool(status.get("policy", {}).get("complete")),
               "sessions_started": status.get("profile_sessions_started"),
               "sessions_stopped": status.get("profile_sessions_stopped"),
               "error": status.get("error"),
               "windows": status.get("policy", {}).get("windows", {})}
        policy = status.get("policy") or {}
        if row["rank_namespace"] != namespace:
            out["errors"].append("%s rank namespace mismatch %r" % (
                namespace, row["rank_namespace"]))
        if row["dp_rank"] != 0:
            out["errors"].append("%s sampled dp_rank %r" % (namespace, row["dp_rank"]))
        if not row["installed"]:
            out["errors"].append("%s observer not installed" % namespace)
        identity = status.get("install_identity") or {}
        observer_file = R / "observers/sglang/r234_sglang_observer.py"
        if identity.get("observer_sha256") != sha(observer_file):
            out["errors"].append("%s observer file SHA mismatch" % namespace)
        if not (identity.get("target") or {}).get("sha256"):
            out["errors"].append("%s native target SHA missing" % namespace)
        row["sitecustomize_chain"] = status.get("chained_sitecustomize")
        if row["error"]:
            out["errors"].append("%s observer error %r" % (namespace, row["error"]))
        if row["sessions_started"] != 3 or row["sessions_stopped"] != 3:
            out["errors"].append("%s sessions %r/%r" % (
                namespace, row["sessions_started"], row["sessions_stopped"]))
        shapes = policy.get("observed_prefill_batch_shapes") or {}
        single_rows = sum(count for key, count in shapes.items()
                          if key.startswith("rows=1;"))
        row["observed_prefill_batch_shapes"] = shapes
        row["single_row_prefill_batches"] = single_rows
        row["mixed_batches_skipped"] = policy.get("mixed_batches_skipped")
        row["expected_active"] = policy.get("expected_active")
        if policy.get("expected_active") != 1:
            out["errors"].append("%s expected_active %r != 1" % (
                namespace, policy.get("expected_active")))
        if single_rows <= 0:
            out["errors"].append("%s no observed single-row prefill batch" % namespace)
        for window in windows:
            got = row["windows"].get(window["name"])
            if not got or not got.get("valid") or got.get("captured_forwards") != window["forwards"]:
                out["errors"].append("%s window %s invalid %r" % (
                    namespace, window["name"], got))
                continue
            if len(got.get("context_ends") or []) != window["forwards"]:
                out["errors"].append("%s window %s missing context ends" % (
                    namespace, window["name"]))
            if len(got.get("query_tokens") or []) != window["forwards"]:
                out["errors"].append("%s window %s missing query tokens" % (
                    namespace, window["name"]))
            for ends, queries in zip(got.get("context_ends") or [],
                                     got.get("query_tokens") or []):
                if len(ends) != 1 or any(
                        type(end) is not int or not (window["lo"] <= end <= window["hi"])
                        for end in ends):
                    out["errors"].append("%s window %s context %r outside %r" % (
                        namespace, window["name"], ends, [window["lo"], window["hi"]]))
                if any(type(q) is not int or q <= 0 for q in queries):
                    out["errors"].append("%s window %s query tokens %r invalid" % (
                        namespace, window["name"], queries))
        out["leaders"][namespace] = row
    if set(out["leaders"]) != set(leader_namespaces(framework)):
        out["errors"].append("sampled leader set %r" % sorted(out["leaders"]))
    out["valid"] = not out["errors"]
    return out


def _required_downloads(host_status):
    export = host_status.get("profile_export") or {}
    required = []
    for row in (export.get("windows") or {}).values():
        for matches in (row.get("required") or {}).values():
            required.extend(matches)
    for matches in (export.get("shared") or {}).values():
        required.extend(matches)
    return sorted(set(required))


def download_profiles(node, host_status, deadline, plan):
    index = host_status.get("profile_index") or {}
    files = index.get("files") or {}
    wanted = []
    for rel, meta in files.items():
        base = rel.rsplit("/", 1)[-1]
        if base in ("operator_details.csv", "kernel_details.csv", "step_trace_time.csv",
                    "trace_view.json", "profiler_metadata.json"):
            wanted.append((rel, meta))
        elif base.startswith("profiler_info") and base.endswith(".json"):
            wanted.append((rel, meta))
        elif (base.startswith("forward-metadata-")
              or base.startswith("observer-status-")
              or base.startswith("observer-install")
              or base.startswith("observer-sitecustomize")
              or base.startswith("observer-patch-error")):
            wanted.append((rel, meta))
    wanted.sort(key=lambda item: item[1]["bytes"])
    cap = int(plan["budgets"]["max_download_bytes"])
    total = 0
    out = {"downloaded": {}, "skipped": {}, "bytes": 0, "deadline": False,
           "required": _required_downloads(host_status), "required_missing": [],
           "complete": False}
    for rel, meta in wanted:
        if time.monotonic() > deadline:
            out["deadline"] = True
            break
        if not meta.get("sha256"):
            out["skipped"][rel] = {"bytes": meta["bytes"], "sha256": None,
                                   "reason": "host_hash_unavailable"}
            continue
        if total + meta["bytes"] > cap:
            out["skipped"][rel] = {"bytes": meta["bytes"], "sha256": meta["sha256"],
                                   "reason": "download_byte_cap"}
            continue
        target = R / "artifacts/sglang" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        code = """
import pathlib,base64
p=pathlib.Path(%r)
f=p.open('rb');f.seek(%d);print(base64.b64encode(f.read(%d)).decode())
"""
        with target.open("wb") as stream:
            offset = 0
            while offset < meta["bytes"]:
                size = min(1024 * 1024, meta["bytes"] - offset)
                q = remote(node, code % (RUN_ROOT + "/" + rel, offset, size), 120)
                if q["rc"] != 0:
                    raise RuntimeError(("DOWNLOAD_FAIL", rel, q["stderr"]))
                chunk = __import__("base64").b64decode(q["stdout"])
                if len(chunk) != size:
                    raise RuntimeError(("DOWNLOAD_SIZE", rel, len(chunk), size))
                stream.write(chunk)
                offset += size
        if target.stat().st_size != meta["bytes"] or sha(target) != meta["sha256"]:
            raise RuntimeError(("DOWNLOAD_SHA", rel))
        out["downloaded"][rel] = {"bytes": meta["bytes"], "sha256": meta["sha256"]}
        total += meta["bytes"]
        out["bytes"] = total
    out["required_missing"] = [rel for rel in out["required"]
                               if rel not in out["downloaded"]]
    out["complete"] = (not out["deadline"] and not out["required_missing"]
                       and (host_status.get("profile_export") or {}).get("complete") is True)
    (R / "results/profile-download-sglang.json").write_text(
        json.dumps(out, indent=2) + "\n")
    return out


def run_client(plan, deadline=None):
    name = "if-glm53-flash-r247-client-sglang"
    output = R / "results/client-sglang.json"
    remote_output = "/current/results/client-sglang.json"
    model = "glm53-flash-r201"
    cmd = ["docker", "run", "-d", "--pull=never", "--name", name,
           "--label", "project_id=glm53-flash", "--label", "run_id=r247",
           "--label", "namespace=glm53-flash-r247", "--label", "owner=zcode",
           "--network=host", "--read-only", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--cpus", "8",
           "--memory", "16g", "--pids-limit", "512", "--shm-size", "512m",
           "--tmpfs", "/tmp:rw,nosuid,size=2g",
           "-v", str(CLIENT_BASE) + ":/work:ro",
           "-v", str(R) + ":/current:rw",
           "-v", str(CLIENT / "") + ":/current/client:ro",
           "-v", PRIVATE_ROOT + ":/current/private-C32:ro",
           "-v", TOKENIZER_ROOT + ":/current/tokenizer:ro",
           "-w", "/current",
           "-e", "PYTHONPATH=/current/client",
           "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
           "-e", "HF_HUB_OFFLINE=1",
           "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "CUDA_VISIBLE_DEVICES=",
           "-e", "ASCEND_RT_VISIBLE_DEVICES=",
           "-e", "OMP_NUM_THREADS=2",
           "-e", "OPENBLAS_NUM_THREADS=1",
           "--entrypoint", "/work/venv/bin/python", CLIENT_IMAGE,
           "/current/client/r234_client.py",
           "--framework", "sglang", "--model", model, "--run-id", "r247",
           "--private-root", "/current/private-C32",
           "--tokenizer-root", "/current/tokenizer",
           "--output", remote_output,
           "--http-seconds", str(plan["budgets"]["http_seconds"]),
           "--cohort-seconds", str(plan["budgets"]["cohort_seconds"])]
    out = {"name": name, "command": cmd}
    review = CLIENT_BASE / "review.json"
    if not review.is_file() or sha(review) != "58efd7a89735b8b1286ddb26ad71cfd232f1cf04055a5b369ea3b5f2e02183c8":
        return {"name": name, "error": "client_container_base_identity_mismatch",
                "command": cmd}
    existing = subprocess.run(["docker", "inspect", name], capture_output=True,
                              text=True, timeout=20)
    if existing.returncode == 0:
        return {"name": name, "error": "owned_client_container_exists", "command": cmd}
    created = False
    try:
        q = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out["launch"] = {"rc": q.returncode, "stdout": q.stdout, "stderr": q.stderr}
        if q.returncode != 0:
            raise RuntimeError(("CLIENT_LAUNCH", q.stderr))
        created = True
        wait = subprocess.run(["docker", "wait", name], capture_output=True,
                              text=True, timeout=client_wait_seconds(plan, deadline))
        out["wait"] = {"rc": wait.returncode, "stdout": wait.stdout,
                       "stderr": wait.stderr}
        inspect = subprocess.run(["docker", "inspect", name, "--format",
                                  "{{.Image}} {{.State.ExitCode}}"],
                                 capture_output=True, text=True, timeout=30)
        out["inspect"] = {"rc": inspect.returncode, "stdout": inspect.stdout,
                          "stderr": inspect.stderr}
        if output.is_file():
            out["result"] = json.loads(output.read_text())
        else:
            out["error"] = "client_result_missing"
    finally:
        if created:
            subprocess.run(["docker", "stop", "-t", "15", name], capture_output=True,
                           text=True, timeout=45)
    return out


def run_framework(plan):
    framework = "sglang"
    p_node, d_node = 46, 47
    result = {"framework": framework, "generation_POST_attempted": 0,
              "HTTP_terminal": 0, "control_GET": 0, "formal_requests": 0,
              "cold_pairs": 0,
              "sampled_rank_namespaces": leader_namespaces(framework),
              "profile": {}, "errors": []}
    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for node in (p_node, d_node):
            futures[node] = pool.submit(start_host_probe, node)
        try:
            for node in (p_node, d_node):
                wait_file(node, "native-preflight-ready.json", work_stage_deadline(300))
            for node in (p_node, d_node):
                wait_file(node, "preflight.json", work_stage_deadline(150))
            for node in (p_node, d_node):
                signal(node, RUN_ROOT + "/start-model.json",
                       {"project_id": "glm53-flash", "run_id": "r247",
                        "both_native_preflights_passed": True})
            deadline = work_stage_deadline(plan["budgets"]["service_startup_seconds"])
            for node in (p_node, d_node):
                wait_file(node, "service-ready.json", deadline)
            idle_deadline = work_stage_deadline(120)
            idle = False
            evidence = {}
            while time.monotonic() < idle_deadline:
                idle, evidence = metrics_idle(framework)
                result["control_GET"] += len(evidence.get("endpoints", []))
                if idle:
                    break
                time.sleep(3)
            result["initial_idle"] = {"idle": idle, "evidence": evidence}
            if not idle:
                raise RuntimeError(("NOT_IDLE_BEFORE_COHORT", framework, evidence))
            arm_path = RUN_ROOT + "/profile-B/ARMED"
            signal(p_node, arm_path, {"project_id": "glm53-flash", "run_id": "r247",
                                      "armed_before_client": True})
            result["observer_armed"] = True
            result["client"] = run_client(
                plan, work_stage_deadline(plan["budgets"]["cohort_seconds"] + 180))
            client = result["client"].get("result") or {}
            result["generation_POST_attempted"] = client.get("generation_POST_attempted", 0)
            result["HTTP_terminal"] = client.get("HTTP_terminal", 0)
            result["cold_pairs"] = client.get("cold_pairs", 0)
            result["client_passed"] = bool(client.get("passed"))
            if not result["client_passed"]:
                result["errors"].append("client_not_passed")
            idle_deadline = work_stage_deadline(120)
            post_idle = False
            post_evidence = {}
            while time.monotonic() < idle_deadline:
                post_idle, post_evidence = metrics_idle(framework)
                result["control_GET"] += len(post_evidence.get("endpoints", []))
                if post_idle:
                    break
                time.sleep(3)
            result["post_client_idle"] = {"idle": post_idle,
                                          "evidence": post_evidence}
            if not post_idle:
                result["errors"].append("post_client_not_idle")
            observer_rows = wait_observer(framework, work_stage_deadline(120))
            result["observer"] = validate_observer(framework, observer_rows,
                                                   plan["windows"])
            if not result["observer"]["valid"]:
                result["errors"].append("observer_incomplete")
            time.sleep(2)
        except BaseException as exc:
            result["errors"].append(repr(exc))
            result["traceback"] = traceback.format_exc()
        finally:
            try:
                cleanup_mod = runpy.run_path(
                    str(STAGE / "B/flash_pd_writer_first_cleanup.py"))
                cleanup = cleanup_mod["cleanup"]
                result["writer_first_cleanup"] = cleanup(
                    lambda node, name, obj: signal(node, RUN_ROOT + "/" + name, obj),
                    lambda: read_json(46, RUN_ROOT + "/host-status.json"),
                    run_id="r247", image=SGLANG_IMAGE,
                    container="if-glm53-flash-r247-b46",
                    bound=B_CLEANUP_BOUND_SECONDS)
                if not result["writer_first_cleanup"].get("decode_stop_permitted"):
                    result["errors"].append("writer_stop_unproven_decode_retained")
                for node in (p_node, d_node):
                    try:
                        result["nodes"] = result.get("nodes", {})
                        result["nodes"][node] = futures[node].result(
                            timeout=HOST_COLLECT_SECONDS)
                    except BaseException as exc:
                        result["errors"].append("host_collect_%d %r" % (node, exc))
            except BaseException as exc:
                result["errors"].append("cleanup " + repr(exc))
        try:
            host_status = read_json(p_node, RUN_ROOT + "/host-status.json") or {}
            result["host_status_present"] = bool(host_status)
            result["profile_export"] = host_status.get("profile_export") or {
                "complete": False, "error": "missing_profile_export"}
            if result["profile_export"].get("complete") is not True:
                result["errors"].append("profile_export_incomplete")
            p_sdma = host_status.get("sdma_log") or {}
            d_status = (result.get("nodes") or {}).get(d_node) or {}
            d_host = read_json(d_node, RUN_ROOT + "/host-status.json") or {}
            d_sdma = d_host.get("sdma_log") or {}
            result["sdma_log"] = {"P": p_sdma, "D": d_sdma}
            sdma_ok = (p_sdma.get("all_initialized_protocol_sdma") is True
                       and d_sdma.get("all_initialized_protocol_sdma") is True
                       and p_sdma.get("expected_role_initialized") is True
                       and d_sdma.get("expected_role_initialized") is True
                       and int(p_sdma.get("push_complete") or 0) >= 1
                       and int(p_sdma.get("failed_calls") or 0) == 0
                       and int(d_sdma.get("failed_calls") or 0) == 0)
            result["sdma_ok"] = bool(sdma_ok)
            if not sdma_ok:
                result["errors"].append("sdma_native_transfer_unproven")
            result["download"] = download_profiles(
                p_node, host_status, download_deadline(plan), plan)
            if not result["download"].get("complete"):
                result["errors"].append("profile_download_incomplete")
        except BaseException as exc:
            result["errors"].append("download " + repr(exc))
    result["status"] = "PASS" if (result.get("client_passed")
                                  and (result.get("observer") or {}).get("valid")
                                  and result.get("sdma_ok")
                                  and not result["errors"]) else "FAIL"
    return result


def analyze(plan):
    artifacts = R / "artifacts/sglang"
    output = R / "results/sglang-analysis.json"
    remaining = max(1.0, min(120.0, work_stage_deadline(120) - time.monotonic()))
    cmd = [sys.executable, str(ANALYSIS / "run_analysis.py"),
           "--artifacts", str(artifacts), "--windows",
           json.dumps(plan["windows"]), "--output", str(output)]
    try:
        q = subprocess.run(cmd, capture_output=True, text=True, timeout=remaining)
        result = {"rc": q.returncode, "stdout": q.stdout, "stderr": q.stderr,
                  "output": str(output.relative_to(R)) if output.is_file() else None}
    except subprocess.TimeoutExpired as exc:
        return {"rc": None, "status": "TIMEOUT", "error": repr(exc),
                "output": str(output.relative_to(R)) if output.is_file() else None}
    if output.is_file():
        try:
            result["status"] = json.loads(output.read_text()).get("status")
        except BaseException as exc:
            result["status_error"] = repr(exc)
    return result


def compute_status(result, analysis, plan):
    reasons = []
    if result.get("generation_POST_attempted") != plan["requests"]["generation_POST_max"]:
        reasons.append("generation_POST")
    if result.get("HTTP_terminal") != plan["requests"]["generation_POST_max"]:
        reasons.append("HTTP_terminal")
    if result.get("cold_pairs") != plan["requests"]["cold_pairs"]:
        reasons.append("cold_pairs")
    if result.get("formal_requests") != 0:
        reasons.append("formal_requests")
    if result.get("status") != "PASS":
        reasons.append("framework_status")
    if (result.get("profile_export") or {}).get("complete") is not True:
        reasons.append("profile_export")
    if (result.get("download") or {}).get("complete") is not True:
        reasons.append("profile_download")
    if (analysis or {}).get("status") != "COMPUTED":
        reasons.append("analysis_not_computed")
    return ("PASS" if not reasons else "FAIL"), reasons


def main():
    global GLOBAL_DEADLINE
    review = load_review()
    plan = json.loads((R / "controller/profile_plan.json").read_text())
    sitecustomize = preflight_sitecustomize(plan)
    r246_gate = verify_r246_gate()
    manifests = preflight_sources()
    GLOBAL_DEADLINE = time.monotonic() + plan["budgets"]["single_controller_deadline_seconds"]
    (R / "results").mkdir(parents=True, exist_ok=True)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    os.close(os.open(R / "controller.once.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    out = {"project_id": "glm53-flash", "unit": UNIT,
           "source_only": False, "formal_requests": 0,
           "performance_baseline": False,
           "profile_time_excluded_from_throughput": True,
           "candidate_delta": True,
           "original_46_overlay_unchanged": False,
           "review_sha256": sha(R / "review.json"),
           "sitecustomize_precheck": sitecustomize,
           "r246_gate": r246_gate,
           "source_files": manifests,
           "profile_scope": {
               "B_sampled": "TP0 only (TP16 DP1)",
               "leader_view_only": True,
               "expected_active_native_rows": 1,
               "mixed_batches": "counted from real metadata and skipped, never forced",
           },
           "frameworks": {}, "errors": []}
    try:
        live_preflight(plan)
        out["frameworks"]["sglang"] = run_framework(plan)
        out["frameworks"]["sglang"]["analysis"] = analyze(plan)
        status, reasons = compute_status(out["frameworks"]["sglang"],
                                         out["frameworks"]["sglang"]["analysis"], plan)
        out["status_reasons"] = reasons
        out["status"] = status
        (R / "results/receipt.json").write_text(json.dumps(out, indent=2) + "\n")
    except BaseException as exc:
        out["errors"].append("controller " + repr(exc))
        out["status"] = "FAIL"
        (R / "results/receipt.json").write_text(json.dumps(out, indent=2) + "\n")
        return 1
    out["generation_POST_total"] = out["frameworks"]["sglang"].get(
        "generation_POST_attempted", 0)
    out["HTTP_terminal_total"] = out["frameworks"]["sglang"].get("HTTP_terminal", 0)
    out["cold_pairs_total"] = out["frameworks"]["sglang"].get("cold_pairs", 0)
    out["formal_requests"] = 0
    out["accepted_baseline"] = None
    (R / "results/receipt.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"],
                      "POST": out["generation_POST_total"],
                      "HTTP": out["HTTP_terminal_total"],
                      "cold_pairs": out["cold_pairs_total"],
                      "reasons": out["status_reasons"]}))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
