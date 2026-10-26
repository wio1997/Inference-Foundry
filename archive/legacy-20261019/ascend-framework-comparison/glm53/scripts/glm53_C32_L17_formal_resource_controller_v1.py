"""DRAFT_PENDING_PRIMARY_REVIEW: L17 formal C32 resource controller.

Derived from the accepted L12C controller.  All runtime identity comes from the
new L17 mixed identity renderer and the immutable P/D capacity records; no old
D container ID or old stage is embedded.  Only C32 is allowed.  ``plans()`` is
local-only and renders the four precheck programs for primary review.
``execute()`` requires actual frozen readiness, startup, functional decision/
review and formal actual-review SHAs, then performs the same accepted prefix,
gate, CPU-stage and RPC orchestration once.
"""
import ast
import base64
import hashlib
import importlib.util
import json
import re
import shlex
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]

CPU_STAGE = r'''import pathlib,subprocess,json,base64,hashlib,socket
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
image='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
r=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert r.returncode==0 and r.stdout.strip()==image
assets=pathlib.Path('/data/wio/aisbench-offline-43-20261014/benchmark/ais_bench/benchmark')
for item in AIS_IDENTITY['files']:assert hashlib.sha256((assets/item['path']).read_bytes()).hexdigest()==item['sha256']
data=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts')/('C'+str(CONCURRENCY));assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()==MANIFEST_SHA
root=pathlib.Path(CPU_ROOT);assert not root.exists(),'No formal host-stage overwrite/replay';root.mkdir(parents=True)
for name,item in FILES.items():
 assert pathlib.Path(name).name==name
 raw=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['sha256'];(root/name).write_bytes(raw)
raw=json.dumps(SETTINGS,indent=2).encode()+b'\n';(root/'host-settings.json').write_bytes(raw)
print(json.dumps({'project_id':'glm53','node':43,'root':str(root),'settings_sha256':hashlib.sha256(raw).hexdigest(),'files_sha256':{k:v['sha256'] for k,v in FILES.items()},'model_requests':0,'SDK_init_calls':0}))
'''


def load(name, relative, sha):
    return contract.load_module(name, relative, sha)


def render(template, values):
    pattern = r"\b(?:" + "|".join(re.escape(key) for key in values) + r")\b"
    result = re.sub(pattern, lambda match: repr(values[match[0]]), template)
    ast.parse(result)
    return result


def encoded(files):
    out = {}
    for name, path in files.items():
        assert Path(name).name == name
        raw = Path(path).read_bytes()
        out[name] = {"base64": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest()}
    return out


def _settings_sha(settings_path):
    path = ROOT / settings_path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structural(settings):
    assert settings["project_id"] == contract.PROJECT
    assert settings["framework"] == "sglang" and settings["concurrency"] == 32
    assert settings["run_id"] == contract.FORMAL_RUN
    assert settings["namespace"] == contract.FORMAL_NAMESPACE
    assert settings.get("profiling") is False and settings.get("retries") == 0
    assert settings.get("cache_policy") == "do_not_clear_L1_L2_after_warm"
    assert settings["p_capacity"] == contract.P_CAPACITY
    assert settings["p_capacity_sha256"] == contract.P_CAPACITY_SHA
    assert settings["d_capacity"] == contract.D_CAPACITY
    assert settings["d_capacity_sha256"] == contract.D_CAPACITY_SHA
    assert settings["model_image"] == contract.IMAGE
    assert settings["meta_container_id"] == contract.META_CID
    assert settings["model_identity_helper"] == contract.MIXED_IDENTITY
    assert settings["model_identity_sha256"] == contract.MIXED_IDENTITY_SHA
    assert settings["gate_renderer_sha256"] == contract.FORMAL_GATE_SHA
    assert settings["prefix_transfer_helper"] == contract.PREFIX_TRANSFER
    assert settings["prefix_transfer_sha256"] == contract.PREFIX_TRANSFER_SHA
    assert settings["input_manifest_sha256"] == contract.FORMAL_MANIFEST_SHA
    assert settings["native_client_config_path"] == contract.NATIVE_CONFIG
    assert settings["native_client_config_sha256"] == contract.NATIVE_CONFIG_SHA
    assert settings["expected_sdk_root"] == contract.EXPECTED_SDK_ROOT
    assert settings["meta_service_url"] == contract.META_SERVICE_URL
    assert settings["config_store_url"] == contract.CONFIG_STORE_URL
    assert settings["max_dram_size"] == contract.MAX_DRAM_SIZE
    assert settings["shared_remote_root"] == contract.P_REMOTE_ROOT
    assert settings["shared_run_id"] == contract.P_REMOTE_ROOT.rsplit("/", 1)[-1]
    assert settings["backend_HTTP_attempts_planned"] == contract.FORMAL_OFFERED_BACKEND
    assert settings["formal_requests"] == 128
    assert settings["warmup_logical_requests"] == 32
    assert settings["successful_output_tokens_if_all_pass"] == contract.FORMAL_OUTPUT_IF_ALL
    p_capacity, d_capacity = contract.check_capacity_pair()
    expected = contract.combined_providers()
    assert set(settings["runtime_containers"]) == {str(node) for node in contract.ALL_NODES}
    for node in contract.ALL_NODES:
        row = settings["runtime_containers"][str(node)]
        src = expected[str(node)]
        assert row["container_id"] == src["container_id"]
        assert row["name"] == src["name"]
        assert row["local_service_id"] == src["local_service_id"]
        assert row["dram_GiB"] == src["dram_GiB"]
        assert row["remote_root"] == src["remote_root"]
        assert row["run_id"] == src["run_id"]
    return p_capacity, d_capacity


def plans(settings, settings_path=None):
    p_capacity, d_capacity = _structural(settings)
    identity = load("l17_mixed_identity", contract.MIXED_IDENTITY, contract.MIXED_IDENTITY_SHA)
    precheck_codes = {}
    for node in contract.ALL_NODES:
        port = 30500 if node in contract.P_NODES else 30600
        code = identity.render(node, p_capacity, d_capacity, "candidate")
        code += r'''
import urllib.request,pathlib
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(URL,timeout=10) as response:assert response.status==200;raw=response.read(8388609);assert len(raw)<=8388608
lines=raw.decode().splitlines();gauges=[float(line.rsplit(' ',1)[1]) for line in lines if line.startswith(GAUGE_PREFIXES)];assert gauges and all(value==0 for value in gauges),'Existing workload active; no formal warmup dispatch'
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert npu.returncode==0
owned=[]
for line in npu.stdout.split('Process id',1)[-1].splitlines():
 cols=[value.strip() for value in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(value.isdigit() for value in cols[1].split()) and cols[2].isdigit():
  pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text(),'Foreign NPU process: do not benchmark'
  owned.append(pid)
assert owned
print(json.dumps({'project_id':'glm53','node':NODE,'model_identity':model_identity,'idle_native_gauges':gauges,'loadavg':pathlib.Path('/proc/loadavg').read_text(),'meminfo':pathlib.Path('/proc/meminfo').read_text(),'netdev':pathlib.Path('/proc/net/dev').read_text(),'owned_NPU_host_PIDs':owned,'native_NPU_snapshot':npu.stdout,'model_requests':0}))
'''
        values = {
            "URL": f"http://172.16.12.{node}:{port}/metrics",
            "GAUGE_PREFIXES": ("sglang:num_running_reqs{", "sglang:num_queue_reqs{"),
        }
        precheck_codes[node] = render(code, values)
    files = {name: ROOT / path for name, path in settings["client_files"].items()}
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == settings["client_files_sha256"][name]
    gate_files = {name: ROOT / path for name, path in settings["gate_files"].items()}
    for name, path in gate_files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == settings["gate_files_sha256"][name]
    return {
        "precheck_codes": precheck_codes,
        "precheck_hashes": {str(node): hashlib.sha256(code.encode()).hexdigest() for node, code in precheck_codes.items()},
        "client_files_sha256": settings["client_files_sha256"],
        "gate_files_sha256": settings["gate_files_sha256"],
        "settings_sha256": _settings_sha(settings_path) if settings_path else None,
    }


def _actual_require(settings, key_path, key_sha, predicate, label):
    relative = settings.get(key_path)
    digest = settings.get(key_sha)
    assert isinstance(relative, str) and isinstance(digest, str) and len(digest) == 64, label + " is not frozen"
    path = ROOT / relative
    assert path.is_file() and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, label + " SHA mismatch"
    doc = json.loads(path.read_text())
    assert predicate(doc), label + " gate failed"
    return doc


def execute(settings, settings_path):
    p_capacity, d_capacity = _structural(settings)
    plans_result = plans(settings, settings_path)
    formal_review = _actual_require(
        settings, "actual_review_path", "actual_review_sha256",
        lambda row: str(row.get("decision", "")).startswith("APPROVED")
        and row.get("host_codes_sha256") == plans_result["precheck_hashes"]
        and row.get("runtime_authorized") is True
        and row.get("zero_retries") is True,
        "formal actual review",
    )
    native = _actual_require(
        settings, "native_qualification_path", "native_qualification_sha256",
        lambda row: row.get("pass") is True and row.get("framework") == "sglang"
        and row.get("formal_ready") is True and row.get("all32_P_actual_native_packed_MTP_firstuse_verified") is True
        and row.get("native_cache_and_SDMA_verified") is True,
        "native readiness",
    )
    _actual_require(
        settings, "startup_qualification_path", "startup_qualification_sha256",
        lambda row: row.get("pass") is True and row.get("runtime_CIDs") == native.get("runtime_CIDs"),
        "startup qualification",
    )
    _actual_require(
        settings, "functional_decision_path", "functional_decision_sha256",
        lambda row: row.get("decision") == "ACCEPTED" and row.get("functional_ready") is True,
        "functional decision",
    )
    _actual_require(
        settings, "functional_actual_review_path", "functional_actual_review_sha256",
        lambda row: str(row.get("decision", "")).startswith("APPROVED") and row.get("formal_ready") is True,
        "functional actual review",
    )
    required = [("vllm", 32)] + [("sglang", concurrency) for concurrency in [16, 32, 64] if concurrency < 32]
    supplied = settings.get("predecessor_formal_certificates", {})
    assert set(supplied) == {framework + "-C" + str(concurrency) for framework, concurrency in required}
    for framework, concurrency in required:
        item = supplied[framework + "-C" + str(concurrency)]
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        previous = json.loads(path.read_text())
        assert previous["project_id"] == "glm53" and previous["framework"] == framework
        assert previous["concurrency"] == concurrency and previous["measurement_valid"] is True
    native_config = ROOT / settings["native_client_config_path"]
    assert hashlib.sha256(native_config.read_bytes()).hexdigest() == settings["native_client_config_sha256"]
    native_values = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in native_config.read_text().splitlines() if "=" in line and not line.lstrip().startswith("#")
    }
    for key, field in [
        ("meta_service_url", "ock.mmc.meta_service_url"),
        ("config_store_url", "ock.mmc.local_service.config_store_url"),
        ("max_dram_size", "ock.mmc.local_service.max.dram.size"),
    ]:
        assert settings[key] == native_values[field]
    assert native_values["ock.mmc.local_service.protocol"] == "device_sdma"
    assert native_values["ock.mmc.local_service.world_size"] == "128"
    assert native_values["ock.mmc.local_service.dram.size"] == "0GB"
    run = ROOT / "runs" / settings["run_id"]
    run.mkdir(parents=True, exist_ok=False)
    (run / "execution-once.lock").write_text(str(__import__("os").getpid()))
    base = load("inventory", contract.INVENTORY, contract.INVENTORY_SHA)
    rpc = load("formal_rpc", contract.FORMAL_RPC, contract.FORMAL_RPC_SHA)
    gate = load("formal_gate", contract.FORMAL_GATE, settings["gate_renderer_sha256"])
    transfer = load("formal_transfer", settings["prefix_transfer_helper"], settings["prefix_transfer_sha256"])
    files = {name: ROOT / path for name, path in settings["client_files"].items()}
    gate_files = {name: ROOT / path for name, path in settings["gate_files"].items()}
    C = settings["concurrency"]
    fw = settings["framework"]
    rid = settings["run_id"]
    namespace = settings["namespace"]
    Pnode = 34
    cpu_root = "/data/wio/inference-foundry/glm53/runs/" + rid

    def invoke(node, code, timeout=120):
        pod, hostname = base.NODES[node]
        mapping = base.invoke(["kubectl get pod -n default " + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=30)
        assert mapping["rc"] == 0 and mapping["stdout"] == f"172.16.12.{node}"
        code = "import socket\nassert socket.gethostname()==" + repr(hostname) + "\n" + code
        ast.parse(code)
        return base.invoke([shlex.join(["kubectl", "exec", "-i", "-n", "default", pod, "-c", "hostctl", "--", "nsenter", "-t", "1", "-m", "--", "python3", "-"])], code, timeout=timeout)

    def save(name, value):
        (run / name).write_text(json.dumps(value, indent=2) + "\n")
        return value

    receipt = {
        "project_id": "glm53",
        "framework": fw,
        "run_id": rid,
        "concurrency": C,
        "namespace": namespace,
        "model_precheck": {},
        "phase": "PRECHECK",
        "formal_actual_review_sha256": hashlib.sha256((ROOT / settings["actual_review_path"]).read_bytes()).hexdigest(),
        "profiling": False,
        "retries": 0,
    }
    precheck_codes = plans_result["precheck_codes"]
    try:
        for node in contract.ALL_NODES:
            result = save(f"precheck-{node}.json", invoke(node, precheck_codes[node]))
            receipt["model_precheck"][str(node)] = result["rc"]
            assert result["rc"] == 0
        receipt["phase"] = "STAGE"
        if settings.get("prefix_data_staged_receipt"):
            existing = ROOT / settings["prefix_data_staged_receipt"]
            assert hashlib.sha256(existing.read_bytes()).hexdigest() == settings["prefix_data_staged_receipt_sha256"]
        else:
            for node in [43, Pnode]:
                pod = base.NODES[node][0]
                mapped = base.invoke(["kubectl get pod -n default " + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=30)
                assert mapped["rc"] == 0 and mapped["stdout"] == f"172.16.12.{node}"
            command, destination = transfer.render(fw, C, settings["input_manifest_sha256"], rid, base.NODES[43][0], base.NODES[Pnode][0])
            result = save("prefix-transfer.json", base.invoke([command], timeout=180))
            assert result["rc"] == 0
            details = json.loads(result["stdout"])
            assert details["manifest_sha256"] == settings["input_manifest_sha256"] and len(details["files"]) == C + 1
        gate_root = "/run/glm53/formal-gates/" + rid + "/control"
        gate_settings = {
            "framework": fw, "expected_sdk_root": settings["expected_sdk_root"],
            "output": gate_root, "dataset": "/run/glm53/formal-gates/" + rid + "/data",
            "meta_service_url": settings["meta_service_url"], "config_store_url": settings["config_store_url"],
            "world_size": 128, "protocol": "device_sdma",
            "max_dram_size": settings["max_dram_size"], "device_id": 0,
        }
        gate_payload = encoded(gate_files)
        raw = json.dumps(gate_settings, indent=2).encode() + b"\n"
        gate_payload["gate-settings.json"] = {"base64": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest()}
        pitem = settings["runtime_containers"][str(Pnode)]
        gate_values = {
            "CID": pitem["container_id"], "NAME": pitem["name"], "IMAGE": settings["model_image"],
            "SHARED_RID": settings["shared_run_id"], "SHARED_ROOT": settings["shared_remote_root"],
            "RID": rid, "FRAMEWORK": fw, "NODE": Pnode,
            "MANIFEST_SHA": settings["input_manifest_sha256"], "FILES": gate_payload,
        }
        result = save("gate-stage.json", invoke(Pnode, gate.render(gate.STAGE, gate_values)))
        assert result["rc"] == 0
        cpu_payload = encoded(files)
        host = {
            "project_id": "glm53", "framework": fw, "concurrency": C, "namespace": namespace,
            "run_id": rid, "root": cpu_root, "input_manifest_sha256": settings["input_manifest_sha256"],
            "files_sha256": {k: v["sha256"] for k, v in cpu_payload.items()},
            "container_name": settings["client_container_name"],
            "client_settings": {"framework": fw, "concurrency": C, "namespace": namespace, "run_id": rid, "dataset": "/formal-inputs", "output": "/run/glm53/output"},
        }
        ais_path = ROOT / contract.AIS_IDENTITY
        assert hashlib.sha256(ais_path.read_bytes()).hexdigest() == contract.AIS_IDENTITY_SHA
        result = save("CPU-stage.json", invoke(43, render(CPU_STAGE, {
            "AIS_IDENTITY": json.loads(ais_path.read_text()), "CONCURRENCY": C,
            "MANIFEST_SHA": settings["input_manifest_sha256"], "CPU_ROOT": cpu_root,
            "FILES": cpu_payload, "SETTINGS": host,
        })))
        assert result["rc"] == 0
        host_details = json.loads(result["stdout"])
        mapped = base.invoke(["kubectl get pod -n default " + shlex.quote(base.NODES[43][0]) + " -o jsonpath='{.status.hostIP}'"], timeout=30)
        assert mapped["rc"] == 0 and mapped["stdout"] == "172.16.12.43"
        command = base.SSH + [shlex.join([
            "kubectl", "exec", "-i", "-n", "default", base.NODES[43][0], "-c", "hostctl", "--",
            "nsenter", "-t", "1", "-m", "--", "python3", "-u",
            cpu_root + "/glm53_formal_host_controller_v2.py",
            "--settings", cpu_root + "/host-settings.json", "--sha256", host_details["settings_sha256"],
        ])]

        def on_gate(request):
            for node, code in precheck_codes.items():
                checked = save(f"preformal-context-{node}.json", invoke(node, code))
                assert checked["rc"] == 0
            values = gate_values | {
                "REQUEST": request, "NAMESPACE": namespace, "CONCURRENCY": C,
                "SDK_PREFIX": "", "LIB_PREFIX": "",
            }
            result = save("cache-gate-native.json", invoke(Pnode, gate.render(gate.INVOKE, values), 280))
            payload = json.loads(result["stdout"])
            assert result["rc"] == 0 and payload["query_rc"] == 0
            report = payload["report"]
            assert hashlib.sha256((json.dumps(report, indent=2) + "\n").encode()).hexdigest() == payload["report_sha256"]
            return report

        receipt["phase"] = "REAL_WARMUP_AND_FORMAL_CLIENT"
        save("resource-controller-status.json", receipt)
        result = rpc.rpc(
            command,
            run,
            {"project_id": "glm53", "framework": fw, "namespace": namespace, "concurrency": C, "manifest_sha256": settings["input_manifest_sha256"]},
            on_gate,
        )
        receipt["RPC"] = result
        receipt["phase"] = "CLIENT_COMPLETE_NATIVE_POST_AUDIT_REQUIRED"
        receipt["formal_review_sha256"] = hashlib.sha256((ROOT / settings["actual_review_path"]).read_bytes()).hexdigest()
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        save("resource-controller-receipt.json", receipt)
    return receipt


def main():
    settings_path = contract.FORMAL_SETTINGS
    settings = json.loads((ROOT / settings_path).read_text())
    result = plans(settings, settings_path)
    print(json.dumps({
        "project_id": "glm53",
        "status": settings.get("status"),
        "settings": settings_path,
        "settings_sha256": result["settings_sha256"],
        "host_codes_sha256": result["precheck_hashes"],
        "runtime_authorized": False,
        "model_requests": 0,
        "SDK_init_calls": 0,
        "max_SSH_future_execution": 2,
    }, indent=2))


if __name__ == "__main__":
    main()
