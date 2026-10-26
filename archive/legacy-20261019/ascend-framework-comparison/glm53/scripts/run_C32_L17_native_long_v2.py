"""DRAFT_PENDING_PRIMARY_REVIEW: once-only L17 native long functional guards v2.

v2 changes only the pre-approval audit provenance: the actual client-only
result must name the v3 short-client auditor source and carry that v3 file's
actual SHA256, which is then pinned as the long review's
short_client_audit_sha256.  Requests, configuration, native inputs, case counts,
namespace and run id are unchanged.  This draft does not execute anything; the
existing namespace/run values remain reusable for a future long run.

Six logical / 12 backend calls: two 80K known facts plus cold/warm/exact-known-
key/cross-P-DP storage, 2408 output tokens if all pass.  The accepted dataset
SHA, AISBench assets and native-ID construction are unchanged.  Metadata query
uses the qualified ``init_bm=False`` helper only.  No extra SDK benchmark,
profile, retry or model request is added.
"""
import hashlib
import importlib.util
import json
import re
import shlex
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

P = Path(__file__).resolve()
ROOT = P.parents[1]
PROJECT_ROOT = P.parents[1]
CLIENT_AUDITOR_SOURCE = "scripts/audit_C32_L17_native_short_client_local_v3.py"
INVENTORY = ROOT / "scripts/inventory_readonly.py"
assert hashlib.sha256(INVENTORY.read_bytes()).hexdigest() == contract.INVENTORY_SHA
spec = importlib.util.spec_from_file_location("inventory", INVENTORY)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

RUN = ROOT / "runs" / contract.LONG_RUN
REMOTE = contract.LONG_REMOTE
CLIENT_IMAGE = "sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7"
META_IMAGE = "sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83"
SG_IMAGE = contract.IMAGE
SDK = "/data/wio/inference-foundry/glm53/overlays/sdk13-20261017"
P_CAPACITY, D_CAPACITY = contract.check_capacity_pair()
IDS = {int(node): row["container_id"] for node, row in contract.combined_providers().items()}
assert set(IDS) == set(contract.ALL_NODES)

FROZEN = {
    "aisbench_glm53_token_ids.py": contract.AIS_TOKEN_IDS_SHA,
    "glm53_sglang_pd_gateway_v3.py": contract.GATEWAY_V3_SHA,
    "glm53_C32_L17_native_long_client_v1.py": "301fdfcd7725471e2043fa4bc1c22e2b5cd5051cfc43ba141c7e34eda85d875f",
    "glm53_sglang_prefix_keys_v1.py": contract.PREFIX_KEYS_SHA,
    "glm53_C32_L17_native_semantic_probe_v1.py": "23a6d2b70c9822250564433be474fc659a3ae5bf54e5763ea0f51e69d4ed27b1",
    "glm53_request_terminal_v1.py": contract.REQUEST_TERMINAL_SHA,
    "glm53_storage_proof_v1.py": contract.STORAGE_PROOF_SHA,
}
FILES = {}
for name, digest in FROZEN.items():
    raw = (P.parent / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    FILES[name] = raw.decode()
FILES["metadata-local.conf"] = (
    "ock.mmc.meta_service_url = tcp://172.16.12.34:29780\n"
    "ock.mmc.local_service.config_store_url = tcp://172.16.12.34:29781\n"
    "ock.mmc.log_level = info\n"
    "ock.mmc.local_service.world_size = 128\n"
    "ock.mmc.local_service.protocol = device_sdma\n"
    "ock.mmc.local_service.dram.size = 0GB\n"
    "ock.mmc.local_service.max.dram.size = 160GB\n"
)
IDENTITY = json.loads((ROOT / contract.AIS_IDENTITY).read_text())
assert hashlib.sha256((ROOT / contract.AIS_IDENTITY).read_bytes()).hexdigest() == contract.AIS_IDENTITY_SHA
PREREQS = {
    "results/sglang-knownkey-fixture-v2.json": "9796e7cefc73698007bdbc5b30bbf7058dd39d8a15be6ecf5af128ec28ab3ed0",
    "results/request-terminal-actual-fixture-v2.json": "e07b66b2df193c641bb47d20a9687f6c0b747452ca32d4365f77fbeccc123942",
}
for name, digest in PREREQS.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def invoke(node, code, timeout=100):
    pod, host = base.NODES[node]
    mapped = base.invoke(
        ["kubectl get pod -n default " + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"],
        timeout=30,
    )
    assert mapped["rc"] == 0 and mapped["stdout"] == f"172.16.12.{node}"
    code = "import socket\nassert socket.gethostname()==" + repr(host) + "\n" + code
    return base.invoke(
        [shlex.join(["kubectl", "exec", "-i", "-n", "default", pod, "-c", "hostctl", "--", "nsenter", "-t", "1", "-m", "--", "python3", "-"])],
        code,
        timeout=timeout,
    )


def _identity():
    path = ROOT / contract.MIXED_IDENTITY
    assert hashlib.sha256(path.read_bytes()).hexdigest() == contract.MIXED_IDENTITY_SHA
    spec = importlib.util.spec_from_file_location("l17_mixed_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_check(node):
    assert node in contract.ALL_NODES
    identity = _identity()
    url = "http://172.16.12." + str(node) + (":30500" if node in contract.P_NODES else ":30600") + "/metrics"
    code = identity.render(node, P_CAPACITY, D_CAPACITY, "candidate")
    code += (
        "\nimport urllib.request\n"
        "with urllib.request.urlopen(" + repr(url) + ",timeout=10) as response:\n"
        " assert response.status==200;raw=response.read(8388609);assert len(raw)<=8388608\n"
        "gauges=[float(line.rsplit(' ',1)[1]) for line in raw.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]\n"
        "assert gauges and all(value==0 for value in gauges),'Existing workload active; no long client dispatch'\n"
        "print(json.dumps({'node':NODE,'container_id':CID,'actual_model_identity':model_identity,'idle_gauges':gauges,'model_requests':0}))\n"
    )
    import ast
    ast.parse(code)
    return code


def check(node):
    return invoke(node, render_check(node))


def render_client():
    args = [
        "docker", "run", "--rm", "--pull=never", "--network=host", "--pids-limit=512", "--memory=8g", "--cpus=4",
        "--label", "project_id=glm53", "--label", "run_id=" + contract.LONG_RUN,
        "--name", "glm53-L17-native-long-v1-20261019-43",
        "--cidfile", REMOTE + "/client.cid", "--workdir", "/run/glm53",
        "-v", "/data/wio/aisbench-offline-43-20261014:/work:ro",
        "-v", REMOTE + ":/run/glm53",
        "-v", "/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1:/semantic:ro",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONNOUSERSITE=1",
        "-e", "PYTHONPATH=/run/glm53:/work/benchmark",
        "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
        "-e", "TOKENIZERS_PARALLELISM=false", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "ASCEND_RT_VISIBLE_DEVICES=",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HOME=/run/glm53/hf",
        "-e", "XDG_CACHE_HOME=/run/glm53/cache", "-e", "MPLCONFIGDIR=/run/glm53/mpl",
        "-e", "TORCH_HOME=/run/glm53/torch", "-e", "NLTK_DATA=/work/nltk_data",
        "-e", "TIKTOKEN_CACHE_DIR=/work/tiktoken_cache",
        "--entrypoint", "timeout", CLIENT_IMAGE, "4000",
        "/work/venv/bin/python", "/run/glm53/glm53_C32_L17_native_long_client_v1.py",
    ]
    boot = (
        "export PYTHONPATH=/opt/glm53/sdk13:/run/glm53; "
        "export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; "
        "exec timeout 150 python3 /run/glm53/glm53_C32_L17_native_semantic_probe_v1.py"
    )
    meta_args = [
        "docker", "run", "--rm", "-i", "--pull=never", "--network=host", "--memory=4g", "--cpus=2", "--pids-limit=512",
        "--label", "project_id=glm53", "--label", "run_id=" + contract.LONG_RUN,
        "--workdir", "/run/glm53", "-v", REMOTE + ":/run/glm53:ro", "-v", SDK + ":/opt/glm53/sdk13:ro",
        "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "MMC_LOCAL_CONFIG_PATH=/run/glm53/metadata-local.conf",
        "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        "-e", "ASCEND_RT_VISIBLE_DEVICES=", "-e", "CUDA_VISIBLE_DEVICES=",
        "--entrypoint", "/bin/bash", META_IMAGE, "-c", boot,
    ]
    code = r'''import pathlib,json,hashlib,subprocess,sys,time
for image in (CLIENT_IMAGE,META_IMAGE):
 r=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==image
core=pathlib.Path('/data/wio/aisbench-offline-43-20261014/benchmark/ais_bench/benchmark')
for entry in IDENTITY['files']:assert hashlib.sha256((core/entry['path']).read_bytes()).hexdigest()==entry['sha256']
dataset=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1')
assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()=='ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece'
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
for name,text in FILES.items():
 assert pathlib.PurePosixPath(name).name==name
 (root/name).write_text(text)
 if name in FROZEN:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==FROZEN[name]
status={'client_rc':None,'gates':{}};deadline=time.monotonic()+4000
with (root/'client-stdout.txt').open('w') as out,(root/'client-stderr.txt').open('w') as err:
 process=subprocess.Popen(ARGS,stdout=out,stderr=err,text=True)
 try:
  while process.poll() is None:
   for index in (0,1):
    request_path=root/f'need-metadata-{index}.json'
    if request_path.is_file() and str(index) not in status['gates']:
     request=json.loads(request_path.read_text());record=json.loads((dataset/f'semantic-{index}.json').read_text())
     assert request['index']==index and request['namespace']==f'glm53-sem-sglang-L17-native-v1-20261019-case{index}-warm'
     assert request['prefix_ids']==record['prefix_ids'] and request['prefix_sha256_le_u32']==record['prefix_sha256_le_u32']
     try:
      meta=subprocess.run(META_ARGS[:2]+['--name',f'glm53-L17-native-semantic-meta-v1-20261019-43-{index}']+META_ARGS[2:],input=json.dumps(request),capture_output=True,text=True,timeout=160)
      (root/f'metadata-{index}-stdout.txt').write_text(meta.stdout);(root/f'metadata-{index}-stderr.txt').write_text(meta.stderr)
      lines=[x[len('GLM53_PREFIX_METADATA='):] for x in meta.stdout.splitlines() if x.startswith('GLM53_PREFIX_METADATA=')];result=json.loads(lines[0]) if len(lines)==1 else {}
      gate={'rc':meta.returncode,'pass':bool(meta.returncode==0 and result.get('pass') and result.get('close_rc')==0),'metadata':result,'stdout_sha256':hashlib.sha256(meta.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(meta.stderr.encode()).hexdigest()}
     except Exception as error:gate={'rc':-1,'pass':False,'error':repr(error)}
     status['gates'][str(index)]=gate;temp=root/f'metadata-gate-{index}.tmp';temp.write_text(json.dumps(gate)+'\n');temp.replace(root/f'metadata-gate-{index}.json')
   if time.monotonic()>deadline:raise TimeoutError('CPUclient4000s timeout; preserve active container and logs, no automatic cancellation')
   time.sleep(.2)
  status['client_rc']=process.returncode
 except Exception as error:status['error']=repr(error)
 finally:(root/'semantic-phase-status.json').write_text(json.dumps(status,indent=2)+'\n')
artifacts={}
for path in sorted(root.iterdir()):
 if path.is_file():
  data=path.read_bytes();artifacts[path.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  if path.name in ('semantic-result.json','semantic-phase-status.json') or path.name.startswith('metadata-gate-'):artifacts[path.name]['text']=data.decode()
print('GLM53_LONG_ARTIFACTS='+json.dumps(artifacts),flush=True)
print((root/'client-stderr.txt').read_text()[-12000:],file=sys.stderr,end='')
print('GLM53_CLIENT_EXIT='+json.dumps(status),flush=True)
sys.exit(0 if status.get('client_rc')==0 and len(status['gates'])==2 and all(x['pass'] for x in status['gates'].values()) else 1)
'''
    for key, value in [
        ("CLIENT_IMAGE", CLIENT_IMAGE), ("META_IMAGE", META_IMAGE), ("IDENTITY", IDENTITY),
        ("REMOTE", REMOTE), ("FILES", FILES), ("FROZEN", FROZEN), ("ARGS", args), ("META_ARGS", meta_args),
    ]:
        code = re.sub(r"\b" + key + r"\b", lambda _: repr(value), code)
    import ast
    ast.parse(code)
    return code


def execute():
    return invoke(43, render_client(), timeout=4120)


def plan():
    return {
        **{"identity-" + str(node): render_check(node) for node in contract.ALL_NODES},
        "client": render_client(),
    }


def client_result_gate(row):
    return (
        row.get("label") == contract.CLIENT_LABEL
        and row.get("pass") is True
        and row.get("native_qualified") is False
        and row.get("formal_ready") is False
        and row.get("successful_backend_inferences") == contract.SHORT_BACKEND
        and row.get("successful_output_tokens") == contract.SHORT_OUTPUT_IF_ALL
        and row.get("runtime_CIDs") == {str(node): cid for node, cid in IDS.items()}
    )


def main():
    client = json.loads((ROOT / contract.CLIENT_RESULT).read_text())
    assert client_result_gate(client), "Client-only short audit is missing or mislabeled"
    client_auditor_sha256 = hashlib.sha256((PROJECT_ROOT / CLIENT_AUDITOR_SOURCE).read_bytes()).hexdigest()
    assert client.get("client_auditor_source") == CLIENT_AUDITOR_SOURCE, "client audit source is not the v3 auditor"
    assert client.get("client_auditor_sha256") == client_auditor_sha256, "client audit source hash differs from v3"
    startup = json.loads((ROOT / contract.STARTUP_RESULT).read_text())
    assert startup.get("pass") is True and startup.get("all64_actual_SDMA_and_BF16_layout_verified") is True
    assert startup.get("all32_D_nativeOps_fullmesh_and_targetGraph_retained") is True
    assert startup.get("all32_P_native_IO_and_existing_MC_mapping_reuse_verified") is True
    assert startup["runtime_CIDs"] == client["runtime_CIDs"] == {str(node): cid for node, cid in IDS.items()}
    startup_result_sha256 = hashlib.sha256((ROOT / contract.STARTUP_RESULT).read_bytes()).hexdigest()
    client_result_sha256 = hashlib.sha256((ROOT / contract.CLIENT_RESULT).read_bytes()).hexdigest()
    client_audit_sha256 = client_auditor_sha256
    expected = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in plan().items()}
    contract.inference_review_check(
        ROOT / contract.LONG_REVIEW,
        expected,
        launcher_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),
        files_sha256=FROZEN,
        logical_client_cases=contract.LONG_LOGICAL,
        backend_model_calls=contract.LONG_BACKEND,
        successful_output_tokens_if_all_pass=contract.LONG_OUTPUT_IF_ALL,
        provenance={
            "startup_result_sha256": startup_result_sha256,
            "short_client_result_sha256": client_result_sha256,
            "short_client_audit_sha256": client_audit_sha256,
        },
    )
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "execution-once.lock").open("x") as file:
        file.write(str(__import__("os").getpid()))
    receipt = {
        "project_id": "glm53",
        "run_id": contract.LONG_RUN,
        "namespace_base": contract.LONG_NAMESPACE,
        "launcher_sha256": hashlib.sha256(P.read_bytes()).hexdigest(),
        "files_sha256": FROZEN,
        "prerequisites_sha256": PREREQS,
        "review_sha256": hashlib.sha256((ROOT / contract.LONG_REVIEW).read_bytes()).hexdigest(),
        "startup_result_sha256": startup_result_sha256,
        "short_client_result_sha256": client_result_sha256,
        "short_client_audit_sha256": client_audit_sha256,
        "phase": "FUNCTIONAL_ONLY_NOT_PERFORMANCE",
        "model_precheck": {},
        "rc": None,
        "logical_client_cases_planned": contract.LONG_LOGICAL,
        "backend_model_calls_planned": contract.LONG_BACKEND,
        "successful_output_tokens_if_all_pass": contract.LONG_OUTPUT_IF_ALL,
        "formal_requests": 0,
        "retries": 0,
    }
    try:
        for node in IDS:
            result = check(node)
            (RUN / f"precheck-{node}.json").write_text(json.dumps(result, indent=2) + "\n")
            receipt["model_precheck"][node] = result["rc"]
            assert type(result.get("rc")) is int and result["rc"] == 0, "Model identity/idle gate failed; no long client starts"
        (RUN / "dispatch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result = execute()
        (RUN / "node-43.json").write_text(json.dumps(result, indent=2) + "\n")
        receipt["rc"] = result["rc"]
    finally:
        (RUN / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("long functional", receipt["rc"], flush=True)


if __name__ == "__main__":
    main()
