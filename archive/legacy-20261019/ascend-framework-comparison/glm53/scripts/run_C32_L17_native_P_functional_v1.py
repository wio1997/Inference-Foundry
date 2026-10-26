"""DRAFT_PENDING_PRIMARY_REVIEW: one bounded L17 native C32 functional matrix.

Exactly one client invocation after all four current identities and idle state
pass.  It uses the new mixed P-native/D-eager identity renderer, a fresh L17
run namespace and the derived 600-output probe.  No retry, no benchmark clock,
no profile and no formal request.
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
INVENTORY = ROOT / "scripts/inventory_readonly.py"
assert hashlib.sha256(INVENTORY.read_bytes()).hexdigest() == contract.INVENTORY_SHA
spec = importlib.util.spec_from_file_location("inventory", INVENTORY)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

RUN = ROOT / "runs" / contract.SHORT_RUN
REMOTE = contract.SHORT_REMOTE
IMAGE = contract.IMAGE
PROBE = "probe_C32_L17_native_pd_payload_v1.py"
PROBE_SHA = "360acd948bfa62dc534054e3e6a459f8aae5e24ed5ab642056c2d97ceeefb179"
GATEWAY = "glm53_sglang_pd_gateway_v2.py"
GATEWAY_SHA = contract.GATEWAY_V2_SHA
FILES = {name: (P.parent / name).read_text(encoding="utf-8") for name in (PROBE, GATEWAY)}
FROZEN = {PROBE: PROBE_SHA, GATEWAY: GATEWAY_SHA}
for name, digest in FROZEN.items():
    assert hashlib.sha256(FILES[name].encode()).hexdigest() == digest, name

P_CAPACITY, D_CAPACITY = contract.check_capacity_pair()
IDS = {
    int(node): row["container_id"]
    for node, row in contract.combined_providers().items()
}
assert set(IDS) == set(contract.ALL_NODES)


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
    code = identity.render(node, P_CAPACITY, D_CAPACITY, "candidate")
    code += (
        "\nimport urllib.request\n"
        "opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
        "with opener.open(" + repr("http://172.16.12." + str(node) + (":30500" if node in contract.P_NODES else ":30600") + "/metrics") + ",timeout=10) as response:\n"
        " raw=response.read(8*1024**2+1);assert len(raw)<=8*1024**2;assert response.status==200\n"
        "gauges=[float(line.rsplit(' ',1)[1]) for line in raw.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]\n"
        "assert gauges and not any(gauges),'Existing workload active; no functional dispatch'\n"
        "print(json.dumps({'node':NODE,'container_id':CID,'actual_model_identity':model_identity,'supervisor_phase':supervisor['phase'],'provider_phase':provider['phase'],'idle_gauges':gauges,'model_requests':0}))\n"
    )
    import ast
    ast.parse(code)
    return code


def check(node):
    return invoke(node, render_check(node))


def render_client():
    args = [
        "docker", "run", "--rm", "--pull=never", "--network=host", "--cpus=4", "--memory=4g",
        "--pids-limit=256", "--label", "project_id=glm53", "--label", "run_id=" + contract.SHORT_RUN,
        "--name", "glm53-L17-native-C32-functional-v1-20261019-34",
        "-v", REMOTE + ":/run/glm53",
        "-v", "/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/run/glm53",
        "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        "-e", "TOKENIZERS_PARALLELISM=false", "-e", "ASCEND_RT_VISIBLE_DEVICES=", "-e", "CUDA_VISIBLE_DEVICES=",
        "--entrypoint", "timeout", IMAGE, "3100", "python3", "/run/glm53/" + PROBE,
    ]
    code = r'''import subprocess,pathlib,json,hashlib,sys
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay';root.mkdir(parents=True)
for name,text in FILES.items():
 p=root/name;p.write_text(text);assert hashlib.sha256(p.read_bytes()).hexdigest()==FROZEN[name]
with (root/'client-stdout.txt').open('w') as out,(root/'client-stderr.txt').open('w') as err:
 r=subprocess.run(ARGS,stdout=out,stderr=err,text=True,timeout=3150)
result=root/'result.json';data=json.loads(result.read_text()) if result.is_file() else None
print('GLM53_SG_MATRIX_ARTIFACTS='+json.dumps({'client_rc':r.returncode,'result':data,'artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()}}),flush=True)
print((root/'client-stderr.txt').read_text()[-16000:],file=sys.stderr,end='');sys.exit(r.returncode)
'''
    for key, value in [("REMOTE", REMOTE), ("FILES", FILES), ("FROZEN", FROZEN), ("ARGS", args)]:
        code = re.sub(r"\b" + key + r"\b", lambda _: repr(value), code)
    import ast
    ast.parse(code)
    return code


def execute():
    return invoke(34, render_client(), timeout=3220)


def plan():
    return {
        **{"identity-" + str(node): render_check(node) for node in contract.ALL_NODES},
        "client": render_client(),
    }


def main():
    fixture = json.loads((ROOT / "runs/20261017T1400-sglang-gateway-fixture-v2/receipt.json").read_text())
    assert type(fixture.get("rc")) is int and fixture["rc"] == 0
    qualified = json.loads((ROOT / contract.STARTUP_RESULT).read_text())
    assert qualified.get("pass") is True and qualified.get("all64_actual_SDMA_and_BF16_layout_verified") is True
    assert qualified.get("all32_D_nativeOps_fullmesh_and_targetGraph_retained") is True
    assert qualified.get("all32_P_native_IO_and_existing_MC_mapping_reuse_verified") is True
    assert qualified["runtime_CIDs"] == {str(node): cid for node, cid in IDS.items()}
    startup_receipt_relative = "runs/" + contract.STARTUP_CAPTURE + "/receipt.json"
    startup_receipt = json.loads((ROOT / startup_receipt_relative).read_text())
    assert startup_receipt["pass"] is True and startup_receipt["mode"] == "startup"
    assert len(startup_receipt["nodes"]) == 4 and all(value == 0 for value in startup_receipt["nodes"].values())
    startup_result_sha256 = hashlib.sha256((ROOT / contract.STARTUP_RESULT).read_bytes()).hexdigest()
    startup_receipt_sha256 = hashlib.sha256((ROOT / startup_receipt_relative).read_bytes()).hexdigest()
    expected = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in plan().items()}
    contract.inference_review_check(
        ROOT / contract.SHORT_REVIEW,
        expected,
        launcher_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),
        files_sha256=FROZEN,
        logical_client_cases=contract.SHORT_LOGICAL,
        backend_model_calls=contract.SHORT_BACKEND,
        successful_output_tokens_if_all_pass=contract.SHORT_OUTPUT_IF_ALL,
        provenance={
            "startup_result_sha256": startup_result_sha256,
            "startup_capture_receipt_sha256": startup_receipt_sha256,
        },
    )
    RUN.mkdir(parents=True, exist_ok=False)
    with (RUN / "execution-once.lock").open("x") as file:
        file.write(str(__import__("os").getpid()))
    receipt = {
        "project_id": "glm53",
        "run_id": contract.SHORT_RUN,
        "namespace": contract.SHORT_NAMESPACE,
        "launcher_sha256": hashlib.sha256(P.read_bytes()).hexdigest(),
        "files_sha256": FROZEN,
        "review_sha256": hashlib.sha256((ROOT / contract.SHORT_REVIEW).read_bytes()).hexdigest(),
        "startup_result_sha256": startup_result_sha256,
        "startup_capture_receipt_sha256": startup_receipt_sha256,
        "prechecks": {},
        "rc": None,
        "logical_client_cases_planned": contract.SHORT_LOGICAL,
        "backend_model_calls_planned": contract.SHORT_BACKEND,
        "successful_output_tokens_if_all_pass": contract.SHORT_OUTPUT_IF_ALL,
        "formal_requests": 0,
        "retries": 0,
    }
    try:
        for node in contract.ALL_NODES:
            result = check(node)
            (RUN / ("precheck-" + str(node) + ".json")).write_text(json.dumps(result, indent=2) + "\n")
            receipt["prechecks"][str(node)] = result["rc"]
            assert type(result.get("rc")) is int and result["rc"] == 0
        result = execute()
        (RUN / "node-34.json").write_text(json.dumps(result, indent=2) + "\n")
        receipt["rc"] = result["rc"]
        print("matrix", result["rc"], flush=True)
    finally:
        (RUN / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
