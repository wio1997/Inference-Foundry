"""DRAFT_PENDING_PRIMARY_REVIEW: parameterized mixed P-native/D-selected identity v1.

Local renderer only. Import, `--check` and `--render` make zero SSH, HTTP, NPU,
SDK or model calls and write no runtime state. There is no model-start launcher.

Roles are explicit and independent:
- P source is always the accepted L12C native candidate
  (results/sglang-C32-L12C-selected-P-pool-capacity-v1.json);
- D source is the separately selected L17 candidate-v2 or original-rollback-v1,
  with actual D container IDs/service IDs supplied by the future reviewed
  capacity-v2 JSON. No current D IDs are guessed or embedded here.

This is a new renderer; the old ff5a9234 helper is not reused unchanged. P keeps
its source manifest, optional extension SHA/mount and CLI/precision/native guards.
D keeps the unchanged CLI/precision/SDMA/Graph/MTP guards and adds only the
candidate-v2 NoCopy/audit expectations, or explicitly expects no audit hook for
rollback.
"""

import argparse
import ast
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATUS = "DRAFT_PENDING_PRIMARY_REVIEW"
IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"
META_CID = "0dd00f5c35ee8c60cbd697a38fc73a8c37ae498106f9ebfc2933139e953b165c"
P_NODES = (34, 35)
D_NODES = (36, 41)
P_CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
P_CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
P_STAGE = "config/sglang-C32-native-HiCache-L12C-candidate-stage-v1.json"
P_STAGE_SHA = "29194e69b322471e91049179731aa0c84a25c0eb7e7f1481e69cfad437a58d7e"
P_RUN_ID = "20261019T-L12C-candidate-v1"
P_REMOTE = "/data/wio/inference-foundry/glm53/overlays/sglang-C32-native-HiCache-L12C-candidate-v1-20261019"
P_OPTIONAL_RELATIVE = "native/libmf_hybm_accoffload.so"
P_OPTIONAL_SHA = "3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46"
P_OPTIONAL_TARGET = "/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so"
AUDIT_SHA = "0df4276b5e5f8d10aec1faf25cd24e847dc3ef9a57570f70f257c0ebf539d56f"
EAGER_AUDIT_ROOT = "/run/glm53/eager-input-audit"
EAGER_SHA = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
ENVIRON_SHA = "0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c"
SDMA_MANIFEST_SHA = "0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528"
MODEL_CONFIG_SHA = "ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca"

D_STAGES = {
    "candidate": {
        "stage": "config/sglang-C32-L17-D-native-eager-stage-v2.json",
        "stage_sha256": "a8d6759af5f29a3da15e60b1312bc81eae046e93c99ca3eb366a78bdf055da0c",
        "bundle_manifest_sha256": "9427ba4316f5b5075b97afecb01eca062caa912786280897fbf962eb1d498491",
        "bundle": "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2",
        "remote": "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-native-eager-v2-20261019",
        "run_id": "20261019T-L17-D-native-eager-v2",
        "run_root": "/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-native-eager-v2",
        "name_prefix": "glm53-L17-D-native-eager-v2",
        "audit": True,
    },
    "rollback": {
        "stage": "config/sglang-C32-L17-D-original-rollback-stage-v1.json",
        "stage_sha256": "f8b49216cd3ac6651cb48f9308a2590d2c0e306b339355e568a522bb882ddcf0",
        "bundle_manifest_sha256": "b3e1d5824eed9e86b61e1a20a0f0a46732c5caca543d83e1310d62e4a171cd7a",
        "bundle": "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1",
        "remote": "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-original-rollback-v1-20261019",
        "run_id": "20261019T-L17-D-original-rollback-v1",
        "run_root": "/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-original-rollback-v1",
        "name_prefix": "glm53-L17-D-original-rollback-v1",
        "audit": False,
    },
}


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_stage(path, expected, nodes, optional_allowed, require_pending):
    assert sha256_file(ROOT / path) == expected, path
    data = read_json(ROOT / path)
    assert data.get("project_id") == "glm53"
    assert len(data["bundles"]) == 1
    bundle = data["bundles"][0]
    assert sorted(bundle["nodes"]) == sorted(nodes), bundle["nodes"]
    assert bundle["image"] == IMAGE
    assert bundle.get("native_source_manifest_sha256") == SDMA_MANIFEST_SHA
    if "bundle_manifest" in bundle:
        assert sha256_file(ROOT / bundle["bundle_manifest"]) == bundle["bundle_manifest_sha256"]
    if require_pending:
        assert bundle.get("review_status", data.get("review_status", "PENDING_PRIMARY_REVIEW")) == "PENDING_PRIMARY_REVIEW"
        assert bundle.get("runtime_authorized", data.get("runtime_authorized")) is False
    optional = bundle.get("private_optional_extension")
    if optional_allowed:
        assert optional and sorted(optional["nodes"]) == sorted(P_NODES)
        assert optional["sha256"] == P_OPTIONAL_SHA
        assert optional["relative"] == P_OPTIONAL_RELATIVE
        assert optional["container_target"] == P_OPTIONAL_TARGET
    else:
        assert optional is None
        assert "private_optional_extension" not in json.dumps(bundle)
        assert "libmf_hybm_accoffload" not in json.dumps(bundle)
    return bundle


def p_capacity_check(p_capacity):
    assert sha256_file(ROOT / P_CAPACITY) == P_CAPACITY_SHA
    assert p_capacity.get("project_id") == "glm53" and p_capacity.get("pass") is True
    assert p_capacity.get("selected_variant") == "candidate"
    assert "P_ONLY" in p_capacity.get("classification", "")
    if p_capacity.get("selected_P_variant") is not None:
        assert p_capacity["selected_P_variant"] == "candidate"
    for node in P_NODES:
        row = p_capacity["providers"][str(node)]
        assert row["run_id"] == P_RUN_ID
        assert row["name"] == "glm53-L12C-candidate-v1-20261019-%d" % node
        assert row["dram_GiB"] in (136, 112)
        assert row["local_service_id"] in (82, 83)
    return p_capacity


def d_capacity_check(d_capacity, variant):
    assert d_capacity.get("project_id") == "glm53" and d_capacity.get("pass") is True
    assert d_capacity.get("selected_P_variant") == "candidate"
    assert d_capacity.get("selected_D_variant") == variant
    assert d_capacity.get("selected_variant") == variant
    assert d_capacity.get("selected_variant_scope") in ("D_ONLY_NOT_P", "D_ONLY")
    assert d_capacity.get("model_started") is False
    spec = D_STAGES[variant]
    for node in D_NODES:
        row = d_capacity["providers"][str(node)]
        assert row["run_id"] == spec["run_id"]
        assert row["remote_root"] == spec["run_root"]
        assert row["name"] == spec["name_prefix"] + "-20261019-" + str(node)
        assert row["dram_GiB"] == 160
        assert type(row["local_service_id"]) is int and row["local_service_id"] >= 0
        assert row["local_service_id"] not in (82, 83)
    return d_capacity


def render(node, p_capacity, d_capacity, variant):
    """Render one actual mixed identity plan; requires reviewed actual capacity."""
    assert node in P_NODES + D_NODES
    p_capacity_check(p_capacity)
    d_capacity_check(d_capacity, variant)
    p_bundle = load_stage(P_STAGE, P_STAGE_SHA, P_NODES + D_NODES, True, False)
    d_bundle = load_stage(D_STAGES[variant]["stage"], D_STAGES[variant]["stage_sha256"],
                          D_NODES, False, True)
    if variant == "candidate":
        assert d_bundle["audit_version"] == "v2"
        assert d_bundle["audit_hook_sha256"] == AUDIT_SHA
        assert d_bundle["eager_source_sha256"] == EAGER_SHA
        assert d_bundle["environ_source_sha256"] == ENVIRON_SHA
    else:
        assert "audit_version" not in d_bundle
        assert "audit_hook_sha256" not in d_bundle
        assert "eager_source_sha256" not in d_bundle
        assert "environ_source_sha256" not in d_bundle
    if node in P_NODES:
        row = p_capacity["providers"][str(node)]
        stage = {"remote": P_REMOTE, "run_id": P_RUN_ID, "run_root": row["remote_root"],
                 "name_prefix": "glm53-L12C-candidate-v1", "audit": False}
        bundle = p_bundle
        role, is_d, native_p, d_variant = "P", False, True, None
    else:
        row = d_capacity["providers"][str(node)]
        stage = D_STAGES[variant]
        bundle = d_bundle
        role, is_d, native_p, d_variant = "D", True, False, variant
    p_cids = [p_capacity["providers"][str(item)]["container_id"] for item in P_NODES]
    old_d_cids = [
        "25ecbd3a55e199caf4c6012a61b702d1b6035c955406d75bb17107991a932dd1",
        "c39fa7c594a8776e70c365baec435d4288abb1db5cc492405183047d57b805c3",
    ]
    assert row["container_id"] not in old_d_cids
    assert row["container_id"] != META_CID
    if is_d:
        assert row["container_id"] not in p_cids
    else:
        assert row["container_id"] in p_cids
    files = {name: meta["sha256"] for name, meta in bundle["files"].items()}
    values = {
        "NODE": node,
        "ROLE": role,
        "IS_D": is_d,
        "NATIVE_P": native_p,
        "D_VARIANT": d_variant,
        "CID": row["container_id"],
        "NAME": row["name"],
        "IMAGE": IMAGE,
        "NODE_RUN_ID": stage["run_id"] if is_d else row["run_id"],
        "NODE_REMOTE_ROOT": stage["run_root"] if is_d else row["remote_root"],
        "STAGED_ROOT": stage["remote"] if is_d else P_REMOTE,
        "STAGED_FILES": files,
        "P_CIDS": p_cids,
        "OLD_D_CIDS": old_d_cids,
        "META_CID": META_CID,
        "AUDIT_SHA": AUDIT_SHA,
        "EAGER_AUDIT_ROOT": EAGER_AUDIT_ROOT,
        "EAGER_SHA": EAGER_SHA,
        "ENVIRON_SHA": ENVIRON_SHA,
        "P_OPTIONAL_RELATIVE": P_OPTIONAL_RELATIVE,
        "P_OPTIONAL_SHA": P_OPTIONAL_SHA,
        "P_OPTIONAL_TARGET": P_OPTIONAL_TARGET,
        "MODEL_CONFIG_SHA": MODEL_CONFIG_SHA,
    }
    header = "".join("%s = %r\n" % (key, value) for key, value in values.items())
    code = header + HOST
    ast.parse(code)
    return code


HOST = r'''
import json,pathlib,subprocess,hashlib
assert NODE in (34,35,36,41)
assert ROLE in ('P','D') and IS_D==(NODE in (36,41))
assert CID not in OLD_D_CIDS and CID!=META_CID
if IS_D:
    assert CID not in P_CIDS
else:
    assert CID in P_CIDS and NATIVE_P
def cmd(args,timeout=30):
    return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
def must(args,timeout=30):
    result=cmd(args,timeout);assert result.returncode==0,(args,result.returncode,result.stderr);return result.stdout
d=json.loads(must(['docker','inspect',CID]))[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running']
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==NODE_RUN_ID
assert not d['HostConfig']['Privileged']
shared_root=pathlib.Path(NODE_REMOTE_ROOT)
supervisor=json.loads((shared_root/'supervisor-status.json').read_text())
provider=json.loads((shared_root/'provider-status.json').read_text())
assert supervisor['run_id']==NODE_RUN_ID
assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] and 'model_exit_code' not in supervisor
assert provider['phase']=='READY'
assert provider['configuration']['node']==NODE
assert provider['configuration']['protocol']=='device_sdma' and provider['configuration']['init_bm'] is True
assert provider['configuration']['dram_size']==('160GB' if IS_D else str((136 if NODE==34 else 112))+'GB')
top=subprocess.run(['docker','top',CID,'-eo','pid,args'],capture_output=True,text=True,timeout=20);assert top.returncode==0
matches=[]
for line in top.stdout.splitlines()[1:]:
    fields=line.split(None,1)
    if len(fields)!=2 or not fields[0].isdigit():continue
    pid=int(fields[0]);proc=pathlib.Path('/proc')/str(pid)
    try:
        nspid=next(row.split()[1:] for row in (proc/'status').read_text().splitlines() if row.startswith('NSpid:'))
    except (FileNotFoundError,StopIteration):
        continue
    if int(nspid[-1])==supervisor['model_pid']:
        assert CID in (proc/'cgroup').read_text()
        env=dict(item.split(b'=',1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
        selected={key:env.get(key.encode(),b'').decode() for key in [
            'DEEP_USE_MODE','PYTHONPATH','ASCEND_MF_TRANSFER_PROTOCOL','SGLANG_MOONCAKE_SEND_AUX_TCP',
            'PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE',
            'GLM53_DEEPEP_METADATA_BATCH','GLM53_NODE','GLM53_BUFFER_DEDUP_VALIDATE','GLM53_PACKED_DISPATCH',
            'HCCL_BUFFSIZE','DEEPEP_HCCL_BUFFSIZE','TASK_QUEUE_ENABLE',
            'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK','GLM53_C32_TARGET_GRAPH_ALIGN',
            'GLM53_C32_OPS_AUDIT','GLM53_C32_NATIVE_HICACHE_MTP','GLM53_C32_NATIVE_MC_MAPPING_REUSE',
            'SGLANG_HICACHE_HOST_MEM_BACKEND','MEMFABRIC_HYBRID_EXTEND_LIB_PATH','LD_LIBRARY_PATH',
            'SGLANG_EAGER_INPUT_NO_COPY','GLM53_EAGER_INPUT_AUDIT','GLM53_EAGER_INPUT_AUDIT_ROOT']}
        assert not selected['GLM53_PACKED_DISPATCH']
        assert selected['HCCL_BUFFSIZE']==('256' if IS_D else '')
        assert selected['DEEPEP_HCCL_BUFFSIZE']=='' and selected['GLM53_C32_OPS_AUDIT']==''
        assert selected['TASK_QUEUE_ENABLE']==('0' if IS_D else '')
        assert selected['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK']==('32' if IS_D else '')
        assert selected['GLM53_C32_TARGET_GRAPH_ALIGN']==('1' if IS_D else '')
        assert selected['DEEP_USE_MODE']==('ops' if IS_D else 'alltoall')
        assert selected['ASCEND_MF_TRANSFER_PROTOCOL']=='sdma'
        assert selected['SGLANG_MOONCAKE_SEND_AUX_TCP']=='0'
        assert selected['PYTHONHASHSEED']=='0' and selected['ASCEND_ENABLE_USE_FABRIC_MEM']=='1' and selected['ACL_OP_INIT_MODE']=='1'
        assert selected['PYTHONPATH'].startswith('/opt/glm53/source/python:') and 'site-packages' in selected['PYTHONPATH']
        assert selected['MMC_LOCAL_CONFIG_PATH']=='/opt/glm53/source/config/local.conf'
        assert selected['GLM53_DEEPEP_METADATA_BATCH']==('1' if IS_D else '')
        assert selected['GLM53_NODE']==(str(NODE) if IS_D or NATIVE_P else '')
        assert selected['GLM53_C32_NATIVE_HICACHE_MTP']==('1' if NATIVE_P else '')
        assert selected['GLM53_C32_NATIVE_MC_MAPPING_REUSE']==('1' if NATIVE_P else '')
        assert selected['SGLANG_HICACHE_HOST_MEM_BACKEND']==('memfabric' if NATIVE_P else '')
        assert selected['MEMFABRIC_HYBRID_EXTEND_LIB_PATH']==''
        assert selected['GLM53_BUFFER_DEDUP_VALIDATE']==''
        if IS_D and D_VARIANT=='candidate':
            assert selected['SGLANG_EAGER_INPUT_NO_COPY']=='1'
            assert selected['GLM53_EAGER_INPUT_AUDIT']=='1'
            assert selected['GLM53_EAGER_INPUT_AUDIT_ROOT']==EAGER_AUDIT_ROOT
        else:
            assert selected['SGLANG_EAGER_INPUT_NO_COPY']==''
            assert selected['GLM53_EAGER_INPUT_AUDIT']==''
            assert selected['GLM53_EAGER_INPUT_AUDIT_ROOT']==''
        argv=[x for x in (proc/'cmdline').read_bytes().decode().split('\0') if x]
        def option(name):
            assert name in argv,name
            return argv[argv.index(name)+1]
        assert option('--model-path')=='/data/wio/models/GLM-5.3-w8a8'
        assert option('--quantization')=='modelslim'
        assert option('--dtype')==option('--kv-cache-dtype')=='bfloat16'
        assert option('--tp-size')==option('--ep-size')=='32'
        assert option('--dp-size')==('16' if IS_D else '4')
        assert option('--speculative-num-steps')=='5' and option('--speculative-num-draft-tokens')=='6'
        assert option('--max-running-requests')==('64' if IS_D else '8')
        assert '--disable-overlap-schedule' in argv and '--disable-prefill-cuda-graph' in argv
        if IS_D:
            assert option('--deepep-mode')=='low_latency'
            assert option('--cuda-graph-backend-decode')=='full'
            assert option('--cuda-graph-max-bs-decode')=='4'
            assert '--disable-decode-cuda-graph' not in argv
        else:
            assert option('--deepep-mode')=='normal'
            assert '--disable-decode-cuda-graph' in argv
        matches.append({'host_pid':pid,'namespace_pid':supervisor['model_pid'],'selected_environment':selected,'actual_model_argv':argv})
assert len(matches)==1,('Cannot identify exact live native model parent',top.stdout,supervisor)
model_identity=matches[0]
for relative,item in STAGED_FILES.items():
    assert hashlib.sha256((pathlib.Path(STAGED_ROOT)/relative).read_bytes()).hexdigest()==item['sha256'],relative
staged_supervisor=json.loads((pathlib.Path(STAGED_ROOT)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())
assert staged_supervisor['run_id']==NODE_RUN_ID and staged_supervisor['node']==NODE
if IS_D and D_VARIANT=='candidate':
    assert staged_supervisor['model_env']['SGLANG_EAGER_INPUT_NO_COPY']=='1'
    assert staged_supervisor['model_env']['GLM53_EAGER_INPUT_AUDIT']=='1'
    assert staged_supervisor['model_env']['GLM53_EAGER_INPUT_AUDIT_ROOT']==EAGER_AUDIT_ROOT
    audit_path=pathlib.Path(STAGED_ROOT)/'python/glm53_C32_L17_eager_input_audit_v2.py'
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==AUDIT_SHA
    site=(pathlib.Path(STAGED_ROOT)/'python/sitecustomize.py').read_text(encoding='utf-8')
    assert 'glm53_C32_L17_eager_input_audit_v2' in site
else:
    assert 'SGLANG_EAGER_INPUT_NO_COPY' not in staged_supervisor['model_env']
    assert 'GLM53_EAGER_INPUT_AUDIT' not in staged_supervisor['model_env']
    assert not (pathlib.Path(STAGED_ROOT)/'python/glm53_C32_L17_eager_input_audit_v2.py').exists()
    site=(pathlib.Path(STAGED_ROOT)/'python/sitecustomize.py').read_text(encoding='utf-8')
    assert 'glm53_C32_L17_eager_input_audit_v2' not in site
mounts=[x for x in d['Mounts'] if x['Destination']==P_OPTIONAL_TARGET]
if NATIVE_P:
    assert len(mounts)==1 and mounts[0]['RW'] is False
    assert mounts[0]['Source']==STAGED_ROOT+'/'+P_OPTIONAL_RELATIVE
    assert hashlib.sha256(pathlib.Path(mounts[0]['Source']).read_bytes()).hexdigest()==P_OPTIONAL_SHA
else:
    assert not mounts
    assert 'libmf_hybm_accoffload' not in json.dumps(staged_supervisor)
print(json.dumps({'project_id':'glm53','node':NODE,'role':ROLE,'selected_P_variant':'candidate','selected_D_variant':D_VARIANT,'selected_variant_scope':'D_ONLY_NOT_P' if IS_D else None,'container_id':CID,'name':NAME,'run_id':NODE_RUN_ID,'model_identity':model_identity,'optional_extension_mounted':bool(mounts),'staged_files':len(STAGED_FILES),'model_requests':0}))
'''


def check_only():
    p_capacity = p_capacity_check(read_json(ROOT / P_CAPACITY))
    stages = {}
    for variant, spec in D_STAGES.items():
        bundle = load_stage(spec["stage"], spec["stage_sha256"], D_NODES, False, True)
        stages[variant] = {"stage_sha256": spec["stage_sha256"], "files": len(bundle["files"]), "run_id": spec["run_id"]}
    load_stage(P_STAGE, P_STAGE_SHA, P_NODES + D_NODES, True, False)
    return {
        "project_id": "glm53",
        "status": STATUS,
        "pass": True,
        "p_source": {"capacity": P_CAPACITY, "sha256": P_CAPACITY_SHA, "stage": P_STAGE, "stage_sha256": P_STAGE_SHA, "run_id": P_RUN_ID, "optional_extension_sha256": P_OPTIONAL_SHA},
        "d_stages": stages,
        "explicit_roles": {"selected_P_variant": "candidate", "selected_D_variants": ["candidate", "rollback"]},
        "actual_D_capacity_required_for_render": True,
        "ssh": 0,
        "network": 0,
        "NPU_calls": 0,
        "SDK_init_calls": 0,
        "model_requests": 0,
        "runtime_authorized": False,
        "model_start_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--p-capacity", default=str(ROOT / P_CAPACITY))
    parser.add_argument("--d-capacity")
    parser.add_argument("--d-variant", choices=("candidate", "rollback"))
    parser.add_argument("--outdir")
    args = parser.parse_args()
    p_capacity_path = pathlib.Path(args.p_capacity)
    if pathlib.Path(args.p_capacity).resolve() == (ROOT / P_CAPACITY).resolve():
        p_capacity = p_capacity_check(read_json(p_capacity_path))
    else:
        assert sha256_file(p_capacity_path) == P_CAPACITY_SHA
        p_capacity = p_capacity_check(read_json(p_capacity_path))
    if args.render:
        assert args.d_capacity and args.d_variant and args.outdir, "--render requires actual D capacity/variant/outdir"
        d_capacity = read_json(args.d_capacity)
        d_capacity_check(d_capacity, args.d_variant)
        plans = {node: render(node, p_capacity, d_capacity, args.d_variant) for node in P_NODES + D_NODES}
        analyses = {name: {"sha256": hashlib.sha256(code.encode()).hexdigest(), "bytes": len(code.encode())} for name, code in plans.items()}
        outdir = pathlib.Path(args.outdir)
        assert not outdir.exists(), outdir
        outdir.mkdir(parents=True)
        for node, code in plans.items():
            (outdir / ("identity-%d.py" % node)).write_text(code, encoding="utf-8")
        manifest = {"project_id": "glm53", "status": STATUS, "selected_P_variant": "candidate", "selected_D_variant": args.d_variant, "d_capacity_sha256": sha256_file(args.d_capacity), "plans": analyses, "runtime_authorized": False, "model_start_authorized": False}
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0
    result = check_only()
    if args.d_capacity and args.d_variant:
        d_capacity = read_json(args.d_capacity)
        d_capacity_check(d_capacity, args.d_variant)
        plans = {node: render(node, p_capacity, d_capacity, args.d_variant) for node in P_NODES + D_NODES}
        result["rendered_plan_hashes"] = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in plans.items()}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
