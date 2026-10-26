"""DRAFT_PENDING_PRIMARY_REVIEW: D36/41-only lifecycle host-plan library.

This module renders host code and performs local hashing/AST work only. Import and
`--write-plan` make zero SSH, HTTP, NPU, SDK or model calls. The remote actions are
owned by scripts/run_C32_L17_controlled_D_transition_and_pool_v1.py after a separate
primary review freezes every plan SHA.

Fixed scope:
- stop only the exact retained D36/D41 containers (old fullmesh-v3 run);
- keep P34/P35, the shared Meta container and vLLM untouched;
- stage-time candidate-v2 and rollback-v1 bundles only, already staged on D36/41;
- prepare new D36/D41 run roots and candidate pools, no Meta start, no model start;
- candidate/rollback model-start and observe plans stay parameterized until the
  independent local pool-capacity audit pins the actual new container IDs.

No lifecycle launcher may use a guessed service ID or an old D container ID.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import pathlib
import re
import symtable
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ID = "glm53"
STATUS = "DRAFT_PENDING_PRIMARY_REVIEW"

TRANSITION_RID = "20261019T-L17-D-transition-and-pool-v1"
STOP_RID = "20261019T-L17-D-stop-current-v1"
RUN = ROOT / "runs" / TRANSITION_RID
STOP_RUN = ROOT / "runs" / STOP_RID
CAPACITY_OUT = "results/sglang-C32-L17-selected-D-pool-capacity-v1.json"
REVIEW = "results/C32-L17-D-transition-actual-review-v1.json"
LOCAL_RESULT = "results/C32-L17-D-transition-local-preparation-v1.json"

STAGE_RID = "20261019T-L17-D-native-eager-stage-v3"
STAGE_RECEIPT = "runs/20261019T-L17-D-native-eager-stage-v3/receipt.json"
STAGE_REVIEW = "results/C32-L17-D-native-eager-stage-actual-review-v3.json"
STAGE_LAUNCHER = "scripts/stage_C32_L17_D_native_eager_and_rollback_v3.py"
STAGE_LAUNCHER_SHA = "6e9b5dcd486b4977521a280248d8fa77ae3b6593e11dc30184437145ff39978a"

CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
RUNTIME = "config/C32-target-graph-runtime-v3/runtime.json"
RUNTIME_SHA = "65bbdf472b6bcdb0c7569836d434a1bde3d59da1a4bcda7d33dc2718ebbd5368"

BASE_POOL = "scripts/start_sglang_C32_target_graph_pool_v3.py"
BASE_POOL_SHA = "8637bad923b3c231cdf0e50616e91f4aab0a39de4673114fa5009d997e377ca3"
STOP_REFERENCE = "scripts/stop_sglang_C32_L04_for_target_graph_v1.py"
STOP_REFERENCE_SHA = "bc4f1fa1e3d42a0a78aaf8372271921f6f83d5c9a8fd729e8008e9670e462846"
MODEL_START_REFERENCE = "scripts/start_sglang_C32_target_graph_models_v4.py"
MODEL_START_REFERENCE_SHA = "13c8673ee518f9c32d81bbaa510f3ca6ffee3086bdc177f68e3178f090545e0f"
P_LIBRARY_REFERENCE = "scripts/glm53_C32_L12C_P_transition_library_v1.py"
P_LIBRARY_REFERENCE_SHA = "ac856e9ead23a242bf07466a2a6b5899cae67f43bc377c34e1e3ec106c7f87d1"
P_RUNNER_REFERENCE = "scripts/run_C32_L12C_controlled_P_transition_and_pool_v1.py"
P_RUNNER_REFERENCE_SHA = "007eb50f1a1a63292cf53bf3ebd8e8eb70d0e879177893d20a20f857f5c27157"
STAGE_HELPER = "scripts/stage_sglang_C32_packed_normal_and_rollback_v1.py"
STAGE_HELPER_SHA = "13545bf1c90368d57113dc810738da12c64d3efa5c01ba3fd38057e21cb1724f"
MIXED_IDENTITY = "scripts/glm53_C32_L12C_mixed_model_identity_v1.py"
MIXED_IDENTITY_SHA = "ff5a92342a3a385256b619e9c004294b881634678326d706eca026413c5e6a98"

IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"
META_CID = "0dd00f5c35ee8c60cbd697a38fc73a8c37ae498106f9ebfc2933139e953b165c"
META_NAME = "glm53-sglang-C32-target-graph-fullmesh-v3-meta-20261018-34"
OLD_D_RID = "20261018T-L10-sglang-C32-target-graph-fullmesh-v3"
OLD_D_ROOT = "/data/wio/inference-foundry/glm53/runs/" + OLD_D_RID
OLD_D = {
    36: {
        "container_id": "25ecbd3a55e199caf4c6012a61b702d1b6035c955406d75bb17107991a932dd1",
        "name": "glm53-sglang-C32-target-graph-fullmesh-v3-20261018-36",
        "local_service_id": 2,
        "dram_GiB": 160,
    },
    41: {
        "container_id": "c39fa7c594a8776e70c365baec435d4288abb1db5cc492405183047d57b805c3",
        "name": "glm53-sglang-C32-target-graph-fullmesh-v3-20261018-41",
        "local_service_id": 3,
        "dram_GiB": 160,
    },
}
D_NODES = (36, 41)
P_NODES = (34, 35)
D_PORTS = (29440, 30600)
MODEL_FILES = 193
MODEL_BYTES = 764240829290
MODEL_CONFIG_SHA = "ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca"
MODEL_TOKENIZER_SHA = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
MEM_AVAILABLE_FLOOR = 1100 * 1024 ** 3
CAPACITY_FLOOR_TOKENS = 161280
OLD_D_CAPACITY_TOKENS = 181632
EAGER_SHA = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
ENVIRON_SHA = "0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c"
AUDIT_SHA = "0df4276b5e5f8d10aec1faf25cd24e847dc3ef9a57570f70f257c0ebf539d56f"
SDMA_MANIFEST_SHA = "0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528"
EAGER_AUDIT_ROOT = "/run/glm53/eager-input-audit"
ALIGNMENT_AUDIT_ROOT = "/run/glm53/alignment-audit"
NATIVE_PREFLIGHT_BASIS = "D-only derivation of BASE_POOL.native_preflight: P buffer_only/9GB, optional-extension maps and guard.prepare_sdk are removed; D NoCopy/audit marker plus local.conf/meta.conf checks are added. Framework import happens only after provider READY; no SDK init or engine creation."

STAGES = {
    "candidate": {
        "stage": "config/sglang-C32-L17-D-native-eager-stage-v2.json",
        "stage_sha256": "a8d6759af5f29a3da15e60b1312bc81eae046e93c99ca3eb366a78bdf055da0c",
        "bundle": "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2",
        "bundle_manifest": "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2/bundle-manifest.json",
        "bundle_manifest_sha256": "9427ba4316f5b5075b97afecb01eca062caa912786280897fbf962eb1d498491",
        "remote": "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-native-eager-v2-20261019",
        "run_id": "20261019T-L17-D-native-eager-v2",
        "run_root": "/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-native-eager-v2",
        "name_prefix": "glm53-L17-D-native-eager-v2",
        "file_count": 23,
        "candidate": True,
    },
    "rollback": {
        "stage": "config/sglang-C32-L17-D-original-rollback-stage-v1.json",
        "stage_sha256": "f8b49216cd3ac6651cb48f9308a2590d2c0e306b339355e568a522bb882ddcf0",
        "bundle": "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1",
        "bundle_manifest": "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1/bundle-manifest.json",
        "bundle_manifest_sha256": "b3e1d5824eed9e86b61e1a20a0f0a46732c5caca543d83e1310d62e4a171cd7a",
        "remote": "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-original-rollback-v1-20261019",
        "run_id": "20261019T-L17-D-original-rollback-v1",
        "run_root": "/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-original-rollback-v1",
        "name_prefix": "glm53-L17-D-original-rollback-v1",
        "file_count": 22,
        "candidate": False,
    },
}

SOURCE_PROVENANCE = {
    STAGE_LAUNCHER: STAGE_LAUNCHER_SHA,
    "results/C32-L17-D-native-eager-stage-actual-review-v3.json": "c176440c0440b9ea41daf057ebca87dbec1b77a26eea2762b11d3c08678f722b",
    STAGE_RECEIPT: None,
    CAPACITY: CAPACITY_SHA,
    RUNTIME: RUNTIME_SHA,
    BASE_POOL: BASE_POOL_SHA,
    STOP_REFERENCE: STOP_REFERENCE_SHA,
    MODEL_START_REFERENCE: MODEL_START_REFERENCE_SHA,
    P_LIBRARY_REFERENCE: P_LIBRARY_REFERENCE_SHA,
    P_RUNNER_REFERENCE: P_RUNNER_REFERENCE_SHA,
    STAGE_HELPER: STAGE_HELPER_SHA,
    MIXED_IDENTITY: MIXED_IDENTITY_SHA,
    "config/sglang-C32-L17-D-native-eager-stage-v2.json": STAGES["candidate"]["stage_sha256"],
    "config/sglang-C32-L17-D-original-rollback-stage-v1.json": STAGES["rollback"]["stage_sha256"],
    "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2/bundle-manifest.json": STAGES["candidate"]["bundle_manifest_sha256"],
    "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1/bundle-manifest.json": STAGES["rollback"]["bundle_manifest_sha256"],
    "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2/python/glm53_C32_L17_eager_input_audit_v2.py": AUDIT_SHA,
    "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2/manifest.json": SDMA_MANIFEST_SHA,
    "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1/manifest.json": SDMA_MANIFEST_SHA,
}

REVIEW_SCHEMA = {
    "review_status": "APPROVED_FOR_D_STOP_AND_CANDIDATE_POOL_ONLY",
    "runtime_authorized": True,
    "model_start_authorized": False,
    "host_plans_sha256": {},
    "capacity_output": CAPACITY_OUT,
    "stage_receipt_sha256": "sha256 of " + STAGE_RECEIPT,
    "note": "Model start/observe require a later review rendered from the actual capacity CIDs.",
}


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_module(name, relative, expected_sha):
    path = ROOT / relative
    assert sha256_file(path) == expected_sha, "source SHA mismatch: %s" % relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transport_host(node):
    helper = load_module("glm53_l17_stage_transport_%d" % time.monotonic_ns(),
                         STAGE_HELPER, STAGE_HELPER_SHA)
    pod, host = helper.b.NODES[node]
    assert pod and host
    return host


def render(template, values):
    """Substitute __TOKEN__ placeholders with Python-literal values."""
    text = template
    for key, value in values.items():
        token = "__%s__" % key
        text = text.replace(token, repr(value))
    leftovers = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", text)))
    assert not leftovers, "unrendered placeholders: %r" % leftovers
    ast.parse(text)
    return text


def render_bound(template, values, keys):
    """Render with an explicit module-level binding header for every constant."""
    missing = [key for key in keys if key not in values]
    assert not missing, "missing bindings: %r" % missing
    header = "".join("%s = %r\n" % (key, values[key]) for key in keys)
    body = render(template, {key: value for key, value in values.items() if key not in keys})
    return header + body


def stage(kind):
    spec = STAGES[kind]
    assert sha256_file(ROOT / spec["stage"]) == spec["stage_sha256"]
    assert sha256_file(ROOT / spec["bundle_manifest"]) == spec["bundle_manifest_sha256"]
    stage_json = read_json(ROOT / spec["stage"])
    assert stage_json.get("project_id") == PROJECT_ID
    assert len(stage_json["bundles"]) == 1
    bundle = stage_json["bundles"][0]
    assert stage_json.get("review_status", bundle.get("review_status")) == "PENDING_PRIMARY_REVIEW"
    assert stage_json.get("runtime_authorized", bundle.get("runtime_authorized")) is False
    assert sorted(bundle["nodes"]) == list(D_NODES)
    assert bundle["remote"] == spec["remote"]
    assert bundle["runtime_run_id"] == spec["run_id"]
    assert bundle["remote_run_root"] == spec["run_root"]
    assert bundle["image"] == IMAGE
    files = dict(bundle["files"])
    assert len(files) == spec["file_count"]
    assert "bundle-manifest.json" not in files
    assert not any(name in files for name in (
        "config/node-34.json", "config/node-35.json",
        "provider-config/node-34.json", "provider-config/node-35.json",
        "supervisor-config/node-34.json", "supervisor-config/node-35.json",
    ))
    assert "private_optional_extension" not in json.dumps(bundle)
    assert "libmf_hybm_accoffload" not in json.dumps(bundle)
    for name, meta in files.items():
        assert sha256_file(ROOT / meta["local"]) == meta["sha256"], name
    candidate = spec["candidate"]
    audit_path = ROOT / spec["bundle"] / "python/glm53_C32_L17_eager_input_audit_v2.py"
    site = (ROOT / spec["bundle"] / "python/sitecustomize.py").read_text(encoding="utf-8")
    if candidate:
        assert stage_json.get("audit_version") == "v2"
        assert stage_json.get("audit_hook_sha256") == AUDIT_SHA
        assert stage_json.get("environ_source_sha256") == ENVIRON_SHA
        assert stage_json.get("eager_source_sha256") == EAGER_SHA
        assert sha256_file(audit_path) == AUDIT_SHA
        assert "glm53_C32_L17_eager_input_audit_v2" in site
        node = read_json(ROOT / spec["bundle"] / "config/node-36.json")
        assert node["environment_intent"]["SGLANG_EAGER_INPUT_NO_COPY"] == "1"
        assert node["environment_intent"]["GLM53_EAGER_INPUT_AUDIT"] == "1"
        assert node["environment_intent"]["GLM53_EAGER_INPUT_AUDIT_ROOT"] == EAGER_AUDIT_ROOT
    else:
        assert "audit_version" not in stage_json
        assert not audit_path.exists()
        assert "glm53_C32_L17_eager_input_audit_v2" not in site
        node = read_json(ROOT / spec["bundle"] / "config/node-36.json")
        assert "SGLANG_EAGER_INPUT_NO_COPY" not in node["environment_intent"]
        assert "GLM53_EAGER_INPUT_AUDIT" not in node["environment_intent"]
    for node_id in D_NODES:
        node = read_json(ROOT / spec["bundle"] / ("config/node-%d.json" % node_id))
        sup = read_json(ROOT / spec["bundle"] / ("supervisor-config/node-%d.json" % node_id))
        prov = read_json(ROOT / spec["bundle"] / ("provider-config/node-%d.json" % node_id))
        assert node["node"] == node_id and sup["node"] == node_id and prov["node"] == node_id
        assert node["parallel_intent"]["launch_tp"] == 32
        assert node["parallel_intent"]["attention_dp"] == 16
        assert node["parallel_intent"]["attention_tp"] == 2
        assert node["parallel_intent"]["expert_parallel"] == 32
        assert sup["run_id"] == spec["run_id"]
        assert prov["dram_size"] == "160GB" and prov["max_dram_size"] == "160GB"
        assert prov["protocol"] == "device_sdma" and prov["init_bm"] is True
        assert "local_service_id" not in prov
    enriched = dict(bundle)
    enriched["files"] = files
    enriched["bundle_manifest_sha256"] = spec["bundle_manifest_sha256"]
    enriched["name_prefix"] = spec["name_prefix"]
    enriched["candidate"] = candidate
    return enriched


def base_pool():
    return load_module("glm53_l17_base_pool_%d" % time.monotonic_ns(),
                       BASE_POOL, BASE_POOL_SHA)


def pool(kind):
    spec = STAGES[kind]
    bundle = stage(kind)
    module = base_pool()
    module.RID = spec["run_id"]
    module.REMOTE = spec["run_root"]
    module.STAGED = spec["remote"]
    module.BUNDLE = bundle
    module.FILES = bundle["files"]
    module.RUN = ROOT / "runs" / spec["run_id"]
    module.name = lambda node, meta=False: spec["name_prefix"] + "-20261019-" + str(node)
    return module, bundle


def pool_code(kind, node):
    module, bundle = pool(kind)
    original = module.invoke
    module.invoke = lambda node_id, code, timeout=120: code
    try:
        code = module.start(node)
    finally:
        module.invoke = original
    # The reviewed base start() imports time but references socket in its
    # retained Meta branch. D uses the provider branch, yet the generated host
    # program must have no unbound global for any branch.
    import_line = "import pathlib,subprocess,json,time\n"
    assert code.count(import_line) == 1
    code = code.replace(import_line, "import pathlib,subprocess,json,time,socket\n")
    assert "libmf_hybm_accoffload" not in code
    assert "private_optional_extension" not in code
    for needle in (
        "'--network=host'",
        "'--pids-limit=32768'",
        "'--memory=1700g'",
        "'--shm-size=64g'",
        "/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro",
        "'ASCEND_VISIBLE_DEVICES=0-15'",
        "/opt/glm53/source/supervisor.py",
        "/opt/glm53/source:ro",
    ):
        assert needle in code, needle
    assert code.count("'--device'") == 19
    return code


def health_plans():
    launcher = load_module("glm53_l17_stage_v3", STAGE_LAUNCHER, STAGE_LAUNCHER_SHA)
    _, _, codes, cpu_code = launcher.plan()
    review = read_json(ROOT / STAGE_REVIEW)
    assert review["review_status"] == "APPROVED_FOR_STAGE_ONLY"
    assert review["host_code_sha256"] == {name: sha256_text(code) for name, code in codes.items()}
    assert review["CPU_code_sha256"] == sha256_text(cpu_code)
    return codes


META_PLAN = r'''
import pathlib,subprocess,json,hashlib,re
def cmd(args,timeout=30):
    process=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    return process
r=cmd(['docker','inspect',META_ID]);assert r.returncode==0,r.stderr
d=json.loads(r.stdout)[0]
assert d['Id']==META_ID and d['Image']==IMAGE and d['Name']=='/'+META_NAME
assert d['Config']['Labels']['project_id']=='glm53'
assert d['Config']['Labels']['run_id']==SHARED_RID
assert d['State']['Running']
root=pathlib.Path(OUTPUT);assert not root.exists();root.mkdir(parents=True)
destination=root/'memcache_hybrid'
r=cmd(['docker','cp',META_ID+':/var/log/memcache_hybrid',str(destination)],120)
assert r.returncode==0,r.stderr
pattern=r'Mount bm on loc|Unmount|UnMount|ock\.mmc\.meta\.lease_ttl_ms|ock\.mmc\.evict_threshold_'
files=[];lines=[]
for path in sorted(destination.rglob('*')):
    if path.is_file():
        raw=path.read_bytes();assert len(raw)<=256*1024**2
        text=raw.decode(errors='replace')
        critical=[line for line in text.splitlines() if re.search(pattern,line,re.I)]
        assert len(critical)<=200000
        files.append({'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'critical_count':len(critical)})
        lines.extend(critical)
print(json.dumps({'project_id':'glm53','container_id':META_ID,'meta_name':META_NAME,'shared_run_id':SHARED_RID,'files':files,'critical_lines':lines,'critical_count':len(lines),'critical_sha256':hashlib.sha256(json.dumps(lines).encode()).hexdigest(),'model_requests':0}))
'''


def meta_plan(phase):
    assert phase in ("before", "after")
    values = {
        "META_ID": META_CID,
        "META_NAME": META_NAME,
        "IMAGE": IMAGE,
        "SHARED_RID": OLD_D_RID,
        "OUTPUT": "/data/wio/inference-foundry/glm53/runs/" + TRANSITION_RID + "-meta-" + phase,
    }
    return render_bound(META_PLAN, values, ["META_ID", "META_NAME", "IMAGE", "SHARED_RID", "OUTPUT"])


STOP_COMMON = r'''
import pathlib,subprocess,json,hashlib,socket,urllib.request,time
assert socket.gethostname()==HOSTNAME

def cmd(args,timeout=30):
    return subprocess.run(args,capture_output=True,text=True,timeout=timeout)

def must(args,timeout=30):
    result=cmd(args,timeout);assert result.returncode==0,(args,result.returncode,result.stderr);return result.stdout

def owned(cid,name):
    d=json.loads(must(['docker','inspect',cid]))[0]
    assert d['Id']==cid and d['Name']=='/'+name and d['Image']==IMAGE
    assert d['Config']['Labels']['project_id']=='glm53'
    assert d['Config']['Labels']['run_id']==OLD_RID
    assert not d['HostConfig']['Privileged']
    return d

def file_identity(path):
    assert path.is_file() and not path.is_symlink(),str(path)
    digest=hashlib.sha256();size=0
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(block);size+=len(block)
    return {'path':str(path),'bytes':size,'sha256':digest.hexdigest()}

def profile_metadata(root):
    entries=[(str(path.relative_to(root)),path.stat().st_size,path.stat().st_mtime_ns) for path in sorted(root.rglob('*')) if path.is_file() and not path.is_symlink()]
    return {'directory':str(root),'exists':root.exists(),'file_count':len(entries),'bytes':sum(item[1] for item in entries),'metadata_sha256':hashlib.sha256(json.dumps(entries).encode()).hexdigest(),'entries':entries}

def idle():
    d=owned(CID,NAME);assert d['State']['Running']
    shared=pathlib.Path(RUN_ROOT)
    supervisor=json.loads((shared/'supervisor-status.json').read_text())
    provider=json.loads((shared/'provider-status.json').read_text())
    assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] is True
    assert 'model_exit_code' not in supervisor
    assert provider['phase']=='READY'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://172.16.12.'+str(NODE)+':'+str(D_PORT)+'/metrics',timeout=15) as response:
        raw=response.read(16*1024**2+1)
    assert len(raw)<=16*1024**2
    gauges=[line for line in raw.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
    assert gauges and all(float(line.rsplit(' ',1)[1])==0 for line in gauges)
    owned_rows=[];empty_cards=0
    for line in must(['npu-smi','info']).split('Process id',1)[-1].splitlines():
        columns=[value.strip() for value in line.split('|')]
        if len(columns)>=5 and len(columns[1].split())==2 and all(value.isdigit() for value in columns[1].split()) and columns[2].isdigit():
            pid=int(columns[2])
            assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text()
            owned_rows.append(pid)
    assert len(owned_rows)==18,'unexpected NPU context; preserve services'
    return {'container_id':CID,'name':NAME,'state':d['State'],'supervisor':supervisor,'provider':provider,'idle_gauges':gauges,'metrics_sha256':hashlib.sha256(raw).hexdigest(),'owned_NPU_process_pids':owned_rows}
'''

STOP_PREFLIGHT = r'''
for bundle in STAGED_BUNDLES:
    for relative,expected in bundle['files'].items():
        assert hashlib.sha256((pathlib.Path(bundle['remote'])/relative).read_bytes()).hexdigest()==expected,relative
    assert hashlib.sha256((pathlib.Path(bundle['remote'])/'bundle-manifest.json').read_bytes()).hexdigest()==bundle['bundle_manifest_sha256']
out=idle()
assert not pathlib.Path(OUTDIR).exists()
out.update(node=NODE,model_requests=0,preflight_pass=True,old_service_id=OLD_SERVICE_ID)
print(json.dumps(out))
'''

STOP_BODY = r'''
before=idle()
shared=pathlib.Path(RUN_ROOT)
profile=shared/'profile'
profile_before=profile_metadata(profile)
logs_before={stream:file_identity(shared/('model.'+stream+'.txt')) for stream in ['stdout','stderr']}
supervisor_before=json.loads((shared/'supervisor-status.json').read_text())
provider_before=json.loads((shared/'provider-status.json').read_text())
root=pathlib.Path(OUTDIR);assert not root.exists();root.mkdir(parents=True)
stop_result=cmd(['docker','stop','--time','240',CID],270)
after=owned(CID,NAME)
assert not after['State']['Running'] and after['State']['Pid']==0
supervisor=json.loads((shared/'supervisor-status.json').read_text())
provider=json.loads((shared/'provider-status.json').read_text())
npu=must(['npu-smi','info'])
empty=npu.count('No running processes')==8
provider_closed=bool(provider.get('events')) and provider['events'][-1].get('event')=='CLOSE_COMPLETE' and provider['events'][-1].get('rc')==0
resource_closure=stop_result.returncode==0 and empty and supervisor.get('phase')=='STOPPED' and provider.get('phase')=='CLOSE_COMPLETE' and provider_closed
clean_model=supervisor.get('model_exit_code')==0 and supervisor.get('timeout_used') is False and after['State']['ExitCode']==0
copies={}
for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:
    destination=root/directory.rsplit('/',1)[1]
    copied=cmd(['docker','cp',CID+':'+directory,str(destination)],120)
    copies[directory]={'rc':copied.returncode,'stderr':copied.stderr}
profile_after=profile_metadata(profile)
logs_after={stream:file_identity(shared/('model.'+stream+'.txt')) for stream in ['stdout','stderr']}
assert profile_after==profile_before,'old D profile metadata changed'
for stream in ['stdout','stderr']:
    assert logs_after[stream]['bytes']>=logs_before[stream]['bytes'],stream
(root/'profile-metadata-before.json').write_text(json.dumps(profile_before['entries'],separators=(',',':'))+'\n')
(root/'profile-metadata-after.json').write_text(json.dumps(profile_after['entries'],separators=(',',':'))+'\n')
profile_before_summary={key:value for key,value in profile_before.items() if key!='entries'}
profile_after_summary={key:value for key,value in profile_after.items() if key!='entries'}
print(json.dumps({'node':NODE,'container_id':CID,'name':NAME,'before':before,'supervisor_before':supervisor_before,'provider_before':provider_before,'stop_command_rc':stop_result.returncode,'after_state':after['State'],'supervisor':supervisor,'provider':provider,'all_NPUs_empty':empty,'provider_closed0':provider_closed,'resource_shutdown_complete':resource_closure,'clean_model_shutdown':clean_model,'model_shutdown_diagnostic':{'model_exit_code':supervisor.get('model_exit_code'),'timeout_used':supervisor.get('timeout_used'),'docker_exit_code':after['State']['ExitCode']},'SDK_copy':copies,'profile_before':profile_before_summary,'profile_after':profile_after_summary,'profile_metadata_list_files':['profile-metadata-before.json','profile-metadata-after.json'],'retained_model_logs_before':logs_before,'retained_model_logs_after':logs_after,'retained_model_and_diagnostic_files_in_place':str(shared),'model_requests':0,'only_exact_D_container_stopped':True}))
'''


def stop_values(node):
    item = OLD_D[node]
    return {
        "HOSTNAME": transport_host(node),
        "NODE": node,
        "CID": item["container_id"],
        "NAME": item["name"],
        "IMAGE": IMAGE,
        "OLD_RID": OLD_D_RID,
        "D_PORT": 30600,
        "RUN_ROOT": OLD_D_ROOT,
        "OUTDIR": "/data/wio/inference-foundry/glm53/runs/" + STOP_RID,
        "OLD_SERVICE_ID": item["local_service_id"],
        "STAGED_BUNDLES": [{"remote": bundle["remote"],
                            "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
                            "files": {name: meta["sha256"] for name, meta in bundle["files"].items()}}
                           for bundle in (stage("candidate"), stage("rollback"))],
    }


def stop_plans():
    plans = {}
    keys = ["HOSTNAME", "NODE", "CID", "NAME", "IMAGE", "OLD_RID", "D_PORT",
            "RUN_ROOT", "OUTDIR", "OLD_SERVICE_ID", "STAGED_BUNDLES"]
    for node in D_NODES:
        values = stop_values(node)
        plans["precheck-%d" % node] = render_bound(STOP_COMMON + STOP_PREFLIGHT, values, keys)
        plans["stop-%d" % node] = render_bound(STOP_COMMON + STOP_BODY, values, keys)
    return plans


PREPARE_PLAN = r'''
import pathlib,subprocess,json,hashlib
assert NODE in [36,41]
def cmd(args,timeout=30):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    return result
def must(args,timeout=30):
    result=cmd(args,timeout);assert result.returncode==0,(args,result.returncode,result.stderr);return result.stdout
d=json.loads(must(['docker','inspect',OLD_CID]))[0]
assert d['Id']==OLD_CID and not d['State']['Running'] and d['State']['Pid']==0
assert d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==OLD_RID
npu=must(['npu-smi','info']);assert npu.count('No running processes')==8
for relative,expected in FILES.items():
    assert hashlib.sha256((pathlib.Path(STAGED)/relative).read_bytes()).hexdigest()==expected,relative
assert hashlib.sha256((pathlib.Path(STAGED)/'bundle-manifest.json').read_bytes()).hexdigest()==BUNDLE_MANIFEST_SHA
root=pathlib.Path(REMOTE);assert not root.exists();root.mkdir(parents=True)
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8')
weights=[path for path in model.rglob('*') if path.is_file()]
assert len(weights)==193 and sum(path.stat().st_size for path in weights)==764240829290
assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()==MODEL_CONFIG_SHA
available=int(next(line.split()[1] for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
assert available>1100*1024**3
print(json.dumps({'node':NODE,'variant':KIND,'old_container_closed':True,'empty_NPU_cards':8,'model_weights_files':len(weights),'model_weights_bytes':sum(path.stat().st_size for path in weights),'MemAvailable':available,'new_runtime_root':str(root),'stage_files':len(FILES),'private_optional_extension_mounted':False,'model_requests':0}))
'''


def prepare_plan(kind, node):
    spec = STAGES[kind]
    bundle = stage(kind)
    return render_bound(PREPARE_PLAN, {
        "NODE": node,
        "KIND": kind,
        "OLD_CID": OLD_D[node]["container_id"],
        "OLD_RID": OLD_D_RID,
        "IMAGE": IMAGE,
        "FILES": {name: meta["sha256"] for name, meta in bundle["files"].items()},
        "STAGED": spec["remote"],
        "REMOTE": spec["run_root"],
        "BUNDLE_MANIFEST_SHA": spec["bundle_manifest_sha256"],
        "MODEL_CONFIG_SHA": MODEL_CONFIG_SHA,
    }, ["NODE", "KIND", "OLD_CID", "OLD_RID", "IMAGE", "FILES", "STAGED",
        "REMOTE", "BUNDLE_MANIFEST_SHA", "MODEL_CONFIG_SHA"])


PREFLIGHT_INNER = r'''
import pathlib,json,hashlib,os,sys
assert sys.path[1]=='/opt/glm53/source/python',sys.path
import torch,torch_npu
count=torch.npu.device_count();assert count==16 and torch.npu.is_available(),count
import acl
acl_count=acl.rt.get_device_count();assert list(acl_count)==[16,0],acl_count
root=pathlib.Path('/opt/glm53/source')
manifest=json.loads((root/'manifest.json').read_text())
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==SDMA_MANIFEST_SHA
meta_text=(root/'config/meta.conf').read_text(encoding='utf-8')
local_text=(root/'config/local.conf').read_text(encoding='utf-8')
for key,value in [('ock.mmc.meta.lease_ttl_ms','60000'),('ock.mmc.evict_threshold_high','99'),('ock.mmc.evict_threshold_low','98')]:
    assert (key+' = '+value) in meta_text,key
assert 'ock.mmc.local_service.dram.size = 0GB' in local_text
assert 'ock.mmc.local_service.max.dram.size = 160GB' in local_text
for entry in manifest['files']:
    assert hashlib.sha256(pathlib.Path(entry['container_path']).read_bytes()).hexdigest()==entry['patched_sha256'],entry['container_path']
for entry in manifest['helpers']:
    assert hashlib.sha256((root/entry['local']).read_bytes()).hexdigest()==entry['sha256'],entry['local']
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
assert AscendTransferEngine._get_transfer_protocol()=='sdma'
assert AscendTransferEngine._resolve_trans_op_type('sdma').value==1
from sglang.srt.environ import envs
assert envs.SGLANG_MOONCAKE_SEND_AUX_TCP.get() is False
from sglang.srt.server_args import prepare_server_args
from sglang.srt.arg_groups.overrides import resolved_view
config=json.loads((root/('config/node-'+str(NODE)+'.json')).read_text())
for key,value in config['environment_intent'].items():
    assert os.environ[key]==value,key
parallel=config['parallel_intent']
assert parallel['launch_tp']==32 and parallel['attention_dp']==16 and parallel['attention_tp']==2 and parallel['expert_parallel']==32
args=prepare_server_args(config['launch_args']);args.resolve_once();args.check_server_args();cfg=resolved_view(args)
assert cfg.tp_size==32 and cfg.ep_size==32 and cfg.dp_size==16
assert cfg.speculative_num_steps==5 and cfg.speculative_num_draft_tokens==6 and cfg.kv_cache_dtype=='bfloat16'
assert cfg.hicache_size==8 and cfg.hicache_host_memory_mode=='cache'
assert cfg.moe_a2a_backend=='deepep' and cfg.deepep_mode=='low_latency'
assert os.environ['DEEP_USE_MODE']=='ops'
assert os.environ['HCCL_BUFFSIZE']=='256' and os.environ['TASK_QUEUE_ENABLE']=='0'
assert os.environ['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK']=='32'
assert cfg.cuda_graph_config.decode.backend=='full' and cfg.cuda_graph_config.decode.max_bs==4
assert list(cfg.cuda_graph_config.decode.bs)==[1,2,3,4]
assert cfg.cuda_graph_config.prefill.backend=='disabled'
import deep_ep.strategies.low_latency_strategy as native_ops
assert native_ops.OpsLowLatencyCommStrategy._glm53_C32_target_graph
assert os.environ['GLM53_NODE']==str(NODE)
assert os.environ['ASCEND_MF_TRANSFER_PROTOCOL']=='sdma'
audit_path=root/'python/glm53_C32_L17_eager_input_audit_v2.py'
site=(root/'python/sitecustomize.py').read_text(encoding='utf-8')
if CANDIDATE:
    assert os.environ['SGLANG_EAGER_INPUT_NO_COPY']=='1'
    assert os.environ['GLM53_EAGER_INPUT_AUDIT']=='1'
    assert os.environ['GLM53_EAGER_INPUT_AUDIT_ROOT']==EAGER_AUDIT_ROOT
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==AUDIT_SHA
    assert 'glm53_C32_L17_eager_input_audit_v2' in site
    no_copy='true'
else:
    assert 'SGLANG_EAGER_INPUT_NO_COPY' not in os.environ
    assert 'GLM53_EAGER_INPUT_AUDIT' not in os.environ
    assert not audit_path.exists()
    assert 'glm53_C32_L17_eager_input_audit_v2' not in site
    no_copy='unset'
print('GLM53_L17_PREFLIGHT='+json.dumps({'node':NODE,'variant':KIND,'pass':True,'torch_device_count':count,'ACL_count':acl_count,'soc_name':acl.get_soc_name(),'tp':cfg.tp_size,'attention_tp':parallel['attention_tp'],'dp':cfg.dp_size,'ep':cfg.ep_size,'MTP_steps':cfg.speculative_num_steps,'MTP_draft_tokens':cfg.speculative_num_draft_tokens,'kv_cache_dtype':cfg.kv_cache_dtype,'hicache_size_decimal_GB':cfg.hicache_size,'hicache_host_memory_mode':cfg.hicache_host_memory_mode,'decode_graph_backend':cfg.cuda_graph_config.decode.backend,'decode_graph_bs':cfg.cuda_graph_config.decode.bs,'decode_graph_max_bs':cfg.cuda_graph_config.decode.max_bs,'HCCL_BUFFSIZE':os.environ.get('HCCL_BUFFSIZE'),'TASK_QUEUE_ENABLE':os.environ.get('TASK_QUEUE_ENABLE'),'dispatch_capacity':os.environ.get('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK'),'SDMA_protocol':'sdma','no_copy':no_copy,'candidate_audit_marker':CANDIDATE,'native_source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'engine_created':False,'NPU_compute_calls':0}))
'''

PREFLIGHT_HOST = r'''
import pathlib,subprocess,json
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20)
assert r.returncode==0,r.stderr
d=json.loads(r.stdout)[0]
assert d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53'
assert d['Config']['Labels']['run_id']==RID and d['State']['Running'] and not d['HostConfig']['Privileged']
root=pathlib.Path(REMOTE)
state=json.loads((root/'supervisor-status.json').read_text())
assert state['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and state['model_started'] is False
provider=json.loads((root/'provider-status.json').read_text())
assert provider['phase']=='READY'
configuration=provider['configuration']
assert configuration['dram_size']=='160GB' and configuration['max_dram_size']=='160GB'
assert configuration['protocol']=='device_sdma' and configuration['init_bm'] is True
provider_config=json.loads((pathlib.Path(STAGED)/('provider-config/node-'+str(NODE)+'.json')).read_text())
assert provider_config['dram_size']=='160GB' and provider_config['max_dram_size']=='160GB'
supervisor=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())
assert supervisor['run_id']==RID and supervisor['node']==NODE and 'PYTHONPATH' not in supervisor['model_env']
if CANDIDATE:
    assert supervisor['model_env']['SGLANG_EAGER_INPUT_NO_COPY']=='1'
    assert supervisor['model_env']['GLM53_EAGER_INPUT_AUDIT']=='1'
    assert supervisor['model_env']['GLM53_EAGER_INPUT_AUDIT_ROOT']==EAGER_AUDIT_ROOT
else:
    assert 'SGLANG_EAGER_INPUT_NO_COPY' not in supervisor['model_env']
    assert 'GLM53_EAGER_INPUT_AUDIT' not in supervisor['model_env']
bootstrap='set -eu; export PYTHONPATH=/opt/glm53/source/python:${PYTHONPATH:-}; exec python3 -c "$1"'
exec_args=['docker','exec',d['Id'],'env']+[key+'='+value for key,value in supervisor['model_env'].items()]+['/bin/bash','-c',bootstrap,'glm53-l17-d-preflight-env',INNER]
run=subprocess.run(exec_args,capture_output=True,text=True,timeout=180)
print(json.dumps({'node':NODE,'variant':KIND,'container_id':d['Id'],'name':NAME,'query_rc':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'supervisor_before':state,'supervisor_after':json.loads((root/'supervisor-status.json').read_text()),'model_started':False}))
raise SystemExit(run.returncode)
'''


def preflight_inner(kind, node):
    spec = STAGES[kind]
    return render_bound(PREFLIGHT_INNER, {
        "NODE": node,
        "KIND": kind,
        "CANDIDATE": spec["candidate"],
        "EAGER_AUDIT_ROOT": EAGER_AUDIT_ROOT,
        "AUDIT_SHA": AUDIT_SHA,
        "SDMA_MANIFEST_SHA": SDMA_MANIFEST_SHA,
    }, ["NODE", "KIND", "CANDIDATE", "EAGER_AUDIT_ROOT", "AUDIT_SHA",
        "SDMA_MANIFEST_SHA"])


def native_preflight_plan(kind, node):
    spec = STAGES[kind]
    inner = preflight_inner(kind, node)
    return render_bound(PREFLIGHT_HOST, {
        "NODE": node,
        "KIND": kind,
        "CANDIDATE": spec["candidate"],
        "NAME": spec["name_prefix"] + "-20261019-" + str(node),
        "IMAGE": IMAGE,
        "RID": spec["run_id"],
        "REMOTE": spec["run_root"],
        "STAGED": spec["remote"],
        "EAGER_AUDIT_ROOT": EAGER_AUDIT_ROOT,
        "INNER": inner,
    }, ["NODE", "KIND", "CANDIDATE", "NAME", "IMAGE", "RID", "REMOTE",
        "STAGED", "EAGER_AUDIT_ROOT", "INNER"])


MODEL_START_PLAN = r'''
import pathlib,json,subprocess,hashlib,socket
assert NODE in (36,41)
capacity=CAPACITY
assert capacity['project_id']=='glm53' and capacity['pass'] is True
assert capacity['selected_variant']==KIND
assert capacity['model_started'] is False
assert capacity['actual_MC_bytes']==568*1024**3
provider_row=capacity['providers'][str(NODE)]
assert provider_row['dram_GiB']==160 and provider_row['run_id']==RID and provider_row['remote_root']==REMOTE
assert provider_row['name']==NAME and provider_row['container_id']!=OLD_CID
service_id=provider_row['local_service_id'];assert type(service_id) is int and service_id>=0
root=pathlib.Path(REMOTE)
supervisor=json.loads((root/'supervisor-status.json').read_text())
provider=json.loads((root/'provider-status.json').read_text())
assert supervisor['run_id']==RID and supervisor['phase']=='POOL_READY_WAITING_MODEL_SIGNAL'
assert supervisor['model_started'] is False and provider['phase']=='READY'
ready_events=[event for event in provider['events'] if event['event']=='READY']
assert len(ready_events)==1 and ready_events[0]['local_service_id']==service_id
configuration=provider['configuration']
assert configuration['dram_size']=='160GB' and configuration['max_dram_size']=='160GB'
assert configuration['protocol']=='device_sdma' and configuration['init_bm'] is True
assert configuration['hbm_size']=='0GB' and configuration['max_hbm_size']=='0GB'
d=json.loads(subprocess.check_output(['docker','inspect',provider_row['container_id']],text=True))[0]
assert d['Id']==provider_row['container_id'] and d['Name']=='/'+NAME and d['Image']==IMAGE
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
assert d['State']['Running'] and not d['HostConfig']['Privileged']
supervisor_config=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())
if KIND=='candidate':
    assert supervisor_config['model_env']['SGLANG_EAGER_INPUT_NO_COPY']=='1'
    assert supervisor_config['model_env']['GLM53_EAGER_INPUT_AUDIT']=='1'
else:
    assert 'SGLANG_EAGER_INPUT_NO_COPY' not in supervisor_config['model_env']
    assert 'GLM53_EAGER_INPUT_AUDIT' not in supervisor_config['model_env']
for port in (29440,30600):
    sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
assert not (root/'model.stdout.txt').exists() and not (root/'model.stderr.txt').exists()
with (root/'model-start-controller.once').open('x') as handle:
    handle.write(d['Id'])
run=subprocess.run(['docker','kill','--signal=USR1',d['Id']],capture_output=True,text=True,timeout=20)
assert run.returncode==0 and run.stdout.strip()==d['Id'],run.stderr
print(json.dumps({'project_id':'glm53','node':NODE,'variant':KIND,'container_id':d['Id'],'name':NAME,'run_id':RID,'actual_local_service_id':service_id,'model_start_signal_sent_once':True,'model_ports_free':[29440,30600],'external_model_requests':0,'capacity_dict_sha256':CAPACITY_DICT_SHA}))
'''


def model_start_plan(kind, node, capacity):
    """Render only with a reviewed capacity dict whose actual CIDs are pinned."""
    spec = STAGES[kind]
    assert capacity.get("pass") is True
    assert capacity.get("selected_variant") == kind
    assert capacity.get("model_started") is False
    row = capacity["providers"][str(node)]
    assert row["dram_GiB"] == 160
    assert row["run_id"] == spec["run_id"]
    assert row["remote_root"] == spec["run_root"]
    assert row["name"] == spec["name_prefix"] + "-20261019-" + str(node)
    assert row["container_id"] != OLD_D[node]["container_id"]
    assert type(row["local_service_id"]) is int and row["local_service_id"] >= 0
    return render_bound(MODEL_START_PLAN, {
        "NODE": node,
        "KIND": kind,
        "NAME": row["name"],
        "IMAGE": IMAGE,
        "RID": spec["run_id"],
        "REMOTE": spec["run_root"],
        "STAGED": spec["remote"],
        "OLD_CID": OLD_D[node]["container_id"],
        "CAPACITY": capacity,
        "CAPACITY_DICT_SHA": sha256_text(json.dumps(capacity, sort_keys=True)),
    }, ["NODE", "KIND", "NAME", "IMAGE", "RID", "REMOTE", "STAGED", "OLD_CID",
        "CAPACITY", "CAPACITY_DICT_SHA"])


OBSERVE_PLAN = r'''
import pathlib,subprocess,json,time,urllib.request,urllib.error,hashlib,re
assert NODE in (36,41)
root=pathlib.Path(REMOTE)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def http(path,timeout=5):
    response=opener.open('http://172.16.12.'+str(NODE)+':'+str(PORT)+path,timeout=timeout)
    raw=response.read(8*1024**2+1);assert len(raw)<=8*1024**2
    return response.status,raw

def read_jsonl(directory,pattern,limit_files=256,limit_total=64*1024**2):
    paths=sorted(pathlib.Path(directory).glob(pattern));assert len(paths)<=limit_files
    rows=[];total=0
    for path in paths:
        raw=path.read_bytes();total+=len(raw);assert len(raw)<=16*1024**2 and total<=limit_total
        assert not raw or raw.endswith(b'\n')
        rows.extend(json.loads(line) for line in raw.splitlines())
    return paths,rows

def safe_jsonl(directory,pattern):
    try:
        return read_jsonl(directory,pattern)[1]
    except Exception:
        return []

def eager_summary(initialized):
    result={'candidate':CANDIDATE,'record_files':0,'record_pids':0,'audit_pids_match_SDK':False,'all_native_no_copy_true':None,'all_clone_distinct':None,'all_clone_fields_identity':None,'bs2_observed_pids':0,'bs2_all16':False,'reason':None}
    if not CANDIDATE:
        return result
    directory=root/'eager-input-audit'
    if not directory.is_dir():
        result['reason']='audit_dir_missing';return result
    files=sorted(directory.glob('pid-*.jsonl'))
    result['record_files']=len(files)
    if len(files)!=16:
        result['reason']='record_file_count';return result
    records=[]
    for path in files:
        raw=path.read_bytes()
        if len(raw)>16*1024**2 or (raw and not raw.endswith(b'\n')):
            result['reason']='record_bytes';return result
        try:
            records.extend(json.loads(line) for line in raw.splitlines())
        except Exception as error:
            result['reason']='record_parse';return result
    by_pid={}
    for record in records:
        by_pid.setdefault(record.get('pid'),[]).append(record)
    result['record_pids']=len(by_pid)
    if set(by_pid)!=set(initialized):
        result['reason']='pid_join';return result
    native_true=clone_distinct=fields_identity=True;bs2_pids=0;reason=None
    for pid,rows in by_pid.items():
        for record in rows:
            if record.get('schema')!='glm53.L17.eager_input_audit.v2' or record.get('audit_version')!='v2' or record.get('node')!=NODE:
                reason='record_identity';break
            if record.get('eager_source_sha256')!=EAGER_SHA or record.get('environ_source_sha256')!=ENVIRON_SHA:
                reason='record_source';break
            native_true=native_true and record.get('native_no_copy_get') is True
            clone_distinct=clone_distinct and record.get('clone_distinct') is True
            verdicts=record.get('field_verdicts') or {}
            fields_identity=fields_identity and verdicts.get('identity')==record.get('dataclass_field_count')
        if reason is not None:
            break
        if any(row.get('batch_size')==2 for row in rows):
            bs2_pids+=1
    result.update(record_pids=len(by_pid),records=len(records),audit_pids_match_SDK=reason is None,all_native_no_copy_true=native_true if reason is None else False,all_clone_distinct=clone_distinct if reason is None else False,all_clone_fields_identity=fields_identity if reason is None else False,bs2_observed_pids=bs2_pids,bs2_all16=reason is None and bs2_pids==16,reason=reason)
    return result

def ops_summary(initialized,align):
    ops={}
    for pid in initialized:
        rows=[event for event in align if event.get('pid')==pid]
        if not rows:
            return {}
        if {event.get('revision') for event in rows}!={'glm53-C32-target-graph-fullmesh-v2'}:
            return {}
        for method in ['_npu_low_latency_dispatch','_npu_low_latency_combine']:
            matches=[event for event in rows if event.get('event')=='ops_first_return' and event.get('method')==method]
            if len(matches)!=1 or matches[0].get('group_size')!=32 or matches[0].get('comm_alg')!='fullmesh_v1':
                return {}
        if not any(event.get('event')=='draft_graph_disabled' for event in rows):
            return {}
        productive=[event for event in rows if event.get('event')=='target_graph_decision' and event.get('eligible') is True and event.get('mode')=='TARGET_VERIFY' and event.get('raw_bs',0)>0 and event.get('is_draft') is False]
        if not productive:
            return {}
        first_return=[event for event in rows if event.get('event')=='ops_first_return']
        ops[str(pid)]={'rank':first_return[0]['rank'],'productive_target_verify_decisions':len(productive),'draft_graph_disabled':True,'fullmesh_v1':True,'native_ops_group_size':32}
    return ops

deadline=time.monotonic()+1800;last=0;summary=None
while True:
    supervisor=json.loads((root/'supervisor-status.json').read_text())
    assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] is True
    assert 'model_exit_code' not in supervisor
    provider=json.loads((root/'provider-status.json').read_text());assert provider['phase']=='READY'
    health_status=None;metrics_status=None;metric=None
    server={'status':None,'selected':None,'error':None}
    try:
        health_status,_=http('/health')
    except Exception as error:
        server['health_error']=repr(error)
    try:
        metrics_status,metric=http('/metrics')
    except Exception as error:
        server['metrics_error']=repr(error)
    gauges=[];idle=False
    if metric is not None:
        gauges=[float(line.rsplit(' ',1)[1]) for line in metric.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
        idle=bool(gauges) and not any(gauges)
    try:
        status,raw=http('/get_server_info',10)
        server={'status':status,'selected':{item['path']:item['value'] for item in json.loads(raw.decode())['selected']},'error':None}
    except urllib.error.HTTPError as error:
        server={'status':error.code,'selected':None,'error':'HTTPError'}
    except Exception as error:
        server={'status':None,'selected':None,'error':repr(error)}
    capacity=None;graph_config=None;server_ok=False
    if NODE==36 and server['status']==200 and server['selected'] is not None:
        selected=server['selected']
        expected=[('tp_size',32),('ep_size',32),('dp_size',16),('speculative_num_steps',5),('speculative_num_draft_tokens',6),('kv_cache_dtype','bfloat16'),('deepep_mode','low_latency')]
        server_ok=all(selected.get(key)==value for key,value in expected)
        capacity=selected.get('max_total_num_tokens')
        graph_config=(selected.get('cuda_graph_config') or {}).get('decode') or {}
        server_ok=server_ok and type(capacity) is int and capacity>=161280 and graph_config.get('backend')=='full' and graph_config.get('bs')==[1,2,3,4]
    elif NODE==41:
        server_ok=server['status']==404 and server['error']=='HTTPError'
    npu_rows=[]
    npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    if npu.returncode==0:
        for line in npu.stdout.split('Process id',1)[-1].splitlines():
            columns=[value.strip() for value in line.split('|')]
            if len(columns)>=5 and len(columns[1].split())==2 and all(value.isdigit() for value in columns[1].split()) and columns[2].isdigit():
                pid=int(columns[2])
                try:
                    cgroup=(pathlib.Path('/proc')/str(pid)/'cgroup').read_text()
                except OSError:
                    continue
                if CID in cgroup:npu_rows.append(pid)
    logs={stream:((root/('model.'+stream+'.txt')).read_text(errors='replace') if (root/('model.'+stream+'.txt')).exists() else '') for stream in ['stdout','stderr']}
    joined='\n'.join(logs.values())
    warm=('Disaggregation warmup requests completed for all 16 DP ranks' in joined) if NODE==36 else None
    fatal=('GLM53_EAGER_INPUT_AUDIT_FATAL' in joined) or ('GLM53_C32_ALIGNMENT_FATAL' in joined)
    events=safe_jsonl(root/'audit','sdma-*.jsonl')
    initialized={event['pid']:event for event in events if event.get('event')=='initialize_complete'}
    init_ok=len(initialized)==16 and all(event.get('rc')==0 and event.get('protocol')=='SDMA' and event.get('protocol_value')==1 for event in initialized.values())
    layouts=[event for event in events if event.get('event')=='model_state_registered']
    layout_ok=init_ok and len(layouts)==16 and {event.get('pid') for event in layouts}==set(initialized)
    if layout_ok:
        for event in layouts:
            metadata=event.get('metadata') or {}
            if not (metadata.get('hidden_kv_layers')==78 and metadata.get('draft_kv_layers')==1 and metadata.get('kv_buf_groups')==3 and len(metadata.get('kv_item_lens') or [])==237 and len(metadata.get('aux_item_lens') or [])==11 and metadata.get('page_size')==128 and metadata.get('kv_cache_dtype_str')=='bfloat16'):
                layout_ok=False;break
    graph={}
    if metric is not None:
        for line in metric.decode().splitlines():
            if not line.startswith('sglang:cuda_graph_passes_total{'):continue
            try:
                label_text,value=line.split('} ',1);labels=dict(re.findall(r'(\w+)="([^"]*)"',label_text))
                rank=int(labels['moe_ep_rank'])
                if labels.get('mode')!='decode_cuda_graph' or labels.get('engine_type')!='decode' or labels.get('model_name')!='glm53-w8a8':
                    continue
                if int(labels.get('tp_rank',-1))!=rank or int(labels.get('dp_rank',-1))!=rank//2:
                    continue
                passes=float(value)
            except (ValueError,KeyError):
                continue
            if passes>0:graph[rank]=passes
    graph_ok=set(graph)==(set(range(0,16)) if NODE==36 else set(range(16,32)))
    native_ops=ops_summary(initialized,safe_jsonl(root/'alignment-audit','worker-*.jsonl')) if init_ok else {}
    eager=eager_summary(initialized) if init_ok else {'candidate':CANDIDATE,'record_files':0,'record_pids':0,'audit_pids_match_SDK':False,'all_native_no_copy_true':None,'all_clone_distinct':None,'all_clone_fields_identity':None,'bs2_observed_pids':0,'bs2_all16':False,'reason':'sdk_not_ready'}
    done=health_status==200 and metrics_status==200 and idle and len(npu_rows)==18 and init_ok and layout_ok and len(native_ops)==16 and graph_ok and not fatal
    if NODE==36:
        done=done and server_ok and warm is True
    else:
        done=done and server_ok
    if CANDIDATE:
        done=done and eager['audit_pids_match_SDK'] and eager['all_native_no_copy_true'] and eager['all_clone_distinct'] and eager['all_clone_fields_identity']
    if time.monotonic()-last>=30 or done:
        last=time.monotonic();print('GLM53_L17_D_OBSERVE_PROGRESS='+json.dumps({'node':NODE,'health':health_status,'idle':idle,'registered':len(layouts),'graph_ranks':len(graph),'native_ops_workers':len(native_ops),'warm_all16dp':warm,'eager_bs2_pids':eager['bs2_observed_pids'],'done':done}),flush=True)
    if done:
        summary={'project_id':'glm53','node':NODE,'variant':KIND,'startup_pass':True,'container_id':CID,'name':NAME,'run_id':RID,'actual_local_service_id':SERVICE_ID,'health_status':health_status,'metrics_status':metrics_status,'idle':idle,'idle_gauges':gauges,'secondary_server_info_404_expected':NODE==41,'primary_server_info_status':server['status'],'reported_KV_capacity':capacity,'reported_KV_capacity_scope':'role-global primary /get_server_info max_total_num_tokens','capacity_floor':161280,'old_D_reported_KV_capacity':181632,'capacity_tuning':False,'warm_all16dp':warm,'actual_SDK_worker_PIDs':sorted(initialized),'registered_worker_PIDs':sorted(event['pid'] for event in layouts),'owned_NPU_context_PIDs':sorted(npu_rows),'graph':{str(rank):passes for rank,passes in sorted(graph.items())},'native_ops_fullmesh':native_ops,'MTP':{'steps':5,'draft_tokens':6,'draft_kv_layers':1,'packed_layout_metadata_workers':len(layouts),'kv_cache_dtype':'bfloat16'},'eager_first_use':eager,'functional_bs2_all32_gate_pending':True,'formal_ready':False,'external_model_requests':0,'P_native_map_checks_on_D':0}
        break
    assert time.monotonic()<deadline,'D startup observation bounded 1800s exceeded'
    time.sleep(5)
d=json.loads(subprocess.check_output(['docker','inspect',CID],text=True))[0]
assert d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['run_id']==RID and d['State']['Running']
print('GLM53_L17_D_OBSERVE='+json.dumps(summary))
'''


def observe_plan(kind, node, capacity):
    """Render only with the future reviewed capacity dict (actual CIDs)."""
    spec = STAGES[kind]
    row = capacity["providers"][str(node)]
    assert capacity.get("selected_variant") == kind
    assert row["name"] == spec["name_prefix"] + "-20261019-" + str(node)
    assert row["container_id"] != OLD_D[node]["container_id"]
    assert type(row["local_service_id"]) is int
    return render_bound(OBSERVE_PLAN, {
        "NODE": node,
        "KIND": kind,
        "PORT": 30600,
        "CID": row["container_id"],
        "NAME": row["name"],
        "IMAGE": IMAGE,
        "RID": spec["run_id"],
        "REMOTE": spec["run_root"],
        "SERVICE_ID": row["local_service_id"],
        "CANDIDATE": spec["candidate"],
        "EAGER_SHA": EAGER_SHA,
        "ENVIRON_SHA": ENVIRON_SHA,
    }, ["NODE", "KIND", "PORT", "CID", "NAME", "IMAGE", "RID", "REMOTE",
        "SERVICE_ID", "CANDIDATE", "EAGER_SHA", "ENVIRON_SHA"])


def plan_all():
    """All plans that do not depend on the future actual D container IDs."""
    plans = {}
    health = health_plans()
    plans.update({"health-%d" % node: health["health-%d" % node]
                  for node in (34, 35, 36, 41)})
    plans.update(stop_plans())
    plans["meta-before"] = meta_plan("before")
    plans["meta-after"] = meta_plan("after")
    for kind in ("candidate", "rollback"):
        for node in D_NODES:
            for key, code in (
                ("prepare", prepare_plan(kind, node)),
                ("pool", pool_code(kind, node)),
                ("preflight", native_preflight_plan(kind, node)),
            ):
                plans["%s-%s-%d" % (kind, key, node)] = code
    return plans


def pending_dynamic_plans():
    return ["%s-%s-%d" % (kind, key, node) for kind in ("candidate", "rollback")
            for key in ("start", "observe") for node in D_NODES]


def generated_codes():
    """Executable plans plus the preflight source embedded as a container argv."""
    codes = plan_all()
    for kind in ("candidate", "rollback"):
        for node in D_NODES:
            codes["%s-preflight-inner-%d" % (kind, node)] = preflight_inner(kind, node)
    return codes


def _unbound_globals(code, filename):
    table = symtable.symtable(code, filename, "exec")
    module_names = {symbol.get_name() for symbol in table.get_symbols()
                    if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
    builtin_names = set(dir(builtins))
    unbound = []

    def walk(scope, nested):
        for symbol in scope.get_symbols():
            name = symbol.get_name()
            if symbol.is_assigned() or symbol.is_parameter() or symbol.is_imported():
                continue
            if not symbol.is_referenced():
                continue
            if name in module_names or name in builtin_names:
                continue
            if symbol.is_global() or (not nested and symbol.is_free()):
                unbound.append({"scope": scope.get_name(), "name": name, "free": symbol.is_free(), "global": symbol.is_global()})
        for child in scope.get_children():
            walk(child, True)

    walk(table, False)
    return unbound


def analyze_code(name, code):
    parsed = False
    compiled = False
    error = None
    try:
        ast.parse(code, filename=name)
        parsed = True
    except BaseException as failure:
        error = repr(failure)
    if parsed:
        try:
            compile(code, name, "exec", dont_inherit=True)
            compiled = True
        except BaseException as failure:
            error = repr(failure)
    return {
        "name": name,
        "sha256": sha256_text(code),
        "bytes": len(code.encode("utf-8")),
        "ast_parse": parsed,
        "compile": compiled,
        "error": error,
        "unbound_globals": _unbound_globals(code, name) if parsed else [],
        "module_binding_count": len({symbol.get_name() for symbol in symtable.symtable(code, name, "exec").get_symbols() if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}) if parsed else 0,
    }


def source_provenance():
    record = {}
    for relative, expected in SOURCE_PROVENANCE.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if expected is not None:
            assert actual == expected, "%s: %s != %s" % (relative, actual, expected)
        record[relative] = actual
    return record


def materialize(outdir):
    outdir = pathlib.Path(outdir)
    assert not outdir.exists(), "materialization directory exists: %s" % outdir
    assert not (ROOT / REVIEW).exists(), "review already exists: %s" % REVIEW
    assert not (ROOT / CAPACITY_OUT).exists(), "capacity output already exists: %s" % CAPACITY_OUT
    plans = plan_all()
    inner = {"%s-preflight-inner-%d" % (kind, node): preflight_inner(kind, node)
             for kind in ("candidate", "rollback") for node in D_NODES}
    codes = dict(plans)
    codes.update(inner)
    analyses = {name: analyze_code(name, code) for name, code in sorted(codes.items())}
    failing = {name: item for name, item in analyses.items() if not item["ast_parse"] or not item["compile"] or item["unbound_globals"]}
    assert not failing, failing
    outdir.mkdir(parents=True)
    for name, code in codes.items():
        (outdir / (name + ".py")).write_text(code, encoding="utf-8")
    executable_hashes = {name: analyses[name]["sha256"] for name in plans}
    inner_hashes = {name: analyses[name]["sha256"] for name in inner}
    manifest = {
        "project_id": PROJECT_ID,
        "status": STATUS,
        "library": str(pathlib.Path(__file__).resolve().relative_to(ROOT)),
        "library_sha256": sha256_file(pathlib.Path(__file__)),
        "transition_run_id": TRANSITION_RID,
        "stop_run_id": STOP_RID,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plans": analyses,
        "executable_plan_hashes": executable_hashes,
        "embedded_preflight_inner_hashes": inner_hashes,
        "pending_dynamic_capacity": pending_dynamic_plans(),
        "base_pool_host_code_correction": "base start() imports time but references socket only in the retained dead Meta branch; generated D code imports socket too. Docker args, readiness logic and bundle values are unchanged.",
        "native_preflight_basis": NATIVE_PREFLIGHT_BASIS,
        "future_budget": {"functional": 76, "canonical": 320, "status": "PROPOSED_NOT_AUTHORIZED"},
        "no_approved_placeholder": True,
        "review_created": False,
        "capacity_output_created": False,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    status = {
        "project_id": PROJECT_ID,
        "status": STATUS,
        "plans_total": len(codes),
        "executable_plans": len(plans),
        "embedded_preflight_inner": len(inner),
        "plans_valid": len(codes) - len(failing),
        "unbound_globals": 0,
        "ssh": 0,
        "network": 0,
        "HTTP_requests": 0,
        "NPU_calls": 0,
        "SDK_init_calls": 0,
        "model_requests": 0,
        "remote_writes": 0,
        "Git_operations": 0,
        "runtime_authorized": False,
        "model_start_authorized": False,
        "pending_dynamic_capacity": pending_dynamic_plans(),
        "blocked_until": ["independent local pool-capacity audit pins actual D CIDs",
                          "separate primary review freezes model-start/observe host SHAs"],
    }
    (outdir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    validation = {
        "project_id": PROJECT_ID,
        "pass": True,
        "checks": len(analyses) * 3,
        "checks_passed": len(analyses) * 3,
        "ast_parse_all": all(item["ast_parse"] for item in analyses.values()),
        "compile_all": all(item["compile"] for item in analyses.values()),
        "unbound_globals_all_empty": all(not item["unbound_globals"] for item in analyses.values()),
        "plan_hashes": {name: item["sha256"] for name, item in analyses.items()},
        "executable_plan_hashes": executable_hashes,
        "embedded_preflight_inner_hashes": inner_hashes,
    }
    (outdir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "project_id": PROJECT_ID,
        "pass": True,
        "status": STATUS,
        "run_id": outdir.name,
        "library_sha256": manifest["library_sha256"],
        "manifest_sha256": sha256_file(outdir / "manifest.json"),
        "status_sha256": sha256_file(outdir / "status.json"),
        "validation_sha256": sha256_file(outdir / "validation.json"),
        "plans_total": len(codes),
        "executable_plans": len(plans),
        "pending_dynamic_capacity": pending_dynamic_plans(),
        "source_provenance": source_provenance(),
        "SSH": 0,
        "network": 0,
        "model_requests": 0,
        "runtime_authorized": False,
        "model_start_authorized": False,
    }
    (outdir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    result = {
        "project_id": PROJECT_ID,
        "pass": True,
        "status": STATUS,
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "model_start_authorized": False,
        "materialization_run": "runs/" + outdir.name,
        "manifest_sha256": receipt["manifest_sha256"],
        "validation_sha256": receipt["validation_sha256"],
        "receipt_sha256": sha256_file(outdir / "receipt.json"),
        "plans_total": len(codes),
        "executable_plans": len(plans),
        "executable_plan_hashes": executable_hashes,
        "embedded_preflight_inner_hashes": inner_hashes,
        "pending_dynamic_capacity": pending_dynamic_plans(),
        "source_provenance": receipt["source_provenance"],
        "not_created": [REVIEW, CAPACITY_OUT, "stop/pool/model-start launcher", "functional/formal client launcher"],
        "future_budget": {"functional": 76, "canonical": 320, "status": "PROPOSED_NOT_AUTHORIZED"},
        "request_ledger_changed": False,
        "performance_claim": False,
        "objective_complete": False,
    }
    result_path = ROOT / LOCAL_RESULT
    assert not result_path.exists(), "local result already exists: %s" % LOCAL_RESULT
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = LOCAL_RESULT
    result["result_sha256"] = sha256_file(result_path)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--write-plan":
        target = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "runs/20261019T-L17-D-transition-local-materialization-v1")
        print(json.dumps(materialize(target), indent=2))
        sys.exit(0)
    raise SystemExit("DRAFT_PENDING_PRIMARY_REVIEW: local plan materialization only; use --write-plan")
