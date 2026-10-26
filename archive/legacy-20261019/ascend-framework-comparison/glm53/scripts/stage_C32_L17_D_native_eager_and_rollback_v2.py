#!/usr/bin/env python3
"""DRAFT ONLY: exact-once stage of candidate-v2 and rollback-v1, then CPU preflight.

Primary v2 correction stages the frozen bundle metadata through separate payload copies. It remains unexecuted until its actual review is frozen:
root review must create results/C32-L17-D-native-eager-stage-actual-review-v2.json
with the exact generated code hashes before main() can run:

    {
      "review_status": "APPROVED_FOR_STAGE_ONLY",
      "runtime_authorized": false,
      "host_code_sha256": {
        "health-34": "<sha256>",
        "health-35": "<sha256>",
        "health-36": "<sha256>",
        "health-41": "<sha256>"
      },
      "CPU_code_sha256": "<sha256>"
    }

The stage itself is read-only with respect to the four current runtime containers:
all four mixed P-new/D-retained ownership plus healthy-idle checks run first, then
candidate-v2 and rollback-v1 are staged exactly once to new D36/D41 remote roots.
The CPU preflight runs in a fixed-image, network-none, cpu2/mem4g container on D36
with both staged roots mounted read-only and no NPU devices.  No lifecycle,
stop, start or model-request launcher is produced here.

The historical P-only CPU qualifier (python/qualify_image_cpu_v1.py) is not run:
it references the removed P files and is not part of this D-only candidate.
"""

from __future__ import annotations

import ast
import copy
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = "20261019T-L17-D-native-eager-stage-v2"
RUN = ROOT / "runs" / RID
STAGE_HELPER = ROOT / "scripts/stage_sglang_C32_packed_normal_and_rollback_v1.py"
STAGE_HELPER_SHA = "13545bf1c90368d57113dc810738da12c64d3efa5c01ba3fd38057e21cb1724f"
MIXED_IDENTITY = ROOT / "scripts/glm53_C32_L12C_mixed_model_identity_v1.py"
MIXED_IDENTITY_SHA = "ff5a92342a3a385256b619e9c004294b881634678326d706eca026413c5e6a98"
CAPACITY = ROOT / "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
REVIEW = ROOT / "results/C32-L17-D-native-eager-stage-actual-review-v2.json"

CANDIDATE_STAGE = ROOT / "config/sglang-C32-L17-D-native-eager-stage-v2.json"
CANDIDATE_STAGE_SHA = "a8d6759af5f29a3da15e60b1312bc81eae046e93c99ca3eb366a78bdf055da0c"
CANDIDATE_BUNDLE = ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2"
CANDIDATE_BUNDLE_SHA = "9427ba4316f5b5075b97afecb01eca062caa912786280897fbf962eb1d498491"
ROLLBACK_STAGE = ROOT / "config/sglang-C32-L17-D-original-rollback-stage-v1.json"
ROLLBACK_STAGE_SHA = "f8b49216cd3ac6651cb48f9308a2590d2c0e306b339355e568a522bb882ddcf0"
ROLLBACK_BUNDLE = ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1"
ROLLBACK_BUNDLE_SHA = "b3e1d5824eed9e86b61e1a20a0f0a46732c5caca543d83e1310d62e4a171cd7a"

AUDIT_SHA = "0df4276b5e5f8d10aec1faf25cd24e847dc3ef9a57570f70f257c0ebf539d56f"
EAGER_SHA = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
ENVIRON_SHA = "0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c"
SDMA_MANIFEST_SHA = "0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528"
IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"

P_NODES = (34, 35)
D_NODES = (36, 41)
ALL_NODES = (34, 35, 36, 41)
P_PORT = 30500
D_PORT = 30600

EXPECTED_RUNTIME = {
    34: {
        "container_id": "a6639da9774e3b62098b21d6e7af803299e11ea3b4f3332705eedb6312902179",
        "name": "glm53-L12C-candidate-v1-20261019-34",
        "local_service_id": 82,
        "dram_GiB": 136,
        "remote_root": "/data/wio/inference-foundry/glm53/runs/20261019T-L12C-candidate-v1",
        "run_id": "20261019T-L12C-candidate-v1",
    },
    35: {
        "container_id": "94dd0021d5343d9c9e9cc5e4c07877e9306bdf93d8bf6382d388c7121e7c57f1",
        "name": "glm53-L12C-candidate-v1-20261019-35",
        "local_service_id": 83,
        "dram_GiB": 112,
        "remote_root": "/data/wio/inference-foundry/glm53/runs/20261019T-L12C-candidate-v1",
        "run_id": "20261019T-L12C-candidate-v1",
    },
    36: {
        "container_id": "25ecbd3a55e199caf4c6012a61b702d1b6035c955406d75bb17107991a932dd1",
        "name": "glm53-sglang-C32-target-graph-fullmesh-v3-20261018-36",
        "local_service_id": 2,
        "dram_GiB": 160,
        "remote_root": "/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3",
        "run_id": "20261018T-L10-sglang-C32-target-graph-fullmesh-v3",
    },
    41: {
        "container_id": "c39fa7c594a8776e70c365baec435d4288abb1db5cc492405183047d57b805c3",
        "name": "glm53-sglang-C32-target-graph-fullmesh-v3-20261018-41",
        "local_service_id": 3,
        "dram_GiB": 160,
        "remote_root": "/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3",
        "run_id": "20261018T-L10-sglang-C32-target-graph-fullmesh-v3",
    },
}
EXPECTED_META_CID = "0dd00f5c35ee8c60cbd697a38fc73a8c37ae498106f9ebfc2933139e953b165c"

P_FILES = (
    "config/node-34.json",
    "config/node-35.json",
    "provider-config/node-34.json",
    "provider-config/node-35.json",
    "supervisor-config/node-34.json",
    "supervisor-config/node-35.json",
)

HEALTH_EXTRA = r'''
import urllib.request
assert option("--port") == ("30500" if NODE in (34, 35) else "30600")
_url = "http://172.16.12." + str(NODE) + ":" + str(30500 if NODE in (34, 35) else 30600)
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with _opener.open(_url + "/health", timeout=10) as _f:
    assert _f.status == 200
with _opener.open(_url + "/metrics", timeout=10) as _f:
    _raw = _f.read(8 * 1024 ** 2 + 1)
    assert len(_raw) <= 8 * 1024 ** 2
_values = [float(_line.rsplit(" ", 1)[1]) for _line in _raw.decode().splitlines() if _line.startswith(("sglang:num_running_reqs{", "sglang:num_queue_reqs{"))]
assert _values and all(_value == 0 for _value in _values)
_npu = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
assert _npu.returncode == 0
_rows = []
for _line in _npu.stdout.split("Process id", 1)[-1].splitlines():
    _cols = [x.strip() for x in _line.split("|")]
    if len(_cols) >= 5 and len(_cols[1].split()) == 2 and all(x.isdigit() for x in _cols[1].split()) and _cols[2].isdigit():
        _pid = int(_cols[2])
        assert CID in (pathlib.Path("/proc") / str(_pid) / "cgroup").read_text()
        _rows.append(_pid)
assert len(_rows) == 18
print("GLM53_L17_HEALTH=" + json.dumps({"node": NODE, "container_id": CID, "healthy_idle": True, "NPU_context_PIDs": sorted(_rows), "metrics_sha256": hashlib.sha256(_raw).hexdigest(), "model_requests": 0, "existing_models_changed": False}), flush=True)
'''

CPU_TEMPLATE = r'''
import __future__
import ast
import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types

EAGER_SHA = __EAGER_SHA__
ENVIRON_SHA = __ENVIRON_SHA__
AUDIT_SHA = __AUDIT_SHA__
SDMA_MANIFEST_SHA = __SDMA_MANIFEST_SHA__
CANDIDATE_FILES = __CANDIDATE_FILES__
ROLLBACK_FILES = __ROLLBACK_FILES__
EAGER_PATH = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/eager_runner.py")
ENVIRON_PATH = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/environ.py")
CANDIDATE_ROOT = pathlib.Path("/opt/glm53/candidate")
ROLLBACK_ROOT = pathlib.Path("/opt/glm53/rollback")
AUDIT_PATH = CANDIDATE_ROOT / "python/glm53_C32_L17_eager_input_audit_v2.py"
STDLIB_ROOTS = {"base64", "contextlib", "enum", "functools", "json", "os", "typing", "warnings"}
CHECKS = []


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(pathlib.Path(path).read_bytes())


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check(name, condition, detail=None):
    passed = bool(condition)
    CHECKS.append({"name": name, "pass": passed, "detail": detail})
    if not passed:
        raise AssertionError("CPU check failed: %s: %r" % (name, detail))
    return passed


def diff(old, new, prefix=""):
    changes = []
    if type(old) is not type(new):
        return [prefix or "<root>"]
    if isinstance(old, dict):
        for key in sorted(set(old) | set(new)):
            child = (prefix + "." + key) if prefix else key
            if key not in old or key not in new:
                changes.append(child)
            else:
                changes.extend(diff(old[key], new[key], child))
        return changes
    if isinstance(old, list):
        if old != new:
            changes.append(prefix or "<root>")
        return changes
    if old != new:
        changes.append(prefix or "<root>")
    return changes


def expect_failure(action, fragment):
    try:
        action()
    except BaseException as error:
        return fragment in repr(error), repr(error)
    return False, "no exception"


def class_def(name, body):
    fields = {"name": name, "bases": [], "keywords": [], "body": body, "decorator_list": []}
    if "type_params" in ast.ClassDef._fields:
        fields["type_params"] = []
    return ast.ClassDef(**fields)


def extract_env_fixture():
    source_bytes = ENVIRON_PATH.read_bytes()
    text = source_bytes.decode("utf-8")
    tree = ast.parse(text, filename=str(ENVIRON_PATH))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for required in ("EnvField", "EnvBool", "Envs"):
        check("environ_class:" + required, required in classes)
    assignment = None
    for item in classes["Envs"].body:
        if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SGLANG_EAGER_INPUT_NO_COPY" for target in item.targets):
            assignment = item
            break
    check("environ_flag_assignment_present", assignment is not None)
    check(
        "environ_flag_assignment_actual_ast",
        (ast.get_source_segment(text, assignment) or "").strip() == "SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)",
        ast.get_source_segment(text, assignment),
    )
    prelude = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(alias.name.split(".")[0] in STDLIB_ROOTS for alias in node.names):
                prelude.append(copy.deepcopy(node))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in STDLIB_ROOTS:
                prelude.append(copy.deepcopy(node))
    module_ast = ast.Module(
        body=prelude
        + [
            copy.deepcopy(classes["EnvField"]),
            copy.deepcopy(classes["EnvBool"]),
            class_def("Envs", [copy.deepcopy(assignment)]),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module_ast)
    namespace = {"__name__": "sglang.srt.environ"}
    exec(compile(module_ast, str(ENVIRON_PATH), "exec", dont_inherit=True), namespace)
    envs = namespace["Envs"]()
    flag = envs.SGLANG_EAGER_INPUT_NO_COPY
    check("envs_class_source_qualified", namespace["Envs"].__module__ == "sglang.srt.environ")
    check("envs_flag_type", type(flag).__name__ == "EnvBool")
    check("envs_flag_name_natural", flag.name == "SGLANG_EAGER_INPUT_NO_COPY")
    check("envs_field_set_name_natural", type(flag)._allow_set_name is True)
    return source_bytes, envs, classes, assignment


def extract_load_batch():
    source_bytes = EAGER_PATH.read_bytes()
    text = source_bytes.decode("utf-8")
    tree = ast.parse(text, filename=str(EAGER_PATH))
    function = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EagerRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "load_batch":
                    function = item
                    break
    check("eager_load_batch_ast_present", function is not None)
    check("eager_load_batch_actual_lines", function.lineno == 182 and function.end_lineno == 210, {"lineno": function.lineno, "end_lineno": function.end_lineno})
    module_ast = ast.Module(body=[copy.deepcopy(function)], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    code = compile(module_ast, str(EAGER_PATH), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
    return source_bytes, code


class T:
    def __init__(self, shape, stride):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self.dtype = "torch.int64"
        self.device = "npu:0"
        self.cpu_calls = 0
        self.item_calls = 0
        self.data_ptr_calls = 0

    def stride(self):
        return self._stride

    def cpu(self):
        self.cpu_calls += 1
        raise AssertionError("audit must not call .cpu()")

    def item(self):
        self.item_calls += 1
        raise AssertionError("audit must not call .item()")

    def data_ptr(self):
        self.data_ptr_calls += 1
        raise AssertionError("audit must not call data_ptr()")


@dataclasses.dataclass
class Spec:
    draft_input_ids: object
    draft_positions: object
    num_steps: int


@dataclasses.dataclass
class Static:
    planned_bs: int
    planned_num_tokens: int
    ready: bool


@dataclasses.dataclass
class Batch:
    forward_mode: object
    batch_size: int
    input_ids: object
    req_pool_indices: object
    seq_lens: object
    out_cache_loc: object
    seq_lens_cpu: object
    positions: object
    spec_info: object
    input_embeds: object
    static_metadata: object
    return_logprob: bool
    capture_hidden_mode: object


class Mode:
    def __init__(self, name):
        self.name = name


class Spy:
    def __init__(self):
        self.fill_calls = 0
        self.extract_calls = 0

    def fill_from(self, *args, **kwargs):
        self.fill_calls += 1

    def extract_buffer(self, *args, **kwargs):
        self.extract_calls += 1
        return "EXTRACTED"


def make_batch(mode="DECODE", batch_size=4):
    return Batch(
        forward_mode=Mode(mode),
        batch_size=batch_size,
        input_ids=T((4, 3), (5, 1)),
        req_pool_indices=T((4,), (1,)),
        seq_lens=T((4,), (1,)),
        out_cache_loc=T((12,), (1,)),
        seq_lens_cpu=None,
        positions=T((4, 3), (3, 1)),
        spec_info=Spec(T((4, 6), (6, 1)), T((4, 6), (6, 1)), 5),
        input_embeds=None,
        static_metadata=Static(4, 12, True),
        return_logprob=False,
        capture_hidden_mode="NONE",
    )


def build_runner(lb_code, envs):
    namespace = {"envs": envs, "replace": dataclasses.replace, "__name__": "glm53_cpu_fixture"}
    exec(lb_code, namespace)
    actual = namespace["load_batch"]
    counted = {"calls": 0}

    def original(self, batch, *args, **kwargs):
        counted["calls"] += 1
        return actual(self, batch, *args, **kwargs)

    class Runner:
        pass

    Runner.load_batch = original
    module = types.ModuleType("glm53_cpu_fixture.eager_runner")
    module.EagerRunner = Runner
    runner = Runner()
    runner.model_runner = types.SimpleNamespace(is_draft_worker=False)
    runner._eager_registry = Spy()
    return module, runner, counted


def install_environ(envs, path=ENVIRON_PATH):
    module = types.ModuleType("sglang.srt.environ")
    module.__file__ = str(path)
    module.envs = envs
    previous = sys.modules.get("sglang.srt.environ")
    sys.modules["sglang.srt.environ"] = module
    return previous


def restore_environ(previous):
    if previous is None:
        sys.modules.pop("sglang.srt.environ", None)
    else:
        sys.modules["sglang.srt.environ"] = previous


def main_cpu():
    check("eager_sha", sha256_file(EAGER_PATH) == EAGER_SHA, sha256_file(EAGER_PATH))
    check("environ_sha", sha256_file(ENVIRON_PATH) == ENVIRON_SHA, sha256_file(ENVIRON_PATH))
    check("audit_sha", sha256_file(AUDIT_PATH) == AUDIT_SHA, sha256_file(AUDIT_PATH))
    check("candidate_sdma_manifest_unchanged", sha256_file(CANDIDATE_ROOT / "manifest.json") == SDMA_MANIFEST_SHA)
    check("rollback_sdma_manifest_unchanged", sha256_file(ROLLBACK_ROOT / "manifest.json") == SDMA_MANIFEST_SHA)
    for name, expected in CANDIDATE_FILES:
        check("candidate_file:" + name, sha256_file(CANDIDATE_ROOT / name) == expected)
    for name, expected in ROLLBACK_FILES:
        check("rollback_file:" + name, sha256_file(ROLLBACK_ROOT / name) == expected)
    for name in ("config/node-34.json", "config/node-35.json", "provider-config/node-34.json", "provider-config/node-35.json", "supervisor-config/node-34.json", "supervisor-config/node-35.json"):
        check("candidate_no_P:" + name, not (CANDIDATE_ROOT / name).exists())
        check("rollback_no_P:" + name, not (ROLLBACK_ROOT / name).exists())
    # Bundle manifests are local proofs verified by the launcher before staging;
    # they are not part of the staged file set.
    for node in (36, 41):
        candidate_node = read_json(CANDIDATE_ROOT / ("config/node-%d.json" % node))
        rollback_node = read_json(ROLLBACK_ROOT / ("config/node-%d.json" % node))
        allowed = ["environment_intent.SGLANG_EAGER_INPUT_NO_COPY", "environment_intent.GLM53_EAGER_INPUT_AUDIT", "environment_intent.GLM53_EAGER_INPUT_AUDIT_ROOT"]
        check("node_%d_env_diff" % node, sorted(diff(rollback_node, candidate_node)) == sorted(allowed), diff(rollback_node, candidate_node))
        check("node_%d_full_cli_unchanged" % node, candidate_node["launch_args"] == rollback_node["launch_args"])
        check(
            "node_%d_other_fields_unchanged" % node,
            all(candidate_node[key] == rollback_node[key] for key in rollback_node if key not in ("environment_intent", "launch_args")),
        )
        candidate_sup = read_json(CANDIDATE_ROOT / ("supervisor-config/node-%d.json" % node))
        rollback_sup = read_json(ROLLBACK_ROOT / ("supervisor-config/node-%d.json" % node))
        allowed_sup = ["run_id", "model_env.SGLANG_EAGER_INPUT_NO_COPY", "model_env.GLM53_EAGER_INPUT_AUDIT", "model_env.GLM53_EAGER_INPUT_AUDIT_ROOT"]
        check("supervisor_%d_diff" % node, sorted(diff(rollback_sup, candidate_sup)) == sorted(allowed_sup), diff(rollback_sup, candidate_sup))
        check("supervisor_%d_model_argv" % node, candidate_sup["model_argv"] == rollback_sup["model_argv"])
        check("supervisor_%d_provider_argv" % node, candidate_sup["provider_argv"] == rollback_sup["provider_argv"])
        candidate_provider = read_json(CANDIDATE_ROOT / ("provider-config/node-%d.json" % node))
        rollback_provider = read_json(ROLLBACK_ROOT / ("provider-config/node-%d.json" % node))
        check("provider_%d_unchanged" % node, candidate_provider == rollback_provider)
        check("provider_%d_no_local_service_id" % node, "local_service_id" not in candidate_provider)
        check("provider_%d_dram" % node, candidate_provider.get("dram_size") == "160GB" and candidate_provider.get("max_dram_size") == "160GB")
    old_site = (ROLLBACK_ROOT / "python/sitecustomize.py").read_bytes()
    new_site = (CANDIDATE_ROOT / "python/sitecustomize.py").read_bytes()
    check("sitecustomize_prefix", new_site.startswith(old_site))
    appended = new_site[len(old_site):]
    check("sitecustomize_audit_sha", AUDIT_SHA.encode("ascii") in appended)
    check("sitecustomize_v2_import", b"glm53_C32_L17_eager_input_audit_v2" in appended)
    check("sitecustomize_no_framework_import", b"import torch" not in appended and b"import sglang" not in appended and b"from sglang" not in appended)
    compile(new_site, "candidate-sitecustomize", "exec")

    environ_bytes, envs, env_classes, env_assignment = extract_env_fixture()
    eager_bytes, lb_code = extract_load_batch()
    check("environ_source_bytes", len(environ_bytes) > 0)
    check("eager_source_bytes", len(eager_bytes) > 0)

    spec = importlib.util.spec_from_file_location("glm53_audit_cpu", AUDIT_PATH)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    check("audit_source_pin", audit.SOURCE_SHA256 == EAGER_SHA)
    check("audit_environ_pin", audit.ENVIRON_SHA256 == ENVIRON_SHA)
    check("audit_version_v2", audit.AUDIT_VERSION == "v2")
    check("audit_schema_v2", audit.SCHEMA == "glm53.L17.eager_input_audit.v2")
    check("audit_stdlib_only", "torch" not in sys.modules and "torch_npu" not in sys.modules and "sglang" not in sys.modules)

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="glm53-l17-cpu-"))
    os.environ["GLM53_EAGER_INPUT_AUDIT"] = "1"
    os.environ["GLM53_NODE"] = "36"
    os.environ["GLM53_EAGER_INPUT_AUDIT_ROOT"] = str(tmpdir)
    os.environ["SGLANG_EAGER_INPUT_NO_COPY"] = "1"

    previous = install_environ(envs)
    module, runner, counted = build_runner(lb_code, envs)
    audit._reset_for_tests()
    audit._patch_module(module, EAGER_SHA, str(tmpdir))
    batch = make_batch()
    result = runner.load_batch(batch)
    check("first_use_original_once", counted["calls"] == 1)
    check("first_use_clone_distinct", result is not batch)
    record_path = tmpdir / ("pid-%d.jsonl" % os.getpid())
    check("first_use_record_created", record_path.is_file())
    record = json.loads(record_path.read_text(encoding="utf-8").strip().splitlines()[0])
    check("record_schema_v2", record.get("schema") == "glm53.L17.eager_input_audit.v2")
    check("record_eager_sha", record.get("eager_source_sha256") == EAGER_SHA)
    check("record_environ_sha", record.get("environ_source_sha256") == ENVIRON_SHA)
    check("record_clone_distinct", record.get("clone_distinct") is True)
    check("record_native_get", record.get("native_no_copy_get") is True)
    check("record_batch_size", record.get("batch_size") == 4)
    check("record_seq_lens_cpu_raw_none", "seq_lens_cpu" in record.get("allowlisted", {}) and record["allowlisted"]["seq_lens_cpu"] is None)
    check("record_identity_all_fields", record.get("field_verdicts", {}).get("identity") == record.get("dataclass_field_count"))
    check("record_input_ids_stride5", record.get("allowlisted", {}).get("input_ids", {}).get("stride") == [5, 1])
    size_after_first = record_path.stat().st_size
    saved_append = audit._append_record
    saved_native = audit._native_no_copy_get_true
    saved_pin = audit._pin_environ_source
    audit._append_record = lambda *a, **k: (_ for _ in ()).throw(AssertionError("fastpath wrote evidence"))
    audit._native_no_copy_get_true = lambda: (_ for _ in ()).throw(AssertionError("fastpath audited env"))
    audit._pin_environ_source = lambda *a, **k: (_ for _ in ()).throw(AssertionError("fastpath checked source"))
    try:
        runner.load_batch(batch)
    finally:
        audit._append_record = saved_append
        audit._native_no_copy_get_true = saved_native
        audit._pin_environ_source = saved_pin
    check("fastpath_original_second_call", counted["calls"] == 2)
    check("fastpath_no_new_record", record_path.stat().st_size == size_after_first)

    # Sequential bs1 -> bs2 is two first-use records; repeated bs2 adds no log.
    audit._reset_for_tests()
    module_bs, runner_bs, counted_bs = build_runner(lb_code, envs)
    audit._patch_module(module_bs, EAGER_SHA, str(tmpdir / "bs_sequence"))
    runner_bs.load_batch(make_batch("DECODE", batch_size=1))
    runner_bs.load_batch(make_batch("DECODE", batch_size=2))
    bs_path = tmpdir / "bs_sequence" / ("pid-%d.jsonl" % os.getpid())
    bs_lines = [json.loads(line) for line in bs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check("bs_sequence_two_records", len(bs_lines) == 2)
    check("bs_sequence_batch_sizes", [line.get("batch_size") for line in bs_lines] == [1, 2])
    check("bs_sequence_modes_preserved", [line.get("forward_mode") for line in bs_lines] == ["DECODE", "DECODE"])
    check("bs_sequence_stride5", all(line.get("allowlisted", {}).get("input_ids", {}).get("stride") == [5, 1] for line in bs_lines))
    check("bs_sequence_identity_all_fields", all(line.get("field_verdicts", {}).get("identity") == line.get("dataclass_field_count") for line in bs_lines))
    size_after_bs2 = bs_path.stat().st_size
    calls_after_bs2 = counted_bs["calls"]
    saved_append = audit._append_record
    saved_native = audit._native_no_copy_get_true
    audit._append_record = lambda *a, **k: (_ for _ in ()).throw(AssertionError("repeat bs2 wrote evidence"))
    audit._native_no_copy_get_true = lambda: (_ for _ in ()).throw(AssertionError("repeat bs2 audited env"))
    try:
        runner_bs.load_batch(make_batch("DECODE", batch_size=2))
    finally:
        audit._append_record = saved_append
        audit._native_no_copy_get_true = saved_native
    check("bs_repeat_original_once", counted_bs["calls"] == calls_after_bs2 + 1)
    check("bs_repeat_no_new_record", bs_path.stat().st_size == size_after_bs2)
    restore_environ(previous)

    # A decode batch above the max4 capture/admission bound fails closed and is
    # never clamped to bs4.
    audit._reset_for_tests()
    module_over, runner_over, counted_over = build_runner(lb_code, envs)
    audit._patch_module(module_over, EAGER_SHA, str(tmpdir / "bs_overflow"))
    ok, text = expect_failure(lambda: runner_over.load_batch(make_batch("DECODE", batch_size=5)), "exceeds the DECODE mode bound 4; not clamped")
    check("decode_bs5_not_clamped", ok, text)
    check("decode_bs5_original_not_called", counted_over["calls"] == 0)

    # False environment: fail closed before the original, then sticky.
    audit._reset_for_tests()
    os.environ["SGLANG_EAGER_INPUT_NO_COPY"] = "0"
    previous = install_environ(envs)
    module2, runner2, counted2 = build_runner(lb_code, envs)
    audit._patch_module(module2, EAGER_SHA, str(tmpdir / "false"))
    ok, text = expect_failure(lambda: runner2.load_batch(make_batch("MIXED")), "is not True")
    check("false_env_fails", ok, text)
    check("false_env_original_not_called", counted2["calls"] == 0)
    os.environ["SGLANG_EAGER_INPUT_NO_COPY"] = "1"
    ok, text = expect_failure(lambda: runner2.load_batch(make_batch("MIXED")), "sticky first-use audit failure")
    check("false_env_retry_sticky", ok, text)
    check("false_env_retry_original_not_called", counted2["calls"] == 0)
    restore_environ(previous)

    # Wrong environ SHA at first use.
    audit._reset_for_tests()
    os.environ["SGLANG_EAGER_INPUT_NO_COPY"] = "1"
    module3, runner3, counted3 = build_runner(lb_code, envs)
    audit._patch_module(module3, EAGER_SHA, str(tmpdir / "bad_environ"))
    bad_environ = tmpdir / "bad_environ.py"
    bad_environ.write_bytes(ENVIRON_PATH.read_bytes() + b"\n# tampered\n")
    previous = install_environ(envs, bad_environ)
    ok, text = expect_failure(lambda: runner3.load_batch(make_batch("DECODE")), "installed environ source SHA mismatch")
    check("wrong_environ_sha_fails", ok, text)
    check("wrong_environ_sha_original_not_called", counted3["calls"] == 0)
    restore_environ(previous)

    # Wrong eager SHA and loader bytes-before-exec.
    audit._reset_for_tests()
    module4, runner4, counted4 = build_runner(lb_code, envs)
    audit._ACTIVE_SOURCE_SHA = EAGER_SHA
    ok, text = expect_failure(lambda: audit._patch_module(module4, "00" * 32, str(tmpdir / "bad_eager")), "patch source SHA mismatch")
    check("wrong_eager_sha_fails", ok, text)
    check("wrong_eager_sha_not_installed", getattr(module4.EagerRunner, "load_batch", None) is not None and not hasattr(module4.EagerRunner.load_batch, "__glm53_l17_eager_input_audit__"))
    bad_eager = tmpdir / "bad_eager.py"
    bad_eager.write_text("EXECUTED = True\nclass EagerRunner:\n    def load_batch(self, batch):\n        return batch\n", encoding="utf-8")
    loader = audit._PinnedSourceLoader(str(bad_eager), EAGER_SHA, str(tmpdir))
    bad_module = types.ModuleType("bad_loader_fixture")
    ok, text = expect_failure(lambda: loader.exec_module(bad_module), "installed source SHA mismatch before exec")
    check("loader_wrong_sha_fails_before_exec", ok, text)
    check("loader_wrong_sha_not_executed", not hasattr(bad_module, "EXECUTED"))

    audit._reset_for_tests()
    for key in ("GLM53_EAGER_INPUT_AUDIT", "GLM53_NODE", "GLM53_EAGER_INPUT_AUDIT_ROOT", "SGLANG_EAGER_INPUT_NO_COPY"):
        os.environ.pop(key, None)
    sys.modules.pop("sglang.srt.environ", None)
    check("no_framework_imported", not any(name == "sglang" or name.startswith("sglang.") for name in sys.modules) and "torch" not in sys.modules and "torch_npu" not in sys.modules)
    print("GLM53_L17_CPU=" + json.dumps({"pass": all(item["pass"] for item in CHECKS), "checks": len(CHECKS), "checks_passed": sum(1 for item in CHECKS if item["pass"]), "SDK_init_calls": 0, "NPU_compute_calls": 0, "model_requests": 0, "installed_eager_sha": sha256_file(EAGER_PATH), "installed_environ_sha": sha256_file(ENVIRON_PATH)}, sort_keys=True), flush=True)


main_cpu()
'''


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_module(path, expected_sha, name):
    assert sha256_file(path) == expected_sha, "source SHA mismatch: %s" % path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stage_helper():
    return load_module(STAGE_HELPER, STAGE_HELPER_SHA, "glm53_l17_stage_helper")


def verify_capacity(capacity):
    assert capacity.get("project_id") == "glm53"
    assert capacity.get("pass") is True
    assert capacity.get("selected_variant") == "candidate"
    assert capacity.get("Meta_retained") is True
    assert capacity.get("D_identities_and_contexts_retained") is True
    assert capacity.get("meta_container_id") == EXPECTED_META_CID
    assert capacity.get("native_fixture_actual", {}).get("all4_healthy_idle") is True
    providers = capacity.get("providers", {})
    for node in ALL_NODES:
        actual = providers.get(str(node), {})
        expected = EXPECTED_RUNTIME[node]
        for key, value in expected.items():
            assert actual.get(key) == value, "runtime identity mismatch node=%d key=%s actual=%r expected=%r" % (node, key, actual.get(key), value)


def verify_bundles(stage_helper):
    assert sha256_file(CANDIDATE_STAGE) == CANDIDATE_STAGE_SHA
    assert sha256_file(CANDIDATE_BUNDLE / "bundle-manifest.json") == CANDIDATE_BUNDLE_SHA
    assert sha256_file(ROLLBACK_STAGE) == ROLLBACK_STAGE_SHA
    assert sha256_file(ROLLBACK_BUNDLE / "bundle-manifest.json") == ROLLBACK_BUNDLE_SHA
    candidate_stage = json.loads(CANDIDATE_STAGE.read_text(encoding="utf-8"))
    rollback_stage = json.loads(ROLLBACK_STAGE.read_text(encoding="utf-8"))
    candidate_bundle = candidate_stage["bundles"][0]
    rollback_bundle = rollback_stage["bundles"][0]
    assert sorted(candidate_bundle["nodes"]) == list(D_NODES)
    assert sorted(rollback_bundle["nodes"]) == list(D_NODES)
    assert candidate_bundle["remote"] == "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-native-eager-v2-20261019"
    assert rollback_bundle["remote"] == "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-original-rollback-v1-20261019"
    assert candidate_bundle["runtime_run_id"] == "20261019T-L17-D-native-eager-v2"
    assert rollback_bundle["runtime_run_id"] == "20261019T-L17-D-original-rollback-v1"
    assert candidate_bundle["image"] == IMAGE and rollback_bundle["image"] == IMAGE
    assert candidate_stage.get("audit_version") == "v2"
    assert candidate_stage.get("audit_hook_sha256") == AUDIT_SHA
    assert candidate_stage.get("environ_source_sha256") == ENVIRON_SHA
    assert candidate_stage.get("eager_source_sha256") == EAGER_SHA
    for bundle in (candidate_bundle, rollback_bundle):
        assert not any(name in bundle["files"] for name in P_FILES)
        assert "private_optional_extension" not in json.dumps(bundle)
        for name, meta in bundle["files"].items():
            assert sha256_file(ROOT / meta["local"]) == meta["sha256"], "staged file SHA mismatch: %s" % name
    candidate_manifest = json.loads((CANDIDATE_BUNDLE / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert candidate_manifest.get("audit_hook", {}).get("sha256") == AUDIT_SHA
    assert candidate_manifest.get("audit_hook", {}).get("version") == "v2"
    assert candidate_manifest.get("supersedes", {}).get("superseded_audit_sha256") == "1be2514ca62a270b274d451d8daf5ba325d42bc9344a0cfcac830b21a8002a5c"
    assert candidate_manifest.get("files", {}).get("manifest.json", {}).get("sha256") == SDMA_MANIFEST_SHA
    assert candidate_manifest.get("runtime_authorized") is False
    assert candidate_manifest.get("review_status") == "PENDING_PRIMARY_REVIEW"
    rollback_manifest = json.loads((ROLLBACK_BUNDLE / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert rollback_manifest.get("files", {}).get("manifest.json", {}).get("sha256") == SDMA_MANIFEST_SHA
    return [candidate_bundle, rollback_bundle]


def build_health_codes(mixed, capacity):
    codes = {}
    for node in ALL_NODES:
        code = mixed.render(node, capacity) + HEALTH_EXTRA
        ast.parse(code)
        codes["health-%d" % node] = code
    return codes


def build_cpu_code(candidate_bundle, rollback_bundle):
    candidate_files = sorted((name, meta["sha256"]) for name, meta in candidate_bundle["files"].items())
    rollback_files = sorted((name, meta["sha256"]) for name, meta in rollback_bundle["files"].items())
    cpu = (
        CPU_TEMPLATE.replace("__EAGER_SHA__", repr(EAGER_SHA))
        .replace("__ENVIRON_SHA__", repr(ENVIRON_SHA))
        .replace("__AUDIT_SHA__", repr(AUDIT_SHA))
        .replace("__SDMA_MANIFEST_SHA__", repr(SDMA_MANIFEST_SHA))
        .replace("__CANDIDATE_FILES__", repr(candidate_files))
        .replace("__ROLLBACK_FILES__", repr(rollback_files))
    )
    ast.parse(cpu)
    return cpu


def plan():
    stage_helper = load_stage_helper()
    bundles = verify_bundles(stage_helper)
    capacity = json.loads(CAPACITY.read_text(encoding="utf-8"))
    assert sha256_file(CAPACITY) == CAPACITY_SHA
    verify_capacity(capacity)
    mixed = load_module(MIXED_IDENTITY, MIXED_IDENTITY_SHA, "glm53_l17_mixed_identity")
    health_codes = build_health_codes(mixed, capacity)
    cpu_code = build_cpu_code(bundles[0], bundles[1])
    return stage_helper, bundles, health_codes, cpu_code


def code_hashes(health_codes, cpu_code):
    return {
        "host_code_sha256": {name: hashlib.sha256(code.encode("utf-8")).hexdigest() for name, code in sorted(health_codes.items())},
        "CPU_code_sha256": hashlib.sha256(cpu_code.encode("utf-8")).hexdigest(),
    }


def render_cpu_host_code(bundles, cpu_code):
    cpu_args = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--cpus=2",
        "--memory=4g",
        "--pids-limit=256",
        "--label",
        "project_id=glm53",
        "--name",
        "glm53-L17-D-CPU-20261019-36",
        "-v",
        bundles[0]["remote"] + ":/opt/glm53/candidate:ro",
        "-v",
        bundles[1]["remote"] + ":/opt/glm53/rollback:ro",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint",
        "python3",
        IMAGE,
        "-c",
        cpu_code,
    ]
    host_cpu_code = (
        "import subprocess,json\n"
        "r=subprocess.run(" + repr(cpu_args) + ",capture_output=True,text=True,timeout=240)\n"
        "print(json.dumps({'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))\n"
        "raise SystemExit(r.returncode)\n"
    )
    ast.parse(host_cpu_code)
    return host_cpu_code


def staging_payloads(bundles):
    out = []
    for bundle in bundles:
        payload = copy.deepcopy(bundle)
        local = payload['bundle_manifest']
        assert isinstance(local, str) and not pathlib.PurePosixPath(local).is_absolute() and '..' not in pathlib.PurePosixPath(local).parts
        assert 'bundle-manifest.json' not in payload['files']
        assert sha256_file(ROOT / local) == payload['bundle_manifest_sha256']
        payload['files']['bundle-manifest.json'] = {'local': local, 'sha256': payload['bundle_manifest_sha256']}
        out.append(payload)
    assert [len(x['files']) for x in out] == [24, 23]
    return out


def stage_host_codes(helper, payloads):
    original = helper.invoke
    helper.invoke = lambda node, code, timeout=120: code
    try:
        return {'stage-%d-%d' % (index, node): helper.stage(payload, node) for index, payload in enumerate(payloads) for node in payload['nodes']}
    finally:
        helper.invoke = original


def extra_review_hashes(helper, bundles, cpu_code):
    payloads = staging_payloads(bundles)
    codes = stage_host_codes(helper, payloads)
    for code in codes.values():
        ast.parse(code)
    return {
        'stage_host_code_sha256': {key: hashlib.sha256(code.encode()).hexdigest() for key, code in codes.items()},
        'staging_payload_sha256': [hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest() for payload in payloads],
        'actual_CPU_host_wrapper_sha256': hashlib.sha256(render_cpu_host_code(bundles, cpu_code).encode()).hexdigest(),
    }


def main():
    stage_helper, bundles, health_codes, cpu_code = plan()
    assert REVIEW.is_file(), "root stage review is missing: %s" % REVIEW
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    expected_hashes = code_hashes(health_codes, cpu_code)
    assert review.get("review_status") == "APPROVED_FOR_STAGE_ONLY"
    assert review.get("runtime_authorized") is False
    assert review.get("host_code_sha256") == expected_hashes["host_code_sha256"]
    assert review.get("CPU_code_sha256") == expected_hashes["CPU_code_sha256"]
    extra_hashes = extra_review_hashes(stage_helper, bundles, cpu_code)
    assert all(review.get(key) == value for key, value in extra_hashes.items())
    payloads = staging_payloads(bundles)

    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "execution-once.lock").write_text(str(os.getpid()), encoding="utf-8")
    receipt = {
        "project_id": "glm53",
        "run_id": RID,
        "launcher_sha256": sha256_file(pathlib.Path(__file__)),
        "stage_helper_sha256": STAGE_HELPER_SHA,
        "candidate_stage_sha256": CANDIDATE_STAGE_SHA,
        "candidate_bundle_manifest_sha256": CANDIDATE_BUNDLE_SHA,
        "rollback_stage_sha256": ROLLBACK_STAGE_SHA,
        "rollback_bundle_manifest_sha256": ROLLBACK_BUNDLE_SHA,
        "review_path": str(REVIEW.relative_to(ROOT)),
        "review_sha256": sha256_file(REVIEW),
        "host_code_sha256": expected_hashes["host_code_sha256"],
        "CPU_code_sha256": expected_hashes["CPU_code_sha256"],
        "extra_review_hashes": extra_hashes,
        "staged_file_counts_including_metadata": [len(x["files"]) for x in payloads],
        "max_SSH_concurrency": 2,
        "health": {},
        "stages": {},
        "CPU_pass": False,
        "model_requests": 0,
        "SDK_init_calls": 0,
        "existing_services_changed": False,
        "lifecycle_launcher_created": False,
        "runtime_authorized": False,
        "pass": False,
    }

    def save(name, result):
        (RUN / name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    try:
        # All four current mixed P-new / D-retained ownership and healthy-idle
        # checks run before any new root exists.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(stage_helper.invoke, node, health_codes["health-%d" % node], 150): node for node in ALL_NODES}
            for future in concurrent.futures.as_completed(jobs):
                node = jobs[future]
                result = save("health-%d.json" % node, future.result())
                receipt["health"][str(node)] = result["rc"]
                assert result["rc"] == 0, "health check failed on node %d: %r" % (node, result)

        # Four exact-once stage jobs, at most two SSH commands in flight.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {
                pool.submit(stage_helper.stage, bundle, node): (index, node)
                for index, bundle in enumerate(payloads)
                for node in bundle["nodes"]
            }
            for future in concurrent.futures.as_completed(jobs):
                index, node = jobs[future]
                key = "%d-%d" % (index, node)
                result = save("stage-%s.json" % key, future.result())
                receipt["stages"][key] = result["rc"]
                assert result["rc"] == 0, "stage failed: %s: %r" % (key, result)

        host_cpu_code = render_cpu_host_code(bundles, cpu_code)
        result = save("CPU-image.json", stage_helper.invoke(36, host_cpu_code, 300))
        assert result["rc"] == 0, "CPU container command failed: %r" % (result,)
        container = json.loads(result["stdout"])
        assert container["rc"] == 0, "CPU container failed: %r" % (container,)
        marker = "GLM53_L17_CPU="
        assert marker in container["stdout"], "CPU marker missing: %r" % (container,)
        cpu_result = json.loads(container["stdout"].split(marker, 1)[1].strip().splitlines()[0])
        assert cpu_result.get("pass") is True and cpu_result.get("checks") == cpu_result.get("checks_passed")
        receipt["CPU_pass"] = True
        receipt["CPU_result"] = cpu_result
        receipt["pass"] = True
    finally:
        save("receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--print-code-hashes":
        _, _, health_codes, cpu_code = plan()
        print(json.dumps(code_hashes(health_codes, cpu_code), indent=2))
        sys.exit(0)
    sys.exit(main())
