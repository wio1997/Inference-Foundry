"""Local stdlib CPU qualification for the corrected L17-D native eager-input v2.

No framework, torch, NPU, network or remote access is used.  The exact installed
``EagerRunner.load_batch`` AST is extracted from the L16 source copy and the
native branch is exercised with the actual installed ``EnvField``/``EnvBool``
AST and the actual ``Envs`` flag assignment, compiled into a source-qualified
minimal ``Envs`` class (descriptor ``__set_name__`` runs naturally).  The v2
hook's native getter, environ/eager source pins, serialized sticky first-use,
fork reset and clone metadata validation are validated with synthetic
dataclasses, metadata-only fake tensors and a spy eager registry.  The
candidate-v2/rollback-v1 bundle diffs and the immutable v1 artifacts are
independently checked.
"""

from __future__ import annotations

import __future__
import ast
import contextlib
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import types
import warnings

PROJECT_ID = "glm53"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTALLED_SOURCE = (
    PROJECT_ROOT
    / "runs/20261019T-L16-sglang-D-eager-input-source-prereq-v1/source/node-36/"
    "sglang.srt.model_executor.runner.eager_runner.py"
)
INSTALLED_ENVIRON = (
    PROJECT_ROOT
    / "runs/20261019T-L16-sglang-D-eager-input-source-prereq-v1/source/node-36/"
    "sglang.srt.environ.py"
)
INSTALLED_SHA256 = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
INSTALLED_ENVIRON_SHA256 = "0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c"
AUDIT_PATH = PROJECT_ROOT / "scripts/glm53_C32_L17_eager_input_audit_v2.py"
PREPARE_PATH = PROJECT_ROOT / "scripts/prepare_C32_L17_D_native_eager_bundles_v2.py"
BASE_STAGE = PROJECT_ROOT / "config/sglang-C32-target-graph-fullmesh-stage-v3.json"
CANDIDATE_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-native-eager-stage-v2.json"
CANDIDATE_DIR = PROJECT_ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v2"
CANDIDATE_V1_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-native-eager-stage-v1.json"
CANDIDATE_V1_DIR = PROJECT_ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v1"
ROLLBACK_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-original-rollback-stage-v1.json"
ROLLBACK_DIR = PROJECT_ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1"
OLD_SITECUSTOMIZE = (
    PROJECT_ROOT / "config/optimization-overlays/C32-target-graph-fullmesh-v1/python/sitecustomize_v2.py"
)
AUDIT_V1_PATH = PROJECT_ROOT / "scripts/glm53_C32_L17_eager_input_audit_v1.py"
QUALIFY_V1_PATH = PROJECT_ROOT / "scripts/qualify_C32_L17_D_native_eager_local_v1.py"
RESULT_V1_PATH = PROJECT_ROOT / "results/C32-L17-D-native-eager-local-preparation-v1.json"
REVIEW_V1_PATH = PROJECT_ROOT / "diagnostics/model-profile/L17-primary-audit-draft-review-v1.md"
REFINEMENT_PATH = PROJECT_ROOT / "diagnostics/model-profile/L17-first-use-batch-coverage-review-v2.md"
RESULT_PATH = PROJECT_ROOT / "results/C32-L17-D-native-eager-local-preparation-v2.json"
RUN_DIR = PROJECT_ROOT / "runs/20261019T-L17-local-audit-correction-executor-v2"
CANDIDATE_REMOTE = (
    "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-native-eager-v2-20261019"
)
ROLLBACK_REMOTE = (
    "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-original-rollback-v1-20261019"
)
CANDIDATE_RUN_ID = "20261019T-L17-D-native-eager-v2"
ROLLBACK_RUN_ID = "20261019T-L17-D-original-rollback-v1"
D_NODES = [36, 41]
P_FILES = [
    "config/node-34.json",
    "config/node-35.json",
    "provider-config/node-34.json",
    "provider-config/node-35.json",
    "supervisor-config/node-34.json",
    "supervisor-config/node-35.json",
]
AUDIT_ENV = {
    "SGLANG_EAGER_INPUT_NO_COPY": "1",
    "GLM53_EAGER_INPUT_AUDIT": "1",
    "GLM53_EAGER_INPUT_AUDIT_ROOT": "/run/glm53/eager-input-audit",
}
ALLOWLISTED = [
    "input_ids",
    "positions",
    "seq_lens",
    "out_cache_loc",
    "req_pool_indices",
    "seq_lens_cpu",
    "spec_info",
]
EXPECTED_V1 = {
    "audit": "1be2514ca62a270b274d451d8daf5ba325d42bc9344a0cfcac830b21a8002a5c",
    "qualifier": "aa745940cb95a30e63c08d332c6c13699f808141742b553fba279243e434a550",
    "result": "441a10f365d0c26dffe49e50adf233479dc39ebd23f9dd2abfa78111f29258ba",
    "review": "ae2c2b9b363c6fdb204ad3147e62bdb124f3a21c09245532d7a4252aa52c10a0",
    "candidate_stage": "72cdfaa288e88ecb06fba5b3cdbb054a15bf3809d8f61625bb98431dc82abecf",
    "candidate_bundle": "31cef859d835cbe6d0835fe7b3b1064f9ede7e2f63a2f71b6838857f8b67b8c2",
    "rollback_stage": "f8b49216cd3ac6651cb48f9308a2590d2c0e306b339355e568a522bb882ddcf0",
    "rollback_bundle": "b3e1d5824eed9e86b61e1a20a0f0a46732c5caca543d83e1310d62e4a171cd7a",
}
EXPECTED_BASE_STAGE_SHA256 = "4a00cb3433b0bc96ee92de899d371a3c3174027e3ceb8ee338ccffa493fe0291"
EXPECTED_NATIVE_SOURCE_MANIFEST = "0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528"
EXPECTED_SDMA_MANIFEST = "0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(pathlib.Path(path).read_bytes())


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def structural_diff(old, new, prefix=""):
    changes = []
    if type(old) is not type(new):
        return [prefix or "<root>"]
    if isinstance(old, dict):
        for key in sorted(set(old) | set(new)):
            child = (prefix + "." + key) if prefix else key
            if key not in old or key not in new:
                changes.append(child)
            else:
                changes.extend(structural_diff(old[key], new[key], child))
        return changes
    if isinstance(old, list):
        if old != new:
            changes.append(prefix or "<root>")
        return changes
    if old != new:
        changes.append(prefix or "<root>")
    return changes


class Checks:
    def __init__(self):
        self.items = []
        self.failures = []

    def add(self, name, condition, detail=None):
        passed = bool(condition)
        self.items.append({"name": name, "pass": passed, "detail": detail})
        if not passed:
            self.failures.append({"check": name, "detail": detail})
        return passed

    def equal(self, name, actual, expected):
        return self.add(name, actual == expected, {"actual": actual, "expected": expected})


class FakeTensor:
    """Metadata-only tensor stand-in; no storage, no value reads."""

    instances = 0

    def __init__(self, shape, stride, dtype="torch.int64", device="npu:0"):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self.dtype = dtype
        self.device = device
        self.cpu_calls = 0
        self.item_calls = 0
        self.data_ptr_calls = 0
        self.eq_calls = 0
        FakeTensor.instances += 1

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

    def __eq__(self, other):
        self.eq_calls += 1
        return self is other

    def __repr__(self):
        return "<FakeTensor shape=%r stride=%r>" % (self.shape, self._stride)


@dataclasses.dataclass
class FakeSpecInfo:
    draft_input_ids: object
    draft_positions: object
    num_steps: int


@dataclasses.dataclass
class FakeStaticMetadata:
    planned_bs: int
    planned_num_tokens: int
    ready: bool


@dataclasses.dataclass
class FakeForwardBatch:
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


class FakeMode:
    def __init__(self, name):
        self.name = name


class SpyRegistry:
    def __init__(self):
        self.fill_calls = 0
        self.extract_calls = 0

    def fill_from(self, *args, **kwargs):
        self.fill_calls += 1

    def extract_buffer(self, *args, **kwargs):
        self.extract_calls += 1
        return "EXTRACTED_FROM_STATIC_REGISTRY"


class FakeRunner:
    def __init__(self, is_draft_worker):
        self.model_runner = types.SimpleNamespace(is_draft_worker=is_draft_worker)
        self._eager_registry = SpyRegistry()


def make_batch(mode_name="DECODE", draft=False, batch_size=4):
    mode = FakeMode(mode_name)
    return FakeForwardBatch(
        forward_mode=mode,
        batch_size=batch_size,
        input_ids=FakeTensor((4, 3), (5, 1)),
        req_pool_indices=FakeTensor((4,), (1,)),
        seq_lens=FakeTensor((4,), (1,)),
        out_cache_loc=FakeTensor((12,), (1,)),
        seq_lens_cpu=None,
        positions=FakeTensor((4, 3), (3, 1)),
        spec_info=FakeSpecInfo(
            draft_input_ids=FakeTensor((4, 6), (6, 1)),
            draft_positions=FakeTensor((4, 6), (6, 1)),
            num_steps=5,
        ),
        input_embeds=None,
        static_metadata=FakeStaticMetadata(planned_bs=4, planned_num_tokens=12, ready=True),
        return_logprob=False,
        capture_hidden_mode="NONE",
    )


def extract_load_batch():
    source_bytes = INSTALLED_SOURCE.read_bytes()
    source_text = source_bytes.decode("utf-8")
    actual_sha = sha256_bytes(source_bytes)
    tree = ast.parse(source_text, filename=str(INSTALLED_SOURCE))
    function = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EagerRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "load_batch":
                    function = item
                    break
    if function is None:
        raise RuntimeError("installed EagerRunner.load_batch was not found")
    module_ast = ast.Module(body=[copy.deepcopy(function)], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    code = compile(
        module_ast,
        str(INSTALLED_SOURCE),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    segment = ast.get_source_segment(source_text, function) or ""
    arg_names = [arg.arg for arg in function.args.args] + [arg.arg for arg in function.args.kwonlyargs]
    return {
        "source_bytes": len(source_bytes),
        "source_sha256": actual_sha,
        "lineno": function.lineno,
        "end_lineno": function.end_lineno,
        "arg_names": arg_names,
        "vararg": function.args.vararg.arg if function.args.vararg else None,
        "kwarg": function.args.kwarg.arg if function.args.kwarg else None,
        "segment_bytes": len(segment.encode("utf-8")),
        "segment_sha256": sha256_bytes(segment.encode("utf-8")),
        "code": code,
    }


_STDLIB_ROOTS = {
    "base64",
    "contextlib",
    "enum",
    "functools",
    "json",
    "os",
    "typing",
    "warnings",
}


def _class_def(name, body):
    fields = {"name": name, "bases": [], "keywords": [], "body": body, "decorator_list": []}
    if "type_params" in ast.ClassDef._fields:
        fields["type_params"] = []
    return ast.ClassDef(**fields)


def extract_env_fixture():
    """Build a minimal Envs from the actual EnvField/EnvBool/flag AST."""
    source_bytes = INSTALLED_ENVIRON.read_bytes()
    source_text = source_bytes.decode("utf-8")
    tree = ast.parse(source_text, filename=str(INSTALLED_ENVIRON))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for required in ("EnvField", "EnvBool", "Envs"):
        if required not in classes:
            raise RuntimeError("installed environ is missing class %s" % required)
    flag_assignment = None
    for item in classes["Envs"].body:
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SGLANG_EAGER_INPUT_NO_COPY"
            for target in item.targets
        ):
            flag_assignment = item
            break
    if flag_assignment is None:
        raise RuntimeError("installed Envs has no SGLANG_EAGER_INPUT_NO_COPY assignment")

    prelude = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(alias.name.split(".")[0] in _STDLIB_ROOTS for alias in node.names):
                prelude.append(copy.deepcopy(node))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _STDLIB_ROOTS:
                prelude.append(copy.deepcopy(node))

    minimal_envs = _class_def("Envs", [copy.deepcopy(flag_assignment)])
    module_ast = ast.Module(
        body=prelude
        + [
            copy.deepcopy(classes["EnvField"]),
            copy.deepcopy(classes["EnvBool"]),
            minimal_envs,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module_ast)
    namespace = {"__name__": "sglang.srt.environ"}
    exec(compile(module_ast, str(INSTALLED_ENVIRON), "exec", dont_inherit=True), namespace)
    envs_class = namespace["Envs"]
    envs = envs_class()
    flag = envs.SGLANG_EAGER_INPUT_NO_COPY
    return {
        "source_bytes": len(source_bytes),
        "source_sha256": sha256_bytes(source_bytes),
        "envs_class": envs_class,
        "envs": envs,
        "flag": flag,
        "flag_type": type(flag),
        "assignment_source": ast.get_source_segment(source_text, flag_assignment) or "",
        "assignment_lineno": flag_assignment.lineno,
        "envfield_lineno": classes["EnvField"].lineno,
        "envbool_lineno": classes["EnvBool"].lineno,
        "envs_lineno": classes["Envs"].lineno,
    }


def build_environ_module(envs, file_path):
    module = types.ModuleType("sglang.srt.environ")
    module.__file__ = str(file_path)
    module.envs = envs
    return module


def install_fake_environ(environ_module):
    previous = sys.modules.get("sglang.srt.environ")
    sys.modules["sglang.srt.environ"] = environ_module
    return previous


def restore_fake_environ(previous):
    if previous is None:
        sys.modules.pop("sglang.srt.environ", None)
    else:
        sys.modules["sglang.srt.environ"] = previous


def build_namespace(replace_function, envs):
    return {"envs": envs, "replace": replace_function, "__name__": "glm53_l17_fixture"}


def build_original_with_replace(lb_code, envs, replace_function):
    namespace = build_namespace(replace_function, envs)
    exec(lb_code, namespace)
    return namespace["load_batch"]


def build_counted_original(lb_code, envs):
    namespace = build_namespace(dataclasses.replace, envs)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    counted = {"calls": 0, "last_result": None}

    def load_batch(self, forward_batch, *args, **kwargs):
        counted["calls"] += 1
        counted["last_result"] = original(self, forward_batch, *args, **kwargs)
        return counted["last_result"]

    load_batch.__name__ = "load_batch"
    load_batch.__qualname__ = "EagerRunner.load_batch"
    return load_batch, counted


def build_module(load_batch):
    class FixtureEagerRunner:
        pass

    FixtureEagerRunner.load_batch = load_batch
    module = types.ModuleType("glm53_l17_fixture.eager_runner")
    module.EagerRunner = FixtureEagerRunner
    return module, FixtureEagerRunner


def make_runner(runner_class, is_draft_worker=False):
    runner = runner_class()
    runner.model_runner = types.SimpleNamespace(is_draft_worker=is_draft_worker)
    runner._eager_registry = SpyRegistry()
    return runner


def rewrite_lines(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextlib.contextmanager
def env_settings(settings):
    previous = {}
    for key, value in settings.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def expect_failure(action, fragment):
    try:
        action()
    except BaseException as error:
        text = repr(error)
        return fragment in text, text
    return False, "no exception"


def test_env_source_semantics(checks):
    checks.equal("environ_source_sha", sha256_file(INSTALLED_ENVIRON), INSTALLED_ENVIRON_SHA256)
    fixture = extract_env_fixture()
    checks.equal("environ_ast_source_sha", fixture["source_sha256"], INSTALLED_ENVIRON_SHA256)
    checks.equal("envfield_actual_ast_line", fixture["envfield_lineno"], 45)
    checks.equal("envbool_actual_ast_line", fixture["envbool_lineno"], 143)
    checks.equal("envs_actual_ast_line", fixture["envs_lineno"], 257)
    checks.equal("envs_flag_actual_ast_line", fixture["assignment_lineno"], 1448)
    checks.equal(
        "envs_flag_actual_ast_source",
        fixture["assignment_source"].strip(),
        "SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)",
    )
    checks.equal("envs_class_qualified_name", fixture["envs_class"].__module__, "sglang.srt.environ")
    checks.equal("envbool_class_qualified_name", fixture["flag_type"].__module__, "sglang.srt.environ")
    checks.equal("envs_instance_type", type(fixture["envs"]).__name__, "Envs")
    checks.equal("envs_flag_descriptor_type", fixture["flag_type"].__name__, "EnvBool")
    checks.equal("envs_flag_descriptor_name_natural", fixture["flag"].name, "SGLANG_EAGER_INPUT_NO_COPY")
    checks.equal("envs_flag_descriptor_set_to_none", fixture["flag"]._set_to_none, False)
    checks.equal("envs_flag_descriptor_default", fixture["flag"].default, False)
    checks.equal("envs_field_allow_set_name_natural", fixture["envs_class"].__dict__["SGLANG_EAGER_INPUT_NO_COPY"].__class__._allow_set_name, True)

    environ_module = build_environ_module(fixture["envs"], INSTALLED_ENVIRON)
    checks.add(
        "no_direct_module_flag_mock",
        not hasattr(environ_module, "SGLANG_EAGER_INPUT_NO_COPY"),
        sorted(vars(environ_module)),
    )
    checks.add(
        "module_envs_is_actual_object",
        getattr(environ_module, "envs", None) is fixture["envs"],
    )

    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": None}):
        checks.equal("env_source_absent_default_false", fixture["flag"].get(), False)
    for raw, expected in (
        ("1", True),
        ("true", True),
        ("yes", True),
        ("y", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("n", False),
    ):
        with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": raw}):
            checks.equal("env_source_parse_%s" % raw, fixture["flag"].get(), expected)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "bogus"}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = fixture["flag"].get()
        checks.equal("env_source_invalid_default_false", value, False)
        checks.add("env_source_invalid_warns", bool(caught))
        checks.add(
            "env_source_no_direct_module_flag_dependency",
            not hasattr(environ_module, "SGLANG_EAGER_INPUT_NO_COPY"),
        )


def test_native_branch(checks, lb_code):
    fixture = extract_env_fixture()
    envs = fixture["envs"]
    namespace = build_namespace(dataclasses.replace, envs)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    runner = FakeRunner(False)
    batch = make_batch()

    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1"}):
        result = original(runner, batch)
    checks.add("native_true_returns_distinct_clone", result is not batch, {"same_object": result is batch})
    checks.equal("native_true_clone_type", type(result).__name__, "FakeForwardBatch")
    checks.add("native_true_preserves_input_ids_identity", result.input_ids is batch.input_ids)
    checks.add("native_true_preserves_positions_identity", result.positions is batch.positions)
    checks.add("native_true_preserves_seq_lens_identity", result.seq_lens is batch.seq_lens)
    checks.add("native_true_preserves_out_cache_loc_identity", result.out_cache_loc is batch.out_cache_loc)
    checks.add("native_true_preserves_req_pool_indices_identity", result.req_pool_indices is batch.req_pool_indices)
    checks.add("native_true_preserves_none_seq_lens_cpu", result.seq_lens_cpu is None)
    checks.add("native_true_preserves_spec_info_identity", result.spec_info is batch.spec_info)
    checks.add("native_true_preserves_static_metadata_identity", result.static_metadata is batch.static_metadata)
    checks.equal("native_true_noncontiguous_stride5", result.input_ids.stride(), (5, 1))
    checks.equal("native_true_noncontiguous_shape", result.input_ids.shape, (4, 3))
    checks.equal("native_true_registry_fill_calls", runner._eager_registry.fill_calls, 0)
    checks.equal("native_true_registry_extract_calls", runner._eager_registry.extract_calls, 0)
    checks.equal("native_true_fake_tensor_cpu_calls", batch.input_ids.cpu_calls, 0)
    checks.equal("native_true_fake_tensor_item_calls", batch.input_ids.item_calls, 0)
    checks.equal("native_true_fake_tensor_data_ptr_calls", batch.input_ids.data_ptr_calls, 0)
    checks.equal("native_true_fake_tensor_eq_calls", batch.input_ids.eq_calls, 0)

    # Actual EnvBool accepts the word "true" as the same native true branch.
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "true"}):
        result_word = original(runner, batch)
    checks.add("native_true_word_returns_distinct_clone", result_word is not batch)

    # Real false source semantics must take the fixed-registry fill/extract path.
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "0"}):
        runner_false = FakeRunner(False)
        result_false = original(runner_false, batch)
    checks.equal("false_env_fill_calls", runner_false._eager_registry.fill_calls, 1)
    checks.equal("false_env_extract_calls", runner_false._eager_registry.extract_calls, 1)
    checks.equal("false_env_result", result_false, "EXTRACTED_FROM_STATIC_REGISTRY")

    # Absent environment uses the actual EnvBool default False.
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": None}):
        runner_absent = FakeRunner(False)
        result_absent = original(runner_absent, batch)
    checks.equal("absent_env_fill_calls", runner_absent._eager_registry.fill_calls, 1)
    checks.equal("absent_env_result", result_absent, "EXTRACTED_FROM_STATIC_REGISTRY")


def setup_patched(audit, original, root, env_fixture=None, install_environ=True, environ_path=None):
    audit._reset_for_tests()
    fixture = env_fixture or extract_env_fixture()
    module, runner_class = build_module(original)
    previous = None
    if install_environ:
        environ_module = build_environ_module(fixture["envs"], environ_path or INSTALLED_ENVIRON)
        previous = install_fake_environ(environ_module)
    audit._patch_module(module, INSTALLED_SHA256, str(root))
    runner = make_runner(runner_class, False)
    return runner, module, runner_class, previous, fixture


def test_hook(checks, audit, lb_code, tmpdir):
    hook_text = AUDIT_PATH.read_text(encoding="utf-8")
    checks.add(
        "audit_module_no_framework_import_at_top",
        "import torch" not in hook_text
        and "import sglang" not in hook_text
        and "from sglang" not in hook_text,
    )
    checks.add(
        "audit_module_pins_both_sources",
        audit.SOURCE_SHA256 == INSTALLED_SHA256
        and audit.ENVIRON_SHA256 == INSTALLED_ENVIRON_SHA256,
    )

    fixture = extract_env_fixture()
    flag_type = fixture["flag_type"]
    original_get = flag_type.get
    get_counter = {"calls": 0}

    def counted_get(self):
        get_counter["calls"] += 1
        return original_get(self)

    flag_type.get = counted_get
    try:
        load_batch, counted = build_counted_original(lb_code, fixture["envs"])
        module, runner_class = build_module(load_batch)
        previous_environ = install_fake_environ(
            build_environ_module(fixture["envs"], INSTALLED_ENVIRON)
        )
        audit._reset_for_tests()
        audit._patch_module(module, INSTALLED_SHA256, str(tmpdir))
        checks.add("wrapper_marker_v2", getattr(runner_class.load_batch, "__glm53_l17_eager_input_audit__", None) == "v2")
        checks.add("wrapper_original_marker", getattr(runner_class.load_batch, "__glm53_l17_original__", None) is load_batch)
        checks.add("wrapper_installed", runner_class.load_batch is not load_batch)
        checks.add("environ_source_pinned_at_patch", audit._ENVIRON_SOURCE_PINNED is True)

        runner = make_runner(runner_class, False)
        batch = make_batch("DECODE")
        with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
            result = runner.load_batch(batch)
        checks.equal("wrapper_one_original_call", counted["calls"], 1)
        checks.add("wrapper_returns_original_outcome", result is counted["last_result"])
        checks.add("wrapper_result_distinct", result is not batch)
        checks.equal("wrapper_env_get_first_use", get_counter["calls"], 2)  # audit get + native get
        checks.add("first_use_key_accepted", (os.getpid(), False, "DECODE", 4) in audit._SEEN)

        pid_file = tmpdir / ("pid-%d.jsonl" % os.getpid())
        checks.add("first_use_jsonl_created", pid_file.is_file(), str(pid_file))
        lines = rewrite_lines(pid_file)
        checks.equal("first_use_line_count", len(lines), 1)
        record = lines[0] if lines else {}
        checks.equal("record_schema", record.get("schema"), "glm53.L17.eager_input_audit.v2")
        checks.equal("record_audit_version", record.get("audit_version"), "v2")
        checks.equal("record_source_sha", record.get("source_sha256"), INSTALLED_SHA256)
        checks.equal("record_eager_source_sha", record.get("eager_source_sha256"), INSTALLED_SHA256)
        checks.equal("record_environ_source_sha", record.get("environ_source_sha256"), INSTALLED_ENVIRON_SHA256)
        checks.equal("record_node", record.get("node"), 36)
        checks.equal("record_pid", record.get("pid"), os.getpid())
        checks.equal("record_forward_mode", record.get("forward_mode"), "DECODE")
        checks.equal("record_is_draft", record.get("is_draft_worker"), False)
        checks.equal("record_batch_size", record.get("batch_size"), 4)
        checks.equal(
            "record_batch_size_source",
            record.get("batch_size_source"),
            "forward_batch.batch_size (Python int, padded live batch)",
        )
        checks.equal(
            "record_first_use_key_fields",
            record.get("first_use_key_fields"),
            ["pid", "is_draft_worker", "forward_mode", "batch_size"],
        )
        checks.equal("record_native_get", record.get("native_no_copy_get"), True)
        checks.equal("record_clone_distinct", record.get("clone_distinct"), True)
        checks.equal("record_field_count", record.get("dataclass_field_count"), len(dataclasses.fields(FakeForwardBatch)))
        checks.equal("record_identity_count", record.get("field_verdicts", {}).get("identity"), len(dataclasses.fields(FakeForwardBatch)))
        allowlisted = record.get("allowlisted", {})
        input_ids_meta = allowlisted.get("input_ids", {})
        checks.equal("record_input_ids_shape", input_ids_meta.get("shape"), [4, 3])
        checks.equal("record_input_ids_stride", input_ids_meta.get("stride"), [5, 1])
        checks.equal("record_input_ids_dtype", input_ids_meta.get("dtype"), "torch.int64")
        checks.equal("record_input_ids_device", input_ids_meta.get("device"), "npu:0")
        checks.equal("record_seq_lens_cpu_raw_none", allowlisted.get("seq_lens_cpu"), None)
        checks.add("record_seq_lens_cpu_preserved_none", "seq_lens_cpu" in allowlisted and allowlisted["seq_lens_cpu"] is None)
        checks.equal("record_spec_info_dataclass", allowlisted.get("spec_info", {}).get("dataclass"), "FakeSpecInfo")
        checks.equal(
            "record_spec_info_nested_shape",
            allowlisted.get("spec_info", {}).get("fields", {}).get("draft_input_ids", {}).get("shape"),
            [4, 6],
        )
        checks.add("record_spec_info_value_not_read", "values" not in json.dumps(record), None)
        checks.add(
            "record_field_names_hashed",
            isinstance(record.get("field_names_sha256"), str) and len(record["field_names_sha256"]) == 64,
        )

        # Accepted fastpath: no audit get/pin/write/fs, original exactly once.
        size_after_first = pid_file.stat().st_size
        gets_after_first = get_counter["calls"]

        def forbidden(*args, **kwargs):
            raise AssertionError("accepted fastpath must not call audit first-use helpers")

        saved_append = audit._append_record
        saved_resolve = audit._resolve_root
        saved_native = audit._native_no_copy_get_true
        saved_pin = audit._pin_environ_source
        audit._append_record = forbidden
        audit._resolve_root = forbidden
        audit._native_no_copy_get_true = forbidden
        audit._pin_environ_source = forbidden
        try:
            with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
                result2 = runner.load_batch(batch)
        finally:
            audit._append_record = saved_append
            audit._resolve_root = saved_resolve
            audit._native_no_copy_get_true = saved_native
            audit._pin_environ_source = saved_pin
        checks.equal("same_key_second_original_call", counted["calls"], 2)
        checks.add("same_key_second_outcome", result2 is counted["last_result"])
        checks.equal("same_key_native_only_get_delta", get_counter["calls"] - gets_after_first, 1)
        checks.equal("same_key_no_new_line", len(rewrite_lines(pid_file)), 1)
        checks.equal("same_key_no_file_growth", pid_file.stat().st_size, size_after_first)

        with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
            batch_verify = make_batch("TARGET_VERIFY")
            runner.load_batch(batch_verify)
        checks.equal("new_mode_line_count", len(rewrite_lines(pid_file)), 2)
        checks.equal("new_mode_env_get", get_counter["calls"], gets_after_first + 1 + 2)

        runner_draft = make_runner(runner_class, True)
        with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
            runner_draft.load_batch(make_batch("DECODE"))
        checks.equal("new_is_draft_line_count", len(rewrite_lines(pid_file)), 3)
        checks.equal("new_is_draft_env_get", get_counter["calls"], gets_after_first + 1 + 2 + 2)

        seen_types_ok = True
        for key in audit._SEEN:
            if not (
                type(key) is tuple
                and len(key) == 4
                and type(key[0]) is int
                and type(key[1]) is bool
                and type(key[2]) is str
                and type(key[3]) is int
            ):
                seen_types_ok = False
        checks.add("seen_keys_primitives_only", seen_types_ok, None)
        tensor_refs = [name for name, value in vars(audit).items() if isinstance(value, FakeTensor)]
        checks.add("no_tensor_refs_in_audit_globals", not tensor_refs, tensor_refs)

        # False environment fails closed before the original and is sticky.
        counted["calls"] = 0
        with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "0", "GLM53_NODE": "36"}):
            ok, text = expect_failure(
                lambda: runner.load_batch(make_batch("MIXED")),
                "native envs.SGLANG_EAGER_INPUT_NO_COPY.get() is not True",
            )
        checks.add("false_env_fails_closed", ok, text)
        checks.equal("false_env_original_not_called", counted["calls"], 0)
        checks.add("false_env_failure_is_bounded_string", isinstance(audit._FAILURE, str) and len(audit._FAILURE) <= 600, audit._FAILURE)
        with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
            ok, text = expect_failure(
                lambda: runner.load_batch(make_batch("MIXED")),
                "sticky first-use audit failure",
            )
        checks.add("false_env_retry_sticky", ok, text)
        checks.equal("false_env_retry_original_not_called", counted["calls"], 0)

        # Controlled PID change produces a dedicated first-use record.
        audit._reset_for_tests()
        load_batch2, counted2 = build_counted_original(lb_code, fixture["envs"])
        module2, runner_class2 = build_module(load_batch2)
        previous2 = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
        audit._patch_module(module2, INSTALLED_SHA256, str(tmpdir))
        runner2 = make_runner(runner_class2, False)
        original_pid = audit._current_pid
        audit._current_pid = lambda: 424242
        try:
            with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
                runner2.load_batch(make_batch("DECODE"))
        finally:
            audit._current_pid = original_pid
            restore_fake_environ(previous2)
        fork_file = tmpdir / "pid-424242.jsonl"
        checks.add("controlled_pid_dedicated_file", fork_file.is_file(), str(fork_file))
        fork_lines = rewrite_lines(fork_file)
        checks.equal("controlled_pid_line_count", len(fork_lines), 1)
        checks.equal("controlled_pid_record_pid", fork_lines[0].get("pid") if fork_lines else None, 424242)

        restore_fake_environ(previous_environ)
    finally:
        flag_type.get = original_get
        audit._reset_for_tests()


def test_batch_key_coverage(checks, audit, lb_code, tmpdir):
    tmpdir.mkdir(parents=True, exist_ok=True)
    fixture = extract_env_fixture()
    load_batch, counted = build_counted_original(lb_code, fixture["envs"])
    module, runner_class = build_module(load_batch)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._reset_for_tests()
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "bs_sequence"))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        runner.load_batch(make_batch("DECODE", batch_size=1))
        runner.load_batch(make_batch("DECODE", batch_size=2))
    pid_file = tmpdir / "bs_sequence" / ("pid-%d.jsonl" % os.getpid())
    lines = rewrite_lines(pid_file)
    checks.equal("batch_key_bs1_then_bs2_two_records", len(lines), 2)
    checks.equal("batch_key_bs1_batch_size", lines[0].get("batch_size") if lines else None, 1)
    checks.equal("batch_key_bs2_batch_size", lines[1].get("batch_size") if len(lines) > 1 else None, 2)
    checks.equal(
        "batch_key_modes_preserved",
        [line.get("forward_mode") for line in lines],
        ["DECODE", "DECODE"],
    )
    checks.equal(
        "batch_key_bs1_identity_all_fields",
        lines[0].get("field_verdicts", {}).get("identity") if lines else None,
        len(dataclasses.fields(FakeForwardBatch)),
    )
    checks.equal(
        "batch_key_bs2_identity_all_fields",
        lines[1].get("field_verdicts", {}).get("identity") if len(lines) > 1 else None,
        len(dataclasses.fields(FakeForwardBatch)),
    )
    checks.equal(
        "batch_key_bs1_input_ids_stride",
        lines[0].get("allowlisted", {}).get("input_ids", {}).get("stride") if lines else None,
        [5, 1],
    )
    checks.equal(
        "batch_key_bs2_input_ids_stride",
        lines[1].get("allowlisted", {}).get("input_ids", {}).get("stride") if len(lines) > 1 else None,
        [5, 1],
    )
    checks.add("batch_key_bs1_accepted", (os.getpid(), False, "DECODE", 1) in audit._SEEN)
    checks.add("batch_key_bs2_accepted", (os.getpid(), False, "DECODE", 2) in audit._SEEN)

    size_after_bs2 = pid_file.stat().st_size
    calls_after_bs2 = counted["calls"]
    saved_native = audit._native_no_copy_get_true
    saved_append = audit._append_record
    audit._native_no_copy_get_true = lambda: (_ for _ in ()).throw(AssertionError("repeat bs2 audited"))
    audit._append_record = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("repeat bs2 wrote"))
    try:
        with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
            runner.load_batch(make_batch("DECODE", batch_size=2))
    finally:
        audit._native_no_copy_get_true = saved_native
        audit._append_record = saved_append
    checks.equal("batch_key_repeat_bs2_original_once", counted["calls"], calls_after_bs2 + 1)
    checks.equal("batch_key_repeat_bs2_no_new_record", len(rewrite_lines(pid_file)), 2)
    checks.equal("batch_key_repeat_bs2_no_growth", pid_file.stat().st_size, size_after_bs2)
    restore_fake_environ(previous)
    audit._reset_for_tests()

    # A larger decode batch fails closed; it is not clamped to bs4.
    fixture2 = extract_env_fixture()
    load_batch2, counted2 = build_counted_original(lb_code, fixture2["envs"])
    module2, runner_class2 = build_module(load_batch2)
    previous2 = install_fake_environ(build_environ_module(fixture2["envs"], INSTALLED_ENVIRON))
    audit._reset_for_tests()
    audit._patch_module(module2, INSTALLED_SHA256, str(tmpdir / "bs_overflow"))
    runner2 = make_runner(runner_class2, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(
            lambda: runner2.load_batch(make_batch("DECODE", batch_size=5)),
            "exceeds the DECODE mode bound 4; not clamped",
        )
        checks.add("decode_bs5_not_clamped", ok, text)
        checks.equal("decode_bs5_original_not_called", counted2["calls"], 0)
        ok, text = expect_failure(
            lambda: runner2.load_batch(make_batch("DECODE", batch_size=5)),
            "sticky first-use audit failure",
        )
        checks.add("decode_bs5_retry_sticky", ok, text)
        checks.equal("decode_bs5_retry_original_not_called", counted2["calls"], 0)
    restore_fake_environ(previous2)
    audit._reset_for_tests()

    # The key never reads tensor-like batch_size.
    fixture3 = extract_env_fixture()
    load_batch3, counted3 = build_counted_original(lb_code, fixture3["envs"])
    module3, runner_class3 = build_module(load_batch3)
    previous3 = install_fake_environ(build_environ_module(fixture3["envs"], INSTALLED_ENVIRON))
    audit._reset_for_tests()
    audit._patch_module(module3, INSTALLED_SHA256, str(tmpdir / "bs_tensor"))
    runner3 = make_runner(runner_class3, False)
    bad_batch = dataclasses.replace(
        make_batch("DECODE", batch_size=1), batch_size=FakeTensor((1,), (1,))
    )
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(
            lambda: runner3.load_batch(bad_batch), "batch_size is not a Python int"
        )
        checks.add("non_int_batch_size_rejected", ok, text)
        checks.equal("non_int_batch_size_original_not_called", counted3["calls"], 0)
    restore_fake_environ(previous3)
    audit._reset_for_tests()

    # Extend batches above 4 use the existing 64-request bound and keep the
    # actual size; they are not clamped to bs4 either.
    fixture4 = extract_env_fixture()
    load_batch4, counted4 = build_counted_original(lb_code, fixture4["envs"])
    module4, runner_class4 = build_module(load_batch4)
    previous4 = install_fake_environ(build_environ_module(fixture4["envs"], INSTALLED_ENVIRON))
    audit._reset_for_tests()
    audit._patch_module(module4, INSTALLED_SHA256, str(tmpdir / "bs_extend"))
    runner4 = make_runner(runner_class4, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        runner4.load_batch(make_batch("EXTEND", batch_size=5))
    extend_file = tmpdir / "bs_extend" / ("pid-%d.jsonl" % os.getpid())
    extend_lines = rewrite_lines(extend_file)
    checks.equal(
        "extend_bs5_uses_actual_size",
        extend_lines[0].get("batch_size") if extend_lines else None,
        5,
    )
    checks.add("extend_bs5_not_clamped", (os.getpid(), False, "EXTEND", 5) in audit._SEEN)
    restore_fake_environ(previous4)
    audit._reset_for_tests()


def test_failures(checks, audit, lb_code, tmpdir):
    tmpdir.mkdir(parents=True, exist_ok=True)
    fixture = extract_env_fixture()

    # Wrong eager source SHA in patch.
    load_batch, _ = build_counted_original(lb_code, fixture["envs"])
    module, runner_class = build_module(load_batch)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._reset_for_tests()
    audit._ACTIVE_SOURCE_SHA = INSTALLED_SHA256
    ok, text = expect_failure(lambda: audit._patch_module(module, "00" * 32, str(tmpdir)), "patch source SHA mismatch")
    checks.add("wrong_patch_eager_sha_fails", ok, text)
    checks.add("wrong_patch_eager_not_installed", runner_class.load_batch is load_batch)
    restore_fake_environ(previous)

    # Loader must verify eager bytes before executing source.
    bad_source = tmpdir / "bad_eager_source.py"
    bad_source.write_text(
        "EXECUTED = True\nclass EagerRunner:\n    def load_batch(self, batch):\n        return batch\n",
        encoding="utf-8",
    )
    loader = audit._PinnedSourceLoader(str(bad_source), INSTALLED_SHA256, str(tmpdir))
    bad_module = types.ModuleType("glm53_bad_fixture")
    ok, text = expect_failure(lambda: loader.exec_module(bad_module), "installed source SHA mismatch before exec")
    checks.add("wrong_loader_eager_sha_fails", ok, text)
    checks.add("wrong_loader_eager_not_executed", not hasattr(bad_module, "EXECUTED"))

    # Wrong environ SHA at patch time.
    bad_environ = tmpdir / "bad_environ.py"
    bad_environ.write_bytes(INSTALLED_ENVIRON.read_bytes() + b"\n# tampered\n")
    load_batch, _ = build_counted_original(lb_code, fixture["envs"])
    module, runner_class = build_module(load_batch)
    previous = install_fake_environ(build_environ_module(fixture["envs"], bad_environ))
    audit._reset_for_tests()
    ok, text = expect_failure(lambda: audit._patch_module(module, INSTALLED_SHA256, str(tmpdir)), "installed environ source SHA mismatch")
    checks.add("wrong_environ_sha_at_patch_fails", ok, text)
    checks.add("wrong_environ_sha_at_patch_not_installed", runner_class.load_batch is load_batch)
    restore_fake_environ(previous)

    # Wrong environ SHA at first use, after the wrapper was installed without environ.
    audit._reset_for_tests()
    calls = {"n": 0}

    def counted_original(self, batch, *args, **kwargs):
        calls["n"] += 1
        return dataclasses.replace(batch)

    module, runner_class = build_module(counted_original)
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "bad_first_use"))
    previous = install_fake_environ(build_environ_module(fixture["envs"], bad_environ))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "installed environ source SHA mismatch")
        checks.add("wrong_environ_sha_at_first_use_fails", ok, text)
        checks.equal("wrong_environ_sha_first_use_original_not_called", calls["n"], 0)
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "sticky first-use audit failure")
        checks.add("wrong_environ_sha_retry_sticky", ok, text)
        checks.equal("wrong_environ_sha_retry_original_not_called", calls["n"], 0)
    restore_fake_environ(previous)

    # Same-object failure.
    audit._reset_for_tests()
    calls = {"n": 0}

    def same_object(self, batch, *args, **kwargs):
        calls["n"] += 1
        return batch

    module, runner_class = build_module(same_object)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "same_object"))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "same ForwardBatch object")
        checks.add("source_clone_identity_fails", ok, text)
        checks.equal("source_clone_identity_original_once", calls["n"], 1)
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "sticky first-use audit failure")
        checks.add("source_clone_identity_retry_sticky", ok, text)
        checks.equal("source_clone_identity_retry_original_not_called", calls["n"], 1)
    restore_fake_environ(previous)

    # Non-allowlisted dataclass field change.
    audit._reset_for_tests()

    def mutating_static(batch):
        clone = dataclasses.replace(batch)
        clone.static_metadata = dataclasses.replace(batch.static_metadata, planned_bs=999)
        return clone

    module, runner_class = build_module(
        build_original_with_replace(lb_code, fixture["envs"], mutating_static)
    )
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "mutated_static"))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "dataclass field changed across replace: static_metadata")
        checks.add("mutated_static_metadata_fails", ok, text)
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "sticky first-use audit failure")
        checks.add("mutated_static_metadata_retry_sticky", ok, text)
    restore_fake_environ(previous)

    # Allowlisted field change.
    audit._reset_for_tests()

    def mutating_input(batch):
        clone = dataclasses.replace(batch)
        clone.input_ids = FakeTensor((4, 4), (4, 1))
        return clone

    module, runner_class = build_module(
        build_original_with_replace(lb_code, fixture["envs"], mutating_input)
    )
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "mutated_input"))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "dataclass field changed across replace: input_ids")
        checks.add("mutated_input_ids_fails", ok, text)
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "sticky first-use audit failure")
        checks.add("mutated_input_ids_retry_sticky", ok, text)
    restore_fake_environ(previous)

    # Evidence write failure: key must not be accepted and retry must be sticky.
    audit._reset_for_tests()
    root_file = tmpdir / "not_a_directory"
    root_file.write_text("x", encoding="utf-8")
    load_batch, counted = build_counted_original(lb_code, fixture["envs"])
    module, runner_class = build_module(load_batch)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._patch_module(module, INSTALLED_SHA256, str(root_file))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "cannot create audit root")
        checks.add("write_failure_fails_closed", ok, text)
        checks.equal("write_failure_original_called_once", counted["calls"], 1)
        checks.add("write_failure_key_not_accepted", (os.getpid(), False, "DECODE", 4) not in audit._SEEN, list(audit._SEEN))
        checks.add("write_failure_sticky_is_string", isinstance(audit._FAILURE, str))
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "sticky first-use audit failure")
        checks.add("write_failure_retry_sticky", ok, text)
        checks.equal("write_failure_retry_original_not_called", counted["calls"], 1)
    restore_fake_environ(previous)

    # Original exception propagates and conservatively poisons the audit.
    audit._reset_for_tests()
    calls = {"n": 0}

    def raising_original(self, batch, *args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("synthetic original failure")

    module, runner_class = build_module(raising_original)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "original_failure"))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "synthetic original failure")
        checks.add("original_failure_propagates", ok, text)
        checks.equal("original_failure_original_once", calls["n"], 1)
        checks.add("original_failure_sticky_is_bounded_string", isinstance(audit._FAILURE, str) and len(audit._FAILURE) <= 600, audit._FAILURE)
        checks.add("original_failure_no_traceback_cached", "Traceback" not in (audit._FAILURE or ""), audit._FAILURE)
        checks.add("original_failure_no_batch_repr_cached", "FakeForwardBatch" not in (audit._FAILURE or ""), audit._FAILURE)
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "sticky first-use audit failure")
        checks.add("original_failure_retry_sticky", ok, text)
        checks.equal("original_failure_retry_original_not_called", calls["n"], 1)
    restore_fake_environ(previous)

    # False environment then retry with the same key: sticky, original never called.
    audit._reset_for_tests()
    load_batch, counted = build_counted_original(lb_code, fixture["envs"])
    module, runner_class = build_module(load_batch)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "false_env"))
    runner = make_runner(runner_class, False)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "0", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "is not True")
        checks.add("false_env_first_fails", ok, text)
        checks.equal("false_env_first_original_not_called", counted["calls"], 0)
    with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
        ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "sticky first-use audit failure")
        checks.add("false_env_retry_sticky", ok, text)
        checks.equal("false_env_retry_original_not_called", counted["calls"], 0)
    restore_fake_environ(previous)


def test_concurrency(checks, audit, lb_code, tmpdir):
    tmpdir.mkdir(parents=True, exist_ok=True)
    fixture = extract_env_fixture()
    calls = {"n": 0}
    call_lock = threading.Lock()

    def slow_original(self, batch, *args, **kwargs):
        with call_lock:
            calls["n"] += 1
        time.sleep(0.15)
        return dataclasses.replace(batch)

    module, runner_class = build_module(slow_original)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._reset_for_tests()
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir))
    runner = make_runner(runner_class, False)
    barrier = threading.Barrier(3)
    outcomes = []

    def worker():
        try:
            barrier.wait(timeout=5)
            with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
                outcomes.append(("ok", runner.load_batch(make_batch("DECODE"))))
        except BaseException as error:
            outcomes.append(("error", error))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    checks.equal("concurrent_first_use_original_calls", calls["n"], 2)
    checks.equal("concurrent_first_use_outcomes", len(outcomes), 2)
    checks.add("concurrent_first_use_no_errors", all(kind == "ok" for kind, _ in outcomes), [repr(v) for _, v in outcomes])
    pid_file = tmpdir / ("pid-%d.jsonl" % os.getpid())
    checks.equal("concurrent_first_use_single_record", len(rewrite_lines(pid_file)), 1)
    checks.add("concurrent_first_use_accepted", (os.getpid(), False, "DECODE", 4) in audit._SEEN)
    checks.add("concurrent_first_use_no_failure", audit._FAILURE is None, audit._FAILURE)
    restore_fake_environ(previous)

    # Competing first use where the first original raises: exactly one original
    # call and one sticky failure for the waiter.
    calls = {"n": 0}

    def concurrent_raising(self, batch, *args, **kwargs):
        with call_lock:
            calls["n"] += 1
        time.sleep(0.15)
        raise RuntimeError("concurrent synthetic original failure")

    module, runner_class = build_module(concurrent_raising)
    previous = install_fake_environ(build_environ_module(fixture["envs"], INSTALLED_ENVIRON))
    audit._reset_for_tests()
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "concurrent_failure"))
    runner = make_runner(runner_class, False)
    barrier = threading.Barrier(3)
    errors = []

    def failing_worker():
        try:
            barrier.wait(timeout=5)
            with env_settings({"SGLANG_EAGER_INPUT_NO_COPY": "1", "GLM53_NODE": "36"}):
                runner.load_batch(make_batch("DECODE"))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=failing_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    checks.equal("concurrent_failure_original_calls", calls["n"], 1)
    checks.equal("concurrent_failure_error_count", len(errors), 2)
    checks.add(
        "concurrent_failure_one_original_one_sticky",
        sorted(type(error).__name__ for error in errors) == ["AuditFailure", "RuntimeError"],
        [type(error).__name__ for error in errors],
    )
    checks.add(
        "concurrent_failure_sticky_message",
        any("sticky first-use audit failure" in str(error) for error in errors),
        [str(error) for error in errors],
    )
    restore_fake_environ(previous)
    audit._reset_for_tests()


def test_fork(checks, audit):
    if not hasattr(os, "fork"):
        checks.add("fork_available", False, "os.fork is unavailable")
        return
    audit._reset_for_tests()
    audit._SEEN.add((1, False, "DECODE", 4))
    audit._FAILURE = "prior failure"
    audit._FIRST_USE_LOCK.acquire()
    pid = os.fork()
    if pid == 0:
        try:
            ok = audit._FAILURE is None and audit._SEEN == set()
            acquired = audit._FIRST_USE_LOCK.acquire(blocking=False)
            if acquired:
                audit._FIRST_USE_LOCK.release()
            os._exit(0 if ok and acquired else 1)
        except BaseException:
            os._exit(2)
    audit._FIRST_USE_LOCK.release()
    _, status = os.waitpid(pid, 0)
    checks.add(
        "fork_child_resets_locks_seen_failure",
        os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0,
        {"exited": os.WIFEXITED(status), "code": os.WEXITSTATUS(status) if os.WIFEXITED(status) else None},
    )
    checks.add("fork_parent_state_unchanged", audit._FAILURE == "prior failure" and (1, False, "DECODE", 4) in audit._SEEN)
    tensors = [name for name, value in vars(audit).items() if isinstance(value, FakeTensor)]
    checks.add("fork_no_tensor_globals", not tensors, tensors)
    audit._reset_for_tests()


def _purge_sglang_modules(saved):
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            sys.modules.pop(name, None)
    for name, module in saved.items():
        sys.modules[name] = module


def test_import_finder(checks, audit, tmpdir):
    tmpdir.mkdir(parents=True, exist_ok=True)
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "sglang" or name.startswith("sglang.")
    }
    _purge_sglang_modules({})

    package_root = tmpdir / "finder_pkg"
    package_dirs = [
        package_root / "sglang",
        package_root / "sglang" / "srt",
        package_root / "sglang" / "srt" / "model_executor",
        package_root / "sglang" / "srt" / "model_executor" / "runner",
    ]
    for directory in package_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    eager_source = (
        "EXECUTED = True\n"
        "class EagerRunner:\n"
        "    def load_batch(self, forward_batch, *args, **kwargs):\n"
        "        return forward_batch\n"
    )
    eager_path = package_dirs[-1] / "eager_runner.py"
    eager_path.write_text(eager_source, encoding="utf-8")
    source_sha = sha256_bytes(eager_path.read_bytes())

    sys.path.insert(0, str(package_root))
    finder = audit._AuditFinder(source_sha, str(tmpdir / "finder_root"))
    sys.meta_path.insert(0, finder)
    try:
        checks.add("finder_ignores_other_modules", finder.find_spec("sglang.srt.model_executor.runner_utils") is None)
        checks.add("finder_ignores_environ", finder.find_spec("sglang.srt.environ") is None)
        module = importlib.import_module(audit.SOURCE_MODULE)
        checks.add("finder_module_executed", getattr(module, "EXECUTED", False) is True)
        checks.add(
            "finder_wrapper_marker_v2",
            getattr(module.EagerRunner.load_batch, "__glm53_l17_eager_input_audit__", None) == "v2",
        )
        checks.add(
            "finder_wrapper_original_present",
            callable(getattr(module.EagerRunner.load_batch, "__glm53_l17_original__", None)),
        )
        parent = sys.modules["sglang.srt.model_executor.runner"]
        spec = finder.find_spec(audit.SOURCE_MODULE, parent.__path__)
        checks.add("finder_spec_uses_pinned_loader", isinstance(spec.loader, audit._PinnedSourceLoader) if spec else False)
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(package_root))
        _purge_sglang_modules(saved)

    # Wrong eager SHA through the finder must fail before the module executes.
    _purge_sglang_modules({})
    sys.path.insert(0, str(package_root))
    wrong_finder = audit._AuditFinder("00" * 32, str(tmpdir / "finder_wrong"))
    sys.meta_path.insert(0, wrong_finder)
    try:
        ok, text = expect_failure(
            lambda: importlib.import_module(audit.SOURCE_MODULE),
            "installed source SHA mismatch before exec",
        )
        checks.add("finder_wrong_eager_sha_fails_before_exec", ok, text)
        checks.add("finder_wrong_eager_sha_module_absent", audit.SOURCE_MODULE not in sys.modules)
    finally:
        sys.meta_path.remove(wrong_finder)
        sys.path.remove(str(package_root))
        _purge_sglang_modules(saved)

    # Direct pinned loader with wrong bytes must fail before exec.
    bad_source = tmpdir / "bad_loader_source.py"
    bad_source.write_text(
        "EXECUTED = True\nclass EagerRunner:\n    def load_batch(self, batch):\n        return batch\n",
        encoding="utf-8",
    )
    loader = audit._PinnedSourceLoader(str(bad_source), INSTALLED_SHA256, str(tmpdir))
    bad_module = types.ModuleType("glm53_bad_loader_fixture")
    ok, text = expect_failure(lambda: loader.exec_module(bad_module), "installed source SHA mismatch before exec")
    checks.add("direct_loader_wrong_sha_fails", ok, text)
    checks.add("direct_loader_wrong_sha_not_executed", not hasattr(bad_module, "EXECUTED"))


def test_no_framework_imports(checks):
    tree = ast.parse(AUDIT_PATH.read_text(encoding="utf-8"), filename=str(AUDIT_PATH))
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ("torch", "torch_npu", "sglang"):
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in ("torch", "torch_npu", "sglang"):
                forbidden.append(node.module)
    checks.add("audit_imports_no_framework_or_npu", not forbidden, forbidden)
    checks.add("torch_not_imported_by_qualifier", "torch" not in sys.modules)
    checks.add("torch_npu_not_imported_by_qualifier", "torch_npu" not in sys.modules)


def validate_bundles(checks):
    checks.equal("base_stage_sha", sha256_file(BASE_STAGE), EXPECTED_BASE_STAGE_SHA256)
    base = read_json(BASE_STAGE)
    base_bundle = base["bundles"][0]
    base_files = base_bundle["files"]

    checks.equal("candidate_v1_stage_sha_immutable", sha256_file(CANDIDATE_V1_STAGE), EXPECTED_V1["candidate_stage"])
    checks.equal("candidate_v1_bundle_sha_immutable", sha256_file(CANDIDATE_V1_DIR / "bundle-manifest.json"), EXPECTED_V1["candidate_bundle"])
    checks.equal("rollback_v1_stage_sha_immutable", sha256_file(ROLLBACK_STAGE), EXPECTED_V1["rollback_stage"])
    checks.equal("rollback_v1_bundle_sha_immutable", sha256_file(ROLLBACK_DIR / "bundle-manifest.json"), EXPECTED_V1["rollback_bundle"])
    checks.equal("audit_v1_sha_immutable", sha256_file(AUDIT_V1_PATH), EXPECTED_V1["audit"])
    checks.equal("qualifier_v1_sha_immutable", sha256_file(QUALIFY_V1_PATH), EXPECTED_V1["qualifier"])
    checks.equal("result_v1_sha_immutable", sha256_file(RESULT_V1_PATH), EXPECTED_V1["result"])
    checks.equal("review_v1_sha_immutable", sha256_file(REVIEW_V1_PATH), EXPECTED_V1["review"])
    v1_result = read_json(RESULT_V1_PATH)
    checks.equal("v1_result_qualification_pass_preserved", v1_result.get("qualification", {}).get("pass"), True)
    checks.equal("v1_result_checks_total_preserved", v1_result.get("qualification", {}).get("checks_total"), 183)
    checks.add(
        "v1_review_rejected_runtime_use",
        "rejects the draft for runtime use" in REVIEW_V1_PATH.read_text(encoding="utf-8"),
    )

    candidate_stage = read_json(CANDIDATE_STAGE)
    rollback_stage = read_json(ROLLBACK_STAGE)
    candidate_manifest = read_json(CANDIDATE_DIR / "bundle-manifest.json")
    rollback_manifest = read_json(ROLLBACK_DIR / "bundle-manifest.json")
    candidate_bundle = candidate_stage["bundles"][0]
    rollback_bundle = rollback_stage["bundles"][0]
    candidate_files = candidate_bundle["files"]
    rollback_files = rollback_bundle["files"]

    checks.equal("candidate_project", candidate_stage.get("project_id"), PROJECT_ID)
    checks.equal("candidate_phase", candidate_stage.get("phase"), "L17_D_NATIVE_EAGER_CANDIDATE_V2_LOCAL_PREPARATION")
    checks.equal("candidate_rollback_reference", candidate_stage.get("rollback_reference"), str(ROLLBACK_STAGE.relative_to(PROJECT_ROOT)))
    checks.equal("candidate_audit_version", candidate_stage.get("audit_version"), "v2")
    checks.equal("candidate_audit_hook_sha", candidate_stage.get("audit_hook_sha256"), sha256_file(AUDIT_PATH))
    checks.equal("candidate_environ_source_sha", candidate_stage.get("environ_source_sha256"), INSTALLED_ENVIRON_SHA256)
    checks.equal("candidate_eager_source_sha", candidate_stage.get("eager_source_sha256"), INSTALLED_SHA256)
    checks.equal("candidate_nodes", sorted(candidate_bundle["nodes"]), D_NODES)
    checks.equal("candidate_remote", candidate_bundle["remote"], CANDIDATE_REMOTE)
    checks.equal("candidate_run_id", candidate_bundle["runtime_run_id"], CANDIDATE_RUN_ID)
    checks.equal("candidate_remote_run_root", candidate_bundle["remote_run_root"], "/data/wio/inference-foundry/glm53/runs/" + CANDIDATE_RUN_ID)
    checks.equal("candidate_image", candidate_bundle["image"], base_bundle["image"])
    checks.equal("candidate_native_source_manifest", candidate_bundle["native_source_manifest_sha256"], EXPECTED_NATIVE_SOURCE_MANIFEST)
    checks.equal("candidate_variant", candidate_bundle["variant"], "candidate")
    checks.equal("candidate_runtime_authorized_false", candidate_bundle.get("runtime_authorized"), False)
    checks.equal("candidate_review_pending", candidate_bundle.get("review_status"), "PENDING_PRIMARY_REVIEW")
    checks.add("candidate_no_P_files", not any(name in candidate_files for name in P_FILES))
    checks.add("candidate_no_optional_extension", "private_optional_extension" not in json.dumps(candidate_stage) and "private_optional_extension" not in json.dumps(candidate_manifest))
    checks.add("rollback_no_P_files", not any(name in rollback_files for name in P_FILES))
    checks.add("rollback_no_optional_extension", "private_optional_extension" not in json.dumps(rollback_stage) and "private_optional_extension" not in json.dumps(rollback_manifest))
    checks.equal("candidate_manifest_sha", sha256_file(CANDIDATE_DIR / "bundle-manifest.json"), candidate_bundle["bundle_manifest_sha256"])
    checks.equal("candidate_manifest_file_count", len(candidate_manifest["files"]), len(candidate_files))
    checks.equal("candidate_manifest_runtime_authorized_false", candidate_manifest.get("runtime_authorized"), False)
    checks.equal("candidate_manifest_review_pending", candidate_manifest.get("review_status"), "PENDING_PRIMARY_REVIEW")
    checks.equal("candidate_manifest_supersedes_v1_audit", candidate_manifest.get("supersedes", {}).get("superseded_audit_sha256"), EXPECTED_V1["audit"])
    checks.equal("candidate_manifest_supersedes_v1_stage", candidate_manifest.get("supersedes", {}).get("superseded_stage_sha256"), EXPECTED_V1["candidate_stage"])
    checks.equal("candidate_manifest_audit_sha", candidate_manifest.get("audit_hook", {}).get("sha256"), sha256_file(AUDIT_PATH))
    checks.equal("candidate_manifest_audit_version", candidate_manifest.get("audit_hook", {}).get("version"), "v2")
    checks.equal("candidate_manifest_environ_sha", candidate_manifest.get("audit_hook", {}).get("environ_source_sha256"), INSTALLED_ENVIRON_SHA256)
    checks.equal("candidate_manifest_eager_sha", candidate_manifest.get("audit_hook", {}).get("source_sha256"), INSTALLED_SHA256)
    checks.equal("candidate_sdma_manifest_unchanged", candidate_manifest["files"]["manifest.json"]["sha256"], EXPECTED_SDMA_MANIFEST)

    candidate_hashes_ok = True
    for staged_name, meta in candidate_files.items():
        local = PROJECT_ROOT / meta["local"]
        if not local.is_file() or sha256_file(local) != meta["sha256"]:
            candidate_hashes_ok = False
        if candidate_manifest["files"].get(staged_name, {}).get("sha256") != meta["sha256"]:
            candidate_hashes_ok = False
    checks.add("candidate_all_stage_and_manifest_hashes", candidate_hashes_ok)

    rollback_hashes_ok = True
    for staged_name, meta in rollback_files.items():
        local = PROJECT_ROOT / meta["local"]
        if not local.is_file() or sha256_file(local) != meta["sha256"]:
            rollback_hashes_ok = False
        if rollback_manifest["files"].get(staged_name, {}).get("sha256") != meta["sha256"]:
            rollback_hashes_ok = False
    checks.add("rollback_all_stage_and_manifest_hashes", rollback_hashes_ok)

    for node in D_NODES:
        node_name = "config/node-%d.json" % node
        sup_name = "supervisor-config/node-%d.json" % node
        provider_name = "provider-config/node-%d.json" % node
        old_node = read_json(PROJECT_ROOT / rollback_files[node_name]["local"])
        new_node = read_json(PROJECT_ROOT / candidate_files[node_name]["local"])
        checks.equal(
            "candidate_node_%d_diff_exact" % node,
            sorted(structural_diff(old_node, new_node)),
            sorted(["environment_intent." + key for key in AUDIT_ENV]),
        )
        checks.equal("candidate_node_%d_cli_exact" % node, new_node["launch_args"], old_node["launch_args"])
        checks.add(
            "candidate_node_%d_old_env_preserved" % node,
            all(new_node["environment_intent"].get(k) == v for k, v in old_node["environment_intent"].items()),
        )
        checks.add(
            "candidate_node_%d_three_env_values" % node,
            all(new_node["environment_intent"].get(k) == v for k, v in AUDIT_ENV.items()),
        )
        old_sup = read_json(PROJECT_ROOT / rollback_files[sup_name]["local"])
        new_sup = read_json(PROJECT_ROOT / candidate_files[sup_name]["local"])
        checks.equal(
            "candidate_supervisor_%d_diff_exact" % node,
            sorted(structural_diff(old_sup, new_sup)),
            sorted(["run_id"] + ["model_env." + key for key in AUDIT_ENV]),
        )
        checks.equal("candidate_supervisor_%d_run_id" % node, new_sup["run_id"], CANDIDATE_RUN_ID)
        checks.add(
            "candidate_supervisor_%d_old_env_preserved" % node,
            all(new_sup["model_env"].get(k) == v for k, v in old_sup["model_env"].items()),
        )
        checks.equal("candidate_supervisor_%d_provider_argv" % node, new_sup["provider_argv"], old_sup["provider_argv"])
        checks.equal("candidate_supervisor_%d_model_argv" % node, new_sup["model_argv"], old_sup["model_argv"])
        checks.equal(
            "candidate_provider_%d_unchanged" % node,
            sha256_file(PROJECT_ROOT / candidate_files[provider_name]["local"]),
            sha256_file(PROJECT_ROOT / base_files[provider_name]["local"]),
        )
        provider = read_json(PROJECT_ROOT / candidate_files[provider_name]["local"])
        checks.add("candidate_provider_%d_no_local_service_id" % node, "local_service_id" not in provider)
        checks.equal("candidate_provider_%d_dram" % node, provider.get("dram_size"), "160GB")
        checks.equal("candidate_provider_%d_max_dram" % node, provider.get("max_dram_size"), "160GB")
        checks.add(
            "rollback_node_%d_byte_exact" % node,
            sha256_file(PROJECT_ROOT / rollback_files[node_name]["local"])
            == sha256_file(PROJECT_ROOT / base_files[node_name]["local"]),
        )
        old_rollback_sup = read_json(PROJECT_ROOT / base_files[sup_name]["local"])
        new_rollback_sup = read_json(PROJECT_ROOT / rollback_files[sup_name]["local"])
        checks.equal(
            "rollback_supervisor_%d_diff_run_id_only" % node,
            sorted(structural_diff(old_rollback_sup, new_rollback_sup)),
            ["run_id"],
        )
        checks.equal("rollback_supervisor_%d_run_id" % node, new_rollback_sup["run_id"], ROLLBACK_RUN_ID)
        checks.add(
            "rollback_provider_%d_unchanged" % node,
            sha256_file(PROJECT_ROOT / rollback_files[provider_name]["local"])
            == sha256_file(PROJECT_ROOT / base_files[provider_name]["local"]),
        )

    common = [
        "manifest.json",
        "source/sglang/srt/disaggregation/ascend/transfer_engine.py",
        "source/sglang/srt/disaggregation/ascend/conn.py",
        "python/glm53_sglang_sdma_audit.py",
        "config/meta.conf",
        "config/local.conf",
        "config/manifest.json",
        "provider.py",
        "sdk-preflight.py",
        "entry.py",
        "supervisor.py",
        "python/glm53_deepep_metadata_batch_v1.py",
        "python/glm53_deepep_metadata_import_hook_v1.py",
        "python/qualify_image_cpu_v1.py",
        "python/glm53_C32_target_graph_fullmesh_v2.py",
    ]
    for staged_name in common:
        checks.add(
            "candidate_common_unchanged:" + staged_name,
            sha256_file(PROJECT_ROOT / candidate_files[staged_name]["local"])
            == sha256_file(PROJECT_ROOT / base_files[staged_name]["local"]),
        )
        checks.add(
            "rollback_common_unchanged:" + staged_name,
            sha256_file(PROJECT_ROOT / rollback_files[staged_name]["local"])
            == sha256_file(PROJECT_ROOT / base_files[staged_name]["local"]),
        )

    # Sitecustomize prefix, audit SHA and absence of eager imports in the append.
    old_site = OLD_SITECUSTOMIZE.read_bytes()
    candidate_site = (CANDIDATE_DIR / "python/sitecustomize.py").read_bytes()
    checks.add("candidate_sitecustomize_prefix", candidate_site.startswith(old_site))
    appended = candidate_site[len(old_site):]
    checks.add("candidate_sitecustomize_appended", len(appended) > 0)
    checks.add("candidate_sitecustomize_pins_v2_audit_sha", sha256_file(AUDIT_PATH).encode("ascii") in appended)
    checks.add("candidate_sitecustomize_imports_v2", b"glm53_C32_L17_eager_input_audit_v2" in appended)
    checks.add("candidate_sitecustomize_no_torch_import", b"import torch" not in appended)
    checks.add("candidate_sitecustomize_no_sglang_import", b"import sglang" not in appended and b"from sglang" not in appended)
    try:
        compile(candidate_site, str(CANDIDATE_DIR / "python/sitecustomize.py"), "exec")
        candidate_site_compiles = True
    except BaseException as error:
        candidate_site_compiles = False
        checks.add("candidate_sitecustomize_compile_error", False, repr(error))
    checks.add("candidate_sitecustomize_compiles", candidate_site_compiles)

    staged_audit = CANDIDATE_DIR / "python/glm53_C32_L17_eager_input_audit_v2.py"
    checks.add("staged_audit_matches_source", sha256_file(staged_audit) == sha256_file(AUDIT_PATH))
    checks.add("candidate_stage_audit_file_present", "python/glm53_C32_L17_eager_input_audit_v2.py" in candidate_files)

    # The old v1 candidate remains byte-exact, including its v1 audit hook.
    candidate_v1_stage = read_json(CANDIDATE_V1_STAGE)
    candidate_v1_files = candidate_v1_stage["bundles"][0]["files"]
    v1_hashes_ok = True
    for staged_name, meta in candidate_v1_files.items():
        local = PROJECT_ROOT / meta["local"]
        if not local.is_file() or sha256_file(local) != meta["sha256"]:
            v1_hashes_ok = False
    checks.add("candidate_v1_all_files_immutable", v1_hashes_ok)
    checks.equal("candidate_v1_file_count", len(candidate_v1_files), 23)
    checks.equal("candidate_v1_audit_sha", candidate_v1_files["python/glm53_C32_L17_eager_input_audit_v1.py"]["sha256"], EXPECTED_V1["audit"])

    rollback_v1_stage = read_json(ROLLBACK_STAGE)
    checks.equal("rollback_v1_file_count", len(rollback_v1_stage["bundles"][0]["files"]), 22)


def collect_changed_files():
    paths = []
    for directory in (CANDIDATE_DIR,):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                paths.append(path)
    for path in (
        CANDIDATE_STAGE,
        AUDIT_PATH,
        PREPARE_PATH,
        PROJECT_ROOT / "scripts/qualify_C32_L17_D_native_eager_local_v2.py",
        PROJECT_ROOT / "scripts/stage_C32_L17_D_native_eager_and_rollback_v1.py",
        RUN_DIR / "bundle-validation-v2.json",
    ):
        paths.append(path)
    seen = set()
    out = []
    for path in paths:
        path = pathlib.Path(path)
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return sorted(out, key=lambda item: item["path"])


def main():
    checks = Checks()
    checks.equal("installed_source_sha", sha256_file(INSTALLED_SOURCE), INSTALLED_SHA256)
    checks.equal("installed_environ_sha", sha256_file(INSTALLED_ENVIRON), INSTALLED_ENVIRON_SHA256)
    extraction = extract_load_batch()
    checks.equal("extracted_source_sha", extraction["source_sha256"], INSTALLED_SHA256)
    checks.add("load_batch_found", extraction["lineno"] > 0, extraction["lineno"])
    checks.add(
        "load_batch_is_actual_ast",
        extraction["lineno"] == 182 and extraction["end_lineno"] == 210,
        {"lineno": extraction["lineno"], "end_lineno": extraction["end_lineno"]},
    )

    audit_spec = importlib.util.spec_from_file_location("glm53_l17_audit_v2_under_test", AUDIT_PATH)
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    checks.equal("audit_module_source_sha_pin", audit.SOURCE_SHA256, INSTALLED_SHA256)
    checks.equal("audit_module_environ_sha_pin", audit.ENVIRON_SHA256, INSTALLED_ENVIRON_SHA256)
    checks.equal("audit_wrapper_version", audit.AUDIT_VERSION, "v2")
    checks.equal("audit_schema_version", audit.SCHEMA, "glm53.L17.eager_input_audit.v2")

    with tempfile.TemporaryDirectory(prefix="glm53-l17-v2-cpu-") as temp_name:
        tmpdir = pathlib.Path(temp_name)
        with env_settings(
            {
                "GLM53_EAGER_INPUT_AUDIT": "1",
                "GLM53_NODE": "36",
                "SGLANG_EAGER_INPUT_NO_COPY": "1",
                "GLM53_EAGER_INPUT_AUDIT_ROOT": str(tmpdir / "runtime-root"),
            }
        ):
            test_env_source_semantics(checks)
            test_native_branch(checks, extraction["code"])
            test_hook(checks, audit, extraction["code"], tmpdir / "hook")
            test_batch_key_coverage(checks, audit, extraction["code"], tmpdir / "batch-key")
            test_failures(checks, audit, extraction["code"], tmpdir / "failures")
            test_concurrency(checks, audit, extraction["code"], tmpdir / "concurrency")
            test_fork(checks, audit)
            test_import_finder(checks, audit, tmpdir / "finder")
    test_no_framework_imports(checks)
    validate_bundles(checks)

    result = {
        "project_id": PROJECT_ID,
        "result_id": "C32-L17-D-native-eager-local-preparation-v2",
        "scope": "LOCAL_STDLIB_CPU_QUALIFICATION_AND_BUNDLE_SHA_DIFF_VALIDATION_ONLY",
        "status": "LOCAL_PREPARATION_V2_COMPLETE_PRIMARY_REVIEW_PENDING",
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "lifecycle_launcher_created": False,
        "model_requests": 0,
        "HTTP_requests": 0,
        "NPU_calls": 0,
        "SDK_init_calls": 0,
        "new_profiles": 0,
        "SSH": 0,
        "network": 0,
        "remote_writes": 0,
        "Git_operations": 0,
        "supersedes": {
            "superseded_audit_version": "v1",
            "superseded_audit": str(AUDIT_V1_PATH.relative_to(PROJECT_ROOT)),
            "superseded_audit_sha256": EXPECTED_V1["audit"],
            "superseded_qualifier": str(QUALIFY_V1_PATH.relative_to(PROJECT_ROOT)),
            "superseded_qualifier_sha256": EXPECTED_V1["qualifier"],
            "superseded_result": str(RESULT_V1_PATH.relative_to(PROJECT_ROOT)),
            "superseded_result_sha256": EXPECTED_V1["result"],
            "superseded_candidate_stage": str(CANDIDATE_V1_STAGE.relative_to(PROJECT_ROOT)),
            "superseded_candidate_stage_sha256": EXPECTED_V1["candidate_stage"],
            "rejection_review": str(REVIEW_V1_PATH.relative_to(PROJECT_ROOT)),
            "rejection_review_sha256": EXPECTED_V1["review"],
            "reason": "Primary review identified a direct module-flag getter and fail-open accepted-key semantics; v1 remains unchanged and is not runtime-authorized.",
        },
        "defects_fixed": [
            "Native getter now uses sglang.srt.environ.envs.SGLANG_EAGER_INPUT_NO_COPY.get(), not a direct module attribute.",
            "Installed environ bytes are SHA-pinned at 0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c before first use; eager source remains pinned at a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5.",
            "The accepted key is inserted only after native get, exactly one original call, complete clone validation and an fsynced evidence write all succeed.",
            "The first-use path is serialized by a lock; competing first uses cannot bypass; accepted keys use a cheap pid/mode/draft/cache fastpath.",
            "Any audit failure is sticky; later calls raise without invoking the original. A normal original-call exception conservatively poisons the audit too.",
            "The failure cache stores only a bounded error string; no args, self, tensor/batch reference or traceback is retained.",
            "os.register_at_fork resets locks, seen keys and failure state in the child without holding tensor/batch references.",
            "Raw None values are preserved in metadata summaries while the None identity verdict remains accepted.",
            "The first-use key now includes the padded Python integer batch_size; sequential bs1 then bs2 produce two records and repeated bs2 adds no log.",
            "Decode-like batch sizes above the existing max4 D capture/admission policy fail closed; a larger batch is never clamped or reinterpreted as bs4.",
        ],
        "primary_refinement": {
            "path": str(REFINEMENT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(REFINEMENT_PATH),
            "incorporated": True,
            "summary": "First-use key includes (pid, draft/target role, actual forward-mode name, padded Python integer batch_size); bs1 then bs2 are distinct; repeated bs2 adds no log; decode-like batches above max4 fail closed without bs4 reinterpretation.",
            "runtime_authorization": False,
        },
        "batch_key_policy": {
            "first_use_key": "(pid, is_draft_worker, actual_forward_mode_name, padded_batch_size)",
            "batch_size_source": "forward_batch.batch_size (Python int only; never tensor contents)",
            "decode_like_modes": ["DECODE", "TARGET_VERIFY", "IDLE"],
            "decode_like_max": 4,
            "extend_mixed_max": 64,
            "clamping": False,
            "larger_batch_behavior": "fail closed; never reinterpreted as bs4",
        },
        "source": {
            "installed_eager_runner": str(INSTALLED_SOURCE.relative_to(PROJECT_ROOT)),
            "source_sha256": INSTALLED_SHA256,
            "source_bytes": extraction["source_bytes"],
            "load_batch_lineno": extraction["lineno"],
            "load_batch_end_lineno": extraction["end_lineno"],
            "load_batch_arg_names": extraction["arg_names"],
            "load_batch_vararg": extraction["vararg"],
            "load_batch_kwarg": extraction["kwarg"],
            "load_batch_source_segment_bytes": extraction["segment_bytes"],
            "load_batch_source_segment_sha256": extraction["segment_sha256"],
            "installed_environ": str(INSTALLED_ENVIRON.relative_to(PROJECT_ROOT)),
            "environ_source_sha256": INSTALLED_ENVIRON_SHA256,
            "environ_source_bytes": INSTALLED_ENVIRON.stat().st_size,
            "env_flag_lineno": 1448,
            "env_flag_assignment": "SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)",
            "env_fixture": "actual EnvField/EnvBool AST plus actual Envs flag assignment in a source-qualified minimal Envs class; descriptor __set_name__ runs naturally",
            "future_annotations_compiler_flag": True,
        },
        "audit_hook": {
            "path": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(AUDIT_PATH),
            "bytes": AUDIT_PATH.stat().st_size,
            "stdlib_only": True,
            "import_hook": "importlib.abc.MetaPathFinder + pinned loader before exec",
            "activation_env": AUDIT_ENV,
            "activation_nodes": [36, 41],
            "first_use_key": "(pid, is_draft_worker, actual_forward_mode_name, padded_batch_size)",
            "first_use_key_batch_size": "forward_batch.batch_size (Python int only; never tensor contents)",
            "decode_batch_bound": 4,
            "extend_batch_bound": 64,
            "batch_size_clamping": False,
            "native_getter": "sglang.srt.environ.envs.SGLANG_EAGER_INPUT_NO_COPY.get()",
            "source_sha256": INSTALLED_SHA256,
            "environ_source_sha256": INSTALLED_ENVIRON_SHA256,
            "seen_cache_primitives_only": True,
            "per_pid_jsonl": True,
            "sticky_failure": True,
            "first_use_serialized": True,
            "fork_reset": True,
            "wrapper_marker": "v2",
        },
        "qualification": {
            "pass": not checks.failures,
            "checks_total": len(checks.items),
            "checks_passed": sum(1 for item in checks.items if item["pass"]),
            "checks": checks.items,
            "failures": checks.failures,
        },
        "bundles": {
            "candidate_stage": str(CANDIDATE_STAGE.relative_to(PROJECT_ROOT)),
            "candidate_stage_sha256": sha256_file(CANDIDATE_STAGE) if CANDIDATE_STAGE.is_file() else None,
            "candidate_remote": CANDIDATE_REMOTE,
            "candidate_run_id": CANDIDATE_RUN_ID,
            "candidate_bundle": str(CANDIDATE_DIR.relative_to(PROJECT_ROOT)),
            "candidate_bundle_manifest_sha256": sha256_file(CANDIDATE_DIR / "bundle-manifest.json") if (CANDIDATE_DIR / "bundle-manifest.json").is_file() else None,
            "candidate_file_count": len(read_json(CANDIDATE_STAGE)["bundles"][0]["files"]) if CANDIDATE_STAGE.is_file() else 0,
            "rollback_stage": str(ROLLBACK_STAGE.relative_to(PROJECT_ROOT)),
            "rollback_stage_sha256": sha256_file(ROLLBACK_STAGE),
            "rollback_remote": ROLLBACK_REMOTE,
            "rollback_run_id": ROLLBACK_RUN_ID,
            "rollback_bundle": str(ROLLBACK_DIR.relative_to(PROJECT_ROOT)),
            "rollback_bundle_manifest_sha256": sha256_file(ROLLBACK_DIR / "bundle-manifest.json"),
            "rollback_file_count": len(read_json(ROLLBACK_STAGE)["bundles"][0]["files"]),
            "base_stage": str(BASE_STAGE.relative_to(PROJECT_ROOT)),
            "base_stage_sha256": sha256_file(BASE_STAGE),
            "removed_P_files": P_FILES,
            "allowed_candidate_diffs": {
                "node_environment_intent": ["environment_intent." + key for key in AUDIT_ENV],
                "supervisor": ["run_id"] + ["model_env." + key for key in AUDIT_ENV],
                "sitecustomize": "append_only_byte_prefix_preserved",
                "new_files": ["python/glm53_C32_L17_eager_input_audit_v2.py"],
            },
            "allowed_rollback_diffs": {"supervisor": ["run_id"], "note": "source/env/CLI byte-exact old v3"},
            "provider_configs_unchanged": True,
            "no_guessed_service_ids": True,
            "no_optional_extension": True,
            "old_sdma_manifest_sha256_unchanged": EXPECTED_SDMA_MANIFEST,
            "candidate_v1_immutable": True,
            "rollback_v1_immutable": True,
        },
        "native_env_runtime_proof": {
            "method": "native envs.SGLANG_EAGER_INPUT_NO_COPY.get() called on first audited forward",
            "actual_envs_object_fixture": True,
            "direct_module_flag_mock": False,
            "true_and_false_source_semantics_tested": True,
            "environ_source_pin_tested": True,
            "eager_source_pin_tested": True,
            "l16_initial_env_was_absent_for_all32": True,
            "candidate_sets_explicit_1": True,
            "note": "The CPU test proves the hook's native getter path and value check; it is not an NPU runtime measurement.",
        },
        "request_ledger": {
            "path": "results/model-request-ledger-through-L13-D-diagnostic-v1.json",
            "sha256": sha256_file(PROJECT_ROOT / "results/model-request-ledger-through-L13-D-diagnostic-v1.json"),
            "backend_HTTP_attempts": 6269,
            "qualified_successful_backend_inferences": 6096,
            "unchanged_by_this_task": True,
            "new_model_requests": 0,
        },
        "limitations": [
            "Synthetic FakeTensor metadata and stdlib CPU only; no torch/NPU tensor or async lifetime was exercised.",
            "No claim that the no-copy branch is safe for asynchronous Graph/draft lifetimes or that memory is saved.",
            "No model correctness, performance, SLA, latency or throughput claim; no performance certificate.",
            "The hook records first-use metadata only; the CPU test is not a runtime proof of the actual NPU service.",
            "Initial L16 /proc environment was absent; the candidate explicitly sets 1 and the audit proves get() true at first use.",
            "The v2 candidate remains local preparation; independent primary review is required before staging or deployment.",
            "Actual functional qualification must report observed bs2 draft coverage and metadata by all 32 SDK worker identities; missing coverage is UNKNOWN and must never be filled from CPU fixtures. Functional failures or missing required coverage prevent formal performance qualification.",
        ],
        "future_runtime_requirements": [
            "Report observed bs2 draft coverage and metadata for all 32 SDK worker identities in actual functional qualification.",
            "Missing bs2 draft coverage remains UNKNOWN; CPU fixtures must not be used to fill it.",
            "Any functional failure or missing required coverage prevents formal performance qualification.",
            "The bounded first-use audit overhead must be named in any performance attribution.",
        ],
        "no_performance_claim": True,
        "performance_certificate_allowed": False,
        "objective_complete": False,
    }
    result["changed_files"] = collect_changed_files()
    result["changed_files_note"] = "This result file is the only new path not self-hashed; all listed files are hashed after writing."
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "pass": result["qualification"]["pass"],
        "checks": "%d/%d" % (result["qualification"]["checks_passed"], result["qualification"]["checks_total"]),
        "failures": result["qualification"]["failures"],
        "result": str(RESULT_PATH.relative_to(PROJECT_ROOT)),
        "result_sha256": sha256_file(RESULT_PATH),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result["qualification"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
