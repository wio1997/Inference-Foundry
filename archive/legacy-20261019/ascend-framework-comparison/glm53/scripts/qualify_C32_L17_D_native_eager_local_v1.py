"""Local stdlib CPU qualification for the L17-D native eager-input candidate.

No framework, torch, NPU, network or remote access is used.  The exact installed
``EagerRunner.load_batch`` AST is extracted from the L16 source copy, compiled
with future annotations, and exercised with synthetic dataclasses, metadata-only
fake tensors and a spy eager registry.  The hook's first-use behavior and the
candidate/rollback bundle diffs are independently validated.
"""

from __future__ import annotations

import __future__
import ast
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types

PROJECT_ID = "glm53"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTALLED_SOURCE = (
    PROJECT_ROOT
    / "runs/20261019T-L16-sglang-D-eager-input-source-prereq-v1/source/node-36/"
    "sglang.srt.model_executor.runner.eager_runner.py"
)
INSTALLED_SHA256 = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
AUDIT_PATH = PROJECT_ROOT / "scripts/glm53_C32_L17_eager_input_audit_v1.py"
BASE_STAGE = PROJECT_ROOT / "config/sglang-C32-target-graph-fullmesh-stage-v3.json"
CANDIDATE_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-native-eager-stage-v1.json"
ROLLBACK_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-original-rollback-stage-v1.json"
CANDIDATE_DIR = PROJECT_ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1/candidate-v1"
ROLLBACK_DIR = PROJECT_ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1/rollback-v1"
OLD_SITECUSTOMIZE = (
    PROJECT_ROOT / "config/optimization-overlays/C32-target-graph-fullmesh-v1/python/sitecustomize_v2.py"
)
RESULT_PATH = PROJECT_ROOT / "results/C32-L17-D-native-eager-local-preparation-v1.json"
RUN_DIR = PROJECT_ROOT / "runs/20261019T-L17-local-preparation-executor-v1"
CANDIDATE_REMOTE = (
    "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-native-eager-v1-20261019"
)
ROLLBACK_REMOTE = (
    "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-original-rollback-v1-20261019"
)
CANDIDATE_RUN_ID = "20261019T-L17-D-native-eager-v1"
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


class FakeEnvBool:
    def __init__(self, value):
        self.value = value
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        return self.value


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


def make_batch(mode_name="DECODE", draft=False):
    mode = FakeMode(mode_name)
    return FakeForwardBatch(
        forward_mode=mode,
        batch_size=4,
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
    module_ast = ast.Module(body=[function], type_ignores=[])
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


def build_namespace(replace_function, env_value=True):
    environ_module = types.ModuleType("sglang.srt.environ")
    flag = FakeEnvBool(env_value)
    environ_module.SGLANG_EAGER_INPUT_NO_COPY = flag
    namespace = {"envs": environ_module, "replace": replace_function, "__name__": "glm53_l17_fixture"}
    return namespace, environ_module, flag


def install_fake_environ(environ_module):
    previous = sys.modules.get("sglang.srt.environ")
    sys.modules["sglang.srt.environ"] = environ_module
    return previous


def restore_fake_environ(previous):
    if previous is None:
        sys.modules.pop("sglang.srt.environ", None)
    else:
        sys.modules["sglang.srt.environ"] = previous


def build_counted_original(lb_code):
    namespace, environ_module, flag = build_namespace(dataclasses.replace, True)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    counted = {"calls": 0, "last_result": None}

    def load_batch(self, forward_batch, *args, **kwargs):
        counted["calls"] += 1
        counted["last_result"] = original(self, forward_batch, *args, **kwargs)
        return counted["last_result"]

    load_batch.__name__ = "load_batch"
    load_batch.__qualname__ = "EagerRunner.load_batch"
    return load_batch, counted, environ_module, flag, namespace


def build_module(load_batch):
    class FixtureEagerRunner:
        pass

    FixtureEagerRunner.load_batch = load_batch
    module = types.ModuleType("glm53_l17_fixture.eager_runner")
    module.EagerRunner = FixtureEagerRunner
    return module, FixtureEagerRunner


def make_runner(runner_class, is_draft_worker):
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


def test_native_branch(checks, lb_code):
    namespace, environ_module, flag = build_namespace(dataclasses.replace, True)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    runner = FakeRunner(False)
    batch = make_batch()
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

    # False environment must take the fixed-registry fill/extract path.
    namespace_false, environ_false, flag_false = build_namespace(dataclasses.replace, False)
    exec(lb_code, namespace_false)
    original_false = namespace_false["load_batch"]
    runner_false = FakeRunner(False)
    result_false = original_false(runner_false, batch)
    checks.equal("false_env_fill_calls", runner_false._eager_registry.fill_calls, 1)
    checks.equal("false_env_extract_calls", runner_false._eager_registry.extract_calls, 1)
    checks.equal("false_env_result", result_false, "EXTRACTED_FROM_STATIC_REGISTRY")


def test_hook(checks, audit, lb_code, tmpdir):
    hook_text = AUDIT_PATH.read_text(encoding="utf-8")
    checks.add("audit_module_no_framework_import_at_top", "import torch" not in hook_text and "import sglang" not in hook_text and "from sglang" not in hook_text)
    audit._reset_for_tests()
    load_batch, counted, environ_module, flag, _ = build_counted_original(lb_code)
    module, runner_class = build_module(load_batch)
    previous_environ = install_fake_environ(environ_module)
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir))
    checks.add("wrapper_marker", getattr(runner_class.load_batch, "__glm53_l17_eager_input_audit__", None) == "v1")
    checks.add("wrapper_original_marker", getattr(runner_class.load_batch, "__glm53_l17_original__", None) is load_batch)
    checks.add("wrapper_installed", runner_class.load_batch is not load_batch)

    runner = make_runner(runner_class, False)
    batch = make_batch("DECODE")
    result = runner.load_batch(batch)
    checks.equal("wrapper_one_original_call", counted["calls"], 1)
    checks.add("wrapper_returns_original_outcome", result is counted["last_result"])
    checks.add("wrapper_result_distinct", result is not batch)
    checks.equal("wrapper_env_get_first_use", flag.get_calls, 2)  # audit + native load_batch

    pid_file = tmpdir / ("pid-%d.jsonl" % os.getpid())
    checks.add("first_use_jsonl_created", pid_file.is_file(), str(pid_file))
    lines = rewrite_lines(pid_file)
    checks.equal("first_use_line_count", len(lines), 1)
    record = lines[0] if lines else {}
    checks.equal("record_schema", record.get("schema"), "glm53.L17.eager_input_audit.v1")
    checks.equal("record_source_sha", record.get("source_sha256"), INSTALLED_SHA256)
    checks.equal("record_node", record.get("node"), 36)
    checks.equal("record_pid", record.get("pid"), os.getpid())
    checks.equal("record_forward_mode", record.get("forward_mode"), "DECODE")
    checks.equal("record_is_draft", record.get("is_draft_worker"), False)
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
    checks.equal("record_seq_lens_cpu_none", allowlisted.get("seq_lens_cpu"), {"none": True})
    checks.equal("record_spec_info_dataclass", allowlisted.get("spec_info", {}).get("dataclass"), "FakeSpecInfo")
    checks.equal(
        "record_spec_info_nested_shape",
        allowlisted.get("spec_info", {}).get("fields", {}).get("draft_input_ids", {}).get("shape"),
        [4, 6],
    )
    checks.add("record_spec_info_value_not_read", "values" not in json.dumps(record), None)

    size_after_first = pid_file.stat().st_size
    gets_after_first = flag.get_calls
    result2 = runner.load_batch(batch)
    checks.equal("same_key_second_original_call", counted["calls"], 2)
    checks.add("same_key_second_outcome", result2 is counted["last_result"])
    checks.equal("same_key_native_only_get_delta", flag.get_calls - gets_after_first, 1)
    checks.equal("same_key_no_new_line", len(rewrite_lines(pid_file)), 1)
    checks.equal("same_key_no_file_growth", pid_file.stat().st_size, size_after_first)

    batch_verify = make_batch("TARGET_VERIFY")
    runner.load_batch(batch_verify)
    checks.equal("new_mode_line_count", len(rewrite_lines(pid_file)), 2)
    checks.equal("new_mode_env_get", flag.get_calls, 5)  # + audit + native

    runner_draft = make_runner(runner_class, True)
    runner_draft.load_batch(make_batch("DECODE"))
    checks.equal("new_is_draft_line_count", len(rewrite_lines(pid_file)), 3)
    checks.equal("new_is_draft_env_get", flag.get_calls, 7)

    original_pid = audit._current_pid
    audit._current_pid = lambda: 424242
    try:
        runner.load_batch(make_batch("DECODE"))
    finally:
        audit._current_pid = original_pid
    fork_file = tmpdir / "pid-424242.jsonl"
    checks.add("fork_pid_dedicated_file", fork_file.is_file(), str(fork_file))
    fork_lines = rewrite_lines(fork_file)
    checks.equal("fork_pid_line_count", len(fork_lines), 1)
    checks.equal("fork_pid_record_pid", fork_lines[0].get("pid") if fork_lines else None, 424242)
    checks.equal("fork_pid_env_get", flag.get_calls, 9)

    seen_types_ok = True
    for key in audit._SEEN:
        if not (type(key) is tuple and len(key) == 3 and type(key[0]) is int and type(key[1]) is bool and type(key[2]) is str):
            seen_types_ok = False
    checks.add("seen_keys_primitives_only", seen_types_ok, None)
    tensor_refs = [name for name, value in vars(audit).items() if isinstance(value, FakeTensor)]
    checks.add("no_tensor_refs_in_audit_globals", not tensor_refs, tensor_refs)
    serialized = json.dumps(record)
    checks.add("record_has_no_tensor_literal", "FakeTensor" not in serialized, None)
    checks.add("record_field_names_hashed", isinstance(record.get("field_names_sha256"), str) and len(record["field_names_sha256"]) == 64)

    # False native env fails closed before the original is invoked.
    flag.value = False
    counted["calls"] = 0
    ok, text = expect_failure(lambda: runner.load_batch(make_batch("MIXED")), "native SGLANG_EAGER_INPUT_NO_COPY.get() is not True")
    checks.add("false_env_fails_closed", ok, text)
    checks.equal("false_env_original_not_called", counted["calls"], 0)
    flag.value = True

    restore_fake_environ(previous_environ)


def test_failures(checks, audit, lb_code, tmpdir):
    tmpdir.mkdir(parents=True, exist_ok=True)
    # Wrong source SHA in patch.
    audit._reset_for_tests()
    audit._ACTIVE_SOURCE_SHA = INSTALLED_SHA256
    load_batch, _, environ_module, _, _ = build_counted_original(lb_code)
    module, runner_class = build_module(load_batch)
    ok, text = expect_failure(lambda: audit._patch_module(module, "00" * 32, str(tmpdir)), "patch source SHA mismatch")
    checks.add("wrong_patch_source_fails", ok, text)
    checks.add("wrong_patch_source_not_installed", runner_class.load_batch is load_batch)

    # Loader must verify before executing source.
    bad_source = tmpdir / "bad_source.py"
    bad_source.write_text("EXECUTED = True\nclass EagerRunner:\n    def load_batch(self, batch):\n        return batch\n", encoding="utf-8")
    loader = audit._PinnedSourceLoader(str(bad_source), INSTALLED_SHA256, str(tmpdir))
    bad_module = types.ModuleType("glm53_bad_fixture")
    ok, text = expect_failure(lambda: loader.exec_module(bad_module), "installed source SHA mismatch before exec")
    checks.add("wrong_loader_source_fails", ok, text)
    checks.add("wrong_loader_source_not_executed", not hasattr(bad_module, "EXECUTED"))

    # Mutated non-allowlisted static_metadata must fail whole-dataclass validation.
    audit._reset_for_tests()
    previous_environ = install_fake_environ(build_namespace(dataclasses.replace, True)[1])

    def mutating_static(batch):
        clone = dataclasses.replace(batch)
        clone.static_metadata = dataclasses.replace(batch.static_metadata, planned_bs=999)
        return clone

    namespace, environ_module, flag = build_namespace(mutating_static, True)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    module, runner_class = build_module(original)
    runner = make_runner(runner_class, False)
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "mutated_static"))
    ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "dataclass field changed across replace: static_metadata")
    checks.add("mutated_static_metadata_fails", ok, text)

    # Mutated allowlisted field.
    audit._reset_for_tests()

    def mutating_input(batch):
        clone = dataclasses.replace(batch)
        clone.input_ids = FakeTensor((4, 4), (4, 1))
        return clone

    namespace, environ_module, flag = build_namespace(mutating_input, True)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    module, runner_class = build_module(original)
    runner = make_runner(runner_class, False)
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "mutated_input"))
    ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "dataclass field changed across replace: input_ids")
    checks.add("mutated_input_ids_fails", ok, text)

    # Source clone identity failure.
    audit._reset_for_tests()
    namespace, environ_module, flag = build_namespace(lambda batch: batch, True)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    module, runner_class = build_module(original)
    runner = make_runner(runner_class, False)
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "identity"))
    ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "same ForwardBatch object")
    checks.add("source_clone_identity_fails", ok, text)

    # First-use write failure fails closed.
    audit._reset_for_tests()
    root_file = tmpdir / "not_a_directory"
    root_file.write_text("x", encoding="utf-8")
    namespace, environ_module, flag = build_namespace(dataclasses.replace, True)
    exec(lb_code, namespace)
    original = namespace["load_batch"]
    module, runner_class = build_module(original)
    runner = make_runner(runner_class, False)
    audit._patch_module(module, INSTALLED_SHA256, str(root_file))
    ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "cannot create audit root")
    checks.add("write_failure_fails_closed", ok, text)

    # Original failure releases the first-use key.
    audit._reset_for_tests()
    namespace, environ_module, flag = build_namespace(dataclasses.replace, True)
    exec(lb_code, namespace)
    original = namespace["load_batch"]

    def raising_load_batch(self, batch, *args, **kwargs):
        raise RuntimeError("synthetic original failure")

    module, runner_class = build_module(raising_load_batch)
    audit._patch_module(module, INSTALLED_SHA256, str(tmpdir / "original_failure"))
    runner = make_runner(runner_class, False)
    ok, text = expect_failure(lambda: runner.load_batch(make_batch("DECODE")), "synthetic original failure")
    checks.add("original_failure_propagates", ok, text)
    key = (os.getpid(), False, "DECODE")
    checks.add("original_failure_releases_key", key not in audit._SEEN, list(audit._SEEN))

    restore_fake_environ(previous_environ)


def validate_bundles(checks):
    base = read_json(BASE_STAGE)
    base_files = base["bundles"][0]["files"]
    candidate_stage = read_json(CANDIDATE_STAGE)
    rollback_stage = read_json(ROLLBACK_STAGE)
    for label, stage, remote, run_id, bundle_dir in (
        ("candidate", candidate_stage, CANDIDATE_REMOTE, CANDIDATE_RUN_ID, CANDIDATE_DIR),
        ("rollback", rollback_stage, ROLLBACK_REMOTE, ROLLBACK_RUN_ID, ROLLBACK_DIR),
    ):
        checks.equal(label + "_project", stage.get("project_id"), PROJECT_ID)
        bundle = stage["bundles"][0]
        checks.equal(label + "_nodes", sorted(bundle["nodes"]), D_NODES)
        checks.equal(label + "_remote", bundle["remote"], remote)
        checks.equal(label + "_run_id", bundle["runtime_run_id"], run_id)
        checks.equal(label + "_image", bundle["image"], base["bundles"][0]["image"])
        checks.equal(
            label + "_native_source_manifest",
            bundle["native_source_manifest_sha256"],
            base["bundles"][0]["native_source_manifest_sha256"],
        )
        stage_files = bundle["files"]
        file_hashes_ok = True
        for staged_name, meta in stage_files.items():
            actual = sha256_file(PROJECT_ROOT / meta["local"])
            if actual != meta["sha256"]:
                file_hashes_ok = False
        checks.add(label + "_all_stage_hashes", file_hashes_ok)
        checks.add(label + "_no_P_files", not any(name in stage_files for name in P_FILES))
        checks.add(label + "_has_D_provider", all("provider-config/node-%d.json" % node in stage_files for node in D_NODES))
        checks.add(label + "_has_D_supervisor", all("supervisor-config/node-%d.json" % node in stage_files for node in D_NODES))
        manifest = read_json(bundle_dir / "bundle-manifest.json")
        checks.equal(label + "_bundle_manifest_sha", sha256_file(bundle_dir / "bundle-manifest.json"), bundle["bundle_manifest_sha256"])
        checks.equal(label + "_bundle_manifest_files", len(manifest["files"]), len(stage_files))
        checks.equal(label + "_runtime_authorized_false", manifest.get("runtime_authorized"), False)
        checks.equal(label + "_review_pending", manifest.get("review_status"), "PENDING_PRIMARY_REVIEW")
        checks.add(label + "_removed_P_files_recorded", sorted(manifest.get("removed_P_files", [])) == sorted(P_FILES))
        checks.equal(label + "_provenance_base_stage", manifest["provenance"]["base_stage"], "config/sglang-C32-target-graph-fullmesh-stage-v3.json")

    candidate_bundle = candidate_stage["bundles"][0]
    rollback_bundle = rollback_stage["bundles"][0]
    for node in D_NODES:
        node_name = "config/node-%d.json" % node
        sup_name = "supervisor-config/node-%d.json" % node
        old_node = read_json(PROJECT_ROOT / base_files[node_name]["local"])
        new_node = read_json(PROJECT_ROOT / candidate_bundle["files"][node_name]["local"])
        checks.equal(
            "candidate_node_%d_diff" % node,
            sorted(structural_diff(old_node, new_node)),
            sorted(["environment_intent." + key for key in AUDIT_ENV]),
        )
        checks.equal("candidate_node_%d_cli" % node, new_node["launch_args"], old_node["launch_args"])
        checks.add(
            "candidate_node_%d_old_env_preserved" % node,
            all(new_node["environment_intent"].get(k) == v for k, v in old_node["environment_intent"].items()),
        )
        checks.add(
            "candidate_node_%d_three_env_values" % node,
            all(new_node["environment_intent"].get(k) == v for k, v in AUDIT_ENV.items()),
        )
        old_sup = read_json(PROJECT_ROOT / base_files[sup_name]["local"])
        new_sup = read_json(PROJECT_ROOT / candidate_bundle["files"][sup_name]["local"])
        checks.equal(
            "candidate_supervisor_%d_diff" % node,
            sorted(structural_diff(old_sup, new_sup)),
            sorted(["run_id"] + ["model_env." + key for key in AUDIT_ENV]),
        )
        checks.add(
            "candidate_supervisor_%d_old_env_preserved" % node,
            all(new_sup["model_env"].get(k) == v for k, v in old_sup["model_env"].items()),
        )
        checks.equal("candidate_supervisor_%d_run_id" % node, new_sup["run_id"], CANDIDATE_RUN_ID)
        checks.add(
            "candidate_provider_%d_unchanged" % node,
            sha256_file(PROJECT_ROOT / candidate_bundle["files"]["provider-config/node-%d.json" % node]["local"])
            == sha256_file(PROJECT_ROOT / base_files["provider-config/node-%d.json" % node]["local"]),
        )

        rollback_node = PROJECT_ROOT / rollback_bundle["files"][node_name]["local"]
        checks.add(
            "rollback_node_%d_byte_exact" % node,
            sha256_file(rollback_node) == sha256_file(PROJECT_ROOT / base_files[node_name]["local"]),
        )
        old_rollback_sup = read_json(PROJECT_ROOT / base_files[sup_name]["local"])
        new_rollback_sup = read_json(PROJECT_ROOT / rollback_bundle["files"][sup_name]["local"])
        checks.equal(
            "rollback_supervisor_%d_diff" % node,
            sorted(structural_diff(old_rollback_sup, new_rollback_sup)),
            ["run_id"],
        )
        checks.equal("rollback_supervisor_%d_run_id" % node, new_rollback_sup["run_id"], ROLLBACK_RUN_ID)
        checks.add(
            "rollback_provider_%d_unchanged" % node,
            sha256_file(PROJECT_ROOT / rollback_bundle["files"]["provider-config/node-%d.json" % node]["local"])
            == sha256_file(PROJECT_ROOT / base_files["provider-config/node-%d.json" % node]["local"]),
        )
        provider = read_json(PROJECT_ROOT / candidate_bundle["files"]["provider-config/node-%d.json" % node]["local"])
        checks.add("provider_%d_no_local_service_id" % node, "local_service_id" not in provider)
        checks.equal("provider_%d_dram" % node, provider.get("dram_size"), "160GB")
        checks.equal("provider_%d_max_dram" % node, provider.get("max_dram_size"), "160GB")

    # Common artifacts must remain byte-exact in both bundles.
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
            sha256_file(PROJECT_ROOT / candidate_bundle["files"][staged_name]["local"])
            == sha256_file(PROJECT_ROOT / base_files[staged_name]["local"]),
        )
        checks.add(
            "rollback_common_unchanged:" + staged_name,
            sha256_file(PROJECT_ROOT / rollback_bundle["files"][staged_name]["local"])
            == sha256_file(PROJECT_ROOT / base_files[staged_name]["local"]),
        )

    # Sitecustomize prefix, pin and absence of eager imports in the appended block.
    old_site = OLD_SITECUSTOMIZE.read_bytes()
    candidate_site = (CANDIDATE_DIR / "python/sitecustomize.py").read_bytes()
    rollback_site = (ROLLBACK_DIR / "python/sitecustomize.py").read_bytes()
    checks.add("candidate_sitecustomize_prefix", candidate_site.startswith(old_site))
    checks.add("rollback_sitecustomize_exact", rollback_site == old_site)
    appended = candidate_site[len(old_site):]
    checks.add("candidate_sitecustomize_appended", len(appended) > 0)
    checks.add("candidate_sitecustomize_pins_audit_sha", sha256_file(AUDIT_PATH).encode("ascii") in appended)
    checks.add("candidate_sitecustomize_no_torch_import", b"import torch" not in appended)
    checks.add("candidate_sitecustomize_no_sglang_import", b"import sglang" not in appended and b"from sglang" not in appended)
    try:
        compile(candidate_site, str(CANDIDATE_DIR / "python/sitecustomize.py"), "exec")
        candidate_site_compiles = True
    except BaseException as error:
        candidate_site_compiles = False
        checks.add("candidate_sitecustomize_compile_error", False, repr(error))
    checks.add("candidate_sitecustomize_compiles", candidate_site_compiles)

    staged_audit = CANDIDATE_DIR / "python/glm53_C32_L17_eager_input_audit_v1.py"
    checks.add("staged_audit_matches_source", sha256_file(staged_audit) == sha256_file(AUDIT_PATH))


def collect_changed_files(extra_paths):
    paths = []
    for directory in (CANDIDATE_DIR, ROLLBACK_DIR):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                paths.append(path)
    for path in (
        CANDIDATE_STAGE,
        ROLLBACK_STAGE,
        AUDIT_PATH,
        PROJECT_ROOT / "scripts/prepare_C32_L17_D_native_eager_bundles_v1.py",
        PROJECT_ROOT / "scripts/qualify_C32_L17_D_native_eager_local_v1.py",
        PROJECT_ROOT / "diagnostics/model-profile/loops/L16-D-native-eager-live-inputs-prerequisite-v1.json",
        PROJECT_ROOT / "diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json",
        PROJECT_ROOT / "STATE.json",
        PROJECT_ROOT / "STATUS_BRIEF.json",
        PROJECT_ROOT / "RECOVERY_LATEST.md",
        RUN_DIR / "bundle-validation.json",
    ):
        paths.append(path)
    paths.extend(extra_paths)
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
    extraction = extract_load_batch()
    checks.equal("extracted_source_sha", extraction["source_sha256"], INSTALLED_SHA256)
    checks.add("load_batch_found", extraction["lineno"] > 0, extraction["lineno"])
    checks.add("load_batch_is_actual_ast", extraction["lineno"] == 182 and extraction["end_lineno"] == 210, {
        "lineno": extraction["lineno"],
        "end_lineno": extraction["end_lineno"],
    })

    audit_spec = importlib.util.spec_from_file_location("glm53_l17_audit_under_test", AUDIT_PATH)
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    checks.equal("audit_module_source_sha_pin", audit.SOURCE_SHA256, INSTALLED_SHA256)

    with tempfile.TemporaryDirectory(prefix="glm53-l17-cpu-") as temp_name:
        tmpdir = pathlib.Path(temp_name)
        with env_settings(
            {
                "GLM53_EAGER_INPUT_AUDIT": "1",
                "GLM53_NODE": "36",
                "SGLANG_EAGER_INPUT_NO_COPY": "1",
                "GLM53_EAGER_INPUT_AUDIT_ROOT": str(tmpdir / "runtime-root"),
            }
        ):
            test_native_branch(checks, extraction["code"])
            test_hook(checks, audit, extraction["code"], tmpdir / "hook")
            test_failures(checks, audit, extraction["code"], tmpdir / "failures")

    validate_bundles(checks)

    result = {
        "project_id": PROJECT_ID,
        "result_id": "C32-L17-D-native-eager-local-preparation-v1",
        "scope": "LOCAL_STDLIB_CPU_QUALIFICATION_AND_BUNDLE_SHA_DIFF_VALIDATION_ONLY",
        "status": "LOCAL_PREPARATION_COMPLETE_PRIMARY_REVIEW_PENDING",
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
            "future_annotations_compiler_flag": True,
            "forward_batch_dataclass_validation": "dataclasses.fields(forward_batch) and dataclasses.fields(result); every field name and identity/immutable verdict is checked dynamically",
            "static_metadata_note": "The L16 fixed source set did not include forward_batch_info.py. The hook covers a static_metadata field if the actual dataclass has one; the CPU fixture adds a non-allowlisted FakeStaticMetadata field to prove whole-dataclass validation.",
        },
        "audit_hook": {
            "path": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(AUDIT_PATH),
            "bytes": AUDIT_PATH.stat().st_size,
            "stdlib_only": True,
            "import_hook": "importlib.abc.MetaPathFinder + pinned loader before exec",
            "activation_env": AUDIT_ENV,
            "activation_nodes": [36, 41],
            "first_use_key": "(pid, is_draft_worker, forward_mode)",
            "seen_cache_primitives_only": True,
            "per_pid_jsonl": True,
            "wrapper_marker": "v1",
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
            "candidate_stage_sha256": sha256_file(CANDIDATE_STAGE),
            "candidate_remote": CANDIDATE_REMOTE,
            "candidate_run_id": CANDIDATE_RUN_ID,
            "candidate_bundle": str(CANDIDATE_DIR.relative_to(PROJECT_ROOT)),
            "candidate_bundle_manifest_sha256": sha256_file(CANDIDATE_DIR / "bundle-manifest.json"),
            "candidate_file_count": len(read_json(CANDIDATE_STAGE)["bundles"][0]["files"]),
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
                "new_files": ["python/glm53_C32_L17_eager_input_audit_v1.py"],
            },
            "allowed_rollback_diffs": {"supervisor": ["run_id"], "note": "source/env/CLI byte-exact old v3"},
        },
        "native_env_runtime_proof": {
            "method": "native envs.SGLANG_EAGER_INPUT_NO_COPY.get() called on first audited forward",
            "fake_get_true_path_tested": True,
            "l16_initial_env_was_absent_for_all32": True,
            "candidate_sets_explicit_1": True,
            "note": "The CPU test proves the hook's get() path and value check; it is not an NPU runtime measurement.",
        },
        "limitations": [
            "Synthetic FakeTensor metadata and stdlib CPU only; no torch/NPU tensor or async lifetime was exercised.",
            "No claim that the no-copy branch is safe for asynchronous Graph/draft lifetimes or that memory is saved.",
            "No model correctness, performance, SLA, latency or throughput claim; no performance certificate.",
            "The hook records first-use metadata only; the CPU test is not a runtime proof of the actual NPU service.",
            "Initial L16 /proc environment was absent; the candidate explicitly sets 1 and the audit proves get() true at first use.",
            "No actual ForwardBatch source was re-read and no literal static_metadata field is asserted for the installed dataclass; coverage is dynamic over dataclasses.fields plus the CPU fixture.",
        ],
        "no_performance_claim": True,
        "performance_certificate_allowed": False,
        "objective_complete": False,
    }
    changed_paths = []
    result["changed_files"] = collect_changed_files(changed_paths)
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
