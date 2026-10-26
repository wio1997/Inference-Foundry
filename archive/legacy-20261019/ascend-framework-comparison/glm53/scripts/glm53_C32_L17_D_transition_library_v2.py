"""DRAFT_PENDING_PRIMARY_REVIEW: D36/41-only lifecycle host-plan library v2.

This module renders host code and performs local hashing/AST work only. Import and
`--write-plan` make zero SSH, HTTP, NPU, SDK or model calls. The remote actions are
owned by scripts/run_C32_L17_controlled_D_transition_and_pool_v2.py after a separate
primary review freezes every plan SHA.

Fixed v2 scope:
- stop only the exact retained D36/D41 containers (old fullmesh-v3 run);
- keep P34/P35, the shared Meta container and vLLM untouched; Meta Docker work is
  rendered only for node34, the sole host that runs the Meta container;
- stage-time candidate-v2 and rollback-v1 bundles only, already staged on D36/41;
- prepare new D36/D41 run roots and candidate pools, no Meta start, no model start;
- P is always the explicit L12C native candidate; only D is selected as
  candidate-v2 or original-rollback-v1;
- candidate/rollback model-start, observe and cleanup plans stay parameterized
  until the independent local pool-capacity audit pins the actual new CIDs.

v1 remains unchanged at its frozen SHA. No lifecycle launcher may use a guessed
service ID or an old D container ID.
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

TRANSITION_RID = "20261019T-L17-D-transition-and-pool-v2"
STOP_RID = "20261019T-L17-D-stop-current-v2"
CLEANUP_RID = "20261019T-L17-D-cleanup-v2"
RUN = ROOT / "runs" / TRANSITION_RID
STOP_RUN = ROOT / "runs" / STOP_RID
CAPACITY_OUT = "results/sglang-C32-L17-selected-D-pool-capacity-v2.json"
REVIEW = "results/C32-L17-D-transition-actual-review-v2.json"
LOCAL_RESULT = "results/C32-L17-D-transition-local-preparation-v2.json"

META_NODE = 34
P_SELECTED_VARIANT = "candidate"
D_SELECTED_VARIANTS = ("candidate", "rollback")
D_ONLY_SCOPE = "D_ONLY_NOT_P"
META_AFTER_WAIT_BOUND_SECONDS = 45
META_AFTER_WAIT_STEP_SECONDS = 3
META_AFTER_WAIT_REASON = (
    "bounded CPU-only wait for the next native Meta 30s metrics summary after the "
    "fresh D mounts; no model/SDK/NPU query"
)
META_METRICS_SOURCE_PREFIX = (
    "[MMC mmc_meta_service.cpp:539 operator()] Metrics summary: "
)
META_MOUNT_SOURCE_PREFIX = "[MMC mmc_global_allocator.h:170 Mount] "
META_TOTAL_BYTES = 609885356032

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
MIXED_IDENTITY_V1 = "scripts/glm53_C32_L17_mixed_model_identity_v1.py"
MIXED_IDENTITY_V1_SHA = "ec8a167da59504dcfcd9dec1347960e93fbab2764d9545f96a6f80768aff1526"

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
NATIVE_PREFLIGHT_BASIS = "D-only derivation of BASE_POOL.native_preflight: P buffer_only/9GB, optional-extension maps and guard.prepare_sdk are removed; D NoCopy/audit marker plus local.conf/meta.conf checks are added. Framework import happens only after provider READY; the actual installed EagerRunner module is imported without constructing a runner/model/engine or allocating an NPU tensor, its eager/environ source SHAs are re-hashed, envs.SGLANG_EAGER_INPUT_NO_COPY.get() is an actual bool, and EagerRunner.load_batch.__glm53_l17_eager_input_audit__ is the actual v2 marker for candidate / absent for rollback. Preflight reports NPU_compute_calls=0 and SDK_init_calls=0 separately from the two provider SETUP/INIT sequences."

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
    MIXED_IDENTITY_V1: MIXED_IDENTITY_V1_SHA,
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
    "review_version": "v2",
    "runtime_authorized": True,
    "model_start_authorized": False,
    "host_plans_sha256": {},
    "capacity_output": CAPACITY_OUT,
    "stage_receipt_sha256": "sha256 of " + STAGE_RECEIPT,
    "selected_P_variant": P_SELECTED_VARIANT,
    "selected_D_variant": "candidate",
    "note": "Model start/observe/cleanup require a later review rendered from the actual capacity CIDs.",
}

CANDIDATE_STAGE_SHA = STAGES["candidate"]["stage_sha256"]
ROLLBACK_STAGE_SHA = STAGES["rollback"]["stage_sha256"]
EXPECTED_PROVIDER_SDK_SETUP_INIT_SEQUENCES = 2


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def p_selection(capacity):
    """P is always the accepted L12C native candidate, never a D selector."""
    explicit = capacity.get("selected_P_variant")
    legacy = capacity.get("selected_variant")
    scope = capacity.get("selected_variant_scope") or capacity.get("classification") or ""
    if explicit is None:
        assert "P_ONLY" in scope, (
            "legacy selected_variant is ambiguous for P; explicit selected_P_variant "
            "or a P_ONLY classification is required")
        explicit = legacy
    assert explicit == P_SELECTED_VARIANT, explicit
    if legacy is not None and legacy != P_SELECTED_VARIANT:
        # A mixed capacity may carry a D-scoped legacy selected_variant.
        assert scope in (D_ONLY_SCOPE, "D_ONLY"), (
            "legacy selected_variant %r is neither the P candidate nor explicitly D-scoped" % legacy)
    return explicit


def d_selection(capacity):
    """D is selected independently after P; legacy fields must be D-scoped."""
    p = capacity.get("selected_P_variant")
    d = capacity.get("selected_D_variant")
    legacy = capacity.get("selected_variant")
    scope = capacity.get("selected_variant_scope")
    assert p == P_SELECTED_VARIANT, p
    assert d in D_SELECTED_VARIANTS, d
    if legacy is not None:
        assert legacy == d, (legacy, d)
        assert scope in (D_ONLY_SCOPE, "D_ONLY"), scope
    return d


def selection_record(role, capacity):
    if role == "P":
        return {"selected_P_variant": p_selection(capacity)}
    d_selection(capacity)
    return {
        "selected_P_variant": capacity["selected_P_variant"],
        "selected_D_variant": capacity["selected_D_variant"],
        "selected_variant": capacity.get("selected_variant", capacity["selected_D_variant"]),
        "selected_variant_scope": capacity.get("selected_variant_scope", D_ONLY_SCOPE),
    }


# Pure stdlib helpers shared by the local qualification fixtures. The rendered
# host code keeps standalone equivalents because it cannot import this module.

def append_only_ok(before, after):
    """True iff `after` retains every byte of `before` at the same offset."""
    return len(after) >= len(before) and after[:len(before)] == before


def provider_closure(provider, supervisor):
    events = provider.get("events") or []
    last = events[-1] if events else {}
    components = {
        "provider_CLOSE_COMPLETE_rc0": last.get("event") == "CLOSE_COMPLETE" and last.get("rc") == 0,
        "provider_phase_CLOSE_COMPLETE": provider.get("phase") == "CLOSE_COMPLETE",
        "supervisor_STOPPED": supervisor.get("phase") == "STOPPED",
        "supervisor_provider_exit_code0": supervisor.get("provider_exit_code") == 0,
    }
    return {"components": components, "pass": all(components.values())}


def parse_server_info(raw):
    """Parse the actual native /get_server_info top-level object."""
    info = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    assert isinstance(info, dict), "native /get_server_info is not a top-level object"
    for key in ("tp_size", "ep_size", "dp_size", "speculative_num_steps",
                "speculative_num_draft_tokens", "kv_cache_dtype",
                "disable_overlap_schedule", "max_total_num_tokens",
                "cuda_graph_config"):
        assert key in info, "native /get_server_info missing %s" % key
    decode = (info.get("cuda_graph_config") or {}).get("decode") or {}
    ok = (
        info["tp_size"] == 32 and info["ep_size"] == 32 and info["dp_size"] == 16
        and info["speculative_num_steps"] == 5 and info["speculative_num_draft_tokens"] == 6
        and info["kv_cache_dtype"] == "bfloat16"
        and info["disable_overlap_schedule"] is True
        and type(info["max_total_num_tokens"]) is int
        and info["max_total_num_tokens"] >= CAPACITY_FLOOR_TOKENS
        and decode.get("backend") == "full"
        and decode.get("bs") == [1, 2, 3, 4]
    )
    return {
        "top_level_object": True,
        "keys": sorted(info),
        "tp_size": info["tp_size"],
        "ep_size": info["ep_size"],
        "dp_size": info["dp_size"],
        "MTP_steps": info["speculative_num_steps"],
        "MTP_draft_tokens": info["speculative_num_draft_tokens"],
        "kv_cache_dtype": info["kv_cache_dtype"],
        "disable_overlap_schedule": info["disable_overlap_schedule"],
        "max_total_num_tokens": info["max_total_num_tokens"],
        "decode_graph": {"backend": decode.get("backend"), "bs": decode.get("bs")},
        "ok": ok,
        "scope": "role-global native primary response; distributed registrations",
    }


_METRICS_PREFIX_LOCAL = "[MMC mmc_meta_service.cpp:539 operator()] Metrics summary: "


def parse_meta_metrics(line):
    """Parse the native aggregate Metrics summary fields; no per-rank attribution."""
    if _METRICS_PREFIX_LOCAL not in line:
        return None
    stamp = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", line.strip())
    assert stamp, "native Meta metrics line without timestamp"
    dram = re.search(r"\bdram_used=(\d+)/(\d+)", line)
    assert dram, "native Meta metrics line without dram_used=used/total"
    values = {}
    for name in ("mount_req", "mount_success", "mount_fail",
                 "unmount_req", "unmount_success", "unmount_fail"):
        match = re.search(r"(?:^| )" + name + r"=(\d+)", line)
        assert match, "native Meta metrics line without %s" % name
        values[name] = int(match.group(1))
    return {
        "timestamp": stamp.group(1),
        "dram_used": int(dram.group(1)),
        "dram_total": int(dram.group(2)),
        "rank": None,
        "per_rank_attribution": False,
        "scope": "native aggregate counters; no per-rank record",
        **values,
    }


def metrics_freshness(before_samples, after_samples, mount_timestamp):
    """Strict freshness: use only an after sample at/after the fresh mount."""
    before = max(before_samples, key=lambda item: item["timestamp"]) if before_samples else None
    fresh = [item for item in after_samples if item["timestamp"] >= mount_timestamp]
    sample = max(fresh, key=lambda item: item["timestamp"]) if fresh else None
    return {"last_before": before, "fresh_after": sample, "fresh": sample is not None}


def classify_bs2_draft(record):
    """Actual draft DECODE bs2 coverage; target-only/idle/extend is not equivalent."""
    if record.get("is_draft_worker") is not True:
        return {"eligible": False, "reason": "not_draft_worker"}
    if record.get("forward_mode") != "DECODE":
        return {"eligible": False, "reason": "forward_mode", "actual": record.get("forward_mode")}
    if record.get("batch_size") != 2:
        return {"eligible": False, "reason": "batch_size", "actual": record.get("batch_size")}
    out = (record.get("allowlisted") or {}).get("out_cache_loc")
    if not (isinstance(out, dict) and out.get("tensor") is True
            and isinstance(out.get("shape"), list) and out["shape"]
            and all(type(x) is int for x in out["shape"])
            and isinstance(out.get("dtype"), str) and out["dtype"]
            and isinstance(out.get("stride"), list) and out["stride"]
            and all(type(x) is int for x in out["stride"])):
        return {"eligible": False, "reason": "out_cache_loc_metadata"}
    return {
        "eligible": True,
        "is_draft_worker": True,
        "forward_mode": "DECODE",
        "batch_size": 2,
        "shape": out["shape"],
        "dtype": out["dtype"],
        "stride": out["stride"],
        "device": out.get("device"),
    }


def load_module(name, relative, expected_sha):
    path = ROOT / relative
    assert sha256_file(path) == expected_sha, "source SHA mismatch: %s" % relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transport():
    """Load the reviewed local transport helper without contacting any node."""
    return load_module("glm53_l17_stage_transport_%d" % time.monotonic_ns(),
                       STAGE_HELPER, STAGE_HELPER_SHA)


def transport_host(node):
    helper = transport()
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
import pathlib,subprocess,json,hashlib,re,shutil,socket,time
assert META_NODE==34 and socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
    process=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    return process
def copy_logs(destination):
    r=cmd(['docker','cp',META_ID+':/var/log/memcache_hybrid',str(destination)],120)
    assert r.returncode==0,r.stderr
def collect(root):
    pattern=r'Mount bm on loc|Unmount|UnMount|ock\.mmc\.meta\.lease_ttl_ms|ock\.mmc\.evict_threshold_'
    files=[];lines=[]
    for path in sorted(root.rglob('*')):
        if path.is_file():
            raw=path.read_bytes();assert len(raw)<=256*1024**2
            text=raw.decode(errors='replace')
            critical=[line for line in text.splitlines() if re.search(pattern,line,re.I)]
            assert len(critical)<=200000
            files.append({'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'critical_count':len(critical)})
            lines.extend(critical)
    return files,lines
def stamp(line):
    match=re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)',line.strip())
    return match.group(1) if match else None
def latest_mount(lines):
    values=[stamp(line) for line in lines if MOUNT_PREFIX in line]
    values=[value for value in values if value]
    return max(values) if values else None
def latest_metrics(lines):
    values=[stamp(line) for line in lines if METRICS_PREFIX in line]
    values=[value for value in values if value]
    return max(values) if values else None
r=cmd(['docker','inspect',META_ID]);assert r.returncode==0,r.stderr
d=json.loads(r.stdout)[0]
assert d['Id']==META_ID and d['Image']==IMAGE and d['Name']=='/'+META_NAME
assert d['Config']['Labels']['project_id']=='glm53'
assert d['Config']['Labels']['run_id']==SHARED_RID
assert d['State']['Running']
root=pathlib.Path(OUTPUT);assert not root.exists();root.mkdir(parents=True)
scratch=pathlib.Path(str(root)+'-scratch')
wait={'bound_seconds':WAIT_BOUND,'step_seconds':WAIT_STEP,'reason':WAIT_REASON,'attempts':0,'converged':False,'skipped':WAIT_BOUND==0,'latest_mount_timestamp':None,'latest_metrics_timestamp':None}
deadline=time.monotonic()+WAIT_BOUND
while True:
    if scratch.exists():shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    copy_logs(scratch/'memcache_hybrid')
    _,scratch_lines=collect(scratch)
    mount_ts=latest_mount(scratch_lines);metric_ts=latest_metrics(scratch_lines)
    wait['attempts']+=1;wait['latest_mount_timestamp']=mount_ts;wait['latest_metrics_timestamp']=metric_ts
    wait['converged']=bool(mount_ts and metric_ts and metric_ts>=mount_ts)
    if WAIT_BOUND==0 or wait['converged'] or time.monotonic()>=deadline:break
    time.sleep(WAIT_STEP)
destination=root/'memcache_hybrid'
copy_logs(destination)
files,lines=collect(root)
if scratch.exists():shutil.rmtree(scratch)
print(json.dumps({'project_id':'glm53','container_id':META_ID,'meta_name':META_NAME,'meta_labels':d['Config']['Labels'],'shared_run_id':SHARED_RID,'meta_node':META_NODE,'files':files,'critical_lines':lines,'critical_count':len(lines),'critical_sha256':hashlib.sha256(json.dumps(lines).encode()).hexdigest(),'meta_after_wait':wait,'model_requests':0}))
'''


def meta_plan(phase):
    assert phase in ("before", "after")
    assert META_NODE == 34
    values = {
        "META_ID": META_CID,
        "META_NAME": META_NAME,
        "IMAGE": IMAGE,
        "SHARED_RID": OLD_D_RID,
        "OUTPUT": "/data/wio/inference-foundry/glm53/runs/" + TRANSITION_RID + "-meta-" + phase,
        "HOSTNAME": transport_host(META_NODE),
        "META_NODE": META_NODE,
        "MOUNT_PREFIX": META_MOUNT_SOURCE_PREFIX,
        "METRICS_PREFIX": META_METRICS_SOURCE_PREFIX,
        "WAIT_BOUND": META_AFTER_WAIT_BOUND_SECONDS if phase == "after" else 0,
        "WAIT_STEP": META_AFTER_WAIT_STEP_SECONDS,
        "WAIT_REASON": META_AFTER_WAIT_REASON,
    }
    return render_bound(META_PLAN, values, [
        "META_ID", "META_NAME", "IMAGE", "SHARED_RID", "OUTPUT", "HOSTNAME",
        "META_NODE", "MOUNT_PREFIX", "METRICS_PREFIX", "WAIT_BOUND", "WAIT_STEP",
        "WAIT_REASON"])


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

def file_identity(path,limit=None):
    assert path.is_file() and not path.is_symlink(),str(path)
    digest=hashlib.sha256();size=0
    with path.open('rb') as handle:
        while limit is None or size<limit:
            want=8*1024**2 if limit is None else min(8*1024**2,limit-size)
            block=handle.read(want)
            if not block:break
            digest.update(block);size+=len(block)
    return {'path':str(path),'bytes':size,'sha256':digest.hexdigest()}

def append_only(before,path):
    current=file_identity(path)
    assert current['bytes']>=before['bytes'],(str(path),before,current)
    prefix=file_identity(path,before['bytes'])
    assert prefix['bytes']==before['bytes'] and prefix['sha256']==before['sha256'],(str(path),before,prefix)
    return {'path':str(path),'prefix_match':True,'prefix_bytes':prefix['bytes'],'prefix_sha256':prefix['sha256'],'before':before,'after_full':current,'shutdown_addition_bytes':current['bytes']-before['bytes']}

def provider_close(provider,supervisor):
    closed=bool(provider.get('events')) and provider['events'][-1].get('event')=='CLOSE_COMPLETE' and provider['events'][-1].get('rc')==0
    return {'provider_phase':provider.get('phase'),'close_event':bool(provider.get('events')) and provider['events'][-1],'CLOSE_COMPLETE_rc0':closed,'supervisor_phase':supervisor.get('phase'),'supervisor_provider_exit_code':supervisor.get('provider_exit_code'),'resource_closure_components':{'provider_CLOSE_COMPLETE_rc0':closed,'provider_phase_CLOSE_COMPLETE':provider.get('phase')=='CLOSE_COMPLETE','supervisor_STOPPED':supervisor.get('phase')=='STOPPED','supervisor_provider_exit_code0':supervisor.get('provider_exit_code')==0}}

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
close=provider_close(provider,supervisor)
components=close['resource_closure_components']
resource_closure=(stop_result.returncode==0 and empty and not after['State']['Running'] and after['State']['Pid']==0 and all(components.values()))
clean_model=supervisor.get('model_exit_code')==0 and supervisor.get('timeout_used') is False and after['State']['ExitCode']==0
copies={}
for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:
    destination=root/directory.rsplit('/',1)[1]
    copied=cmd(['docker','cp',CID+':'+directory,str(destination)],120)
    copies[directory]={'rc':copied.returncode,'stderr':copied.stderr}
profile_after=profile_metadata(profile)
logs_after={stream:append_only(logs_before[stream],shared/('model.'+stream+'.txt')) for stream in ['stdout','stderr']}
assert profile_after==profile_before,'old D profile metadata changed'
(root/'profile-metadata-before.json').write_text(json.dumps(profile_before['entries'],separators=(',',':'))+'\n')
(root/'profile-metadata-after.json').write_text(json.dumps(profile_after['entries'],separators=(',',':'))+'\n')
profile_before_summary={key:value for key,value in profile_before.items() if key!='entries'}
profile_after_summary={key:value for key,value in profile_after.items() if key!='entries'}
print(json.dumps({'node':NODE,'container_id':CID,'name':NAME,'before':before,'supervisor_before':supervisor_before,'provider_before':provider_before,'stop_command_rc':stop_result.returncode,'after_state':after['State'],'supervisor':supervisor,'provider':provider,'all_NPUs_empty':empty,'provider_closed0':close['CLOSE_COMPLETE_rc0'],'resource_closure_components':components,'resource_shutdown_complete':resource_closure,'clean_model_shutdown':clean_model,'model_shutdown_diagnostic':{'model_exit_code':supervisor.get('model_exit_code'),'timeout_used':supervisor.get('timeout_used'),'docker_exit_code':after['State']['ExitCode']},'SDK_copy':copies,'profile_before':profile_before_summary,'profile_after':profile_after_summary,'profile_metadata_list_files':['profile-metadata-before.json','profile-metadata-after.json'],'retained_model_logs_before':logs_before,'retained_model_logs_after':logs_after,'retained_model_and_diagnostic_files_in_place':str(shared),'model_requests':0,'only_exact_D_container_stopped':True}))
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
import sglang.srt.environ as environ_module
import sglang.srt.model_executor.runner.eager_runner as eager_module
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
eager_source_sha256=hashlib.sha256(pathlib.Path(eager_module.__file__).read_bytes()).hexdigest()
environ_source_sha256=hashlib.sha256(pathlib.Path(environ_module.__file__).read_bytes()).hexdigest()
assert eager_source_sha256==EAGER_SHA,(eager_source_sha256,EAGER_SHA)
assert environ_source_sha256==ENVIRON_SHA,(environ_source_sha256,ENVIRON_SHA)
native_no_copy_get=envs.SGLANG_EAGER_INPUT_NO_COPY.get()
eager_audit_marker=getattr(EagerRunner.load_batch,'__glm53_l17_eager_input_audit__',None)
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
    assert native_no_copy_get is True,native_no_copy_get
    assert eager_audit_marker=='v2',eager_audit_marker
    assert os.environ['SGLANG_EAGER_INPUT_NO_COPY']=='1'
    assert os.environ['GLM53_EAGER_INPUT_AUDIT']=='1'
    assert os.environ['GLM53_EAGER_INPUT_AUDIT_ROOT']==EAGER_AUDIT_ROOT
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==AUDIT_SHA
    assert 'glm53_C32_L17_eager_input_audit_v2' in site
    no_copy='true'
else:
    assert native_no_copy_get is False,native_no_copy_get
    assert eager_audit_marker is None,eager_audit_marker
    assert 'SGLANG_EAGER_INPUT_NO_COPY' not in os.environ
    assert 'GLM53_EAGER_INPUT_AUDIT' not in os.environ
    assert not audit_path.exists()
    assert 'glm53_C32_L17_eager_input_audit_v2' not in site
    no_copy='unset'
print('GLM53_L17_PREFLIGHT='+json.dumps({'node':NODE,'variant':KIND,'pass':True,'torch_device_count':count,'ACL_count':acl_count,'soc_name':acl.get_soc_name(),'tp':cfg.tp_size,'attention_tp':parallel['attention_tp'],'dp':cfg.dp_size,'ep':cfg.ep_size,'MTP_steps':cfg.speculative_num_steps,'MTP_draft_tokens':cfg.speculative_num_draft_tokens,'kv_cache_dtype':cfg.kv_cache_dtype,'hicache_size_decimal_GB':cfg.hicache_size,'hicache_host_memory_mode':cfg.hicache_host_memory_mode,'decode_graph_backend':cfg.cuda_graph_config.decode.backend,'decode_graph_bs':cfg.cuda_graph_config.decode.bs,'decode_graph_max_bs':cfg.cuda_graph_config.decode.max_bs,'HCCL_BUFFSIZE':os.environ.get('HCCL_BUFFSIZE'),'TASK_QUEUE_ENABLE':os.environ.get('TASK_QUEUE_ENABLE'),'dispatch_capacity':os.environ.get('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK'),'SDMA_protocol':'sdma','no_copy':no_copy,'native_no_copy_get':native_no_copy_get,'eager_audit_v2_marker':eager_audit_marker,'eager_source_sha256':eager_source_sha256,'environ_source_sha256':environ_source_sha256,'eager_runner_module':str(eager_module.__file__),'environ_module':str(environ_module.__file__),'static_variant_is_candidate':CANDIDATE,'native_source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'preflight_SDK_init_calls':0,'engine_created':False,'model_constructed':False,'NPU_tensor_allocations':0,'NPU_compute_calls':0,'NPU_allocation_attempted':False}))
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
        "EAGER_SHA": EAGER_SHA,
        "ENVIRON_SHA": ENVIRON_SHA,
    }, ["NODE", "KIND", "CANDIDATE", "EAGER_AUDIT_ROOT", "AUDIT_SHA",
        "SDMA_MANIFEST_SHA", "EAGER_SHA", "ENVIRON_SHA"])


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
assert capacity['selected_P_variant']=='candidate'
assert capacity['selected_D_variant']==KIND
assert capacity.get('selected_variant')==KIND
assert capacity.get('selected_variant_scope') in ('D_ONLY_NOT_P','D_ONLY')
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
print(json.dumps({'project_id':'glm53','node':NODE,'variant':KIND,'selected_P_variant':'candidate','selected_D_variant':KIND,'selected_variant_scope':'D_ONLY_NOT_P','container_id':d['Id'],'name':NAME,'run_id':RID,'actual_local_service_id':service_id,'model_start_signal_sent_once':True,'model_ports_free':[29440,30600],'external_model_requests':0,'capacity_dict_sha256':CAPACITY_DICT_SHA}))
'''


def model_start_plan(kind, node, capacity):
    """Render only with a reviewed capacity dict whose actual CIDs are pinned."""
    spec = STAGES[kind]
    assert capacity.get("pass") is True
    d_selection(capacity)
    assert capacity["selected_D_variant"] == kind
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

def read_jsonl_checked(directory,pattern,limit_files=256,limit_total=64*1024**2):
    result={'paths':[],'rows':[],'files':0,'bytes':0,'errors':[]}
    paths=sorted(pathlib.Path(directory).glob(pattern))
    if len(paths)>limit_files:
        result['errors'].append({'path':str(directory),'error':'file_count_limit','count':len(paths)})
        return result
    total=0
    for path in paths:
        try:
            raw=path.read_bytes();total+=len(raw)
            if len(raw)>16*1024**2 or total>limit_total:
                result['errors'].append({'path':str(path),'error':'byte_limit','file_bytes':len(raw),'total_bytes':total})
                continue
            if raw and not raw.endswith(b'\n'):
                result['errors'].append({'path':str(path),'error':'missing_final_newline'})
                continue
            result['paths'].append(str(path))
            for number,line in enumerate(raw.splitlines(),1):
                try:
                    result['rows'].append(json.loads(line))
                except Exception as error:
                    result['errors'].append({'path':str(path),'line':number,'error':repr(error)})
        except Exception as error:
            result['errors'].append({'path':str(path),'error':repr(error)})
    result['files']=len(result['paths']);result['bytes']=total
    return result

def eager_summary(initialized):
    result={'candidate':CANDIDATE,'status':'PENDING_FIRST_USE','record_files':0,'records':0,'record_pids':0,'expected_pids':len(initialized),'expected_global_pids':32,'audit_pids_match_SDK':False,'all_rows_schema_valid':None,'all_native_no_copy_true':None,'all_clone_distinct':None,'all_clone_fields_identity':None,'observed_forward_modes':{},'observed_draft_roles':{},'bs2_draft_decode_pids':[],'bs2_draft_decode_coverage':{},'bs2_observed_pids':0,'bs2_all16':False,'raw_read_errors':[],'parse_errors':[],'schema_errors':[],'reason':'first_use_pending'}
    if not CANDIDATE:
        result['status']='NOT_APPLICABLE_ROLLBACK';result['reason']='rollback has no v2 audit hook';return result
    directory=root/'eager-input-audit'
    if not directory.is_dir():
        result['reason']='audit_dir_missing_pending_first_use';return result
    read=read_jsonl_checked(directory,'pid-*.jsonl')
    result['raw_read_errors']=read['errors'];result['record_files']=read['files']
    if read['errors']:
        result['status']='AUDIT_READ_OR_PARSE_ERROR';result['reason']='raw_audit_error';return result
    records=read['rows'];result['records']=len(records)
    if not records:
        result['reason']='no_first_use_records_pending';return result
    by_pid={};schema_errors=[]
    for record in records:
        pid=record.get('pid')
        if type(pid) is not int:
            schema_errors.append({'pid':repr(pid),'error':'pid_not_int'});continue
        by_pid.setdefault(pid,[]).append(record)
    result['record_pids']=len(by_pid)
    native_true=clone_distinct=fields_identity=True
    bs2={};modes={};roles={}
    for pid,rows in by_pid.items():
        if pid not in initialized:
            schema_errors.append({'pid':pid,'error':'pid_not_in_SDK_workers'})
        for record in rows:
            modes[str(record.get('forward_mode'))]=modes.get(str(record.get('forward_mode')),0)+1
            roles[str(record.get('is_draft_worker'))]=roles.get(str(record.get('is_draft_worker')),0)+1
            if record.get('schema')!='glm53.L17.eager_input_audit.v2' or record.get('audit_version')!='v2' or record.get('project_id')!='glm53':
                schema_errors.append({'pid':pid,'error':'schema_identity'});continue
            if record.get('node')!=NODE:
                schema_errors.append({'pid':pid,'error':'node','actual':record.get('node')});continue
            if record.get('source_sha256')!=EAGER_SHA or record.get('eager_source_sha256')!=EAGER_SHA or record.get('environ_source_sha256')!=ENVIRON_SHA:
                schema_errors.append({'pid':pid,'error':'source_sha'});continue
            if record.get('native_no_copy_get') is not True:
                native_true=False;schema_errors.append({'pid':pid,'error':'native_no_copy_get'})
            if record.get('clone_distinct') is not True:
                clone_distinct=False;schema_errors.append({'pid':pid,'error':'clone_distinct'})
            verdicts=record.get('field_verdicts') or {};count=record.get('dataclass_field_count')
            if not (type(count) is int and verdicts.get('identity')==count):
                fields_identity=False;schema_errors.append({'pid':pid,'error':'field_identity'})
            if record.get('is_draft_worker') is True and record.get('forward_mode')=='DECODE' and record.get('batch_size')==2:
                out=(record.get('allowlisted') or {}).get('out_cache_loc')
                valid=isinstance(out,dict) and out.get('tensor') is True and isinstance(out.get('shape'),list) and bool(out['shape']) and all(type(x) is int for x in out['shape']) and isinstance(out.get('dtype'),str) and bool(out['dtype']) and isinstance(out.get('stride'),list) and bool(out['stride']) and all(type(x) is int for x in out['stride'])
                if valid:
                    bs2[pid]={'shape':out['shape'],'dtype':out['dtype'],'stride':out['stride'],'device':out.get('device'),'batch_size':2,'forward_mode':'DECODE','is_draft_worker':True}
                else:
                    schema_errors.append({'pid':pid,'error':'bs2_draft_out_cache_loc_metadata_missing'})
    joined=set(by_pid)==set(initialized)
    complete=(joined and not schema_errors and native_true and clone_distinct and fields_identity)
    bs2_all=(complete and len(bs2)==16 and set(bs2)==set(initialized))
    result.update(status='COMPLETE_ALL16' if bs2_all else ('ALL_ROWS_VALID_BS2_PENDING' if complete else 'PARTIAL_PENDING'),audit_pids_match_SDK=joined,all_rows_schema_valid=not schema_errors,all_native_no_copy_true=native_true,all_clone_distinct=clone_distinct,all_clone_fields_identity=fields_identity,observed_forward_modes=modes,observed_draft_roles=roles,bs2_draft_decode_pids=sorted(bs2),bs2_draft_decode_coverage={str(pid):value for pid,value in sorted(bs2.items())},bs2_observed_pids=len(bs2),bs2_all16=bs2_all,schema_errors=schema_errors,reason=None if complete else 'schema_or_pid_join_pending')
    return result

def ops_summary(initialized,align):
    ops={};errors=[];counts={}
    for event in align:
        pid=event.get('pid');kind=event.get('event')
        counts[str(kind)]=counts.get(str(kind),0)+1
        if pid not in initialized:
            errors.append({'pid':pid,'event':kind,'error':'pid_not_in_SDK_workers'})
    for pid in initialized:
        rows=[event for event in align if event.get('pid')==pid]
        pid_errors=[]
        if not rows:
            pid_errors.append({'pid':pid,'error':'no_native_rows'})
        elif {event.get('revision') for event in rows}!={'glm53-C32-target-graph-fullmesh-v2'}:
            pid_errors.append({'pid':pid,'error':'revision'})
        else:
            for method in ['_npu_low_latency_dispatch','_npu_low_latency_combine']:
                matches=[event for event in rows if event.get('event')=='ops_first_return' and event.get('method')==method]
                if len(matches)!=1 or matches[0].get('group_size')!=32 or matches[0].get('comm_alg')!='fullmesh_v1':
                    pid_errors.append({'pid':pid,'error':'ops_first_return','method':method})
            if not any(event.get('event')=='draft_graph_disabled' for event in rows):
                pid_errors.append({'pid':pid,'error':'draft_graph_disabled_missing'})
            productive=[event for event in rows if event.get('event')=='target_graph_decision' and event.get('eligible') is True and event.get('mode')=='TARGET_VERIFY' and event.get('raw_bs',0)>0 and event.get('is_draft') is False]
            if not productive:
                pid_errors.append({'pid':pid,'error':'productive_target_verify_missing'})
        if pid_errors:
            errors.extend(pid_errors);continue
        first_return=[event for event in rows if event.get('event')=='ops_first_return']
        ops[str(pid)]={'rank':first_return[0]['rank'],'productive_target_verify_decisions':len(productive),'draft_graph_disabled':True,'fullmesh_v1':True,'native_ops_group_size':32}
    return ops,errors,counts

deadline=time.monotonic()+1800;last=0;summary=None
while True:
    supervisor=json.loads((root/'supervisor-status.json').read_text())
    assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] is True
    assert 'model_exit_code' not in supervisor
    provider=json.loads((root/'provider-status.json').read_text());assert provider['phase']=='READY'
    health_status=None;metrics_status=None;metric=None
    server={'status':None,'info':None,'info_keys':None,'info_sha256':None,'error':None,'scope':'role_global_primary' if NODE==36 else 'secondary_expected_404'}
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
        try:
            info=json.loads(raw.decode())
        except Exception as error:
            server={'status':status,'info':None,'info_keys':None,'info_sha256':hashlib.sha256(raw).hexdigest(),'error':'JSONDecodeError: '+repr(error),'scope':server['scope']}
        else:
            if not isinstance(info,dict):
                server={'status':status,'info':None,'info_keys':None,'info_sha256':hashlib.sha256(raw).hexdigest(),'error':'top_level_not_object','scope':server['scope']}
            else:
                filtered={key:info.get(key) for key in ('tp_size','ep_size','dp_size','speculative_num_steps','speculative_num_draft_tokens','kv_cache_dtype','disable_overlap_schedule','max_total_num_tokens','cuda_graph_config')}
                server={'status':status,'info':filtered,'info_keys':sorted(info),'info_sha256':hashlib.sha256(raw).hexdigest(),'error':None,'scope':server['scope']}
    except urllib.error.HTTPError as error:
        server={'status':error.code,'info':None,'info_keys':None,'info_sha256':None,'error':'HTTPError','scope':server['scope']}
    except Exception as error:
        server={'status':None,'info':None,'info_keys':None,'info_sha256':None,'error':repr(error),'scope':server['scope']}
    capacity=None;graph_config=None;server_ok=False
    if NODE==36:
        info=server.get('info')
        if server['status']==200 and isinstance(info,dict):
            expected=[('tp_size',32),('ep_size',32),('dp_size',16),('speculative_num_steps',5),('speculative_num_draft_tokens',6),('kv_cache_dtype','bfloat16'),('disable_overlap_schedule',True)]
            server_ok=all(info.get(key)==value for key,value in expected)
            capacity=info.get('max_total_num_tokens')
            graph_config=(info.get('cuda_graph_config') or {}).get('decode') or {}
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
    sdma=read_jsonl_checked(root/'audit','sdma-*.jsonl')
    events=sdma['rows']
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
    alignment=read_jsonl_checked(root/'alignment-audit','worker-*.jsonl')
    if init_ok:
        native_ops,native_ops_errors,native_event_counts=ops_summary(initialized,alignment['rows'])
    else:
        native_ops={};native_ops_errors=[{'error':'sdk_not_initialized'}];native_event_counts={}
    expected_ops_ranks=set(range(0,16)) if NODE==36 else set(range(16,32))
    native_ops_ranks={item['rank'] for item in native_ops.values()}
    rank_half_ok=(len(native_ops)==16 and native_ops_ranks==expected_ops_ranks)
    native_audit_errors=list(sdma['errors'])+list(alignment['errors'])+list(native_ops_errors)
    native_audit={'reader':'read_jsonl_checked; raw read/parse errors retained, never replaced by []','sdma_files':sdma['files'],'sdma_bytes':sdma['bytes'],'sdma_rows':len(events),'sdma_read_errors':sdma['errors'],'alignment_files':alignment['files'],'alignment_bytes':alignment['bytes'],'alignment_rows':len(alignment['rows']),'alignment_read_errors':alignment['errors'],'native_ops_errors':native_ops_errors,'native_event_counts':native_event_counts,'sdma_rows_belong_to_SDK_workers':all(event.get('pid') in initialized for event in events)}
    eager=eager_summary(initialized)
    eager_errors=[]
    if CANDIDATE:
        eager_errors=list(eager.get('raw_read_errors') or [])+list(eager.get('parse_errors') or [])+list(eager.get('schema_errors') or [])
    done=health_status==200 and metrics_status==200 and idle and len(npu_rows)==18 and init_ok and layout_ok and rank_half_ok and graph_ok and not fatal and not native_audit_errors and not eager_errors
    if NODE==36:
        done=done and server_ok and warm is True
    else:
        done=done and server_ok
    if time.monotonic()-last>=30 or done:
        last=time.monotonic();print('GLM53_L17_D_OBSERVE_PROGRESS='+json.dumps({'node':NODE,'health':health_status,'idle':idle,'registered':len(layouts),'graph_ranks':len(graph),'native_ops_workers':len(native_ops),'native_audit_errors':len(native_audit_errors),'warm_all16dp':warm,'eager_status':eager['status'],'eager_bs2_draft_decode_pids':eager['bs2_observed_pids'],'done':done}),flush=True)
    if done:
        summary={'project_id':'glm53','node':NODE,'variant':KIND,'selected_P_variant':'candidate','selected_D_variant':KIND,'selected_variant_scope':'D_ONLY_NOT_P','startup_pass':True,'container_id':CID,'name':NAME,'run_id':RID,'actual_local_service_id':SERVICE_ID,'health_status':health_status,'metrics_status':metrics_status,'idle':idle,'idle_gauges':gauges,'server_info':server,'secondary_server_info_404_expected':NODE==41,'primary_server_info_status':server['status'],'reported_KV_capacity':capacity,'reported_KV_capacity_scope':'role-global native primary /get_server_info top-level max_total_num_tokens; distributed worker registrations; node41 404 expected','capacity_floor':161280,'old_D_reported_KV_capacity':181632,'capacity_tuning':False,'warm_all16dp':warm,'actual_SDK_worker_PIDs':sorted(initialized),'sdk_pid_join_scope':'node-local 16 actual SDK workers; global32 requires node36+node41 summaries','expected_global_worker_count':32,'registered_worker_PIDs':sorted(event['pid'] for event in layouts),'owned_NPU_context_PIDs':sorted(npu_rows),'graph':{str(rank):passes for rank,passes in sorted(graph.items())},'native_ops_fullmesh':native_ops,'native_ops_ranks':sorted(native_ops_ranks),'native_ops_rank_half_scope':'node-local native Ops ranks; node36 0-15 plus node41 16-31 covers all32','native_ops_rank_half_ok':rank_half_ok,'native_audit':native_audit,'audit_read_errors':native_audit_errors,'audit_last_parse_errors':(sdma['errors']+alignment['errors'])[-10:],'MTP':{'steps':5,'draft_tokens':6,'draft_kv_layers':1,'packed_layout_metadata_workers':len(layouts),'kv_cache_dtype':'bfloat16'},'eager_first_use':eager,'eager_first_use_gate':'OBSERVED_IN_STARTUP' if eager['bs2_all16'] else 'PENDING_UNTIL_BOUNDED_FUNCTIONAL','functional_bs2_all32_gate_pending':not eager['bs2_all16'],'functional_bs2_all32_gate_requirement':'all32 native first-use plus draft DECODE bs2 out_cache_loc metadata before formal','formal_ready':False,'external_model_requests':0,'P_native_map_checks_on_D':0}
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
    d_selection(capacity)
    assert capacity["selected_D_variant"] == kind
    row = capacity["providers"][str(node)]
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


CLEANUP_PLAN = r'''
import pathlib,subprocess,json,hashlib,socket
assert socket.gethostname()==HOSTNAME
assert NODE in (36,41) and KIND in ('candidate','rollback')
assert CID not in P_CIDS and CID!=META_ID and CID!=OLD_CID
def cmd(args,timeout=30):
    return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
def must(args,timeout=30):
    result=cmd(args,timeout);assert result.returncode==0,(args,result.returncode,result.stderr);return result.stdout
def inspect(cid):
    return json.loads(must(['docker','inspect',cid]))[0]
def file_identity(path):
    assert path.is_file() and not path.is_symlink(),str(path)
    digest=hashlib.sha256();size=0
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(block);size+=len(block)
    return {'path':str(path),'bytes':size,'sha256':digest.hexdigest()}
def append_only(before,path):
    current=file_identity(path)
    assert current['bytes']>=before['bytes'],(str(path),before,current)
    digest=hashlib.sha256();read=0
    with path.open('rb') as handle:
        while read<before['bytes']:
            block=handle.read(min(8*1024**2,before['bytes']-read))
            if not block:break
            digest.update(block);read+=len(block)
    assert read==before['bytes'] and digest.hexdigest()==before['sha256'],(str(path),before)
    return {'path':str(path),'prefix_match':True,'prefix_bytes':before['bytes'],'prefix_sha256':before['sha256'],'before':before,'after_full':current,'shutdown_addition_bytes':current['bytes']-before['bytes']}
def profile_metadata(root):
    entries=[(str(path.relative_to(root)),path.stat().st_size,path.stat().st_mtime_ns) for path in sorted(root.rglob('*')) if path.is_file() and not path.is_symlink()]
    return {'directory':str(root),'exists':root.exists(),'file_count':len(entries),'bytes':sum(item[1] for item in entries),'metadata_sha256':hashlib.sha256(json.dumps(entries).encode()).hexdigest(),'entries':entries}
def provider_close(provider,supervisor):
    closed=bool(provider.get('events')) and provider['events'][-1].get('event')=='CLOSE_COMPLETE' and provider['events'][-1].get('rc')==0
    components={'provider_CLOSE_COMPLETE_rc0':closed,'provider_phase_CLOSE_COMPLETE':provider.get('phase')=='CLOSE_COMPLETE','supervisor_STOPPED':supervisor.get('phase')=='STOPPED','supervisor_provider_exit_code0':supervisor.get('provider_exit_code')==0}
    return {'CLOSE_COMPLETE_rc0':closed,'components':components}
def npu_rows():
    text=must(['npu-smi','info']);rows=[]
    for line in text.split('Process id',1)[-1].splitlines():
        columns=[value.strip() for value in line.split('|')]
        if len(columns)>=5 and len(columns[1].split())==2 and all(value.isdigit() for value in columns[1].split()) and columns[2].isdigit():
            rows.append(int(columns[2]))
    return text,rows
d=inspect(CID)
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
assert not d['HostConfig']['Privileged']
root=pathlib.Path(REMOTE);assert root.is_dir() and str(root)==REMOTE
staged=pathlib.Path(STAGED);assert staged.is_dir(),str(staged)
npu_before,owned_before=npu_rows();owned_ids=[];foreign=[]
for pid in owned_before:
    try:
        cgroup=(pathlib.Path('/proc')/str(pid)/'cgroup').read_text()
    except OSError:
        foreign.append(pid);continue
    if CID in cgroup:owned_ids.append(pid)
    else:foreign.append(pid)
assert not foreign,('foreign NPU PID present; refusing cleanup',foreign)
supervisor_path=root/'supervisor-status.json';provider_path=root/'provider-status.json'
supervisor=json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
provider=json.loads(provider_path.read_text()) if provider_path.exists() else {}
assert provider.get('phase') in ('READY','CLOSE_COMPLETE'),provider.get('phase')
if supervisor.get('model_started') is True or supervisor.get('phase') not in (None,'POOL_READY_WAITING_MODEL_SIGNAL'):
    state='MODEL_STARTED_OR_FAILED'
else:
    state='READY_PROVIDER_ONLY'
logs_before={}
for stream in ['stdout','stderr']:
    path=root/('model.'+stream+'.txt')
    logs_before[stream]=file_identity(path) if path.exists() else None
profile_before=profile_metadata(root/'profile')
out=pathlib.Path(OUTDIR);assert not out.exists();out.mkdir(parents=True)
was_running=d['State']['Running']
stop_result=None
if was_running:
    stop_result=cmd(['docker','stop','--time','240',CID],270)
    assert stop_result.returncode==0,stop_result.stderr
after=inspect(CID)
assert not after['State']['Running'] and after['State']['Pid']==0
supervisor_after=json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
provider_after=json.loads(provider_path.read_text()) if provider_path.exists() else {}
npu_after,_=npu_rows();empty=npu_after.count('No running processes')==8
close=provider_close(provider_after,supervisor_after);components=close['components']
resource_closure=(not after['State']['Running'] and after['State']['Pid']==0 and empty and all(components.values()))
clean_model=supervisor_after.get('model_exit_code')==0 and supervisor_after.get('timeout_used') is False and after['State']['ExitCode']==0
logs_after={}
for stream in ['stdout','stderr']:
    path=root/('model.'+stream+'.txt')
    logs_after[stream]=append_only(logs_before[stream],path) if logs_before[stream] else {'absent':True,'path':str(path)}
profile_after=profile_metadata(root/'profile')
assert profile_after==profile_before,'profile metadata changed during cleanup'
copies={}
for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:
    destination=out/directory.rsplit('/',1)[1]
    copied=cmd(['docker','cp',CID+':'+directory,str(destination)],120)
    copies[directory]={'rc':copied.returncode,'stderr':copied.stderr}
print(json.dumps({'project_id':'glm53','node':NODE,'variant':KIND,'selected_P_variant':'candidate','selected_D_variant':KIND,'selected_variant_scope':'D_ONLY_NOT_P','cleanup_scope':'OWNED_D_ONLY_ACTUAL_CAPACITY_CID','container_id':CID,'name':NAME,'run_id':RID,'root':str(root),'staged_source':str(staged),'state_detected':state,'was_running':was_running,'stop_command_rc':None if stop_result is None else stop_result.returncode,'after_state':after['State'],'supervisor_before':supervisor,'supervisor_after':supervisor_after,'provider_before':provider,'provider_after':provider_after,'provider_close':close,'resource_closure_components':components,'resource_shutdown_complete':resource_closure,'clean_model_shutdown':clean_model,'model_shutdown_diagnostic':{'model_exit_code':supervisor_after.get('model_exit_code'),'timeout_used':supervisor_after.get('timeout_used'),'docker_exit_code':after['State']['ExitCode']},'all_NPUs_empty':empty,'owned_NPU_context_PIDs_before':sorted(owned_ids),'foreign_NPU_PIDs_refused':[],'model_logs_before':logs_before,'model_logs_after':logs_after,'profile_before':{key:value for key,value in profile_before.items() if key!='entries'},'profile_after':{key:value for key,value in profile_after.items() if key!='entries'},'SDK_copy':copies,'retained_model_profile_and_SDK_logs_in_place':str(root),'P_Meta_vllm_touched':False,'foreign_PID_killed':False,'fallback_or_retry':False,'model_requests':0,'capacity_dict_sha256':CAPACITY_DICT_SHA}))
'''


def cleanup_plan(kind, node, capacity):
    """Template only; rendered after a future capacity audit pins actual CIDs."""
    spec = STAGES[kind]
    d_selection(capacity)
    assert capacity["selected_D_variant"] == kind
    row = capacity["providers"][str(node)]
    assert row["run_id"] == spec["run_id"] and row["remote_root"] == spec["run_root"]
    assert row["name"] == spec["name_prefix"] + "-20261019-" + str(node)
    cid = row["container_id"]
    assert cid != OLD_D[node]["container_id"]
    p_cids = [capacity["providers"][str(item)]["container_id"] for item in P_NODES]
    assert cid not in p_cids and cid != META_CID
    return render_bound(CLEANUP_PLAN, {
        "NODE": node,
        "KIND": kind,
        "CID": cid,
        "NAME": row["name"],
        "IMAGE": IMAGE,
        "RID": spec["run_id"],
        "REMOTE": spec["run_root"],
        "STAGED": spec["remote"],
        "OLD_CID": OLD_D[node]["container_id"],
        "P_CIDS": p_cids,
        "META_ID": META_CID,
        "HOSTNAME": transport_host(node),
        "OUTDIR": "/data/wio/inference-foundry/glm53/runs/" + CLEANUP_RID + "-" + kind + "-" + str(node),
        "CAPACITY_DICT_SHA": sha256_text(json.dumps(capacity, sort_keys=True)),
    }, ["NODE", "KIND", "CID", "NAME", "IMAGE", "RID", "REMOTE", "STAGED",
        "OLD_CID", "P_CIDS", "META_ID", "HOSTNAME", "OUTDIR", "CAPACITY_DICT_SHA"])


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
            for key in ("start", "observe", "cleanup") for node in D_NODES]


def dynamic_plans(capacity):
    """Start/observe/cleanup templates, rendered only with actual capacity CIDs."""
    d_selection(capacity)
    plans = {}
    for kind in D_SELECTED_VARIANTS:
        if capacity.get("selected_D_variant") != kind:
            continue
        for node in D_NODES:
            plans["%s-start-%d" % (kind, node)] = model_start_plan(kind, node, capacity)
            plans["%s-observe-%d" % (kind, node)] = observe_plan(kind, node, capacity)
            plans["%s-cleanup-%d" % (kind, node)] = cleanup_plan(kind, node, capacity)
    return plans


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
        "role_selectors": {"selected_P_variant": P_SELECTED_VARIANT, "selected_D_variants": list(D_SELECTED_VARIANTS), "legacy_D_scope": D_ONLY_SCOPE},
        "rollback_sequence": {"status": "PENDING_CANDIDATE_CLOSE_AND_ACTUAL_CAPACITY_AUDIT", "plans": ["rollback-prepare", "rollback-pool", "rollback-preflight"], "executed": False, "note": "static plans remain unexecuted until candidate close and a pass capacity-v2 audit"},
        "meta_scope": {"docker_host_node": META_NODE, "container_id": META_CID, "before_after_plans_only": True, "after_wait_bound_seconds": META_AFTER_WAIT_BOUND_SECONDS, "after_wait_reason": META_AFTER_WAIT_REASON},
        "stop_log_rule": "append-only: after full file may grow; bytes [0,before_bytes) must hash exactly to the pre-stop SHA; full after size/SHA retained",
        "resource_closure_rule": "provider CLOSE_COMPLETE rc0 + supervisor provider_exit_code0/STOPPED + stopped owned CID Pid0 + 8 empty NPUs; clean model exit reported separately",
        "cleanup_dynamic": "parameterized owned-D-only templates; no execution/fallback/retry",
        "future_budget": {"status": "PROPOSED_NOT_AUTHORIZED", "functional_calls": 76, "functional_output_tokens": 21640, "canonical_calls": 320, "canonical_output_tokens": 76992, "total_calls": 396, "total_output_tokens": 98632, "new_profiles": 0, "retries": 0},
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
        "preflight_SDK_init_calls": 0,
        "expected_transition_provider_SDK_setup_init_sequences": EXPECTED_PROVIDER_SDK_SETUP_INIT_SEQUENCES,
        "model_requests": 0,
        "remote_writes": 0,
        "Git_operations": 0,
        "runtime_authorized": False,
        "model_start_authorized": False,
        "role_selectors": {"selected_P_variant": P_SELECTED_VARIANT, "selected_D_variants": list(D_SELECTED_VARIANTS), "legacy_D_scope": D_ONLY_SCOPE},
        "rollback_sequence": {"status": "PENDING_CANDIDATE_CLOSE_AND_ACTUAL_CAPACITY_AUDIT", "plans": ["rollback-prepare", "rollback-pool", "rollback-preflight"], "executed": False},
        "pending_dynamic_capacity": pending_dynamic_plans(),
        "blocked_until": ["independent local pool-capacity audit pins actual D CIDs",
                          "separate primary review freezes model-start/observe/cleanup host SHAs"],
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
        "static_executable_plans": len(plans),
        "embedded_preflight_inner": len(inner),
        "pending_dynamic_capacity": pending_dynamic_plans(),
        "role_selectors": {"selected_P_variant": P_SELECTED_VARIANT, "selected_D_variants": list(D_SELECTED_VARIANTS), "legacy_D_scope": D_ONLY_SCOPE},
        "rollback_sequence": {"status": "PENDING_CANDIDATE_CLOSE_AND_ACTUAL_CAPACITY_AUDIT", "plans": ["rollback-prepare", "rollback-pool", "rollback-preflight"], "executed": False, "note": "static plans remain unexecuted until candidate close and a pass capacity-v2 audit"},
        "meta_scope": {"docker_host_node": META_NODE, "container_id": META_CID, "after_wait_bound_seconds": META_AFTER_WAIT_BOUND_SECONDS},
        "source_provenance": source_provenance(),
        "SSH": 0,
        "network": 0,
        "model_requests": 0,
        "SDK_init_calls": 0,
        "preflight_SDK_init_calls": 0,
        "expected_transition_provider_SDK_setup_init_sequences": EXPECTED_PROVIDER_SDK_SETUP_INIT_SEQUENCES,
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
        "static_executable_plans": len(plans),
        "embedded_preflight_inner": len(inner),
        "executable_plan_hashes": executable_hashes,
        "embedded_preflight_inner_hashes": inner_hashes,
        "pending_dynamic_capacity": pending_dynamic_plans(),
        "role_selectors": {"selected_P_variant": P_SELECTED_VARIANT, "selected_D_variants": list(D_SELECTED_VARIANTS), "legacy_D_scope": D_ONLY_SCOPE},
        "rollback_sequence": {"status": "PENDING_CANDIDATE_CLOSE_AND_ACTUAL_CAPACITY_AUDIT", "plans": ["rollback-prepare", "rollback-pool", "rollback-preflight"], "executed": False, "note": "static plans remain unexecuted until candidate close and a pass capacity-v2 audit"},
        "meta_scope": {"docker_host_node": META_NODE, "container_id": META_CID, "after_wait_bound_seconds": META_AFTER_WAIT_BOUND_SECONDS},
        "source_provenance": receipt["source_provenance"],
        "not_created": [REVIEW, CAPACITY_OUT, "stop/pool/model-start launcher", "functional/formal client launcher", "cleanup execution"],
        "future_budget": {"status": "PROPOSED_NOT_AUTHORIZED", "functional_calls": 76, "functional_output_tokens": 21640, "canonical_calls": 320, "canonical_output_tokens": 76992, "total_calls": 396, "total_output_tokens": 98632, "new_profiles": 0, "retries": 0},
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
