"""Build and validate the corrected L17-D native-eager candidate-v2 bundle.

Local only.  Candidate-v1 and rollback-v1 are immutable inputs: their exact
stage and bundle SHAs are checked before and after this build.  Candidate-v2
copies the exact D36/D41 files from the staged v3 base, adds only the three
reviewed audit environment variables, appends the SHA-pinned v2 audit
activation to sitecustomize, and records the v2 audit source/environ pins.
No lifecycle launcher, deployment or service action is produced.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

PROJECT_ID = "glm53"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_STAGE = PROJECT_ROOT / "config/sglang-C32-target-graph-fullmesh-stage-v3.json"
AUDIT_MODULE = PROJECT_ROOT / "scripts/glm53_C32_L17_eager_input_audit_v2.py"
OVERLAY_ROOT = PROJECT_ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1"
CANDIDATE_DIR = OVERLAY_ROOT / "candidate-v2"
CANDIDATE_V1_DIR = OVERLAY_ROOT / "candidate-v1"
ROLLBACK_DIR = OVERLAY_ROOT / "rollback-v1"
CANDIDATE_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-native-eager-stage-v2.json"
CANDIDATE_V1_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-native-eager-stage-v1.json"
ROLLBACK_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-original-rollback-stage-v1.json"
VALIDATION_PATH = (
    PROJECT_ROOT / "runs/20261019T-L17-local-audit-correction-executor-v2/bundle-validation-v2.json"
)

D_NODES = [36, 41]
REMOVE_P_KEYS = [
    "config/node-34.json",
    "config/node-35.json",
    "provider-config/node-34.json",
    "provider-config/node-35.json",
    "supervisor-config/node-34.json",
    "supervisor-config/node-35.json",
]
CANDIDATE_REMOTE = (
    "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-native-eager-v2-20261019"
)
CANDIDATE_RUN_ID = "20261019T-L17-D-native-eager-v2"
AUDIT_ENV = {
    "SGLANG_EAGER_INPUT_NO_COPY": "1",
    "GLM53_EAGER_INPUT_AUDIT": "1",
    "GLM53_EAGER_INPUT_AUDIT_ROOT": "/run/glm53/eager-input-audit",
}
CANDIDATE_NODE_DIFFS = ["environment_intent." + key for key in AUDIT_ENV]
CANDIDATE_SUPERVISOR_DIFFS = [
    "run_id",
    "model_env.SGLANG_EAGER_INPUT_NO_COPY",
    "model_env.GLM53_EAGER_INPUT_AUDIT",
    "model_env.GLM53_EAGER_INPUT_AUDIT_ROOT",
]
EXPECTED_BASE_STAGE_SHA256 = "4a00cb3433b0bc96ee92de899d371a3c3174027e3ceb8ee338ccffa493fe0291"
EXPECTED_V1 = {
    "candidate_stage": "72cdfaa288e88ecb06fba5b3cdbb054a15bf3809d8f61625bb98431dc82abecf",
    "candidate_bundle": "31cef859d835cbe6d0835fe7b3b1064f9ede7e2f63a2f71b6838857f8b67b8c2",
    "rollback_stage": "f8b49216cd3ac6651cb48f9308a2590d2c0e306b339355e568a522bb882ddcf0",
    "rollback_bundle": "b3e1d5824eed9e86b61e1a20a0f0a46732c5caca543d83e1310d62e4a171cd7a",
    "audit": "1be2514ca62a270b274d451d8daf5ba325d42bc9344a0cfcac830b21a8002a5c",
    "review": "ae2c2b9b363c6fdb204ad3147e62bdb124f3a21c09245532d7a4252aa52c10a0",
}
ENVIRON_SHA256 = "0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c"
EAGER_SHA256 = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
REVIEW_PATH = PROJECT_ROOT / "diagnostics/model-profile/L17-primary-audit-draft-review-v1.md"

SITECUSTOMIZE_APPEND = """

if os.environ.get('GLM53_EAGER_INPUT_AUDIT')=='1':
 try:
  if os.environ.get('SGLANG_EAGER_INPUT_NO_COPY')!='1': raise RuntimeError('SGLANG_EAGER_INPUT_NO_COPY is not 1')
  if os.environ.get('GLM53_NODE') not in ('36','41'): raise RuntimeError('GLM53_NODE is not D36/D41')
  import hashlib,pathlib
  audit_path=pathlib.Path(__file__).with_name('glm53_C32_L17_eager_input_audit_v2.py')
  if hashlib.sha256(audit_path.read_bytes()).hexdigest()!='{audit_sha}': raise RuntimeError('L17 v2 audit module SHA mismatch')
  import glm53_C32_L17_eager_input_audit_v2 as eager_input_audit
  eager_input_audit.activate()
 except BaseException as error:
  os.write(2,('GLM53_EAGER_INPUT_AUDIT_FATAL='+repr(error)+'\\n').encode());os._exit(86)
"""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(pathlib.Path(path).read_bytes())


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


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


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def diffs_match(actual, expected):
    return sorted(actual) == sorted(expected)


def copy_exact(source, destination):
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError("copy SHA mismatch: %s -> %s" % (source, destination))


def build_base_inventory(failures):
    stage_sha = sha256_file(BASE_STAGE)
    require(stage_sha == EXPECTED_BASE_STAGE_SHA256, "base stage SHA differs", failures)
    stage = read_json(BASE_STAGE)
    require(stage.get("project_id") == PROJECT_ID, "base stage project_id", failures)
    require(len(stage.get("bundles", [])) == 1, "base stage must have one bundle", failures)
    bundle = stage["bundles"][0]
    require(bundle.get("framework") == "sglang", "base framework", failures)
    require(sorted(bundle.get("nodes", [])) == [34, 35, 36, 41], "base nodes", failures)
    files = bundle.get("files", {})
    inventory = {}
    for staged_name, meta in files.items():
        local = PROJECT_ROOT / meta["local"]
        require(local.is_file(), "base local missing: %s" % meta["local"], failures)
        if not local.is_file():
            continue
        actual = sha256_file(local)
        require(
            actual == meta["sha256"],
            "base SHA differs for %s: actual=%s expected=%s"
            % (meta["local"], actual, meta["sha256"]),
            failures,
        )
        inventory[staged_name] = {
            "staged_name": staged_name,
            "local": meta["local"],
            "sha256": meta["sha256"],
            "actual_sha256": actual,
            "bytes": local.stat().st_size,
            "source_path": str(local),
        }
    require(set(REMOVE_P_KEYS).issubset(files), "base stage is missing expected P files", failures)
    for node in D_NODES:
        for prefix in ("config", "provider-config", "supervisor-config"):
            require(
                "%s/node-%d.json" % (prefix, node) in files,
                "base stage missing %s/node-%d.json" % (prefix, node),
                failures,
            )
    return {"path": str(BASE_STAGE.relative_to(PROJECT_ROOT)), "sha256": stage_sha, "bundle": bundle, "inventory": inventory}


def verify_immutable_inputs(failures):
    require(sha256_file(CANDIDATE_V1_STAGE) == EXPECTED_V1["candidate_stage"], "candidate-v1 stage changed", failures)
    require(
        sha256_file(CANDIDATE_V1_DIR / "bundle-manifest.json") == EXPECTED_V1["candidate_bundle"],
        "candidate-v1 bundle changed",
        failures,
    )
    require(sha256_file(ROLLBACK_STAGE) == EXPECTED_V1["rollback_stage"], "rollback-v1 stage changed", failures)
    require(
        sha256_file(ROLLBACK_DIR / "bundle-manifest.json") == EXPECTED_V1["rollback_bundle"],
        "rollback-v1 bundle changed",
        failures,
    )
    require(
        sha256_file(PROJECT_ROOT / "scripts/glm53_C32_L17_eager_input_audit_v1.py") == EXPECTED_V1["audit"],
        "audit-v1 source changed",
        failures,
    )
    require(sha256_file(REVIEW_PATH) == EXPECTED_V1["review"], "primary review changed", failures)
    for stage_path, manifest_path in (
        (CANDIDATE_V1_STAGE, CANDIDATE_V1_DIR / "bundle-manifest.json"),
        (ROLLBACK_STAGE, ROLLBACK_DIR / "bundle-manifest.json"),
    ):
        stage = read_json(stage_path)
        for staged_name, meta in stage["bundles"][0]["files"].items():
            local = PROJECT_ROOT / meta["local"]
            require(local.is_file() and sha256_file(local) == meta["sha256"], "immutable input file changed: %s" % staged_name, failures)
        manifest = read_json(manifest_path)
        for staged_name, record in manifest["files"].items():
            local = manifest_path.parent / record["local"]
            require(local.is_file() and sha256_file(local) == record["sha256"], "immutable manifest file changed: %s" % staged_name, failures)


def build_candidate(base, audit_sha, audit_bytes, failures):
    bundle = base["bundle"]
    inventory = base["inventory"]
    selected = [name for name in bundle["files"] if name not in REMOVE_P_KEYS]
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=False)

    file_records = {}
    for staged_name in selected:
        record = inventory[staged_name]
        source = pathlib.Path(record["source_path"])
        target = CANDIDATE_DIR / staged_name
        copy_exact(source, target)
        file_records[staged_name] = {
            "local": staged_name,
            "base_local": record["local"],
            "base_sha256": record["sha256"],
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "change": "unchanged",
        }

    # Node environment_intent: three additive keys only.
    for node in D_NODES:
        staged_name = "config/node-%d.json" % node
        old = read_json(CANDIDATE_DIR / staged_name)
        new = json.loads(json.dumps(old))
        for key, value in AUDIT_ENV.items():
            new["environment_intent"][key] = value
        write_json(CANDIDATE_DIR / staged_name, new)
        diffs = structural_diff(old, new)
        require(diffs_match(diffs, CANDIDATE_NODE_DIFFS), "candidate node %d diffs %r" % (node, diffs), failures)
        file_records[staged_name].update(
            {
                "sha256": sha256_file(CANDIDATE_DIR / staged_name),
                "bytes": (CANDIDATE_DIR / staged_name).stat().st_size,
                "change": "three_additive_environment_intent_keys",
            }
        )

    # Supervisor: new run_id plus the same three model_env keys.
    for node in D_NODES:
        staged_name = "supervisor-config/node-%d.json" % node
        old = read_json(CANDIDATE_DIR / staged_name)
        new = json.loads(json.dumps(old))
        new["run_id"] = CANDIDATE_RUN_ID
        for key, value in AUDIT_ENV.items():
            new["model_env"][key] = value
        write_json(CANDIDATE_DIR / staged_name, new)
        diffs = structural_diff(old, new)
        require(diffs_match(diffs, CANDIDATE_SUPERVISOR_DIFFS), "candidate supervisor %d diffs %r" % (node, diffs), failures)
        file_records[staged_name].update(
            {
                "sha256": sha256_file(CANDIDATE_DIR / staged_name),
                "bytes": (CANDIDATE_DIR / staged_name).stat().st_size,
                "change": "run_id_and_three_additive_model_env_keys",
            }
        )

    # Sitecustomize: exact old byte prefix plus the SHA-pinned v2 activation.
    old_sitecustomize = (CANDIDATE_DIR / "python/sitecustomize.py").read_bytes()
    appended = SITECUSTOMIZE_APPEND.format(audit_sha=audit_sha).encode("ascii")
    if not old_sitecustomize.endswith(b"\n"):
        appended = b"\n" + appended
    new_sitecustomize = old_sitecustomize + appended
    (CANDIDATE_DIR / "python/sitecustomize.py").write_bytes(new_sitecustomize)
    require(new_sitecustomize.startswith(old_sitecustomize), "candidate sitecustomize does not preserve old byte prefix", failures)
    file_records["python/sitecustomize.py"].update(
        {
            "sha256": sha256_bytes(new_sitecustomize),
            "bytes": len(new_sitecustomize),
            "change": "append_sha_pinned_v2_audit_activate",
            "prefix_bytes": len(old_sitecustomize),
            "prefix_sha256": sha256_bytes(old_sitecustomize),
        }
    )

    # Copy the corrected audit hook into the staged python directory.
    target = CANDIDATE_DIR / "python/glm53_C32_L17_eager_input_audit_v2.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(audit_bytes)
    require(sha256_file(target) == audit_sha, "staged v2 audit SHA mismatch", failures)
    file_records["python/glm53_C32_L17_eager_input_audit_v2.py"] = {
        "local": "python/glm53_C32_L17_eager_input_audit_v2.py",
        "base_local": None,
        "base_sha256": None,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "change": "new_source_pinned_v2_audit_hook",
    }

    # Explicit provenance checks against the immutable rollback-v1 node configs.
    rollback_files = read_json(ROLLBACK_STAGE)["bundles"][0]["files"]
    for node in D_NODES:
        node_name = "config/node-%d.json" % node
        supervisor_name = "supervisor-config/node-%d.json" % node
        old_node = read_json(PROJECT_ROOT / rollback_files[node_name]["local"])
        new_node = read_json(CANDIDATE_DIR / node_name)
        require(diffs_match(structural_diff(old_node, new_node), CANDIDATE_NODE_DIFFS), "candidate node %d provenance diff" % node, failures)
        require(new_node["launch_args"] == old_node["launch_args"], "candidate node %d CLI changed" % node, failures)
        for field in (
            "parallel_intent",
            "mtp_intent",
            "cache_budget_intent",
            "moe_dispatch_intent",
            "hicache_host_io_intent",
            "standalone_pool_intent",
            "deployment_intent",
            "meta_eviction_intent",
            "target_graph_alignment_intent",
            "checksum_flag_disabled_reason",
        ):
            require(new_node[field] == old_node[field], "candidate node %d changed %s" % (node, field), failures)
        old_env = old_node["environment_intent"]
        new_env = new_node["environment_intent"]
        for key, value in old_env.items():
            require(new_env.get(key) == value, "candidate node %d lost env %s" % (node, key), failures)

        provider_name = "provider-config/node-%d.json" % node
        require(
            sha256_file(CANDIDATE_DIR / provider_name) == file_records[provider_name]["base_sha256"],
            "provider config %d changed" % node,
            failures,
        )
        provider = read_json(CANDIDATE_DIR / provider_name)
        require("local_service_id" not in provider, "provider config %d has local_service_id" % node, failures)
        require(provider.get("dram_size") == "160GB", "provider config %d dram" % node, failures)
        require(provider.get("max_dram_size") == "160GB", "provider config %d max dram" % node, failures)
        require(provider.get("init_bm") is True, "provider config %d init_bm" % node, failures)

        old_sup = read_json(PROJECT_ROOT / rollback_files[supervisor_name]["local"])
        new_sup = read_json(CANDIDATE_DIR / supervisor_name)
        require(new_sup["run_id"] == CANDIDATE_RUN_ID, "candidate run_id %d" % node, failures)
        require(diffs_match(structural_diff(old_sup, new_sup), CANDIDATE_SUPERVISOR_DIFFS), "candidate supervisor %d provenance diff" % node, failures)
        for key, value in old_sup["model_env"].items():
            require(new_sup["model_env"].get(key) == value, "candidate supervisor %d lost model_env %s" % (node, key), failures)
        require(new_sup["provider_argv"] == old_sup["provider_argv"], "candidate provider argv %d" % node, failures)
        require(new_sup["model_argv"] == old_sup["model_argv"], "candidate model argv %d" % node, failures)

    # Every unchanged common artifact must remain byte-exact.
    for staged_name, record in file_records.items():
        if staged_name in (
            "config/node-36.json",
            "config/node-41.json",
            "supervisor-config/node-36.json",
            "supervisor-config/node-41.json",
            "python/sitecustomize.py",
            "python/glm53_C32_L17_eager_input_audit_v2.py",
        ):
            continue
        require(
            sha256_file(CANDIDATE_DIR / staged_name) == record["base_sha256"],
            "common artifact changed: %s" % staged_name,
            failures,
        )

    manifest = {
        "project_id": PROJECT_ID,
        "bundle_id": "C32-L17-D-native-eager-v1/candidate-v2",
        "variant": "candidate",
        "audit_version": "v2",
        "remote": CANDIDATE_REMOTE,
        "run_id": CANDIDATE_RUN_ID,
        "remote_run_root": "/data/wio/inference-foundry/glm53/runs/" + CANDIDATE_RUN_ID,
        "nodes": D_NODES,
        "image": bundle["image"],
        "base_stage": str(BASE_STAGE.relative_to(PROJECT_ROOT)),
        "base_stage_sha256": base["sha256"],
        "base_remote": bundle["remote"],
        "native_source_manifest_sha256": bundle["native_source_manifest_sha256"],
        "selected_from_base": selected,
        "removed_P_files": REMOVE_P_KEYS,
        "allowed_diffs": {
            "node_environment_intent": CANDIDATE_NODE_DIFFS,
            "supervisor": CANDIDATE_SUPERVISOR_DIFFS,
            "sitecustomize": "append_only_sha_pinned_v2_audit_activate",
            "new_files": ["python/glm53_C32_L17_eager_input_audit_v2.py"],
        },
        "files": file_records,
        "audit_hook": {
            "staged_path": "python/glm53_C32_L17_eager_input_audit_v2.py",
            "source_path": "scripts/glm53_C32_L17_eager_input_audit_v2.py",
            "sha256": audit_sha,
            "bytes": len(audit_bytes),
            "version": "v2",
            "source_module": "sglang.srt.model_executor.runner.eager_runner",
            "source_sha256": EAGER_SHA256,
            "environ_module": "sglang.srt.environ",
            "environ_source_sha256": ENVIRON_SHA256,
            "native_getter": "sglang.srt.environ.envs.SGLANG_EAGER_INPUT_NO_COPY.get()",
            "first_use_semantics": "accepted key only after native get, exactly one original call, complete clone validation and fsynced evidence write; serialized first use; sticky bounded failure; fork reset",
        },
        "supersedes": {
            "superseded_bundle": "C32-L17-D-native-eager-v1/candidate-v1",
            "superseded_stage": str(CANDIDATE_V1_STAGE.relative_to(PROJECT_ROOT)),
            "superseded_stage_sha256": EXPECTED_V1["candidate_stage"],
            "superseded_bundle_manifest_sha256": EXPECTED_V1["candidate_bundle"],
            "superseded_audit": "scripts/glm53_C32_L17_eager_input_audit_v1.py",
            "superseded_audit_sha256": EXPECTED_V1["audit"],
            "rejection_review": str(REVIEW_PATH.relative_to(PROJECT_ROOT)),
            "rejection_review_sha256": EXPECTED_V1["review"],
            "reason": "v1 used a direct module flag getter and fail-open accepted-key semantics; v1 remains unchanged and is not runtime-authorized.",
        },
        "provenance": {
            "base_stage": str(BASE_STAGE.relative_to(PROJECT_ROOT)),
            "base_stage_sha256": base["sha256"],
            "base_bundle_remote": bundle["remote"],
            "base_image": bundle["image"],
            "P_files_removed": REMOVE_P_KEYS,
            "all_base_files_sha_verified": True,
            "candidate_v1_immutable": True,
            "candidate_v1_stage_sha256": EXPECTED_V1["candidate_stage"],
            "candidate_v1_bundle_manifest_sha256": EXPECTED_V1["candidate_bundle"],
            "rollback_v1_immutable": True,
            "rollback_v1_stage_sha256": EXPECTED_V1["rollback_stage"],
            "rollback_v1_bundle_manifest_sha256": EXPECTED_V1["rollback_bundle"],
        },
        "forbidden_changes": [
            "No P config/provider/supervisor file is part of this D-only bundle.",
            "No CLI, precision, Graph, MTP, admission, HCCL, TASK_QUEUE, SDMA or pool value changes.",
            "No provider config or service-ID change; no guessed local_service_id.",
            "No system, driver, SDK, model, weight, Meta or P-cache change.",
            "No lifecycle launcher or deployment is produced by this local bundle.",
            "No optional extension or binary artifact is part of this candidate.",
        ],
        "runtime_authorized": False,
        "review_status": "PENDING_PRIMARY_REVIEW",
    }
    write_json(CANDIDATE_DIR / "bundle-manifest.json", manifest)
    return manifest


def build_stage(manifest):
    files = {}
    for staged_name, record in manifest["files"].items():
        files[staged_name] = {
            "local": str((CANDIDATE_DIR / staged_name).relative_to(PROJECT_ROOT)),
            "sha256": record["sha256"],
        }
    stage = {
        "project_id": PROJECT_ID,
        "phase": "L17_D_NATIVE_EAGER_CANDIDATE_V2_LOCAL_PREPARATION",
        "rollback_reference": str(ROLLBACK_STAGE.relative_to(PROJECT_ROOT)),
        "audit_version": "v2",
        "audit_hook": str(AUDIT_MODULE.relative_to(PROJECT_ROOT)),
        "audit_hook_sha256": manifest["audit_hook"]["sha256"],
        "environ_source_sha256": ENVIRON_SHA256,
        "eager_source_sha256": EAGER_SHA256,
        "supersedes": manifest["supersedes"],
        "bundles": [
            {
                "framework": "sglang",
                "nodes": D_NODES,
                "remote": manifest["remote"],
                "image": manifest["image"],
                "native_source_manifest_sha256": manifest["native_source_manifest_sha256"],
                "files": files,
                "runtime_run_id": manifest["run_id"],
                "remote_run_root": manifest["remote_run_root"],
                "variant": manifest["variant"],
                "audit_version": "v2",
                "audit_hook_sha256": manifest["audit_hook"]["sha256"],
                "environ_source_sha256": ENVIRON_SHA256,
                "eager_source_sha256": EAGER_SHA256,
                "bundle_manifest": str((CANDIDATE_DIR / "bundle-manifest.json").relative_to(PROJECT_ROOT)),
                "bundle_manifest_sha256": sha256_file(CANDIDATE_DIR / "bundle-manifest.json"),
                "allowed_diffs": manifest["allowed_diffs"],
                "provenance": manifest["provenance"],
                "runtime_authorized": False,
                "review_status": "PENDING_PRIMARY_REVIEW",
            }
        ],
    }
    write_json(CANDIDATE_STAGE, stage)
    return stage


def main():
    failures = []
    if CANDIDATE_DIR.exists():
        failures.append("candidate-v2 already exists: %s" % CANDIDATE_DIR)
    if CANDIDATE_STAGE.exists():
        failures.append("candidate stage v2 already exists: %s" % CANDIDATE_STAGE)
    if not AUDIT_MODULE.is_file():
        failures.append("missing v2 audit module: %s" % AUDIT_MODULE)
    verify_immutable_inputs(failures)
    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        return 1

    base = build_base_inventory(failures)
    audit_sha = sha256_file(AUDIT_MODULE)
    audit_bytes = AUDIT_MODULE.read_bytes()
    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        return 1

    manifest = build_candidate(base, audit_sha, audit_bytes, failures)
    stage = build_stage(manifest)

    # Independent stage-level verification.
    stage_files = stage["bundles"][0]["files"]
    require(sorted(stage["bundles"][0]["nodes"]) == D_NODES, "stage nodes", failures)
    require(stage["audit_version"] == "v2", "stage audit version", failures)
    require(stage["audit_hook_sha256"] == audit_sha, "stage audit SHA", failures)
    require(stage["environ_source_sha256"] == ENVIRON_SHA256, "stage environ SHA", failures)
    require(stage["eager_source_sha256"] == EAGER_SHA256, "stage eager SHA", failures)
    require(len(stage_files) == len(manifest["files"]), "stage file count", failures)
    for staged_name, meta in stage_files.items():
        local = PROJECT_ROOT / meta["local"]
        require(local.is_file(), "stage local missing: %s" % meta["local"], failures)
        if local.is_file():
            require(sha256_file(local) == meta["sha256"], "stage SHA mismatch: %s" % staged_name, failures)
            require(manifest["files"][staged_name]["sha256"] == meta["sha256"], "stage/manifest SHA disagreement: %s" % staged_name, failures)
    for forbidden in REMOVE_P_KEYS:
        require(forbidden not in stage_files, "P file staged: %s" % forbidden, failures)
    for required in (
        "provider-config/node-36.json",
        "provider-config/node-41.json",
        "supervisor-config/node-36.json",
        "supervisor-config/node-41.json",
        "python/glm53_C32_L17_eager_input_audit_v2.py",
    ):
        require(required in stage_files, "required stage file missing: %s" % required, failures)

    for node in D_NODES:
        node_config = read_json(CANDIDATE_DIR / ("config/node-%d.json" % node))
        supervisor = read_json(CANDIDATE_DIR / ("supervisor-config/node-%d.json" % node))
        for key, value in AUDIT_ENV.items():
            require(node_config["environment_intent"][key] == value, "candidate node env %s/%s" % (node, key), failures)
            require(supervisor["model_env"][key] == value, "candidate supervisor env %s/%s" % (node, key), failures)

    old_site = (PROJECT_ROOT / "config/optimization-overlays/C32-target-graph-fullmesh-v1/python/sitecustomize_v2.py").read_bytes()
    new_site = (CANDIDATE_DIR / "python/sitecustomize.py").read_bytes()
    require(new_site.startswith(old_site), "sitecustomize byte prefix", failures)
    require(audit_sha.encode("ascii") in new_site[len(old_site):], "sitecustomize v2 audit SHA", failures)
    require(b"glm53_C32_L17_eager_input_audit_v2" in new_site[len(old_site):], "sitecustomize v2 import", failures)
    require(b"import torch" not in new_site[len(old_site):], "sitecustomize appended eager torch import", failures)

    # Post-build immutable input check.
    verify_immutable_inputs(failures)

    validation = {
        "project_id": PROJECT_ID,
        "preparation_scope": "LOCAL_ONLY_CANDIDATE_V2_BUILD_AND_HASH_DIFF_VALIDATION",
        "base_stage": base["path"],
        "base_stage_sha256": base["sha256"],
        "base_bundle_remote": base["bundle"]["remote"],
        "candidate_stage": str(CANDIDATE_STAGE.relative_to(PROJECT_ROOT)),
        "candidate_stage_sha256": sha256_file(CANDIDATE_STAGE),
        "candidate_bundle": str(CANDIDATE_DIR.relative_to(PROJECT_ROOT)),
        "candidate_bundle_manifest_sha256": sha256_file(CANDIDATE_DIR / "bundle-manifest.json"),
        "audit_hook": {
            "path": str(AUDIT_MODULE.relative_to(PROJECT_ROOT)),
            "sha256": audit_sha,
            "bytes": len(audit_bytes),
            "version": "v2",
            "eager_source_sha256": EAGER_SHA256,
            "environ_source_sha256": ENVIRON_SHA256,
        },
        "supersedes": manifest["supersedes"],
        "candidate_file_count": len(manifest["files"]),
        "removed_P_files": REMOVE_P_KEYS,
        "candidate_allowed_diffs": manifest["allowed_diffs"],
        "rollback_reference": str(ROLLBACK_STAGE.relative_to(PROJECT_ROOT)),
        "rollback_stage_sha256": sha256_file(ROLLBACK_STAGE),
        "rollback_bundle_manifest_sha256": sha256_file(ROLLBACK_DIR / "bundle-manifest.json"),
        "candidate_v1_stage_sha256": sha256_file(CANDIDATE_V1_STAGE),
        "candidate_v1_bundle_manifest_sha256": sha256_file(CANDIDATE_V1_DIR / "bundle-manifest.json"),
        "checks": {
            "base_all_sha_match": not any("base SHA differs" in failure for failure in failures),
            "candidate_node_diff_exact": not any("candidate node" in failure and "diffs" in failure for failure in failures),
            "candidate_supervisor_diff_exact": not any("candidate supervisor" in failure and "diffs" in failure for failure in failures),
            "provider_configs_unchanged": not any("provider config" in failure for failure in failures),
            "no_P_files": not any("P file staged" in failure for failure in failures),
            "sitecustomize_prefix_preserved": not any("sitecustomize byte prefix" in failure for failure in failures),
            "sitecustomize_v2_audit_sha": not any("sitecustomize v2 audit SHA" in failure for failure in failures),
            "stage_hashes_match": not any("stage SHA mismatch" in failure for failure in failures),
            "candidate_v1_immutable": not any("candidate-v1" in failure for failure in failures),
            "rollback_v1_immutable": not any("rollback-v1" in failure for failure in failures),
        },
        "failures": failures,
        "pass": not failures,
        "runtime_authorized": False,
        "review_status": "PENDING_PRIMARY_REVIEW",
        "model_requests": 0,
        "SSH": 0,
        "network": 0,
        "NPU": 0,
        "performance_certificate_allowed": False,
    }
    write_json(VALIDATION_PATH, validation)
    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        print("validation: %s" % VALIDATION_PATH, file=sys.stderr)
        return 1
    print("PASS candidate-v2=%d files" % len(manifest["files"]))
    print("candidate_stage_sha256=%s" % sha256_file(CANDIDATE_STAGE))
    print("candidate_bundle_manifest_sha256=%s" % sha256_file(CANDIDATE_DIR / "bundle-manifest.json"))
    print("validation: %s" % VALIDATION_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
