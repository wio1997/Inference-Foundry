"""Build and validate the L17-D native-eager candidate/rollback local bundles.

Local only.  The script copies the exact D36/D41 files from the staged v3 base,
adds only the three reviewed audit environment variables to the candidate, and
produces a byte-exact rollback except for the supervisor run_id.  It fails if
any base SHA differs from config/sglang-C32-target-graph-fullmesh-stage-v3.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys

PROJECT_ID = "glm53"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_STAGE = PROJECT_ROOT / "config/sglang-C32-target-graph-fullmesh-stage-v3.json"
AUDIT_MODULE = PROJECT_ROOT / "scripts/glm53_C32_L17_eager_input_audit_v1.py"
OVERLAY_ROOT = PROJECT_ROOT / "config/optimization-overlays/C32-L17-D-native-eager-v1"
CANDIDATE_DIR = OVERLAY_ROOT / "candidate-v1"
ROLLBACK_DIR = OVERLAY_ROOT / "rollback-v1"
CANDIDATE_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-native-eager-stage-v1.json"
ROLLBACK_STAGE = PROJECT_ROOT / "config/sglang-C32-L17-D-original-rollback-stage-v1.json"
VALIDATION_PATH = (
    PROJECT_ROOT / "runs/20261019T-L17-local-preparation-executor-v1/bundle-validation.json"
)

D_NODES = [36, 41]
P_NODES = [34, 35]
REMOVE_P_KEYS = [
    "config/node-34.json",
    "config/node-35.json",
    "provider-config/node-34.json",
    "provider-config/node-35.json",
    "supervisor-config/node-34.json",
    "supervisor-config/node-35.json",
]
CANDIDATE_REMOTE = (
    "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-native-eager-v1-20261019"
)
ROLLBACK_REMOTE = (
    "/data/wio/inference-foundry/glm53/overlays/sglang-C32-L17-D-original-rollback-v1-20261019"
)
CANDIDATE_RUN_ID = "20261019T-L17-D-native-eager-v1"
ROLLBACK_RUN_ID = "20261019T-L17-D-original-rollback-v1"
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
ROLLBACK_SUPERVISOR_DIFFS = ["run_id"]

SITECUSTOMIZE_APPEND = """

if os.environ.get('GLM53_EAGER_INPUT_AUDIT')=='1':
 try:
  if os.environ.get('SGLANG_EAGER_INPUT_NO_COPY')!='1': raise RuntimeError('SGLANG_EAGER_INPUT_NO_COPY is not 1')
  if os.environ.get('GLM53_NODE') not in ('36','41'): raise RuntimeError('GLM53_NODE is not D36/D41')
  import hashlib,pathlib
  audit_path=pathlib.Path(__file__).with_name('glm53_C32_L17_eager_input_audit_v1.py')
  if hashlib.sha256(audit_path.read_bytes()).hexdigest()!='{audit_sha}': raise RuntimeError('L17 audit module SHA mismatch')
  import glm53_C32_L17_eager_input_audit_v1 as eager_input_audit
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
            if key not in old:
                changes.append(child)
            elif key not in new:
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
    base_stage_sha = sha256_file(BASE_STAGE)
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
    require(
        set(REMOVE_P_KEYS).issubset(files),
        "base stage is missing expected P files",
        failures,
    )
    for node in D_NODES:
        for prefix in ("config", "provider-config", "supervisor-config"):
            require(
                "%s/node-%d.json" % (prefix, node) in files,
                "base stage missing %s/node-%d.json" % (prefix, node),
                failures,
            )
    return {
        "path": str(BASE_STAGE.relative_to(PROJECT_ROOT)),
        "sha256": base_stage_sha,
        "bundle": bundle,
        "inventory": inventory,
    }


def build_bundle(bundle_name, destination, base, candidate, audit_sha, failures):
    bundle = base["bundle"]
    inventory = base["inventory"]
    selected = [name for name in bundle["files"] if name not in REMOVE_P_KEYS]
    destination.mkdir(parents=True, exist_ok=False)

    file_records = {}
    for staged_name in selected:
        record = inventory[staged_name]
        source = pathlib.Path(record["source_path"])
        target = destination / staged_name
        copy_exact(source, target)
        file_records[staged_name] = {
            "local": staged_name,
            "base_local": record["local"],
            "base_sha256": record["sha256"],
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "change": "unchanged",
        }

    if candidate:
        # Node environment_intent: three additive keys only.
        for node in D_NODES:
            staged_name = "config/node-%d.json" % node
            old = read_json(destination / staged_name)
            new = json.loads(json.dumps(old))
            for key, value in AUDIT_ENV.items():
                new["environment_intent"][key] = value
            write_json(destination / staged_name, new)
            diffs = structural_diff(old, new)
            require(diffs_match(diffs, CANDIDATE_NODE_DIFFS), "candidate node %d diffs %r" % (node, diffs), failures)
            file_records[staged_name].update(
                {
                    "sha256": sha256_file(destination / staged_name),
                    "bytes": (destination / staged_name).stat().st_size,
                    "change": "three_additive_environment_intent_keys",
                }
            )

        # Supervisor: new run_id plus the same three model_env keys.
        for node in D_NODES:
            staged_name = "supervisor-config/node-%d.json" % node
            old = read_json(destination / staged_name)
            new = json.loads(json.dumps(old))
            new["run_id"] = CANDIDATE_RUN_ID
            for key, value in AUDIT_ENV.items():
                new["model_env"][key] = value
            write_json(destination / staged_name, new)
            diffs = structural_diff(old, new)
            require(
                diffs_match(diffs, CANDIDATE_SUPERVISOR_DIFFS),
                "candidate supervisor %d diffs %r" % (node, diffs),
                failures,
            )
            file_records[staged_name].update(
                {
                    "sha256": sha256_file(destination / staged_name),
                    "bytes": (destination / staged_name).stat().st_size,
                    "change": "run_id_and_three_additive_model_env_keys",
                }
            )

        # sitecustomize: exact old byte prefix plus the SHA-pinned activation.
        old_sitecustomize = (destination / "python/sitecustomize.py").read_bytes()
        appended = SITECUSTOMIZE_APPEND.format(audit_sha=audit_sha).encode("ascii")
        if not old_sitecustomize.endswith(b"\n"):
            appended = b"\n" + appended
        new_sitecustomize = old_sitecustomize + appended
        (destination / "python/sitecustomize.py").write_bytes(new_sitecustomize)
        require(
            new_sitecustomize.startswith(old_sitecustomize),
            "candidate sitecustomize does not preserve old byte prefix",
            failures,
        )
        file_records["python/sitecustomize.py"].update(
            {
                "sha256": sha256_bytes(new_sitecustomize),
                "bytes": len(new_sitecustomize),
                "change": "append_sha_pinned_audit_activate",
                "prefix_bytes": len(old_sitecustomize),
                "prefix_sha256": sha256_bytes(old_sitecustomize),
            }
        )

        # Copy the audit hook into the staged python directory.
        target = destination / "python/glm53_C32_L17_eager_input_audit_v1.py"
        copy_exact(AUDIT_MODULE, target)
        file_records["python/glm53_C32_L17_eager_input_audit_v1.py"] = {
            "local": "python/glm53_C32_L17_eager_input_audit_v1.py",
            "base_local": None,
            "base_sha256": None,
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "change": "new_source_pinned_audit_hook",
        }
    else:
        # Rollback: exact old source/env/CLI; only supervisor run_id changes.
        for node in D_NODES:
            staged_name = "supervisor-config/node-%d.json" % node
            old = read_json(destination / staged_name)
            new = json.loads(json.dumps(old))
            new["run_id"] = ROLLBACK_RUN_ID
            write_json(destination / staged_name, new)
            diffs = structural_diff(old, new)
            require(
                diffs_match(diffs, ROLLBACK_SUPERVISOR_DIFFS),
                "rollback supervisor %d diffs %r" % (node, diffs),
                failures,
            )
            file_records[staged_name].update(
                {
                    "sha256": sha256_file(destination / staged_name),
                    "bytes": (destination / staged_name).stat().st_size,
                    "change": "run_id_only",
                }
            )

    # Explicit original-provenance checks.
    for node in D_NODES:
        node_name = "config/node-%d.json" % node
        supervisor_name = "supervisor-config/node-%d.json" % node
        if candidate:
            old_node = read_json(PROJECT_ROOT / file_records[node_name]["base_local"])
            new_node = read_json(destination / node_name)
            require(
                diffs_match(structural_diff(old_node, new_node), CANDIDATE_NODE_DIFFS),
                "candidate node %d provenance diff" % node,
                failures,
            )
            require(
                new_node["launch_args"] == old_node["launch_args"],
                "candidate node %d CLI changed" % node,
                failures,
            )
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
                require(
                    new_node[field] == old_node[field],
                    "candidate node %d changed %s" % (node, field),
                    failures,
                )
            old_env = old_node["environment_intent"]
            new_env = new_node["environment_intent"]
            for key, value in old_env.items():
                require(new_env.get(key) == value, "candidate node %d lost env %s" % (node, key), failures)
        else:
            require(
                sha256_file(destination / node_name) == file_records[node_name]["base_sha256"],
                "rollback node %d is not byte-exact" % node,
                failures,
            )

        # Provider configs must be byte-exact and carry no guessed service IDs.
        provider_name = "provider-config/node-%d.json" % node
        require(
            sha256_file(destination / provider_name) == file_records[provider_name]["base_sha256"],
            "provider config %d changed" % node,
            failures,
        )
        provider = read_json(destination / provider_name)
        require("local_service_id" not in provider, "provider config %d has local_service_id" % node, failures)
        require(provider.get("dram_size") == "160GB", "provider config %d dram" % node, failures)
        require(provider.get("max_dram_size") == "160GB", "provider config %d max dram" % node, failures)
        require(provider.get("init_bm") is True, "provider config %d init_bm" % node, failures)

        old_sup = read_json(PROJECT_ROOT / file_records[supervisor_name]["base_local"])
        new_sup = read_json(destination / supervisor_name)
        if candidate:
            require(new_sup["run_id"] == CANDIDATE_RUN_ID, "candidate run_id %d" % node, failures)
            require(
                diffs_match(structural_diff(old_sup, new_sup), CANDIDATE_SUPERVISOR_DIFFS),
                "candidate supervisor %d provenance diff" % node,
                failures,
            )
            for key, value in old_sup["model_env"].items():
                require(
                    new_sup["model_env"].get(key) == value,
                    "candidate supervisor %d lost model_env %s" % (node, key),
                    failures,
                )
            require(new_sup["provider_argv"] == old_sup["provider_argv"], "candidate provider argv %d" % node, failures)
            require(new_sup["model_argv"] == old_sup["model_argv"], "candidate model argv %d" % node, failures)
        else:
            require(new_sup["run_id"] == ROLLBACK_RUN_ID, "rollback run_id %d" % node, failures)
            require(
                diffs_match(structural_diff(old_sup, new_sup), ROLLBACK_SUPERVISOR_DIFFS),
                "rollback supervisor %d provenance diff" % node,
                failures,
            )

    # The common artifacts must be byte-exact in both bundles.
    for staged_name, record in file_records.items():
        if staged_name in (
            "config/node-36.json",
            "config/node-41.json",
            "supervisor-config/node-36.json",
            "supervisor-config/node-41.json",
            "python/sitecustomize.py",
        ) or staged_name == "python/glm53_C32_L17_eager_input_audit_v1.py":
            continue
        require(
            sha256_file(destination / staged_name) == record["base_sha256"],
            "common artifact changed: %s" % staged_name,
            failures,
        )

    manifest = {
        "project_id": PROJECT_ID,
        "bundle_id": bundle_name,
        "variant": "candidate" if candidate else "rollback",
        "remote": CANDIDATE_REMOTE if candidate else ROLLBACK_REMOTE,
        "run_id": CANDIDATE_RUN_ID if candidate else ROLLBACK_RUN_ID,
        "remote_run_root": "/data/wio/inference-foundry/glm53/runs/"
        + (CANDIDATE_RUN_ID if candidate else ROLLBACK_RUN_ID),
        "nodes": D_NODES,
        "image": bundle["image"],
        "base_stage": str(BASE_STAGE.relative_to(PROJECT_ROOT)),
        "base_stage_sha256": base["sha256"],
        "base_remote": bundle["remote"],
        "native_source_manifest_sha256": bundle["native_source_manifest_sha256"],
        "selected_from_base": selected,
        "removed_P_files": REMOVE_P_KEYS,
        "allowed_diffs": (
            {
                "node_environment_intent": CANDIDATE_NODE_DIFFS,
                "supervisor": CANDIDATE_SUPERVISOR_DIFFS,
                "sitecustomize": "append_only_sha_pinned_audit_activate",
                "new_files": ["python/glm53_C32_L17_eager_input_audit_v1.py"],
            }
            if candidate
            else {
                "supervisor": ROLLBACK_SUPERVISOR_DIFFS,
                "note": "source/env/CLI byte-exact old v3; only supervisor run_id is new",
            }
        ),
        "files": file_records,
        "audit_hook": (
            {
                "staged_path": "python/glm53_C32_L17_eager_input_audit_v1.py",
                "source_path": "scripts/glm53_C32_L17_eager_input_audit_v1.py",
                "sha256": audit_sha,
                "source_module": "sglang.srt.model_executor.runner.eager_runner",
                "source_sha256": "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5",
            }
            if candidate
            else None
        ),
        "provenance": {
            "base_stage": str(BASE_STAGE.relative_to(PROJECT_ROOT)),
            "base_stage_sha256": base["sha256"],
            "base_bundle_remote": bundle["remote"],
            "base_image": bundle["image"],
            "P_files_removed": REMOVE_P_KEYS,
            "all_base_files_sha_verified": True,
        },
        "forbidden_changes": [
            "No P config/provider/supervisor file is part of this D-only bundle.",
            "No CLI, precision, Graph, MTP, admission, HCCL, TASK_QUEUE, SDMA or pool value changes.",
            "No provider config or service-ID change; no guessed local_service_id.",
            "No system, driver, SDK, model, weight, Meta or P-cache change.",
            "No lifecycle launcher or deployment is produced by this local bundle.",
        ],
        "runtime_authorized": False,
        "review_status": "PENDING_PRIMARY_REVIEW",
    }
    write_json(destination / "bundle-manifest.json", manifest)
    return manifest, file_records


def build_stage(stage_path, bundle_dir, manifest, phase, rollback_reference):
    files = {}
    for staged_name, record in manifest["files"].items():
        files[staged_name] = {
            "local": str((bundle_dir / staged_name).relative_to(PROJECT_ROOT)),
            "sha256": record["sha256"],
        }
    stage = {
        "project_id": PROJECT_ID,
        "phase": phase,
        "rollback_reference": rollback_reference,
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
                "bundle_manifest": str(
                    (bundle_dir / "bundle-manifest.json").relative_to(PROJECT_ROOT)
                ),
                "bundle_manifest_sha256": sha256_file(bundle_dir / "bundle-manifest.json"),
                "allowed_diffs": manifest["allowed_diffs"],
                "provenance": manifest["provenance"],
                "runtime_authorized": False,
                "review_status": "PENDING_PRIMARY_REVIEW",
            }
        ],
    }
    write_json(stage_path, stage)
    return stage


def main():
    failures = []
    if OVERLAY_ROOT.exists():
        failures.append("overlay root already exists: %s" % OVERLAY_ROOT)
    if CANDIDATE_STAGE.exists():
        failures.append("candidate stage already exists: %s" % CANDIDATE_STAGE)
    if ROLLBACK_STAGE.exists():
        failures.append("rollback stage already exists: %s" % ROLLBACK_STAGE)
    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        return 1
    if not AUDIT_MODULE.is_file():
        print("FAIL: missing audit module: %s" % AUDIT_MODULE, file=sys.stderr)
        return 1

    base = build_base_inventory(failures)
    audit_sha = sha256_file(AUDIT_MODULE)
    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        return 1

    candidate_manifest, candidate_files = build_bundle(
        "C32-L17-D-native-eager-v1/candidate-v1",
        CANDIDATE_DIR,
        base,
        True,
        audit_sha,
        failures,
    )
    rollback_manifest, rollback_files = build_bundle(
        "C32-L17-D-native-eager-v1/rollback-v1",
        ROLLBACK_DIR,
        base,
        False,
        audit_sha,
        failures,
    )

    candidate_stage = build_stage(
        CANDIDATE_STAGE,
        CANDIDATE_DIR,
        candidate_manifest,
        "L17_D_NATIVE_EAGER_CANDIDATE_LOCAL_PREPARATION",
        str(ROLLBACK_STAGE.relative_to(PROJECT_ROOT)),
    )
    rollback_stage = build_stage(
        ROLLBACK_STAGE,
        ROLLBACK_DIR,
        rollback_manifest,
        "L17_D_ORIGINAL_ROLLBACK_LOCAL_PREPARATION",
        str(BASE_STAGE.relative_to(PROJECT_ROOT)),
    )

    # Stage-level independent verification.
    for stage_path, bundle_dir, manifest in (
        (CANDIDATE_STAGE, CANDIDATE_DIR, candidate_manifest),
        (ROLLBACK_STAGE, ROLLBACK_DIR, rollback_manifest),
    ):
        stage = read_json(stage_path)
        stage_files = stage["bundles"][0]["files"]
        require(sorted(stage["bundles"][0]["nodes"]) == D_NODES, "stage nodes", failures)
        require(len(stage["bundles"][0]["files"]) == len(manifest["files"]), "stage file count", failures)
        for staged_name, meta in stage_files.items():
            actual_path = PROJECT_ROOT / meta["local"]
            require(actual_path.is_file(), "stage local missing: %s" % meta["local"], failures)
            if actual_path.is_file():
                require(
                    sha256_file(actual_path) == meta["sha256"],
                    "stage SHA mismatch: %s" % staged_name,
                    failures,
                )
                require(
                    manifest["files"][staged_name]["sha256"] == meta["sha256"],
                    "stage manifest SHA disagreement: %s" % staged_name,
                    failures,
                )
        for forbidden in REMOVE_P_KEYS:
            require(forbidden not in stage_files, "P file staged: %s" % forbidden, failures)
        require("provider-config/node-36.json" in stage_files, "D provider 36 staged", failures)
        require("provider-config/node-41.json" in stage_files, "D provider 41 staged", failures)
        require("supervisor-config/node-36.json" in stage_files, "D supervisor 36 staged", failures)
        require("supervisor-config/node-41.json" in stage_files, "D supervisor 41 staged", failures)

    # Candidate-only explicit values.
    for node in D_NODES:
        node_config = read_json(CANDIDATE_DIR / ("config/node-%d.json" % node))
        supervisor = read_json(CANDIDATE_DIR / ("supervisor-config/node-%d.json" % node))
        for key, value in AUDIT_ENV.items():
            require(node_config["environment_intent"][key] == value, "candidate node env %s/%s" % (node, key), failures)
            require(supervisor["model_env"][key] == value, "candidate supervisor env %s/%s" % (node, key), failures)
    old_site = (PROJECT_ROOT / "config/optimization-overlays/C32-target-graph-fullmesh-v1/python/sitecustomize_v2.py").read_bytes()
    new_site = (CANDIDATE_DIR / "python/sitecustomize.py").read_bytes()
    require(new_site.startswith(old_site), "sitecustomize byte prefix", failures)
    require(b"import torch" not in new_site[ len(old_site): ], "sitecustomize appended eager import", failures)

    # Rollback-only explicit values.
    for node in D_NODES:
        require(
            sha256_file(ROLLBACK_DIR / ("config/node-%d.json" % node))
            == sha256_file(PROJECT_ROOT / ("config/optimization-overlays/C32-target-graph-fullmesh-v1/candidate-v3/config/node-%d.json" % node)),
            "rollback node %d byte-exact" % node,
            failures,
        )
        require(
            sha256_file(ROLLBACK_DIR / ("supervisor-config/node-%d.json" % node))
            != sha256_file(PROJECT_ROOT / ("config/optimization-overlays/C32-target-graph-fullmesh-v1/candidate-v3/supervisor-config/node-%d.json" % node)),
            "rollback supervisor %d must differ by run_id" % node,
            failures,
        )

    validation = {
        "project_id": PROJECT_ID,
        "preparation_scope": "LOCAL_ONLY_BUNDLE_BUILD_AND_HASH_DIFF_VALIDATION",
        "base_stage": base["path"],
        "base_stage_sha256": base["sha256"],
        "base_bundle_remote": base["bundle"]["remote"],
        "candidate_stage": str(CANDIDATE_STAGE.relative_to(PROJECT_ROOT)),
        "candidate_stage_sha256": sha256_file(CANDIDATE_STAGE),
        "rollback_stage": str(ROLLBACK_STAGE.relative_to(PROJECT_ROOT)),
        "rollback_stage_sha256": sha256_file(ROLLBACK_STAGE),
        "candidate_bundle": str(CANDIDATE_DIR.relative_to(PROJECT_ROOT)),
        "rollback_bundle": str(ROLLBACK_DIR.relative_to(PROJECT_ROOT)),
        "audit_hook": {
            "path": str(AUDIT_MODULE.relative_to(PROJECT_ROOT)),
            "sha256": audit_sha,
            "bytes": AUDIT_MODULE.stat().st_size,
        },
        "candidate_file_count": len(candidate_files),
        "rollback_file_count": len(rollback_files),
        "removed_P_files": REMOVE_P_KEYS,
        "candidate_allowed_diffs": candidate_manifest["allowed_diffs"],
        "rollback_allowed_diffs": rollback_manifest["allowed_diffs"],
        "checks": {
            "base_all_sha_match": not any("base SHA differs" in failure for failure in failures),
            "candidate_node_diff_exact": not any("candidate node" in failure and "diffs" in failure for failure in failures),
            "candidate_supervisor_diff_exact": not any("candidate supervisor" in failure and "diffs" in failure for failure in failures),
            "rollback_supervisor_diff_exact": not any("rollback supervisor" in failure and "diffs" in failure for failure in failures),
            "rollback_node_byte_exact": not any("rollback node" in failure for failure in failures),
            "provider_configs_unchanged": not any("provider config" in failure for failure in failures),
            "no_P_files": not any("P file staged" in failure for failure in failures),
            "sitecustomize_prefix_preserved": not any("sitecustomize byte prefix" in failure for failure in failures),
            "stage_hashes_match": not any("stage SHA mismatch" in failure for failure in failures),
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
    print("PASS candidate=%d files rollback=%d files" % (len(candidate_files), len(rollback_files)))
    print("validation: %s" % VALIDATION_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
