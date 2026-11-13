"""Audit host-only CPU build and create a NEW vendor; never enable or install.

Exactly 20 old-filter payloads are copied. Only three host-library paths change.
No source/original vendor/filter is edited; no device kernel or opapi is built.
CPU audit cannot establish runtime registration isolation or NPU correctness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import struct
import subprocess
import traceback

OPS = {"VllmAscendQuantLightningIndexerV2", "VllmAscendQuantLightningIndexerV2Metadata",
       "SparseFlashMla", "SparseFlashMlaMetadata"}
PROTO_TYPES = OPS - {"SparseFlashMlaMetadata"}
TILING_TYPES = {"VllmAscendQuantLightningIndexerV2", "SparseFlashMla"}
DATA_TYPES = TILING_TYPES | {"SparseFlashMlaSwaParamsOp", "SparseFlashMlaCmpParamsOp"}
PROTO_REL = "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
TILING_REL = "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
TILING_ALIAS = "op_impl/ai_core/tbe/op_tiling/liboptiling.so"
REPLACED = {PROTO_REL, TILING_REL, TILING_ALIAS}
INPUTS_SHA = "9f6a599eb0bdc8a697db46d8bc7c5946b50094e365a356d25648dcef8a6ded39"
PROTO_TU_SHA = "db81fdeef657cf3fa6ead7e1b9c26b84d4594c2f249fd9b20de7459a3a3a41e6"
# Filled from the exact saved four-op plan and actual donor registration JSON.
PLAN_SHA = "5aeb29abd07ee5cf2d495f68d979a3a5045199ed9eb09a62ffb5e361c01acd42"
FORBIDDEN_OPS = ['AddRmsNormBias', 'AttnResFwd', 'ChunkFwdOVllm', 'ChunkGatedDeltaRuleFwdHVllm', 'Compressor', 'CompressorMetadata', 'CopyAndExpandEagleInputs', 'DequantSituQuant', 'DequantSwigluQuant', 'DispatchFFNCombine', 'DispatchFFNCombineBF16', 'DispatchFFNCombineW4A8', 'FusedLightningIndexerManage', 'FusedQuantLightningIndexerManage', 'FusedScatterCopySparseFlashAttention', 'FusedSparseAttentionOverlap', 'GmmDequantSituQuant', 'GroupedMatmulSwigluQuant', 'GroupedMatmulSwigluQuantV2', 'GroupedMatmulSwigluQuantWeightNzTensorList', 'HcPost', 'HcPre', 'InplacePartialRotaryMul', 'K2qCsrHist', 'K2qCsrMeta', 'K2qCsrRowPrefix', 'K2qCsrScatter', 'K2qCsrTilePrefix', 'KdaGateCumsumVllm', 'KdaLayoutSwap12', 'KvQuantSparseFlashAttentionVllm', 'LightningIndexerQuant', 'MlaPreprocess', 'MoeGatingTopK', 'MoeGatingTopKHash', 'MsaIndexScore', 'RmsNormCast', 'RmsNormDynamicQuant', 'ScatterNdUpdateSk', 'SparseAttentionScore', 'SparseAttnSharedkv', 'SparseFlashAttention', 'StoreKVBlock', 'TransposeKvCacheByBlock']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe payload path: " + str(value))
    return str(path)


def run(argv, cwd, env):
    p = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
    if p.returncode:
        raise RuntimeError("CPU audit failed: " + repr(argv) + "\n" + p.stdout + p.stderr)
    return p.stdout + p.stderr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filter-root", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--recipe-root", type=Path, required=True)
    p.add_argument("--proto-sha256", required=True)
    p.add_argument("--tiling-sha256", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--ld-library-dir", action="append", type=Path, default=[])
    a = p.parse_args()
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Execute on parent-owned target aarch64 CPU container")
    here = Path(__file__).resolve().parent
    old, build, source, recipe = [x.resolve(strict=True) for x in (
        a.filter_root, a.build_dir, a.source_root, a.recipe_root)]
    out = a.output_root.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Only a wholly new output path is allowed: " + str(out))
    out = out.parent.resolve(strict=True) / out.name
    if any(out.is_relative_to(x) or x.is_relative_to(out) for x in (old, build, source, recipe)):
        raise ValueError("Output must be disjoint from every input tree")
    planpath = here / "assembly_g2_plan.json"
    if digest(planpath) != PLAN_SHA:
        raise ValueError("Pinned four-op payload plan changed")
    plan = json.loads(planpath.read_text())
    if set(plan["allowed_operator_types"]) != OPS or len(plan["selected_files"]) != 20:
        raise ValueError("Unexpected four-op plan")
    if digest(here / "inputs_manifest.json") != INPUTS_SHA:
        raise ValueError("Pinned actual source manifest changed")
    inputs = json.loads((here / "inputs_manifest.json").read_text())
    for entry in inputs["files"]:
        file = (source / relative(entry["path"])).resolve(strict=True)
        if not file.is_relative_to(source) or digest(file) != entry["sha256"]:
            raise ValueError("Actual source SHA mismatch: " + entry["path"])
    if digest(recipe / "proto_subset.cpp") != PROTO_TU_SHA:
        raise ValueError("Proto TU differs from exact three-type include list")

    # Exact compiler TU closure, not a broad 'attention/*' or whole archive.
    host = "csrc/attention/"
    tu = {
        host + "quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp",
        host + "quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_infershape.cpp",
        host + "quant_lightning_indexer_v2_metadata/op_host/quant_lightning_indexer_v2_metadata_infershape.cpp",
        host + "sparse_flash_mla/op_host/sparse_flash_mla_tiling.cpp",
        host + "sparse_flash_mla/op_host/sparse_flash_mla_infershape.cpp",
    }
    for name in ("base_checker", "checker_runner", "common_checker", "mask_checker",
                 "metadata_checker", "paged_attention_checker", "seq_len_checker",
                 "sinks_checker", "softmax_lse_checker", "sparse_compression_checker",
                 "sparse_flash_mla_checker"):
        tu.add(host + "sparse_flash_mla/op_host/checkers/" + name + ".cpp")
    expected = {(source / x).resolve() for x in tu} | {(recipe / "proto_subset.cpp").resolve()}
    commands = json.loads((build / "compile_commands.json").read_text())
    actual = []
    for command in commands:
        file = Path(command["file"])
        if not file.is_absolute():
            file = Path(command["directory"]) / file
        actual.append(file.resolve(strict=True))
    if len(actual) != 17 or set(actual) != expected:
        raise ValueError("Compiler translation-unit closure differs: " + repr(actual))
    # Only actual sources entering those TUs and actual proto headers are scanned.
    proto_headers = [source / "vendor_proto/vllm_ascend_quant_lightning_indexer_v2_proto.h",
                     source / "vendor_proto/sparse_flash_mla_proto.h",
                     source / host / "quant_lightning_indexer_v2_metadata/op_graph/quant_lightning_indexer_v2_metadata_proto.h"]
    scan = list(expected) + proto_headers + [
        source / host / "quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.h",
        source / host / "sparse_flash_mla/op_host/sparse_flash_mla_tiling.h"]
    registrations = []
    patterns = {"REG_OP": PROTO_TYPES, "IMPL_OP_INFERSHAPE": PROTO_TYPES,
                "IMPL_OP_OPTILING": TILING_TYPES, "REGISTER_TILING_DATA_CLASS": DATA_TYPES}
    found = {name: set() for name in patterns}
    for file in scan:
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", file.read_text(), flags=re.S)
        for macro, allowed in patterns.items():
            for typ in re.findall(r"\b" + macro + r"\s*\(\s*([A-Za-z_]\w*)", text):
                if typ not in allowed:
                    raise ValueError("Forbidden source registration: " + macro + " " + typ)
                found[macro].add(typ)
                registrations.append({"path": str(file), "sha256": digest(file), "macro": macro, "type": typ})
        if re.search(r"\b(OP_ADD|OP_TYPE_REGISTER)\s*\(", text):
            raise ValueError("Unexpected OpDef/opapi registration in selected host TUs: " + str(file))
    for macro, allowed in patterns.items():
        if found[macro] != allowed:
            raise ValueError("Incomplete registration set for " + macro + ": " + repr(found[macro]))

    env = dict(os.environ)
    env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    if a.ld_library_dir:
        env["LD_LIBRARY_PATH"] = ":".join(str(x.resolve(strict=True)) for x in a.ld_library_dir)
    binaries = {PROTO_REL: build / "libcust_opsproto_rt2.0.so",
                TILING_REL: build / "libcust_opmaster_rt2.0.so"}
    wanted_sha = {PROTO_REL: a.proto_sha256, TILING_REL: a.tiling_sha256}
    evidence, binary_audits = {}, {}
    for rel, file in binaries.items():
        if digest(file) != wanted_sha[rel]:
            raise ValueError("Built host library SHA changed: " + str(file))
        data = file.read_bytes()
        if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 183:
            raise ValueError("Expected little-endian aarch64 ELF64")
        name = file.name
        for tool, flags in (("readelf", ["-d"]), ("nm", ["-D", "-C"]),
                            ("strings", ["-a"]), ("ldd", [])):
            evidence[name + "." + tool + ".txt"] = run([tool, *flags, str(file)], build, env)
        if "not found" in evidence[name + ".ldd.txt"]:
            raise ValueError("Unresolved runtime dependencies: " + name)
        for symbol in ("GetOpImplFunctions", "GetOpImplFunctionsV2"):
            if not re.search(r"[TW] " + symbol + r"$", evidence[name + ".nm.txt"], re.M):
                raise ValueError("Missing private CANN plugin registry entrypoint: " + name + " " + symbol)
        if any("libopp_registry.so" in ln for ln in evidence[name + ".readelf.txt"].splitlines() if "(NEEDED)" in ln):
            raise ValueError("Plugin must carry private static rt2_registry, not shared opp_registry")
        if any(x in evidence[name + ".readelf.txt"] for x in (
                "libcust_opapi.so", "libsglang_v41_a3_opapi.so", "libcust_opsproto_rt2.0.so", "libcust_opmaster_rt2.0.so")):
            # A SONAME is harmless; only DT_NEEDED dependencies are forbidden.
            needed = "\n".join(x for x in evidence[name + ".readelf.txt"].splitlines() if "(NEEDED)" in x)
            if any(x in needed for x in ("libcust_opapi.so", "libsglang_v41_a3_opapi.so",
                                         "libcust_opsproto_rt2.0.so", "libcust_opmaster_rt2.0.so")):
                raise ValueError("Host library depends on broad old vendor library: " + name)
        mapname = "v41_proto.map" if rel == PROTO_REL else "v41_tiling.map"
        evidence[mapname] = (build / mapname).read_text()
        combined = "\n".join(evidence[name + "." + k + ".txt"] for k in ("nm", "strings")) + evidence[mapname]
        bad = [typ for typ in FORBIDDEN_OPS if re.search(r"(?<![A-Za-z_])" + re.escape(typ) + r"(?![A-Za-z_])", combined)]
        log_label_exception = None
        if "SparseFlashAttention" in bad:
            # Actual pinned SparseFlashMla source has one diagnostic OP_LOGE label,
            # not an extra registration. Retain strict source macro/TU checks.
            label = "SparseFlashAttention"
            src = source / "csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.cpp"
            text = src.read_text()
            literal = 'OP_LOGE("SparseFlashAttention", "tiling context is nullptr!");'
            matcher = r"(?<![A-Za-z_])SparseFlashAttention(?![A-Za-z_])"
            if (rel != TILING_REL or text.count('"SparseFlashAttention"') != 1 or
                literal not in text or
                evidence[name + ".strings.txt"].splitlines().count(label) != 1 or
                re.search(matcher, evidence[name + ".nm.txt"] + evidence[mapname])):
                raise ValueError("Unexpected SparseFlashAttention occurrence; only pinned log label allowed")
            log_label_exception = {"text": label, "source": str(src), "sha256": digest(src),
                                   "kind": "single pinned OP_LOGE diagnostic label; no symbol/map/registration"}
            bad.remove(label)
        if bad:
            raise ValueError("Nonallowlisted known op names in host symbols/strings/map: " + repr(bad))
        binary_audits[rel] = {"sha256": digest(file), "bytes": file.stat().st_size,
                              "forbidden_known_type_hits": [], "ldd_no_missing": True, "pinned_diagnostic_label": log_label_exception}

    oldmanifest = json.loads((old / "vendor_filter_manifest.json").read_text())
    if oldmanifest["inventory_sha256"] != plan["inventory_sha256"]:
        raise ValueError("Old filter original inventory differs")
    records = {relative(x["output"]): x for x in oldmanifest["selected_files"]}
    names = {relative(x["output"]) for x in plan["selected_files"]}
    if set(records) != names or {str(x.relative_to(old)) for x in old.rglob("*") if x.is_file()} != names | {"vendor_filter_manifest.json"}:
        raise ValueError("Old filter must have exactly pinned 20 payloads and its manifest")
    for rel in names:
        file = (old / rel).resolve(strict=True)
        if not file.is_relative_to(old) or digest(file) != records[rel]["output_sha256"]:
            raise ValueError("Old filtered payload changed: " + rel)
        planentry = next(x for x in plan["selected_files"] if x["output"] == rel)
        if not planentry["transformed"] and rel != plan["private_opapi_relative_path"] and digest(file) != planentry["source_sha256"]:
            raise ValueError("Untransformed original payload differs: " + rel)
    for rel, counts in plan["registration_counts"].items():
        registration = json.loads((old / relative(rel)).read_text())
        if not isinstance(registration, dict) or set(registration) != set(counts["keys"]):
            raise ValueError("Filtered registration type keys differ: " + rel)
    # All checks precede output creation. Never overwrite or delete an input.
    out.mkdir()
    marker = out / "assembly_g2_INCOMPLETE.json"
    marker.write_text('{"status":"incomplete_do_not_enable"}\n')
    payloads = []
    for rel in sorted(names):
        inputfile = (binaries[PROTO_REL] if rel == PROTO_REL else binaries[TILING_REL]
                     if rel in (TILING_REL, TILING_ALIAS) else old / rel)
        destination = out / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inputfile, destination)
        if digest(destination) != digest(inputfile):
            raise RuntimeError("Output copy changed: " + rel)
        payloads.append({"path": rel, "sha256": digest(destination),
                         "bytes": destination.stat().st_size, "host_replaced": rel in REPLACED,
                         "old_sha256": digest(old / rel),
                         "old_payload_preserved_exactly": rel not in REPLACED and digest(destination) == digest(old / rel)})
    auditdir = out / "assembly_g2_audit"
    auditdir.mkdir()
    for name, text in evidence.items():
        (auditdir / name).write_text(text)
    shutil.copy2(build / "compile_commands.json", auditdir / "compile_commands.json")
    manifest = {"status": "CPU_AUDITED_NEW_VENDOR_NOT_ENABLED_NPU_COEXISTENCE_UNVERIFIED",
                "task_id": "167-v41-strict-host-subset-g1", "generation": 2,
                "source_filter": str(old), "source_filter_manifest_sha256": digest(old / "vendor_filter_manifest.json"),
                "source_build": str(build), "source_root": str(source), "recipe_root": str(recipe),
                "recipe_CMake_sha256": digest(recipe / "CMakeLists.txt"),
                "assembly_script_sha256": digest(Path(__file__)),
                "inputs_manifest_sha256": INPUTS_SHA, "plan_sha256": PLAN_SHA,
                "host_registered_proto_types": sorted(PROTO_TYPES),
                "host_registered_tiling_types": sorted(TILING_TYPES),
                "SparseFlashMlaMetadata": "pure existing opapi/AICPU path; no invented host proto",
                "source_registrations": registrations, "host_binary_audits": binary_audits,
                "payloads": payloads, "payload_count": len(payloads), "replaced_paths": sorted(REPLACED),
                "preserved_payloads_exact": 17, "tiling_two_paths_same_sha": digest(out / TILING_REL) == digest(out / TILING_ALIAS),
                "runtime_acceptance": "pending fresh-process old4+currentcore5+Hash6 eager/multistream graph NPU checks",
                "residual_broad_binaries": ["unchanged private opapi and AICPU binaries retain other code; CPU/static checks do not prove absence of runtime registration side effects"]}
    (out / "strict_host_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    marker.unlink()
    print(json.dumps({"status": manifest["status"], "output": str(out),
                      "manifest": str(out / "strict_host_manifest.json")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
