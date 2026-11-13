"""Create a new, hash-checked four-op vendor subset; never enable it or edit source.

Plan locally with --inventory --registration-root --plan-output. Build on target
by also passing --source-root and --output-root (must not exist). Common host
registration binaries remain broad: manifest explicitly withholds full runtime
isolation acceptance. No NPU, shell, service or operator execution occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess

PINNED_INVENTORY_SHA256 = "6878bcd289f3d62fa4b63309c7aae4adf428b4bfad2004df7894615848368520"
OPS = {"VllmAscendQuantLightningIndexerV2", "VllmAscendQuantLightningIndexerV2Metadata",
       "SparseFlashMla", "SparseFlashMlaMetadata"}
CORE_OPS = {"VllmAscendQuantLightningIndexerV2", "SparseFlashMla"}
CPU_OPS = OPS - CORE_OPS
ROOT_CORE = "op_impl/ai_core/tbe/"
CONFIG = ROOT_CORE + "kernel/config/ascend910_93/"
INFO = ROOT_CORE + "config/ascend910_93/aic-ascend910_93-ops-info.json"
CPU_INFO = "op_impl/cpu/config/cust_aicpu_kernel.json"
OPAPI_OLD = "op_api/lib/libcust_opapi.so"
OPAPI_NEW = "op_api/lib/libsglang_v41_a3_opapi.so"
SHARED = {OPAPI_OLD, ROOT_CORE+"op_tiling/liboptiling.so",
          ROOT_CORE+"op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
          "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so",
          "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"}
HEADERS = {"op_api/include/aclnnop/"+n for n in (
    "aclnn_quant_lightning_indexer_v2_metadata.h", "aclnn_sparse_flash_mla.h",
    "aclnn_sparse_flash_mla_metadata.h", "aclnn_vllm_ascend_quant_lightning_indexer_v2.h")}
API_BASES = ("aclnnVllmAscendQuantLightningIndexerV2", "aclnnVllmAscendQuantLightningIndexerV2Metadata",
             "aclnnSparseFlashMla", "aclnnSparseFlashMlaMetadata")


def sha(data): return hashlib.sha256(data).hexdigest()


def safe_rel(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe inventory/reference path: {value}")
    return str(p)


def json_bytes(value): return (json.dumps(value, indent=2, sort_keys=True)+"\n").encode()


def elf_dynamic(data):
    """Read ELF64 LE dynamic identity without loading/executing the library."""
    if data[:6] != b"\x7fELF\x02\x01": raise ValueError("Expected little-endian ELF64 shared object")
    hdr = struct.unpack_from("<16sHHIQQQIHHHHHH",data)
    machine, phoff, phentsize, phnum = hdr[2],hdr[5],hdr[9],hdr[10]
    if machine != 183: raise ValueError(f"Expected aarch64 ELF machine183; got {machine}")
    segments = [struct.unpack_from("<IIQQQQQQ",data,phoff+i*phentsize) for i in range(phnum)]
    dynamic = next(s for s in segments if s[0] == 2)
    entries = []
    for offset in range(dynamic[2],dynamic[2]+dynamic[5],16):
        tag,value = struct.unpack_from("<qQ",data,offset)
        if tag == 0: break
        entries.append((tag,value))
    string_addr = next(v for t,v in entries if t == 5)
    load = next(s for s in segments if s[0] == 1 and s[3] <= string_addr < s[3]+s[5])
    strings_offset = load[2]+string_addr-load[3]
    def name(index):
        start = strings_offset+index
        return data[start:data.index(b"\0",start)].decode()
    return {"machine":"aarch64", "soname":next((name(v) for t,v in entries if t==14),None),
            "needed":[name(v) for t,v in entries if t==1],
            "has_init_or_init_array":any(t in (12,25) for t,v in entries)}


def make_plan(inventory, registration_root):
    raw = inventory.read_bytes()
    if sha(raw) != PINNED_INVENTORY_SHA256: raise ValueError("Pinned donor inventory SHA mismatch")
    inv = json.loads(raw)
    entries = {safe_rel(x["path"]):x for x in inv["files"]}
    if len(entries) != len(inv["files"]) or len(entries) != inv["file_count"]:
        raise ValueError("Inventory count or duplicate paths")
    transformed = {}
    registration_counts = {}
    def read(rel):
        data = (registration_root/rel).read_bytes()
        item = entries[rel]
        if sha(data) != item["sha256"] or len(data) != item["bytes"]:
            raise ValueError("Registration reference SHA/size mismatch: "+rel)
        return json.loads(data)
    for rel, names in ((INFO,CORE_OPS),(CPU_INFO,CPU_OPS),
                       (CONFIG+"binary_info_config.json",CORE_OPS),
                       (CONFIG+"relocatable_kernel_info_config.json",set())):
        before = read(rel)
        if not names <= before.keys(): raise ValueError("Missing required registrations: "+rel)
        after = {k:v for k,v in before.items() if k in names}
        transformed[rel] = json_bytes(after)
        registration_counts[rel] = {"before":len(before),"after":len(after),"keys":sorted(after)}
    kernel_paths = set()
    binary = json.loads(transformed[CONFIG+"binary_info_config.json"])
    for op in CORE_OPS:
        for row in binary[op]["binaryList"]:
            for field in ("binPath","jsonPath"):
                kernel_paths.add(ROOT_CORE+"kernel/"+safe_rel(row[field]))
    for name in ("sparse_flash_mla", "vllm_ascend_quant_lightning_indexer_v2"):
        rel=CONFIG+name+".json"
        value=read(rel)
        if set(value) != {"binList"} or not value["binList"]: raise ValueError("Unexpected kernel config")
        for row in value["binList"]:
            path=ROOT_CORE+"kernel/"+safe_rel(row["binInfo"]["jsonFilePath"])
            if path not in kernel_paths: raise ValueError("Kernel config/binary index disagree")
            if not all(key.split("/",1)[0] in CORE_OPS for key in row["simplifiedKey"]):
                raise ValueError("Unwanted op in selected binList")
        # Preserve exact bytes for the two already single-op configs.
        transformed[rel]=(registration_root/rel).read_bytes()
    selected=SHARED|HEADERS|kernel_paths|set(transformed)|{"version.info"}
    if not selected <= entries.keys(): raise ValueError("Selected dependencies missing from inventory")
    for value in json.loads(transformed[CPU_INFO]).values():
        if value["opInfo"]["kernelSo"] != "libtransformer_aicpu_kernels.so":
            raise ValueError("Unexpected AICPU dependency")
    files=[]
    for rel in sorted(selected):
        dest=OPAPI_NEW if rel==OPAPI_OLD else rel
        files.append({"source":rel,"output":dest,"source_sha256":entries[rel]["sha256"],
                      "source_bytes":entries[rel]["bytes"],"transformed":rel in transformed})
    plan={"status":"reviewable_four_op_subset_plan_not_enabled",
          "inventory_sha256":sha(raw),"source_image":inv["source_image"],
          "source_inventory_files":len(entries),"source_inventory_bytes":inv["bytes"],
          "allowed_operator_types":sorted(OPS),"registration_counts":registration_counts,
          "selected_files":files,"selected_input_bytes":sum(x["source_bytes"] for x in files),
          "excluded_file_count":len(entries)-len(selected),
          "private_opapi_relative_path":OPAPI_NEW,
          "private_opapi_environment_variable":"SGLANG_V41_A3_OPAPI_LIBRARY",
          "private_dlsym_allowlist":[n+s for n in API_BASES for s in ("","GetWorkspaceSize")],
          "residual_shared_binary_registration":"UNFILTERED proto/tiling/AICPU and opapi binary contents retained; full runtime isolation NOT established",
          "runtime_isolation_validation":"not_run",
          "never_export_from_subset":["op_api/lib/libcust_opapi.so","bin/set_env.bash","scripts/uninstall.sh"]}
    return plan,entries,transformed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--registration-root",type=Path,required=True)
    p.add_argument("--plan-output",type=Path)
    p.add_argument("--source-root",type=Path)
    p.add_argument("--output-root",type=Path)
    p.add_argument("--rewrite-opapi-soname",action="store_true",
                   help="If needed, patchelf ONLY the newly copied private opapi's SONAME")
    a=p.parse_args()
    plan,entries,transformed=make_plan(a.inventory,a.registration_root)
    if a.plan_output:
        if not a.plan_output.name.startswith("vendor_filter"): raise ValueError("Plan filename must start vendor_filter")
        if a.plan_output.exists(): raise FileExistsError(a.plan_output)
        a.plan_output.parent.mkdir(parents=True,exist_ok=True)
        a.plan_output.write_bytes(json_bytes(plan))
    if bool(a.source_root) != bool(a.output_root): raise ValueError("Supply both source-root and output-root")
    if not a.source_root:
        print(json.dumps({"status":plan["status"],"selected_file_count":len(plan["selected_files"]),
                          "selected_input_bytes":plan["selected_input_bytes"],"plan_output":str(a.plan_output)},indent=2))
        return
    source=a.source_root.resolve(strict=True)
    out=a.output_root.absolute()
    if out.exists() or out.is_symlink(): raise FileExistsError("Output must be a new path: "+str(out))
    parent=out.parent.resolve(strict=True)
    out=parent/out.name
    if out.is_relative_to(source) or source.is_relative_to(out): raise ValueError("Source/output must be disjoint")
    # Verify all inventoried source files BEFORE creating output. Resolve any
    # internal symlink, reject links escaping the source, copy as regular files.
    actual={str(x.relative_to(source)) for x in source.rglob("*") if x.is_file()}
    if actual != set(entries): raise ValueError("Full source file set differs from pinned inventory")
    elf={}
    for rel,item in entries.items():
        path=(source/rel).resolve(strict=True)
        if not path.is_relative_to(source) or not path.is_file(): raise ValueError("Source path escaped vendor")
        data=path.read_bytes()
        if len(data)!=item["bytes"] or sha(data)!=item["sha256"]: raise ValueError("Source SHA/size mismatch: "+rel)
        if rel in SHARED: elf[rel]=elf_dynamic(data)
    forbidden_needed={"libcust_opapi.so"}
    if any(forbidden_needed & set(info["needed"]) for info in elf.values()):
        raise ValueError("Shared dependency needs original libcust_opapi.so; selected identity rewrite requires dependency review")
    soname=elf[OPAPI_OLD]["soname"]
    rename_soname=soname not in (None,Path(OPAPI_NEW).name)
    if rename_soname and (not a.rewrite_opapi_soname or not shutil.which("patchelf")):
        raise ValueError("Old SONAME would alias native lookup; review and rerun with --rewrite-opapi-soname and patchelf installed")
    # Atomic mkdir refuses racing creation. On later failure retain an explicit
    # incomplete new directory for review; never delete source or unrelated data.
    out.mkdir(exist_ok=False)
    (out/"vendor_filter_INCOMPLETE.json").write_bytes(json_bytes({"status":"incomplete_do_not_enable"}))
    written=[]
    for item in plan["selected_files"]:
        rel=item["source"]; dest=out/item["output"]
        dest.parent.mkdir(parents=True,exist_ok=True)
        if rel in transformed: dest.write_bytes(transformed[rel])
        else:
            shutil.copy2((source/rel).resolve(),dest,follow_symlinks=True)
            if sha(dest.read_bytes()) != item["source_sha256"]:
                raise ValueError("Copied source changed after preflight: "+rel)
        if rel==OPAPI_OLD and rename_soname:
            subprocess.run([shutil.which("patchelf"),"--set-soname",Path(OPAPI_NEW).name,str(dest)],check=True,
                           capture_output=True,text=True,timeout=30)
        record={**item,"output_bytes":dest.stat().st_size,"output_sha256":sha(dest.read_bytes())}
        written.append(record)
    if (out/OPAPI_OLD).exists(): raise AssertionError("Forbidden original opapi filename exists")
    final_elf=elf_dynamic((out/OPAPI_NEW).read_bytes())
    if final_elf["soname"] not in (None,Path(OPAPI_NEW).name): raise AssertionError("Unsafe output SONAME")
    manifest={**plan,"status":"filtered_vendor_created_NOT_RUNTIME_VALIDATED_NOT_ENABLED",
              "source_root":str(source),"output_root":str(out),"source_elf_dynamic":elf,
              "private_opapi_elf_dynamic":final_elf,"private_soname_rewritten":rename_soname,
              "selected_files":written,"output_payload_bytes":sum(x["output_bytes"] for x in written)}
    (out/"vendor_filter_manifest.json").write_bytes(json_bytes(manifest))
    (out/"vendor_filter_INCOMPLETE.json").unlink()
    print(json.dumps({"status":manifest["status"],"files":len(written),"bytes":manifest["output_payload_bytes"],
                      "manifest":str(out/"vendor_filter_manifest.json"),
                      "private_opapi":str(out/OPAPI_NEW)},indent=2))


if __name__=="__main__":main()
