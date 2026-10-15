#!/usr/bin/env python3
"""memcache-20261014 :: apply the Ascend MemCache HiCache-L3 port to an installed SGLang tree.

This is a MINIMAL port of sgl-project/sglang PR #26043 (head
08246a9e20fef1d75b4ee6c3d9646c415f215a54) onto the tree that is actually installed in the
`sglang-tianjin:arm64-cann9.0.0-a3-v0.5.16` image (sglang 0.5.17.dev386+gc5bd3d7dc). It does
NOT merge the PR's branch and does NOT upgrade SGLang: every edit is an exact-anchor
replacement against the installed sources and fails loudly if the anchor is missing.

Beyond the PR, one container-specific adaptation is required (see E7): the installed
DSA indexer host pool rejects the `page_first_kv_split` layout that NPU+MLA selection
forces, which would abort HiCache startup for GLM-5.3 (glm_moe_dsa).

Usage (inside the container):
    python3 mc-apply-patch.py --sglang-root /sgl-workspace/sglang/python/sglang [--dry-run]
    python3 mc-apply-patch.py --revert
"""
import argparse
import hashlib
import os
import shutil
import sys

STORE_DELTA_MARKER = "INDEX_STATE_DELTA"

NEW_FILES = [
    "mem_cache/storage/ascend_memcache/__init__.py",
    "mem_cache/storage/ascend_memcache/ascend_memcache_store.py",
    "mem_cache/storage/ascend_memcache/start_meta_service.py",
    "mem_cache/storage/ascend_memcache/README.md",
]

# (edit_id, relative path, anchor, replacement)  -- replacement must contain the anchor
EDITS = []


def edit(eid, rel, anchor, replacement, note=""):
    EDITS.append((eid, rel, anchor, replacement, note))


# ---- E1 environ.py: the two new env knobs -------------------------------------------
edit(
    "E1-environ",
    "srt/environ.py",
    '    MOONCAKE_TENANT_ID = EnvStr("default")\n',
    '    MOONCAKE_TENANT_ID = EnvStr("default")\n'
    "\n"
    "    # Ascend MemCache (HiCache L3); see https://gitcode.com/Ascend/memcache\n"
    "    SGLANG_HICACHE_MEMCACHE_CONFIG_PATH = EnvStr(None)\n"
    "    SGLANG_ASCEND_MEMCACHE_ENABLE_WARMUP = EnvBool(False)\n",
    "env knobs used by the backend and by the PR's start_meta_service launcher",
)

# ---- E2 backend_factory.py: builtin branch + registration ---------------------------
edit(
    "E2a-factory-branch",
    "srt/mem_cache/storage/backend_factory.py",
    '        elif backend_name == "simm":\n'
    "            return backend_class(storage_config, mem_pool_host)\n"
    '        elif backend_name == "mori":\n',
    '        elif backend_name == "simm":\n'
    "            return backend_class(storage_config, mem_pool_host)\n"
    '        elif backend_name == "ascend_memcache":\n'
    "            return backend_class(storage_config, mem_pool_host)\n"
    '        elif backend_name == "mori":\n',
    "construct AscendMemcacheStore like the other zero-copy stores",
)
edit(
    "E2b-factory-register",
    "srt/mem_cache/storage/backend_factory.py",
    "StorageBackendFactory.register_backend(\n"
    '    "hf3fs",\n',
    "StorageBackendFactory.register_backend(\n"
    '    "ascend_memcache",\n'
    '    "sglang.srt.mem_cache.storage.ascend_memcache.ascend_memcache_store",\n'
    '    "AscendMemcacheStore",\n'
    ")\n"
    "\n"
    "StorageBackendFactory.register_backend(\n"
    '    "hf3fs",\n',
    "register the backend name",
)

# ---- E3 cache_controller: route through the zero-copy v1/v2 page path ---------------
edit(
    "E3-zerocopy-list",
    "srt/managers/cache_controller.py",
    '                in ["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]\n',
    '                in [\n'
    '                    "hf3fs",\n'
    '                    "mooncake",\n'
    '                    "ascend_memcache",\n'
    '                    "eic",\n'
    '                    "nixl",\n'
    '                    "simm",\n'
    '                    "mori",\n'
    '                ]\n',
    "without this the generic (non-zero-copy) page path is used and the store sees no keys",
)

# ---- E4 MLA host pool: page_first_kv_split in get_page_buffer_meta ------------------
edit(
    "E4-mla-kvsplit-meta",
    "srt/mem_cache/pool_host/mla.py",
    "        assert len(indices) % self.page_size == 0\n"
    "        ptr_list = []\n"
    "        kv_buffer_data_ptr = self.kv_buffer.data_ptr()\n"
    "        indices = indices.tolist()\n"
    '        if self.layout == "layer_first":\n',
    "        assert len(indices) % self.page_size == 0\n"
    "        ptr_list = []\n"
    "        indices = indices.tolist()\n"
    '        if self.layout == "page_first_kv_split":\n'
    "            # Ascend MLA/DSA keeps K and V in separate per-page tensors\n"
    "            # (page_num, layer_num, page_size, 1, dim). When the model has a sparse\n"
    "            # index head (GLM-5.3 / DeepSeek DSA) the index state lives in its OWN\n"
    "            # tensor as well, so each logical page maps to THREE storage objects.\n"
    "            # Omitting the index component would restore KV without its DSA index\n"
    "            # state, which changes attention results - hence the third tuple.\n"
    "            k_buffer_data_ptr = self.k_buffer.data_ptr()\n"
    "            v_buffer_data_ptr = self.v_buffer.data_ptr()\n"
    "            idx_buf = getattr(self, \"index_k_buffer\", None)\n"
    "            index_k_data_ptr = idx_buf.data_ptr() if idx_buf is not None else None\n"
    "            index_head_dim = idx_buf.shape[-1] if idx_buf is not None else 0\n"
    "            for index in range(0, len(indices), self.page_size):\n"
    "                ptr_list.append(\n"
    "                    k_buffer_data_ptr\n"
    "                    + indices[index] * self.layer_num * self.kv_lora_rank * self.dtype.itemsize\n"
    "                )\n"
    "                ptr_list.append(\n"
    "                    v_buffer_data_ptr\n"
    "                    + indices[index] * self.layer_num * self.qk_rope_head_dim * self.dtype.itemsize\n"
    "                )\n"
    "                if index_k_data_ptr is not None:\n"
    "                    ptr_list.append(\n"
    "                        index_k_data_ptr\n"
    "                        + indices[index] * self.layer_num * index_head_dim * self.dtype.itemsize\n"
    "                    )\n"
    "            k_element_size = (\n"
    "                self.layer_num * self.dtype.itemsize * self.page_size * self.kv_lora_rank\n"
    "            )\n"
    "            v_element_size = (\n"
    "                self.layer_num * self.dtype.itemsize * self.page_size * self.qk_rope_head_dim\n"
    "            )\n"
    "            i_element_size = (\n"
    "                self.layer_num * self.dtype.itemsize * self.page_size * index_head_dim\n"
    "            )\n"
    "            element_size_list = []\n"
    "            for _ in range(0, len(indices), self.page_size):\n"
    "                element_size_list.append(k_element_size)\n"
    "                element_size_list.append(v_element_size)\n"
    "                if index_k_data_ptr is not None:\n"
    "                    element_size_list.append(i_element_size)\n"
    "            return ptr_list, element_size_list\n"
    "\n"
    "        kv_buffer_data_ptr = self.kv_buffer.data_ptr()\n"
    '        if self.layout == "layer_first":\n',
    "PR's kv-split pointer layout, EXTENDED with the DSA index-state component: the PR emits\n"
    "only k/v, so a restore would bring back KV without the sparse index state and change\n"
    "attention results. Kept consistent with the store key layout (see the store delta).",
)


# ---- E5 server_args: allow the backend name + layout compatibility ------------------
edit(
    "E5a-backend-choice",
    "srt/server_args.py",
    '            choices=[\n'
    '                "file",\n'
    '                "mooncake",\n'
    '                "hf3fs",\n',
    '            choices=[\n'
    '                "file",\n'
    '                "mooncake",\n'
    '                "ascend_memcache",\n'
    '                "hf3fs",\n',
    "so --hicache-storage-backend ascend_memcache parses",
)
edit(
    "E5b-layout-compat",
    "srt/server_args.py",
    '            self.hicache_storage_backend != "mooncake"\n'
    '            or self.hicache_mem_layout != "layer_first"\n',
    '            self.hicache_storage_backend not in ("mooncake", "ascend_memcache")\n'
    '            or self.hicache_mem_layout != "layer_first"\n',
    "same layer_first guard as mooncake",
)

# ---- E6 npu utils: honour an explicit io backend, cover decode offload --------------
edit(
    "E6-npu-defaults",
    "srt/hardware_backend/npu/utils.py",
    "    # handles hierarchical cache configs\n"
    "    if args.enable_hierarchical_cache:\n"
    '        args.hicache_io_backend = "kernel_ascend"\n'
    "        if args.use_mla_backend():\n"
    '            args.hicache_mem_layout = "page_first_kv_split"\n'
    "        else:\n"
    '            args.hicache_mem_layout = "page_first_direct"\n',
    "    # handles hierarchical cache configs\n"
    "    if (\n"
    "        args.enable_hierarchical_cache\n"
    "        or args.disaggregation_decode_enable_offload_kvcache\n"
    "    ):\n"
    "        # Respect an explicit user choice; only upgrade the generic default.\n"
    '        if args.hicache_io_backend == "kernel":\n'
    '            args.hicache_io_backend = "kernel_ascend"\n'
    "        if args.use_mla_backend():\n"
    '            args.hicache_mem_layout = "page_first_kv_split"\n'
    "        else:\n"
    '            args.hicache_mem_layout = "page_first_direct"\n',
    "PR behaviour: explicit --hicache-io-backend kernel_ascend survives",
)

# ---- (removed) E7 DSAIndexerPoolHost layout edits --------------------------------
# An earlier draft also taught DSAIndexerPoolHost to accept page_first_kv_split. That is NOT
# required here and was removed to keep the port minimal: NPUMLATokenToKVPool (the real
# Ascend DSA pool) is a SIBLING of DSATokenToKVPool, not a subclass, so the hybrid-cache
# assembler's _DsaStrategy (`isinstance(kvcache, DSATokenToKVPool)`) never matches and
# DSAIndexerPoolHost is never constructed. Verified read-only against the installed
# hybrid_pool_assembler.py (see evidence/container-src-19.txt). The DSA index state therefore
# travels only through MLATokenToKVPoolHost.index_k_buffer, which is what the E4 extension and
# the store index-state delta carry.



# ---- E8 scope the legacy HiCache path to Ascend MemCache ----------------------------
# Current DSA defaults to UnifiedRadixCache with HostPoolGroup. The port implements
# direct NPU MLA host pages (K/V/index), rather than the hybrid-pool v2 schema.
edit(
    "E8-ascend-memcache-direct-host-pool",
    "srt/mem_cache/registry.py",
    "        if ctx.is_hybrid_ssm or ctx.is_hybrid_swa or ctx.is_dsa:\n",
    "        if (\n"
    "            ctx.is_hybrid_ssm\n"
    "            or ctx.is_hybrid_swa\n"
    "            or (ctx.is_dsa and server_args.hicache_storage_backend != \"ascend_memcache\")\n"
    "        ):\n",
    "Ascend MemCache DSA uses HiRadixCache with the tested NPU MLA K/V/index host pool",
)


# Definitive post-patch sentinels. An anchor-based "already applied" test is NOT safe for
# pure-insertion edits (the anchor survives the patch), which caused E2a/E5a to be applied
# twice. Each edit therefore names a unique string that only exists AFTER patching.
SENTINELS = {
    "E1-environ": "SGLANG_ASCEND_MEMCACHE_ENABLE_WARMUP = EnvBool(False)",
    "E2a-factory-branch": 'backend_name == "ascend_memcache"',
    "E2b-factory-register": '"sglang.srt.mem_cache.storage.ascend_memcache.ascend_memcache_store"',
    "E3-zerocopy-list": '"ascend_memcache",\n                    "eic",',
    "E4-mla-kvsplit-meta": 'if self.layout == "page_first_kv_split":\n            # Ascend MLA/DSA keeps K and V',
    "E5a-backend-choice": '"mooncake",\n                "ascend_memcache",',
    "E5b-layout-compat": 'not in ("mooncake", "ascend_memcache")',
    "E6-npu-defaults": 'if args.hicache_io_backend == "kernel":',
    "E5c-layout-kernel-ascend": 'elif self.hicache_io_backend == "kernel_ascend":',
    "E5d-backend-help": "file, mooncake, ascend_memcache, hf3fs, nixl, aibrix",
}

# ---- E5c server_args: the kernel_ascend branch of the layout resolver -----------------
# PR #26043 changes this function in TWO places. Porting only the guard (E5b) left
# "ascend_memcache + kernel_ascend + layer_first" silently keeping layer_first, a layout the
# Ascend MLA host pool does not implement. Caught by the real entry-point probe, not by grep.
edit(
    "E5c-layout-kernel-ascend",
    "srt/server_args.py",
    '        elif self.hicache_io_backend == "kernel":\n'
    '            new_layout = "page_first"\n'
    "        else:\n"
    "            # Keep current behavior for unknown backends (e.g., kernel_ascend).\n"
    "            new_layout = self.hicache_mem_layout\n",
    '        elif self.hicache_io_backend == "kernel":\n'
    '            new_layout = "page_first"\n'
    '        elif self.hicache_io_backend == "kernel_ascend":\n'
    "            # Align with the NPU defaults: MLA requires the kv-split layout while\n"
    "            # non-MLA uses the direct page-first layout.\n"
    "            new_layout = (\n"
    '                "page_first_kv_split" if self.use_mla_backend() else "page_first_direct"\n'
    "            )\n"
    "        else:\n"
    "            # Keep current behavior for unknown backends.\n"
    "            new_layout = self.hicache_mem_layout\n",
    "second PR hunk: layer_first + ascend_memcache + kernel_ascend must be rewritten",
)

# ---- E5d server_args: the CLI help text mentions the backend --------------------------
edit(
    "E5d-backend-help",
    "srt/server_args.py",
    "help=\"The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, hf3fs, nixl, aibrix.",
    "help=\"The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, ascend_memcache, hf3fs, nixl, aibrix.",
    "keep the help text truthful about the available backends",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-root", default="/sgl-workspace/sglang/python/sglang")
    ap.add_argument("--src-dir", default=None,
                    help="dir holding the 4 new files (default: alongside this script)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--backup-suffix", default=".prememcache")
    args = ap.parse_args()

    root = args.sglang_root
    if not os.path.isdir(root):
        print(f"FATAL: sglang root not found: {root}")
        return 2
    src_dir = args.src_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ascend_memcache_src"
    )

    print(f"=== mc-apply-patch {'REVERT' if args.revert else 'APPLY'}"
          f"{' (dry-run)' if args.dry_run else ''} root={root} ===")

    if args.revert:
        # Restore EVERY touched file from its .prememcache original (one backup per file,
        # shared by all edits that target it), then remove the added package.
        restored = 0
        for rel in sorted({e[1] for e in EDITS}):
            p = os.path.join(root, rel)
            bak = p + args.backup_suffix
            if os.path.exists(bak):
                if not args.dry_run:
                    shutil.copy2(bak, p)
                    os.remove(bak)
                print(f"  RESTORED {rel} from {os.path.basename(bak)}")
                restored += 1
            else:
                print(f"  NO-BACKUP {rel} (nothing to restore)")
        d = os.path.join(root, "srt/mem_cache/storage/ascend_memcache")
        if os.path.isdir(d) and not args.dry_run:
            shutil.rmtree(d)
            print(f"  REMOVED {d}")
        print(f"=== revert: {restored} files restored ===")
        return 0

    # --- PLAN against CUMULATIVE per-file text; write each file exactly ONCE ------------
    # A previous version stored the original text per edit and wrote that stored text for
    # every edit. With two edits in one file (E2a/E2b in backend_factory.py, E5a/E5b in
    # server_args.py) the second write discarded the first, so a FRESH install silently
    # produced a tree with the factory registration but no constructor branch, or the CLI
    # choice but no layout guard. Reapplying on an already-patched tree hid it. The plan is
    # therefore accumulated per file and each file is written once, after all validation.
    file_text = {}
    plan = []
    for eid, rel, anchor, repl, note in EDITS:
        p = os.path.join(root, rel)
        if rel not in file_text:
            if not os.path.isfile(p):
                print(f"  FAIL {eid}: no such file {p}")
                return 2
            file_text[rel] = open(p, encoding="utf-8").read()
        t = file_text[rel]
        if t.count(repl) == 1:
            plan.append((eid, rel, "ALREADY", note))
            continue
        n_anchor = t.count(anchor)
        if n_anchor == 1:
            file_text[rel] = t.replace(anchor, repl, 1)
            plan.append((eid, rel, "PATCH", note))
        elif n_anchor == 0:
            print(f"  FAIL {eid}: anchor absent from {rel} and the intended full replacement "
                  f"is not present either")
            print(f"       anchor head: {anchor.splitlines()[0][:100]!r}")
            return 2
        else:
            print(f"  FAIL {eid}: anchor occurs {n_anchor} times in {rel}; refusing an ambiguous edit")
            return 2

    # Post-plan invariant: EVERY intended full replacement must be present exactly once.
    # (SENTINEL presence alone is too weak: a generic sentinel can match unrelated code.)
    for eid, rel, anchor, repl, note in EDITS:
        n = file_text[rel].count(repl)
        if n != 1:
            print(f"  FAIL {eid}: after planning, the intended replacement appears {n} times in {rel}")
            return 2

    # Validate the store delta and the new-file sources BEFORE any write, so that a failure
    # here cannot leave a half-patched tree. The delta is always validated against the
    # PRISTINE SOURCE copy (the destination may legitimately be absent or already patched).
    store_rel = "srt/mem_cache/storage/ascend_memcache/ascend_memcache_store.py"
    store_dst = os.path.join(root, store_rel)
    delta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc-store-index-delta.py")
    if not os.path.isfile(delta):
        print(f"  FAIL: delta helper missing: {delta}")
        return 2
    src_store = os.path.join(src_dir, "ascend_memcache_store.py")
    if not os.path.isfile(src_store):
        print(f"  FAIL: missing new-file source {src_store}")
        return 2
    import subprocess
    import tempfile
    # Apply the delta to a TEMP copy of the pristine source. This both proves the kit is
    # applicable (pre-write) and yields the exact expected bytes of the installed store, so
    # the "already installed" test can be an exact hash comparison instead of a marker scan.
    # A marker scan would wrongly keep an older, incomplete delta version.
    with tempfile.TemporaryDirectory() as td:
        tmp_store = os.path.join(td, "ascend_memcache_store.py")
        shutil.copy2(src_store, tmp_store)
        r = subprocess.run([sys.executable, delta, tmp_store], capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL: store index-state delta cannot be applied (pre-write check)")
            print("  " + "\n  ".join((r.stdout or r.stderr).strip().splitlines()))
            return 2
        expected_store_sha = sha256(tmp_store)
        n_delta = sum(1 for ln in (r.stdout or "").splitlines() if ln.strip().startswith("DELTA "))
    print(f"  pre-write delta check OK: {n_delta} delta edits; expected store sha256="
          f"{expected_store_sha[:16]}")
    store_has_delta = (
        os.path.isfile(store_dst) and sha256(store_dst) == expected_store_sha
    )
    if os.path.isfile(store_dst) and not store_has_delta:
        print("  installed store differs from the expected post-delta content -> will be rebuilt")
    for rel in NEW_FILES:
        src = os.path.join(src_dir, os.path.basename(rel))
        if not os.path.isfile(src):
            print(f"  FAIL: missing new-file source {src}")
            return 2

    print(f"  prevalidation: {len(plan)} edits, "
          f"{sum(1 for x in plan if x[2]=='PATCH')} to apply, "
          f"{sum(1 for x in plan if x[2]=='ALREADY')} already applied")

    if args.dry_run:
        for eid, rel, action, note in plan:
            print(f"  WOULD-{'PATCH' if action=='PATCH' else 'SKIP'} {eid} {rel}  ({note})")
        print("  [dry-run] new files + store delta would be installed after this validation")
        print(f"=== edits applied=0 (dry-run) total={len(EDITS)} ===")
        return 0

    # --- VALIDATED: now perform the writes ---------------------------------------------
    ok = already = 0
    for eid, rel, action, note in plan:
        if action == "ALREADY":
            print(f"  SKIP {eid}: intended replacement already present ({rel})")
            already += 1
            continue
        ok += 1
    for rel, text in file_text.items():
        p = os.path.join(root, rel)
        cur = open(p, encoding="utf-8").read()
        if cur == text:
            continue
        bak = p + args.backup_suffix
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        open(p, "w", encoding="utf-8").write(text)
        print(f"  WROTE {rel} (cumulative, {len(text)} bytes)")
    for eid, rel, action, note in plan:
        if action == "PATCH":
            print(f"  PATCH {eid} {rel}  ({note})")

    tgt_dir = os.path.join(root, "srt/mem_cache/storage/ascend_memcache")
    for rel in NEW_FILES:
        src = os.path.join(src_dir, os.path.basename(rel))
        dst = os.path.join(tgt_dir, os.path.basename(rel))
        os.makedirs(tgt_dir, exist_ok=True)
        # CRITICAL: the store file carries the index-state delta. Re-copying the pristine
        # source over an already-delta'd destination would silently revert it (this happened:
        # a second installer run wiped the delta while reporting success). Never overwrite a
        # destination that already carries the delta.
        if os.path.basename(rel) == os.path.basename(store_rel) and store_has_delta:
            print(f"  KEEP {os.path.basename(rel)} (index-state delta already present) "
                  f"sha256={sha256(dst)[:16]}")
            continue
        shutil.copy2(src, dst)
        print(f"  COPY {os.path.basename(rel)} -> {dst} sha256={sha256(dst)[:16]}")
    if not store_has_delta:
        r = subprocess.run([sys.executable, delta, store_dst], capture_output=True, text=True)
        print("  " + "\n  ".join((r.stdout or r.stderr).strip().splitlines()))
        if r.returncode != 0:
            print("FATAL: store index-state delta failed after validation")
            return 2
    # POST-CONDITION: the installed store must be byte-identical to the expected post-delta
    # content. Without this the pristine file could be left behind while the installer
    # reports success (which is exactly what a re-copy used to do).
    if sha256(store_dst) != expected_store_sha:
        print("FATAL: post-condition failed - installed store sha256 "
              f"{sha256(store_dst)[:16]} != expected {expected_store_sha[:16]}")
        return 2
    comps = open(store_dst, encoding="utf-8").read().count("_mla_component_suffixes")
    if comps < 2:
        print(f"FATAL: post-condition failed - _mla_component_suffixes appears {comps} times")
        return 2
    print(f"  POST-CONDITION OK: store carries the index-state delta "
          f"({comps} component-suffix references)")

    print("  --- source receipts (sha256, first 16) ---")
    for rel in sorted(file_text):
        print(f"    {sha256(os.path.join(root, rel))[:16]}  {rel}")
    print(f"    {sha256(os.path.join(root, store_rel))[:16]}  {store_rel} (post-delta)")

    print(f"=== edits applied={ok} already={already} total={len(EDITS)} ===")
    print("=== NOTE: module bytecode caches must be invalidated "
          "(remove __pycache__ under the patched packages) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
