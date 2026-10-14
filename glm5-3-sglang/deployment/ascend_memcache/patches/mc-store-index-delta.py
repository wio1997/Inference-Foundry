#!/usr/bin/env python3
"""memcache-20261014 :: apply the index-state delta to the ported AscendMemcacheStore.

WHY: sgl-project/sglang PR #26043 emits only K and V objects per page. On Ascend the MLA
device pool for GLM-5.3 is NPUMLATokenToKVPool, which allocates THREE per-page tensors:
k_buffer, v_buffer and index_k_buffer (the latter whenever the model has a sparse index
head, i.e. GLM-5.3 / DeepSeek DSA with index_head_dim=128). The PR registers index_k_buffer
with the SDK but never creates an object key or pointer for it, so a restored prefix would
come back with KV but no DSA index state -> different attention results. This delta makes
the key layout, pointer layout and hit accounting all three-component and consistent.

Usage:
    python3 mc-store-index-delta.py <path to ascend_memcache_store.py> [--revert]
"""
import argparse
import hashlib
import shutil
import sys

DELTA_MARKER = "INDEX_STATE_DELTA"

EDITS = [
    # (id, anchor, replacement)
    (
        "D1-header",
        '"""HiCache L3 storage backend for Ascend MemCache.',
        '"""HiCache L3 storage backend for Ascend MemCache.\n'
        "\n"
        "INDEX_STATE_DELTA (memcache-20261014): ported from sgl-project/sglang PR #26043 with\n"
        "one correctness extension. The PR stores TWO objects per page (K and V). Ascend's MLA\n"
        "device pool NPUMLATokenToKVPool allocates a THIRD per-page tensor, index_k_buffer, for\n"
        "models with a sparse index head (GLM-5.3 / DeepSeek DSA, index_head_dim=128). Writing\n"
        "KV without the matching index state would silently change attention results, so the\n"
        "component list here is k, v, index and it is generated in ONE place\n"
        "(`_mla_component_suffixes`) that the key layout, pointer layout and hit accounting all\n"
        "share.",
    ),
    (
        "D2-init",
        "        self.registered_pools = {}\n",
        "        # INDEX_STATE_DELTA: set for real in register_mem_pool_host().\n"
        "        self.mla_has_index = False\n"
        "        self.registered_pools = {}\n",
    ),
    (
        "D3-register",
        "        if envs.SGLANG_ASCEND_MEMCACHE_ENABLE_WARMUP.get():\n"
        "            self.warmup()\n",
        "        # INDEX_STATE_DELTA: the DSA index tensor is a first-class object component.\n"
        "        self.mla_has_index = (\n"
        "            self._mla_uses_kv_split()\n"
        '            and getattr(self.mem_pool_host, "index_k_buffer", None) is not None\n'
        "        )\n"
        "        logger.info(\n"
        '            "Ascend memcache MLA components: %s (index_state=%s)",\n'
        "            self._mla_component_suffixes(),\n"
        "            self.mla_has_index,\n"
        "        )\n"
        "\n"
        "        if envs.SGLANG_ASCEND_MEMCACHE_ENABLE_WARMUP.get():\n"
        "            self.warmup()\n",
    ),
    (
        "D4-components",
        "    def _get_mla_buffer_meta(self, keys, indices):\n"
        "        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)\n"
        "        key_list = []\n"
        "        for key_ in keys:\n"
        '            key_list.append(f"{key_}_{self.mla_suffix}_k")\n'
        "            if self._mla_uses_kv_split():\n"
        '                key_list.append(f"{key_}_{self.mla_suffix}_v")\n'
        "        assert len(key_list) == len(ptr_list)\n"
        "        return key_list, ptr_list, element_size_list\n",
        "    def _mla_component_suffixes(self):\n"
        "        # INDEX_STATE_DELTA: single source of truth for the per-page component set.\n"
        "        # Keep the order in step with MLATokenToKVPoolHost.get_page_buffer_meta\n"
        "        # (page_first_kv_split -> k, v, [index]).\n"
        "        if not self._mla_uses_kv_split():\n"
        '            return ["k"]\n'
        '        suffixes = ["k", "v"]\n'
        '        if getattr(self, "mla_has_index", False):\n'
        '            suffixes.append("index")\n'
        "        return suffixes\n"
        "\n"
        "    def _mla_key_multiplier(self):\n"
        "        return len(self._mla_component_suffixes())\n"
        "\n"
        "    def _get_mla_buffer_meta(self, keys, indices):\n"
        "        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)\n"
        "        suffixes = self._mla_component_suffixes()\n"
        "        key_list = []\n"
        "        for key_ in keys:\n"
        "            for s in suffixes:\n"
        '                key_list.append(f"{key_}_{self.mla_suffix}_{s}")\n'
        "        assert len(key_list) == len(ptr_list), (\n"
        "            f\"component/pointer mismatch: {len(key_list)} keys vs {len(ptr_list)} ptrs; \"\n"
        "            f\"model index state present={getattr(self, 'mla_has_index', False)}\"\n"
        "        )\n"
        "        return key_list, ptr_list, element_size_list\n",
    ),
    (
        "D5-postprocess",
        "        if key_multiplier is None:\n"
        "            if self.is_mla_backend:\n"
        "                key_multiplier = 2 if self._mla_uses_kv_split() else 1\n"
        "            else:\n",
        "        if key_multiplier is None:\n"
        "            if self.is_mla_backend:\n"
        "                key_multiplier = self._mla_key_multiplier()\n"
        "            else:\n",
    ),
    (
        "D6-batch_get",
        "        if self.is_mla_backend:\n"
        "            key_multiplier = 2 if self._mla_uses_kv_split() else 1\n"
        "        else:\n"
        "            key_multiplier = 2\n",
        "        if self.is_mla_backend:\n"
        "            key_multiplier = self._mla_key_multiplier()\n"
        "        else:\n"
        "            key_multiplier = 2\n",
    ),
    (
        "D9-copy-write-fallback",
        "    def _put_batch_zero_copy_impl(\n"
        "        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]\n"
        "    ) -> List[int]:\n"
        "        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)\n",
        "    def _put_batch_zero_copy_impl(\n"
        "        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]\n"
        "    ) -> List[int]:\n"
        "        # INDEX_STATE_DELTA / measured barrier: `batch_put_from` (zero-copy) writes data\n"
        "        # and makes the object readable IN THE WRITER, but its metadata state sync to the\n"
        "        # MetaService fails (-3102 SyncUpdateState) so a PEER rank never sees the key.\n"
        "        # Cross-node reads therefore returned nothing for zero-copy objects while objects\n"
        "        # written through the copying put() API read back byte-exact at 4 KiB and 1 MiB\n"
        "        # (evidence/xnode-{writer-xn5,reader-35}.log and xnode-big1m-*.log).\n"
        "        # SGLANG_ASCEND_MEMCACHE_COPY_WRITE=1 switches to the copying path, which is the\n"
        "        # tested-safe option when peers must read the objects.\n"
        "        import ctypes as _ctypes\n"
        "        import os as _os\n"
        "        if _os.environ.get(\"SGLANG_ASCEND_MEMCACHE_COPY_WRITE\", \"0\") not in (\"0\", \"\", \"false\", \"False\"):\n"
        "            out: List[int] = []\n"
        "            for k, ptr, sz in zip(key_strs, buffer_ptrs, buffer_sizes):\n"
        "                try:\n"
        "                    buf = (_ctypes.c_char * int(sz)).from_address(int(ptr))\n"
        "                    out.append(int(self.store.put(k, buf)))\n"
        "                except Exception as exc:  # noqa: BLE001\n"
        "                    logger.error(\"copy-write put failed for %s: %s\", k, exc)\n"
        "                    out.append(-1)\n"
        "            return out\n"
        "        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)\n",
    ),
    (
        "D10-atexit-clean-close",
        "            self._init_runtime_fields(storage_config)\n",
        "            self._init_runtime_fields(storage_config)\n"
        "            self._install_atexit_clean_close()\n",
    ),
    (
        "D10b-atexit-method",
        "    def _init_runtime_fields(\n"
        "        self, storage_config: Optional[HiCacheStorageConfig]\n"
        "    ) -> None:\n"
        "        self.enable_storage_metrics = False\n",
        "    def _install_atexit_clean_close(self) -> None:\n"
        '        \"\"\"Guarantee the SDK\'s clean-teardown API runs even without an explicit detach.\n'
        "\n"
        "        ROOT CAUSE (measured on memfabric_hybrid/memcache_hybrid 1.1.5): a process that does a\n"
        "        cross-node object read and then exits WITHOUT close() aborts in the native teardown with\n"
        "        \'corrupted size vs. prev_size in fastbins\' (SIGABRT rc 134), 3/3 runs; the SAME binary\n"
        "        with an explicit close() exits cleanly, 3/3 runs. Payloads are byte-exact either way, so\n"
        "        this is a destruction-ORDER defect in the SDK teardown, not data corruption and not a\n"
        "        driver issue. close() is the SDK\'s own public teardown path and is already called by\n"
        "        HiCache detach_storage_backend(); this hook only covers a process that exits without\n"
        "        detaching, so it is not a way to hide the defect. close() is idempotent.\n"
        '        \"\"\"\n'
        "        import atexit\n"
        "\n"
        "        def _cleanup():\n"
        "            try:\n"
        "                if getattr(self, \"store\", None) is not None:\n"
        '                    logger.info("Ascend memcache: atexit clean close (SDK teardown order)")\n'
        "                    self.close()\n"
        "            except Exception as exc:  # noqa: BLE001\n"
        '                logger.warning("Ascend memcache: atexit close failed: %s", exc)\n'
        "\n"
        "        atexit.register(_cleanup)\n"
        "\n"
        "    def _init_runtime_fields(\n"
        "        self, storage_config: Optional[HiCacheStorageConfig]\n"
        "    ) -> None:\n"
        "        self.enable_storage_metrics = False\n",
    ),
    (
        "D8-kvsplit-robust",
        "    def _mla_uses_kv_split(self) -> bool:\n"
        "        return (\n"
        "            self.is_mla_backend\n"
        "            and self.mem_pool_host is not None\n",
        "    def _mla_uses_kv_split(self) -> bool:\n"
        "        # INDEX_STATE_DELTA: tolerate a store whose host pool is not registered yet\n"
        "        # (create_backend() and register_mem_pool_host() are separate calls; an early\n"
        "        # query must not raise AttributeError).\n"
        "        return (\n"
        "            self.is_mla_backend\n"
        '            and getattr(self, "mem_pool_host", None) is not None\n',
    ),
    (
        "D7-batch_exists",
        "        if self.is_mla_backend:\n"
        "            query_keys = []\n"
        "            for key in page_keys:\n"
        '                query_keys.append(f"{key}_{self.mla_suffix}_k")\n'
        "                if self._mla_uses_kv_split():\n"
        '                    query_keys.append(f"{key}_{self.mla_suffix}_v")\n'
        "            key_multiplier = 2 if self._mla_uses_kv_split() else 1\n",
        "        if self.is_mla_backend:\n"
        "            suffixes = self._mla_component_suffixes()\n"
        "            query_keys = []\n"
        "            for key in page_keys:\n"
        "                for s in suffixes:\n"
        '                    query_keys.append(f"{key}_{self.mla_suffix}_{s}")\n'
        "            key_multiplier = len(suffixes)\n",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="validate anchors only; write nothing (for pre-write planning)")
    args = ap.parse_args()
    if args.check:
        text = open(args.path, encoding="utf-8").read()
        if DELTA_MARKER in text:
            print("SKIP: delta already applied")
            return 0
        missing = [eid for eid, anchor, repl in EDITS if anchor not in text]
        if missing:
            print(f"FAIL check: anchors missing for {missing}")
            return 2
        print(f"CHECK OK: all {len(EDITS)} delta anchors present; nothing written")
        return 0

    if args.revert:
        bak = args.path + ".deltabak"
        shutil.copy2(bak, args.path)
        print(f"reverted {args.path} from {bak}")
        return 0

    shutil.copy2(args.path, args.path + ".deltabak")
    text = open(args.path, encoding="utf-8").read()
    if DELTA_MARKER in text and "D1-header" in text:
        print("SKIP: delta already applied")
        return 0
    for eid, anchor, repl in EDITS:
        if anchor not in text:
            print(f"FAIL {eid}: anchor not found")
            return 2
        text = text.replace(anchor, repl, 1)
        print(f"  DELTA {eid}")
    open(args.path, "w", encoding="utf-8").write(text)
    print(f"OK applied {len(EDITS)} edits; sha256={hashlib.sha256(open(args.path,'rb').read()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
