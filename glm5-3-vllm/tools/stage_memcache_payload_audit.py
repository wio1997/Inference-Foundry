#!/usr/bin/env python3
"""Stage bounded opt-in byte/layout observation; no model or operator changes."""
import ast
import hashlib
import json
from pathlib import Path
import socket
import subprocess

PREFIX = '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
METHOD = '''    def _glm53_audit_buffers(self, keys, addresses, sizes, phase):
        if os.getenv("GLM53_KV_PAYLOAD_AUDIT") != "1":
            return
        import acl, ctypes, hashlib, json
        from pathlib import Path
        counts = getattr(self, "_glm53_audit_counts", {})
        self._glm53_audit_counts = counts
        remaining = 128 - counts.get(phase, 0)
        if remaining <= 0:
            return
        selected = list(zip(keys, addresses, sizes))[:remaining]
        capacity = max(sum(lengths) for _, _, lengths in selected)
        if not 0 < capacity <= 16777216:
            raise RuntimeError("Unexpected diagnostic KV object size")
        host, rc = acl.rt.malloc_host(capacity)
        if rc != 0:
            raise RuntimeError(f"Audit host allocation failed: {rc}")
        root = Path("/out/kv-payload-audit")
        root.mkdir(exist_ok=True)
        try:
            with (root / f"buffers-pid{os.getpid()}.jsonl").open("a") as stream:
                for key, pointers, lengths in selected:
                    offset = 0
                    for pointer, length in zip(pointers, lengths, strict=True):
                        rc = acl.rt.memcpy(host + offset, length, pointer, length, 2)
                        if rc != 0:
                            raise RuntimeError(f"Audit D2H copy failed: {rc}")
                        offset += length
                    raw = ctypes.string_at(host, offset)
                    pieces = []
                    cursor = 0
                    for length in lengths:
                        pieces.append(hashlib.sha256(memoryview(raw)[cursor:cursor+length]).hexdigest())
                        cursor += length
                    row = {"phase": phase, "key": key, "pid": os.getpid(),
                           "device_id": str(self.device_id), "bytes": offset,
                           "sizes": lengths, "sha256": hashlib.sha256(raw).hexdigest(),
                           "entry_sha256_concatenated": "".join(pieces),
                           "monotonic_ns": time.monotonic_ns()}
                    stream.write(json.dumps(row) + "\\n")
                    stream.flush()
                    counts[phase] = counts.get(phase, 0) + 1
        finally:
            rc = acl.rt.free_host(host)
            if rc != 0:
                raise RuntimeError(f"Audit host free failed: {rc}")

'''
LAYOUT = '''        if os.getenv("GLM53_KV_PAYLOAD_AUDIT") == "1":
            import json, torch_npu
            from pathlib import Path
            labels = {}
            for name, caches in self.kv_caches.items():
                for slot, cache in enumerate(self._as_cache_tuple(caches)):
                    length, stride, _, scale = self._get_cache_block_metadata(cache)
                    labels.setdefault(cache.data_ptr(), []).append({
                        "layer": name, "slot": slot, "shape": list(cache.shape),
                        "stride": list(cache.stride()), "dtype": str(cache.dtype),
                        "format": torch_npu.get_npu_format(cache),
                        "inner_contiguous": cache[0].is_contiguous(),
                        "block_bytes": length, "block_stride": stride,
                        "blocks_per_external_page": scale})
            groups = {}
            for group, addresses in self.group_kv_caches_base_addr.items():
                groups[group] = [{"entry": i, "block_bytes": size,
                                  "block_stride": stride, "labels": labels.get(address, [])}
                                 for i, (address, size, stride) in enumerate(zip(
                                     addresses, self.group_block_len[group], self.group_block_stride[group], strict=True))]
            root = Path("/out/kv-payload-audit")
            root.mkdir(exist_ok=True)
            (root / f"layout-pid{os.getpid()}.json").write_text(json.dumps({
                "pid": os.getpid(), "tp_rank": self.tp_rank,
                "num_blocks": self.num_blocks, "groups": groups}, indent=2) + "\\n")
'''


def main():
    assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32', 'B5-5U-Atlas900RCKA3-33')
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/payload-audit-v1')
    root.mkdir(exist_ok=False)
    result = []
    for name, expected in [
        ('pool_worker.py', '3d39b85efa8afb750abc179a030a82d2309fd7b0770eb9fcc0da073af0c141ff'),
        ('backend/memcache_backend.py', 'cf2d7a1ccf488b28dc6a4ae1416e694c348837d7786ece20f0522cafdfc3cd92'),
    ]:
        raw = subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
                                       '--read-only', '--cap-drop', 'ALL', '--entrypoint', 'python3', IMAGE,
                                       '-c', 'from pathlib import Path;import sys;sys.stdout.buffer.write(Path('+repr(PREFIX+name)+').read_bytes())'], timeout=40)
        assert hashlib.sha256(raw).hexdigest() == expected
        text = raw.decode()
        if name == 'pool_worker.py':
            marker = '        self.page_size_bytes = sum(self.block_len)\n'
            assert text.count(marker) == 1
            text = text.replace(marker, marker+LAYOUT)
        else:
            marker = '    def get(\n'
            assert text.count(marker) == 1
            text = text.replace(marker, METHOD+marker)
            get = '            res = self.store.batch_get_into_layers(key, addr, size, MmcDirect.COPY_G2L.value)\n'
            put = '            res = self.store.batch_put_from_layers(key, addr, size, MmcDirect.COPY_L2G.value)\n'
            assert text.count(get) == text.count(put) == 1
            text = text.replace(get, get+'            if res is not None and all(r == 0 for r in res):\n                self._glm53_audit_buffers(key, addr, size, "get_after")\n')
            text = text.replace(put, '            self._glm53_audit_buffers(key, addr, size, "put_before")\n'+put+'            if res is not None and all(r == 0 for r in res):\n                self._glm53_audit_buffers(key, addr, size, "put_after")\n')
        ast.parse(text)
        target = root/name.replace('/', '-')
        target.write_text(text)
        result.append({'source': str(target), 'target': PREFIX+name, 'original_sha256': expected,
                       'sha256': hashlib.sha256(text.encode()).hexdigest()})
    report = {'patches': result, 'scope': 'Opt-in GLM53_KV_PAYLOAD_AUDIT=1. Up to128 keys/phase/process. P before/after PUT and D after GET bytes/entry hashes; real cache layout recorded. Readback adds synchronization/CPU work, so no performance acceptance; negative results do not alone locate a transport versus source/target race.'}
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
