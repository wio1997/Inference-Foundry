#!/usr/bin/env python3
"""Stage bounded opt-in byte/layout observation; no model or operator changes."""
import ast
import hashlib
import json
from pathlib import Path
import socket
import subprocess

CHECKSUM_FUNCTION = 'def _glm53_control_checksum_tensor(value):\n    data = value.contiguous().view(torch.uint8).reshape(-1).to(torch.int32)\n    position = torch.arange(data.numel(), dtype=torch.int32, device=data.device)\n    weight1 = position.remainder(65521)+1\n    weight2 = (position.remainder(65519)*17).remainder(65519)+1\n    return torch.stack((data.sum(dtype=torch.int64),\n                        (data*weight1).sum(dtype=torch.int64),\n                        (data*weight2).sum(dtype=torch.int64),\n                        (data*data).sum(dtype=torch.int64)))'

PREFIX = '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
METHOD = '    def _glm53_audit_buffers(self, keys, addresses, sizes, phase):\n        if os.getenv("GLM53_KV_CONTROL_AUDIT") != "1":\n            return\n        import hashlib, json\n        from pathlib import Path\n        counts = getattr(self, "_glm53_audit_counts", {})\n        self._glm53_audit_counts = counts\n        remaining = 128 - counts.get(phase, 0)\n        if remaining <= 0:\n            return\n        selected = list(zip(keys, addresses, sizes))[:remaining]\n        entries = self._glm53_audit_entries\n        audit_stream = getattr(self, "_glm53_audit_stream", None)\n        if audit_stream is None:\n            audit_stream = torch.npu.Stream()\n            self._glm53_audit_stream = audit_stream\n        root = Path("/out/kv-control-audit")\n        root.mkdir(exist_ok=True)\n        with (root / f"buffers-pid{os.getpid()}.jsonl").open("a") as output:\n            with torch.npu.stream(audit_stream):\n                ready = getattr(self, "_glm53_audit_ready_event", None)\n                if ready is not None:\n                    audit_stream.wait_event(ready)\n                for key, pointers, lengths in selected:\n                    if len(pointers) != len(entries):\n                        raise RuntimeError("Diagnostic layout entry count mismatch")\n                    values = []\n                    for pointer, length, entry in zip(pointers, lengths, entries, strict=True):\n                        reference, stride, scale, block_bytes = entry\n                        cache = reference()\n                        if cache is None:\n                            raise RuntimeError("KV tensor released before its audit")\n                        delta = pointer - cache.data_ptr()\n                        if delta < 0 or delta % stride != 0 or length != block_bytes:\n                            raise RuntimeError("Diagnostic address/stride mismatch")\n                        start = delta // stride * scale\n                        block = cache[start:start+scale]\n                        if block.numel()*block.element_size() != length:\n                            raise RuntimeError("Diagnostic logical block length mismatch")\n                        values.append(_glm53_control_checksum_tensor(block))\n                    # Only these small integer control values leave the NPU.\n                    control = torch.stack(values).cpu().tolist()\n                    row = {"phase": phase, "key": key, "pid": os.getpid(),\n                           "device_id": str(self.device_id), "bytes": sum(lengths),\n                           "sizes": lengths, "checksums": control,\n                           "control_sha256": hashlib.sha256(json.dumps(control).encode()).hexdigest(),\n                           "monotonic_ns": time.monotonic_ns()}\n                    output.write(json.dumps(row) + "\\n")\n                    output.flush()\n                    counts[phase] = counts.get(phase, 0) + 1\n\n'
LAYOUT = '        if os.getenv("GLM53_KV_CONTROL_AUDIT") == "1":\n            import json, torch_npu, weakref\n            from pathlib import Path\n            labels = {}\n            views = {}\n            for name, caches in self.kv_caches.items():\n                for slot, cache in enumerate(self._as_cache_tuple(caches)):\n                    length, stride, _, scale = self._get_cache_block_metadata(cache)\n                    views.setdefault(cache.data_ptr(), []).append((weakref.ref(cache), stride, scale, length))\n                    labels.setdefault(cache.data_ptr(), []).append({\n                        "layer": name, "slot": slot, "shape": list(cache.shape),\n                        "stride": list(cache.stride()), "dtype": str(cache.dtype),\n                        "format": torch_npu.get_npu_format(cache),\n                        "inner_contiguous": cache[0].is_contiguous(),\n                        "block_bytes": length, "block_stride": stride,\n                        "blocks_per_external_page": scale})\n            if self.num_kv_cache_groups != 1:\n                raise RuntimeError("Diagnostic supports current single-group GLM layout only")\n            entries = []\n            for address, length, stride in zip(self.group_kv_caches_base_addr[0], self.group_block_len[0], self.group_block_stride[0], strict=True):\n                matches = [v for v in views[address] if v[1] == stride and v[3] == length]\n                if not matches:\n                    raise RuntimeError("Diagnostic cache view not found")\n                entries.append(matches[0])\n            self.m_store._glm53_audit_entries = entries\n            groups = {}\n            for group, addresses in self.group_kv_caches_base_addr.items():\n                groups[group] = [{"entry": i, "block_bytes": size,\n                                  "block_stride": stride, "labels": labels.get(address, [])}\n                                 for i, (address, size, stride) in enumerate(zip(\n                                     addresses, self.group_block_len[group], self.group_block_stride[group], strict=True))]\n            root = Path("/out/kv-control-audit")\n            root.mkdir(exist_ok=True)\n            (root / f"layout-pid{os.getpid()}.json").write_text(json.dumps({\n                "pid": os.getpid(), "tp_rank": self.tp_rank,\n                "num_blocks": self.num_blocks, "groups": groups}, indent=2) + "\\n")\n'


def main():
    assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32', 'B5-5U-Atlas900RCKA3-33')
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/payload-control-audit-v2')
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
            get_marker = '            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)\n'
            assert text.count(get_marker) == 1
            event_code = '            if os.getenv("GLM53_KV_CONTROL_AUDIT") == "1":\n                event = torch.npu.Event()\n                event.record()\n                self.m_store._glm53_audit_ready_event = event\n'
            text = text.replace(get_marker, event_code+get_marker)
        else:
            assert text.count('import torch\n') == 1
            text = text.replace('import torch\n', 'import torch\n\n'+CHECKSUM_FUNCTION+'\n\n')
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
    report = {'patches': result, 'scope': 'Opt-in GLM53_KV_CONTROL_AUDIT=1. Existing NPU operators compute four integer moments per logical byte view; only control values copied toCPU. Not cryptographic proof. Private stream waits pre-GET default-stream event after SDKcopy, without adding a pre-GET fence. P before/afterPUT and D afterGET plus exact layouts; no performance acceptance.'}
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
