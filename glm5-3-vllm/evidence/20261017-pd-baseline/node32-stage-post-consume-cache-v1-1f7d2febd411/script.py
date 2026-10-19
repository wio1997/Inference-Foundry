#!/usr/bin/env python3
"""Stage native-cache observation triggered by standard request IDs after D finished."""
import ast
import hashlib
import json
from pathlib import Path
import socket
import subprocess

CHECKSUM_FUNCTION = 'def _glm53_control_checksum_tensor(value):\n    data = value.contiguous().view(torch.uint8).reshape(-1).to(torch.int32)\n    position = torch.arange(data.numel(), dtype=torch.int32, device=data.device)\n    weight1 = position.remainder(65521)+1\n    weight2 = (position.remainder(65519)*17).remainder(65519)+1\n    return torch.stack((data.sum(dtype=torch.int64),\n                        (data*weight1).sum(dtype=torch.int64),\n                        (data*weight2).sum(dtype=torch.int64),\n                        (data*data).sum(dtype=torch.int64)))'
LAYOUT = '        if os.getenv("GLM53_KV_CONTROL_AUDIT") == "1":\n            import json, torch_npu, weakref\n            from pathlib import Path\n            labels = {}\n            views = {}\n            for name, caches in self.kv_caches.items():\n                for slot, cache in enumerate(self._as_cache_tuple(caches)):\n                    length, stride, _, scale = self._get_cache_block_metadata(cache)\n                    views.setdefault(cache.data_ptr(), []).append((weakref.ref(cache), stride, scale, length))\n                    labels.setdefault(cache.data_ptr(), []).append({\n                        "layer": name, "slot": slot, "shape": list(cache.shape),\n                        "stride": list(cache.stride()), "dtype": str(cache.dtype),\n                        "format": torch_npu.get_npu_format(cache),\n                        "inner_contiguous": cache[0].is_contiguous(),\n                        "block_bytes": length, "block_stride": stride,\n                        "blocks_per_external_page": scale})\n            if self.num_kv_cache_groups != 1:\n                raise RuntimeError("Diagnostic supports current single-group GLM layout only")\n            entries = []\n            for address, length, stride in zip(self.group_kv_caches_base_addr[0], self.group_block_len[0], self.group_block_stride[0], strict=True):\n                matches = [v for v in views[address] if v[1] == stride and v[3] == length]\n                if not matches:\n                    raise RuntimeError("Diagnostic cache view not found")\n                entries.append(matches[0])\n            self.m_store._glm53_audit_entries = entries\n            groups = {}\n            for group, addresses in self.group_kv_caches_base_addr.items():\n                groups[group] = [{"entry": i, "block_bytes": size,\n                                  "block_stride": stride, "labels": labels.get(address, [])}\n                                 for i, (address, size, stride) in enumerate(zip(\n                                     addresses, self.group_block_len[group], self.group_block_stride[group], strict=True))]\n            root = Path("/out/kv-control-audit")\n            root.mkdir(exist_ok=True)\n            (root / f"layout-pid{os.getpid()}.json").write_text(json.dumps({\n                "pid": os.getpid(), "tp_rank": self.tp_rank,\n                "num_blocks": self.num_blocks, "groups": groups}, indent=2) + "\\n")\n'
METHOD = '    def _glm53_audit_buffers(self, keys, addresses, sizes, phase):\n        if os.getenv("GLM53_KV_CONTROL_AUDIT") != "1":\n            return\n        import hashlib, json\n        from pathlib import Path\n        counts = getattr(self, "_glm53_audit_counts", {})\n        self._glm53_audit_counts = counts\n        remaining = 512 - counts.get(phase, 0)\n        if remaining <= 0:\n            return\n        selected = list(zip(keys, addresses, sizes))[:remaining]\n        entries = self._glm53_audit_entries\n        audit_stream = getattr(self, "_glm53_audit_stream", None)\n        if audit_stream is None:\n            audit_stream = torch.npu.Stream()\n            self._glm53_audit_stream = audit_stream\n        root = Path("/out/kv-control-audit")\n        root.mkdir(exist_ok=True)\n        with (root / f"buffers-pid{os.getpid()}.jsonl").open("a") as output:\n            with torch.npu.stream(audit_stream):\n                ready = getattr(self, "_glm53_audit_ready_event", None)\n                if ready is not None:\n                    audit_stream.wait_event(ready)\n                for key, pointers, lengths in selected:\n                    if len(pointers) != len(entries):\n                        raise RuntimeError("Diagnostic layout entry count mismatch")\n                    values = []\n                    for pointer, length, entry in zip(pointers, lengths, entries, strict=True):\n                        reference, stride, scale, block_bytes = entry\n                        cache = reference()\n                        if cache is None:\n                            raise RuntimeError("KV tensor released before its audit")\n                        delta = pointer - cache.data_ptr()\n                        if delta < 0 or delta % stride != 0 or length != block_bytes:\n                            raise RuntimeError("Diagnostic address/stride mismatch")\n                        start = delta // stride * scale\n                        block = cache[start:start+scale]\n                        if block.numel()*block.element_size() != length:\n                            raise RuntimeError("Diagnostic logical block length mismatch")\n                        values.append(_glm53_control_checksum_tensor(block))\n                    # Only these small integer control values leave the NPU.\n                    control = torch.stack(values).cpu().tolist()\n                    row = {"phase": phase, "key": key, "pid": os.getpid(), "request_id": self._glm53_audit_request_id, "tp_rank": self._glm53_audit_tp_rank,\n                           "device_id": str(self.device_id), "bytes": sum(lengths),\n                           "sizes": lengths, "checksums": control,\n                           "control_sha256": hashlib.sha256(json.dumps(control).encode()).hexdigest(),\n                           "monotonic_ns": time.monotonic_ns()}\n                    output.write(json.dumps(row) + "\\n")\n                    output.flush()\n                    counts[phase] = counts.get(phase, 0) + 1\n\n'
FENCE_ADDITION = '        if (\n            self.vllm_config.kv_transfer_config.kv_connector_extra_config.get("wait_for_final_prompt_put", False)\n            and any(request.is_last_chunk for request in requests)\n        ):\n            # The existing FIFO event follows every PUT queued in this batch.\n            # Earlier prompt chunks may still overlap; the handoff observes\n            # the final chunk only after all preceding SDK PUT calls return.\n            wait_start = time.perf_counter()\n            while not self._previous_save_batch.done.wait(timeout=SAVE_BATCH_FAILURE_POLL_INTERVAL_S):\n                send_thread.raise_if_failed()\n            send_thread.raise_if_failed()\n            logger.info(\n                "Final prompt PUT batch drained before handoff tp_rank=%d requests=%s wait_ms=%.3f",\n                self.tp_rank,\n                ",".join(request.req_id for request in requests if request.is_last_chunk),\n                (time.perf_counter() - wait_start) * 1000,\n            )\n'

POST_METHOD = '''    def _glm53_post_consume_cache_audit(self, metadata):
        if os.getenv("GLM53_KV_POST_CONSUME_AUDIT") != "1":
            return
        seen = getattr(self, "_glm53_post_audited_requests", set())
        self._glm53_post_audited_requests = seen
        for request in metadata.requests:
            if "glm53-postkv-" not in request.req_id or request.req_id in seen:
                continue
            if request.target_token_len != 4096:
                continue
            if request.load_spec is not None and request.load_spec.can_load:
                raise RuntimeError("Post-consume control requires native local reuse, not a new external load")
            if self.num_kv_cache_groups != 1:
                raise RuntimeError("Post-consume control supports this GLM single group only")
            keys, addresses, sizes = [], [], []
            blocks = request.block_ids_by_group[0]
            for start, end, key, _block_hash, block_id in self.token_database.process_token_key_strings_with_block_ids(
                3968, request.block_hashes, blocks, 0, kv_cache_group_id=0,
                skip_null_blocks=False,
            ):
                pointers, lengths, _ = self.token_database.prepare_value(
                    start, end, blocks, kv_cache_group_id=0, block_id=block_id,
                )
                keys.append(key)
                addresses.append(pointers)
                sizes.append(lengths)
            if len(keys) != 31:
                raise RuntimeError("Post-consume local-prefix key count is not31")
            event = torch.npu.Event()
            event.record()
            self.m_store._glm53_audit_ready_event = event
            self.m_store._glm53_audit_request_id = request.req_id
            self.m_store._glm53_audit_buffers(keys, addresses, sizes, "native_prefix")
            seen.add(request.req_id)

'''


def main():
    assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32', 'B5-5U-Atlas900RCKA3-33')
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/post-consume-cache-audit-v1')
    root.mkdir(exist_ok=False)
    prefix = '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/'
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    sources = [('pool_worker.py', '3d39b85efa8afb750abc179a030a82d2309fd7b0770eb9fcc0da073af0c141ff'),
               ('backend/memcache_backend.py', 'cf2d7a1ccf488b28dc6a4ae1416e694c348837d7786ece20f0522cafdfc3cd92'),
               ('metadata.py', '32930f236b360f90d202610faa44699757fca669347ff5efa386adb9ddc4d156')]
    result = []
    for name, expected in sources:
        raw = subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
                                       '--read-only', '--cap-drop', 'ALL', '--entrypoint', 'python3', image,
                                       '-c', 'from pathlib import Path;import sys;sys.stdout.buffer.write(Path('+repr(prefix+name)+').read_bytes())'], timeout=40)
        assert hashlib.sha256(raw).hexdigest() == expected
        text = raw.decode()
        if name == 'pool_worker.py':
            marker = '        self.page_size_bytes = sum(self.block_len)\n'
            assert text.count(marker) == 1
            text = text.replace(marker, marker+LAYOUT+'            self.m_store._glm53_audit_tp_rank = self.tp_rank\n')
            marker = '    def start_load_kv(self, metadata: AscendConnectorMetadata):\n'
            assert text.count(marker) == 1
            text = text.replace(marker, POST_METHOD+marker+'        self._glm53_post_consume_cache_audit(metadata)\n')
            marker = '    def wait_for_save(self, connector_metadata: AscendConnectorMetadata) -> None:\n'
            assert text.count(marker) == 1
            text = text.replace(marker, marker+'        self._glm53_post_consume_cache_audit(connector_metadata)\n')
            marker = '        self._previous_save_batch = send_thread.add_save_batch(requests)\n'
            assert text.count(marker) == 1
            text = text.replace(marker, marker+FENCE_ADDITION)
        elif name == 'metadata.py':
            if 'import os\n' not in text:
                marker = 'from dataclasses import '
                assert text.count(marker) == 1
                text = text.replace(marker, 'import os\n\n'+marker)
            marker = '        if skip_save and load_spec is None:\n'
            assert text.count(marker) == 1
            replacement = '        if skip_save and load_spec is None and not (os.getenv("GLM53_KV_POST_CONSUME_AUDIT") == "1" and "glm53-postkv-" in tracker.req_id):\n'
            text = text.replace(marker, replacement)
        else:
            assert text.count('import torch\n') == 1
            text = text.replace('import torch\n', 'import torch\n\n'+CHECKSUM_FUNCTION+'\n\n')
            marker = '    def get(\n'
            assert text.count(marker) == 1
            text = text.replace(marker, METHOD+marker)
        ast.parse(text)
        path = root/name.replace('/', '-')
        path.write_text(text)
        result.append({'source': str(path), 'target': prefix+name, 'original_sha256': expected,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'patches': result, 'scope': 'Read-only native-prefix controls only for existing standard request IDs containing glm53-postkv-, sent after the unobserved D128 completed. No pre-PUT/GET or post-PUT/GET audit, no new endpoint, RPC, file trigger or hot-update mechanism. P uses existing final FIFO drain as diagnostic boundary guard. All16 native P replicas and existing D prefix,115 entries/key,31 keys/request. Four integer moments, not cryptographic hashes. Local-hit and zero-extra-GET metrics must be checked.'}
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
