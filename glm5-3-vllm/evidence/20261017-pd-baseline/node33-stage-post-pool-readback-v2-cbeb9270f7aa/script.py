#!/usr/bin/env python3
"""Stage post-D private-arena pool readback; no running model is modified."""
import ast
import hashlib
import json
from pathlib import Path
import socket

METHOD = '    def _glm53_post_pool_readback(self, selected):\n        if (os.getenv("GLM53_KV_POST_POOL_READBACK") != "1"\n                or self._glm53_audit_tp_rank != 0\n                or not any(marker in self._glm53_audit_request_id for marker in ("glm53-postkv-target-", "glm53-postkv-source-"))):\n            return\n        import hashlib, json\n        from pathlib import Path\n        counts = getattr(self, "_glm53_pool_readback_counts", 0)\n        selected = selected[:max(0, 512-counts)]\n        if not selected:\n            return\n        lengths = selected[0][2]\n        if len(lengths) != 115 or any(row[2] != lengths for row in selected):\n            raise RuntimeError("Private pool readback requires the fixed GLM115-entry layout")\n        total = sum(lengths)\n        if total != 6979584:\n            raise RuntimeError("Unexpected private pool readback object size")\n        arena = torch.empty(total, dtype=torch.int8, device=f"npu:{self.device_id}")\n        base = arena.data_ptr()\n        pointers, offsets, offset = [], [], 0\n        for length in lengths:\n            pointers.append(base+offset)\n            offsets.append(offset)\n            offset += length\n        registered = self.store.register_buffer(base, total)\n        if registered != 0:\n            raise RuntimeError(f"Private readback register failed: {registered}")\n        root = Path("/out/kv-control-audit")\n        try:\n            with (root/f"buffers-pool-pid{os.getpid()}.jsonl").open("a") as output:\n                for key, _real_addresses, _sizes in selected:\n                    # A distinct sentinel detects unwritten private destinations.\n                    # All of this occurs after the original D128 has completed.\n                    arena.fill_(-128)\n                    torch.npu.synchronize()\n                    started = time.perf_counter()\n                    result = self.store.batch_get_into_layers([key], [pointers], [lengths], 1)\n                    if list(result) != [0]:\n                        raise RuntimeError(f"Private pool readback GET failed: {result}")\n                    get_ms = (time.perf_counter()-started)*1000\n                    values = [_glm53_control_checksum_tensor(arena.narrow(0, start, length))\n                              for start, length in zip(offsets, lengths, strict=True)]\n                    control = torch.stack(values).cpu().tolist()\n                    row = {"phase":"pool_readback", "key":key, "pid":os.getpid(),\n                           "request_id":self._glm53_audit_request_id, "tp_rank":0,\n                           "device_id":str(self.device_id), "bytes":total,\n                           "sizes":lengths, "checksums":control,\n                           "control_sha256":hashlib.sha256(json.dumps(control).encode()).hexdigest(),\n                           "sdk_result":list(result), "sdk_get_ms":get_ms,\n                           "initial_sentinel":128, "monotonic_ns":time.monotonic_ns()}\n                    output.write(json.dumps(row)+"\\n")\n                    output.flush()\n                    counts += 1\n                    self._glm53_pool_readback_counts = counts\n        finally:\n            unregistered = self.store.unregister_buffer(base, total)\n            if unregistered != 0:\n                raise RuntimeError(f"Private readback unregister failed: {unregistered}")\n\n'

def main():
    assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
    original = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/post-consume-cache-audit-v1/backend-memcache_backend.py')
    raw = original.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '09d86b054146d78a265e7309fd094460a75a1479142b22bcc206e52c4c22443b'
    text=raw.decode();tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MemcacheBackend')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_glm53_audit_buffers')
    lines=text.splitlines(True)
    lines.insert(node.end_lineno, '        self._glm53_post_pool_readback(selected)\n\n'+METHOD+'\n')
    patched=''.join(lines);ast.parse(patched)
    path=Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/post-pool-readback-v2/backend-memcache_backend.py');path.parent.mkdir(exist_ok=False)
    path.write_text(patched)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'a0e95bdfb0951c0d50466f35f56cd2430757a5390e11faa22f77e486bee1cea8'
    report={'source':str(path),'sha256':'a0e95bdfb0951c0d50466f35f56cd2430757a5390e11faa22f77e486bee1cea8','base_sha256':'09d86b054146d78a265e7309fd094460a75a1479142b22bcc206e52c4c22443b','scope':'Only post-D target native request, P/D rank0, existing SDK SDMA GET to independent7MB NPU arena initialized0x80,115 entries/key,four moments,one register/unregister per request. No real model KV destination write or bulk D2H.512 keys maximum. No first PUT/GET observer, model/kernel/precision/endpoint/RPC change. Diagnostic SDK GETs counted separately from model loader metrics.'}
    (path.parent/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
