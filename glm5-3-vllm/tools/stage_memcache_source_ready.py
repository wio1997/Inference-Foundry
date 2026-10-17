#!/usr/bin/env python3
"""Stage an opt-in all-stream readiness fence before raw-address SDK PUT."""
import ast
import hashlib
import json
from pathlib import Path
import socket
import subprocess

EXPECTED = 'c7af0a052e1cbada0b300b6872585ffeafe159fb355b73547c0ccc34755fe67b'
PATCHED_SHA256 = '363547b700e78f9ec5bac65de39617c944488b49386ec3afbb2ea328f3626a6b'
ADDITION = '            source_ready_mode = os.getenv("GLM53_KV_SOURCE_READY_FENCE", "0")\n            if source_ready_mode in ("tagged", "1"):\n                source_ready_apply = source_ready_mode == "1" or "glm53-source-ready-" in req_id\n                source_ready_start = time.perf_counter()\n                if source_ready_apply:\n                    # Raw-address SDK copies must observe writes on every producer stream.\n                    torch.npu.synchronize()\n                logger.info(\n                    "Source ready before PUT req_id=%s tp_rank=%d keys=%d fence=%s device=%d wait_ms=%.3f",\n                    req_id, self.tp_rank, len(keys), source_ready_apply,\n                    torch.npu.current_device(), (time.perf_counter() - source_ready_start) * 1000,\n                )\n'

def main():
    assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32', 'B5-5U-Atlas900RCKA3-33')
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/source-ready-fence-v2')
    root.mkdir(exist_ok=False)
    target = '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py'
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    raw = subprocess.check_output(['docker','run','--rm','--pull','never','--network','none','--read-only','--cap-drop','ALL','--entrypoint','python3',image,'-c','from pathlib import Path;import sys;sys.stdout.buffer.write(Path('+repr(target)+').read_bytes())'],timeout=40)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    text = raw.decode().replace('import json\n', 'import json\nimport os\n', 1)
    marker = '            self.m_store.put(keys, addrs, sizes)\n'
    assert text.count(marker) == 1
    text = text.replace(marker, ADDITION+marker)
    ast.parse(text)
    path = root/'kv_transfer.py'
    path.write_text(text)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PATCHED_SHA256
    report = {'source':str(path),'target':target,'original_sha256':EXPECTED,'sha256':PATCHED_SHA256,'scope':'Existing request_id tag selects torch.npu.synchronize immediately BEFORE SDK PUT; default off, no KV reads before/during PUT/GET. CPU log for both arms. No endpoint, RPC, file trigger, kernel, precision or transport change.'}
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
