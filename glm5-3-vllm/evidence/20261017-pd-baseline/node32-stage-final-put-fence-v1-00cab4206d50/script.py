#!/usr/bin/env python3
"""Stage an opt-in wait on the existing final-prompt FIFO save completion."""
import ast
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ORIGINAL = '3d39b85efa8afb750abc179a030a82d2309fd7b0770eb9fcc0da073af0c141ff'
TARGET = '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/final-put-fence-v1')
    root.mkdir(exist_ok=False)
    raw = subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
                                   '--read-only', '--cap-drop', 'ALL', '--entrypoint', 'python3', IMAGE,
                                   '-c', 'from pathlib import Path;import sys;sys.stdout.buffer.write(Path('+repr(TARGET)+').read_bytes())'], timeout=40)
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL
    text = raw.decode()
    marker = '        self._previous_save_batch = send_thread.add_save_batch(requests)\n'
    assert text.count(marker) == 1
    addition = '''        if (
            self.vllm_config.kv_transfer_config.kv_connector_extra_config.get("wait_for_final_prompt_put", False)
            and any(request.is_last_chunk for request in requests)
        ):
            # The existing FIFO event follows every PUT queued in this batch.
            # Earlier prompt chunks may still overlap; the handoff observes
            # the final chunk only after all preceding SDK PUT calls return.
            wait_start = time.perf_counter()
            while not self._previous_save_batch.done.wait(timeout=SAVE_BATCH_FAILURE_POLL_INTERVAL_S):
                send_thread.raise_if_failed()
            send_thread.raise_if_failed()
            logger.info(
                "Final prompt PUT committed before handoff tp_rank=%d requests=%s wait_ms=%.3f",
                self.tp_rank,
                ",".join(request.req_id for request in requests if request.is_last_chunk),
                (time.perf_counter() - wait_start) * 1000,
            )
'''
    text = text.replace(marker, marker+addition)
    ast.parse(text)
    path = root/'pool_worker.py'
    path.write_text(text)
    report = {'source': str(path), 'target': TARGET, 'original_sha256': ORIGINAL,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'scope': 'Opt-in wait_for_final_prompt_put. Existing batch.done event and failure polling only, final prompt chunk only. No audit operations, model/kernel/precision/transport changes. Staging alone does not validate or activate the candidate.'}
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
