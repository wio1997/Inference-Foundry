#!/usr/bin/env python3
"""Summarize the completed model profile; retain raw evidence for separate backup."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    revision = 7 if node == 32 else 6
    root = Path(f'/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node{node}-fixed-v{revision}')
    out = root/'pd-functional-evidence-v2'
    out.mkdir(exist_ok=False)
    cid = json.loads((root/'launch.json').read_text())['container_id']
    state = json.loads(subprocess.check_output(['docker', 'inspect', cid, '--format', '{{json .State}}']))
    proc = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=40)
    raw = proc.stdout+b'\nDOCKER_STDERR_STREAM\n'+proc.stderr
    (out/'console.txt').write_bytes(raw)
    lines = raw.decode(errors='replace').splitlines()
    initialize = [s for s in lines if 'Initialize success:' in s and 'bmDataOpType:' in s]
    types = [int(re.search(r'bmDataOpType: (\d+)', s)[1]) for s in initialize]
    selected = [s for s in lines if re.search(r'bmDataOpType|Registering KV_Caches|kvpool_cached|vllm_cached', s)]
    (out/'runtime-log-evidence.txt').write_text('\n'.join(selected)+'\n')
    manifest = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                for p in sorted((root/'profile').rglob('*')) if p.is_file() and not p.is_symlink()]
    (out/'profile-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    traces = []
    for path in sorted((root/'profile').rglob('trace_view.json')):
        parsed = json.loads(path.read_bytes())
        events = parsed.get('traceEvents', []) if isinstance(parsed, dict) else parsed
        complete = [e for e in events if e.get('ph') == 'X']
        counts = Counter(e.get('name', '') for e in complete)
        sdma = [e for e in complete if e.get('name') == 'SDMA_SQE']
        rdma = [e for e in complete if re.search(r'RDMA', e.get('name', ''), re.I)]
        traces.append({'path': str(path.relative_to(root)), 'sha256': digest(path),
                       'event_count': len(events), 'sdma_sqe_count': len(sdma),
                       'rdma_named_event_count': len(rdma),
                       'sdma_samples': sdma[:2], 'rdma_samples': rdma[:2],
                       'selected_operation_counts': {k: v for k, v in counts.items()
                            if re.search(r'(?i)sdma|rdma|hccl|all.?reduce|all.?gather|reduce.?scatter|lightning|indexer|store.?kv|reshape.*cache|quant.*matmul|matmul.*quant|dynamicquant|inferattention', k)}})
        del parsed, events, complete
    matrices = []
    def transports(value, result):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'Transport Type':
                    result[str(child)] += 1
                else:
                    transports(child, result)
        elif isinstance(value, list):
            for child in value:
                transports(child, result)
    for path in sorted((root/'profile').rglob('communication_matrix.json')):
        counts = Counter()
        transports(json.loads(path.read_bytes()), counts)
        matrices.append({'path': str(path.relative_to(root)), 'sha256': digest(path),
                         'link_type_entry_counts': dict(counts)})
    report = {'node': node, 'observed_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'state': state, 'console': {'path': str(out/'console.txt'), 'bytes': len(raw),
                                        'sha256': hashlib.sha256(raw).hexdigest()},
              'hybm_initialize_count': len(types), 'hybm_data_op_types': types,
              'kv_registration_lines': [s for s in selected if 'Registering KV_Caches' in s],
              'external_cache_evidence': [s for s in selected if re.search(r'kvpool_cached|vllm_cached', s)],
              'profile_file_count': len(manifest), 'profile_bytes': sum(p['bytes'] for p in manifest),
              'traces': traces, 'communication_matrices': matrices,
              'scope': 'Actual model request profile includes both TP communication and KV-pool copies. Counts alone do not attribute each SDMA task to a KV transfer or prove absence of every possible transport. HYBM type 2 independently selects SDMA for MemCache.'}
    (out/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert state['Running'] and len(types) == 16 and set(types) == {2}
    assert len(traces) == 16 and all(t['sdma_sqe_count'] > 0 for t in traces)
    assert all(t['rdma_named_event_count'] == 0 for t in traces)
    assert len(matrices) == 16
    assert all(set(m['link_type_entry_counts']) == {'LOCAL', 'SIO', 'HCCS'} for m in matrices)
    if node == 33:
        assert any('vllm_cached=0 kvpool_cached=384' in s for s in selected)


if __name__ == '__main__':
    main()
