#!/usr/bin/env python3
"""Derive data-path evidence from preserved traces without treating a failed exit as success."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    metrics = json.loads((a.run/'metrics.json').read_text())
    result = json.loads((a.run/'result.json').read_text())
    receipt = json.loads((a.run/'artifacts-backup.json').read_text())
    archive = Path(receipt['local_archive'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
    manifest = {row['path']: row for row in json.loads((a.run/'artifacts_manifest.json').read_text())}
    with tarfile.open(archive) as tar:
        trace_member = next(m for m in tar.getmembers() if m.name.endswith('/trace_view.json'))
        raw = tar.extractfile(trace_member).read()
    assert hashlib.sha256(raw).hexdigest() == manifest[trace_member.name]['sha256']
    parsed = json.loads(raw)
    events = parsed.get('traceEvents', []) if isinstance(parsed, dict) else parsed
    regions = [e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('GLM53_MEMCACHE_')]
    sdma = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'SDMA_SQE']
    mapped = []
    for region in regions:
        begin = float(region['ts'])
        end = begin+float(region['dur'])
        copies = [e for e in sdma if begin <= float(e['ts']) and float(e['ts'])+float(e['dur']) <= end]
        mapped.append({'region': region['name'], 'sdma_task_count': len(copies),
                       'tasks': [{'type': e['name'], 'stream': e['tid'], 'task_id': e['args']['Task Id']} for e in copies]})
    stdout = (a.run/'stdout.json').read_text(errors='replace')
    bm_types = [int(x) for x in re.findall(r'Initialize success:.*?bmDataOpType: (\d+)', stdout)]
    payload_ok = all(row.get('payload_valid') is True for row in metrics['checks'] if 'get_rc' in row)
    put_get_ok = all(row.get('put_rc', 0) == 0 and row.get('get_rc', [0]) == [0] for row in metrics['checks'])
    verified = len(mapped) == 4 and len(sdma) == 4 and all(x['sdma_task_count'] == 1 for x in mapped)
    verified = verified and bm_types == [2] and payload_ok and put_get_ok
    report = {'protocol': 'device_sdma', 'hybm_data_op_type': bm_types,
              'enum_evidence': 'installed include/hybm/hybm_def.h: HYBM_DOP_TYPE_SDMA = 1U << 1',
              'trace_path': trace_member.name, 'trace_sha256': hashlib.sha256(raw).hexdigest(),
              'regions': mapped, 'runtime_sdma_data_path_verified': verified,
              'payload_hashes_match': payload_ok, 'probe_returncode': result['returncode'],
              'close_rc_zero_logged': 'GLM53_STORE_CLOSE_RC=0' in stdout,
              'lifecycle_passed': result['returncode'] == 0,
              'scope': 'synthetic HBM->DRAM pool->peer HBM; same-host trace association only; no timing/PD/model-correctness claim'}
    (a.run/'memcache-trace-analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'run': a.run.name, **report}, indent=2))
    assert verified, 'Do not promote an incomplete data-path proof'


if __name__ == '__main__':
    main()
