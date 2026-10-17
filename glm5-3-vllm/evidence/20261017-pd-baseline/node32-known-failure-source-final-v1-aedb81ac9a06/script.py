#!/usr/bin/env python3
"""Freeze source fence selection, sender devices and FIFO guards after ABBA ends."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.request


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-run', default='source-ready-abba-v1')
    parser.add_argument('--run-id', default='source-ready-source-final-v1')
    parser.add_argument('--expected-waves', type=int, default=12)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.test_run) and re.fullmatch(r'[a-z0-9-]+', args.run_id)
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    run = json.loads((base/'tests'/args.test_run/'result.json').read_text())
    assert run['completed'] and len(run['waves']) == args.expected_waves
    out = base/'consumer-audits'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    cid = json.loads((base/'pd-node32-v14/launch.json').read_text())['container_id']
    proc = subprocess.run(['docker','logs','--timestamps',cid],capture_output=True,timeout=45)
    assert proc.returncode == 0
    raw = proc.stdout+b'\nDOCKER_STDERR_STREAM\n'+proc.stderr
    (out/'console.txt.gz').write_bytes(gzip.compress(raw,compresslevel=1))
    text = raw.decode(errors='replace')
    pattern = r'Source ready before PUT req_id=(\S+) tp_rank=(\d+) keys=(\d+) fence=(True|False) device=(\d+) wait_ms=([0-9.]+)'
    rows = [{'request_id':m[0],'tp_rank':int(m[1]),'keys':int(m[2]),'fence':m[3]=='True','device':int(m[4]),'wait_ms':float(m[5])} for m in re.findall(pattern,text)]
    results = []
    for wave in run['waves']:
        selected = [r for r in rows if r['request_id'].startswith(wave['producer_id']+'-')]
        drained = []
        for line in text.splitlines():
            if 'Final prompt PUT batch drained before handoff' in line and wave['producer_id']+'-' in line:
                match = re.search(r'tp_rank=(\d+)',line)
                drained.append(int(match[1]))
        row = {'wave':wave['wave'],'fence_selected':wave['source_fence_selected'],'source_fence_records':selected,
               'source_put_key_count':sum(r['keys'] for r in selected),
               'sender_ranks':sorted({r['tp_rank'] for r in selected}),
               'final_drain_ranks':sorted(set(drained))}
        row['selection_and_devices_correct'] = (row['sender_ranks'] == list(range(16)) and
            row['final_drain_ranks'] == list(range(16)) and
            all(r['fence'] == wave['source_fence_selected'] and r['device'] == r['tp_rank'] for r in selected))
        results.append(row)
    with urllib.request.urlopen('http://172.16.12.32:29481/metrics',timeout=10) as response:
        metrics = response.read()
    (out/'metrics.txt').write_bytes(metrics)
    report = {'waves':results,'hybm_data_op_types':[int(x) for x in re.findall(r'Initialize success:[^\n]*bmDataOpType: (\d+)',text)],
              'native_error_lines':[s for s in text.splitlines() if 'ERROR' in s or 'Traceback' in s],
              'raw_console_bytes':len(raw),'raw_console_sha256':hashlib.sha256(raw).hexdigest(),
              'scope':'Source fence wait includes outstanding device work, not pure SDMA/network time or end-to-end critical path. Existing final FIFO drain is only a matched-boundary guard, not SDK durable success ACK.'}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert all(r['selection_and_devices_correct'] for r in results), 'Missing/incorrect fence selection, device or FIFO evidence'
    assert not report['native_error_lines']


if __name__ == '__main__':
    main()
