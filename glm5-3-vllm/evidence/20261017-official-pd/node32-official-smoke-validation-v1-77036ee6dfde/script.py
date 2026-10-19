#!/usr/bin/env python3
"""Validate existing official responses against native completed-transfer logs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--proof-b64', required=True); args=parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    proof=json.loads(base64.b64decode(args.proof_b64))
    assert proof['node']==33 and proof['api_ready'] and not proof['errors']
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    smoke=json.loads((base/'tests/smoke-v1/result.json').read_text())
    source_path='/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py'
    cid=json.loads((base/'prefill-node32-v1/launch.json').read_text())['container_id']
    source=subprocess.check_output(['docker','exec',cid,'cat',source_path])
    assert hashlib.sha256(source).hexdigest()==proof['pull_worker_sha256']
    assert b'record_transfer(' not in source and b'xfer_stats' not in source
    report={'completed':False,'basis':'Same already completed HTTP requests; no model configuration changes or new inference. Original metric-gate failure retained unchanged.',
        'metric_gap':{'pull_worker_sha256':proof['pull_worker_sha256'],'description':'Pinned V2 executes batch_transfer_sync_read and logs completed ranges/bytes but never records its base xfer_stats. Zero Prometheus counter is not a valid remote-transfer absence gate in this implementation.'},
        'raw_D_console_sha256':proof['raw_console_sha256'],'requests':[], 'checks':smoke['checks'],
        'graph_replay':proof['actual_graph_replay'],'sdma_runtime_verification':'pending real-request trace; fixed HCCS/SDMA configuration alone is not full path evidence'}
    for request in smoke['requests']:
        if request['destination']!='proxy':continue
        request_id=request['request_id']; rows=[]
        for line in proof['completed_transfer_lines']:
            if request_id not in line:continue
            match=re.search(r'\((\d+) ranges, (\d+) bytes\).*local_ip ([\d.]+) local_device_id (\d+) remote_session_id ([\d.]+):(\d+)',line)
            assert match,line
            ranges,size,local,rank,remote,port=match.groups()
            assert local=='172.16.12.33' and remote=='172.16.12.32' and int(size)>0
            rows.append({'rank':int(rank),'ranges':int(ranges),'bytes':int(size),'remote_session':remote+':'+port,'line':line})
        assert {r['rank'] for r in rows}==set(range(16))
        report['requests'].append({'request_id':request_id,'input_tokens':len(request['request']['prompt']),
            'http_status':request['http_status'],'text':request['response']['choices'][0]['text'],
            'all_16_decode_ranks_completed_remote_read':True,'logical_received_bytes':sum(r['bytes'] for r in rows),'transfers':rows})
    assert len(report['requests'])==2 and all(c['arithmetic_smoke_begins_with_5'] for c in report['checks'])
    assert report['graph_replay']
    report['completed']=True
    out=base/'consumer-audits/smoke-validation-v1';out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
