#!/usr/bin/env python3
"""Close one exact owned official model only after its baseline clients finish."""
import argparse
import base64
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import urllib.request


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,required=True);parser.add_argument('--expected-cid',required=True);parser.add_argument('--gate-b64',required=True);args=parser.parse_args()
    node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
    role='prefill' if node==32 else 'decode';base=Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    gate=json.loads(base64.b64decode(args.gate_b64));assert len(gate)==2 and all(x['completed'] and x['passed'] for x in gate)
    root=base/f'{role}-node{node}-v{args.revision}';cid=json.loads((root/'launch.json').read_text())['container_id'];assert cid==args.expected_cid
    doc=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
    assert doc['Name']==f'/glm53-official-pd-{role}-{node}-v{args.revision}-20261017'
    assert doc['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    if node==32:
        for g in gate:
            r=json.loads((base/'tests'/g['run_id']/'result.json').read_text());assert r['completed'] and r['passed']
    if doc['State']['Running']:
        with urllib.request.urlopen(f'http://172.16.12.{node}:{29481 if node==32 else 29482}/metrics',timeout=10) as response:metrics=response.read().decode()
        live=[float(s.rsplit(' ',1)[1]) for s in metrics.splitlines() if s.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
        assert live and all(x==0 for x in live),'A request is active; do not interrupt measurement'
        subprocess.run(['docker','stop','-t','90',cid],capture_output=True,check=True,timeout=100)
    state=json.loads(subprocess.check_output(['docker','inspect',cid,'--format','{{json .State}}']))
    logs=subprocess.run(['docker','logs','--timestamps',cid],capture_output=True,timeout=30);raw=logs.stdout+b'\nSTDERR\n'+logs.stderr
    out=base/'consumer-audits'/f'node{node}-opt-v{args.revision}-lifecycle';out.mkdir(parents=True,exist_ok=False)
    (out/'console.txt.gz').write_bytes(gzip.compress(raw,compresslevel=1))
    report={'node':node,'container_id':cid,'state':state,'client_gate':gate,'host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw_console_sha256':hashlib.sha256(raw).hexdigest(),'reason':'Necessary bounded mechanism configuration switch after both completed comparison repeats. Proxy and six-hour resource holder remain.'}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert not state['Running'] and not state['OOMKilled']


if __name__=='__main__':main()
