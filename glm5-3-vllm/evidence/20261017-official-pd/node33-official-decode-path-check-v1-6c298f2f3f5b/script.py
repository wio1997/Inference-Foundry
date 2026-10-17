#!/usr/bin/env python3
"""Summarize stopped real-PD traces using CPU-only file reads."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='profile-summary-v1');args=parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+',args.label)
    node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
    role='prefill' if node==32 else 'decode'
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    root=base/f'{role}-node{node}-v1'; out=base/'consumer-audits'/f'node{node}-{args.label}'
    out.mkdir(parents=True,exist_ok=False)
    files=sorted(p for p in (root/'profile').rglob('*') if p.is_file())
    assert not any(p.is_symlink() for p in files)
    report={'node':node,'compute_devices_used_by_analysis':[],'profile_files':len(files),
        'profile_bytes':sum(p.stat().st_size for p in files),'traces':[],'matrices':[],
        'scope':'One real stock-proxy request after delayed native arming steps; no performance timing. SDMA task presence and retained HCCL transport matrices are bounded path evidence, not per-byte attribution or full model accuracy.'}
    for p in sorted((root/'profile').rglob('trace_view.json')):
        raw=p.read_bytes()
        row={'path':str(p.relative_to(root)),'sha256':hashlib.sha256(raw).hexdigest()}
        try:
            obj=json.loads(raw); events=obj.get('traceEvents',[]) if isinstance(obj,dict) else obj
            target=out/'parsed'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            counts=Counter(e.get('name','') for e in events if e.get('ph')=='X')
            row.update(valid_json=True,event_count=len(events),
                sdma_tasks=sum(v for k,v in counts.items() if re.search(r'SDMA_SQE|PCIE_DMA_SQE',k)),
                rdma_named_tasks={k:v for k,v in counts.items() if re.search('rdma',k,re.I)},
                selected_operations={k:v for k,v in counts.items() if re.search(r'SDMA|DMA_SQE|hccl|all.?reduce|all.?gather',k,re.I)})
            del obj,events
        except ValueError as exc: row.update(valid_json=False,error=str(exc))
        report['traces'].append(row)
    def walk(obj,counts):
        if isinstance(obj,dict):
            for key,val in obj.items():
                if key=='Transport Type':counts[str(val)]+=1
                else:walk(val,counts)
        elif isinstance(obj,list):
            for val in obj:walk(val,counts)
    for p in sorted((root/'profile').rglob('communication_matrix.json')):
        raw=p.read_bytes();counts=Counter();walk(json.loads(raw),counts)
        target=out/'parsed'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        report['matrices'].append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(raw).hexdigest(),'transport_types':dict(counts)})
    report['observed_all_16_rank_traces_with_sdma']=len(report['traces'])==16 and all(t.get('valid_json') and t.get('sdma_tasks',0)>0 and not t.get('rdma_named_tasks') for t in report['traces'])
    report['all_16_rank_hccl_matrices_local_sio_hccs_only']=len(report['matrices'])==16 and all(set(t['transport_types'])<= {'LOCAL','SIO','HCCS'} and 'HCCS' in t['transport_types'] for t in report['matrices'])
    report['completed']=report['observed_all_16_rank_traces_with_sdma'] and report['all_16_rank_hccl_matrices_local_sio_hccs_only']
    (out/'profile-manifest.json').write_text(json.dumps([{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files],indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
