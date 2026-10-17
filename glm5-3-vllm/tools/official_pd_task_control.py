#!/usr/bin/env python3
"""Check the already failed short task on native D and stock PD, outside timing."""
import json
from pathlib import Path
import socket
import urllib.request
import uuid


def main():
    assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    reports=[json.loads((base/'tests'/r/'result.json').read_text()) for r in ('official-baseline-v2-r1','official-baseline-v2-r2')]
    assert all(r['completed'] and r['passed'] for r in reports)
    original=next(x for x in reports[0]['task_checks'] if not x['numeric_pass'])
    ids=original['response']['choices'][0]['prompt_token_ids'];assert ids
    out=base/'tests/official-task-control-v2';out.mkdir(parents=True,exist_ok=False)
    result={'completed':False,'basis':'Existing numeric failure, original prompt token IDs, after both timing repeats; no protocol/source/precision changes. Output alone does not identify a KV payload fault.','original':original,'requests':[]}
    for role in ('D','D','proxy'):
        url='http://172.16.12.33:29482' if role=='D' else 'http://172.16.12.32:29880'
        payload={'model':'glm5','prompt':ids,'max_tokens':16,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'stream':False,'stop':['\n'],'cache_salt':'control-'+uuid.uuid4().hex}
        q=urllib.request.Request(url+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(q,timeout=120) as response:value=json.load(response)
        result['requests'].append({'role':role,'request':payload,'response':value})
    result['completed']=True
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
