#!/usr/bin/env python3
"""Compare the counterexample against full native P and full native D paths."""
import json
from pathlib import Path
import urllib.request
import uuid


def main():
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32')
    case=json.loads((base/'adopt-cases-v1/cases.json').read_text())['cases'][0]
    previous=json.loads((base/'client-adopt-record-repeat-v1/pairs.json').read_text())
    refs={mode:next(r['token_ids'] for r in previous['rows'] if r['mode']==mode) for mode in ('control','adopt')}
    report={'completed':False,'rows':[],'case':case,'scope':'Counterexample against actual full-prefill native P/D, unchanged W8A8C8, no remote KV or adoption'}
    root=base/'native-path-counterexample-v1';root.mkdir(exist_ok=False)
    try:
        for node in (32,33):
            for repeat in range(2):
                rid='native-path-'+uuid.uuid4().hex
                body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':64,'min_tokens':1,'temperature':0,'seed':1024,'ignore_eos':True,'return_token_ids':True,'stream':False,'cache_salt':rid}
                req=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(node,29481 if node==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
                with urllib.request.urlopen(req,timeout=90) as r:response=json.load(r)
                ids=response['choices'][0]['token_ids'];assert len(ids)==64
                report['rows'].append({'node':node,'repeat':repeat,'request':body,'response':response,'matches_PD_control':ids==refs['control'],'matches_PD_adopt':ids==refs['adopt']})
                (root/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END')
    return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
