#!/usr/bin/env python3
"""Freeze 24 numerical/extraction fixtures at the two current prototype lengths."""
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    def tokens(text):
        req=urllib.request.Request('http://172.16.12.32:29481/tokenize',json.dumps({'model':'glm5','prompt':text,'add_special_tokens':False}).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=15) as f:return json.load(f)['tokens']
    header=tokens('Read the validation notes. Return only the requested numeric answer.\n')
    filler=tokens('Unrelated note: preserve the order of supplied facts.\n')
    cases=[]
    def add(name,expected,tail,index):
        n=256+index%2;q=tokens(tail);left=n-len(header)-len(q);assert left>0
        ids=header+(filler*((left+len(filler)-1)//len(filler)))[:left]+q
        cases.append({'case_id':name,'expected':expected,'input_tokens':n,'prompt_token_ids':ids,'prompt_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
    for i in range(8):
        expected=731942+i*113;key=['alpha','beta','gamma','delta'][i%4];values={k:expected+(j+1)*71 for j,k in enumerate(['alpha','beta','gamma','delta'])};values[key]=expected
        text='\nRecord: alpha=14; beta=29. Key: beta. Answer: 29\nRecord: alpha=38; beta=57. Key: alpha. Answer: 38\nRecord: '+'; '.join(k+'='+str(v) for k,v in values.items())+'. Key: '+key+'. Answer:'
        add('record-'+str(i),expected,text,i)
    for i,(a,b,c) in enumerate([(17,25,9),(38,14,27),(41,12,18),(63,16,8),(12,29,43),(28,34,17),(51,23,19),(39,15,26)]):
        add('sum-three-'+str(i),a+b+c,'\nComplete arithmetic expressions. Return only the final number.\n1+2+3=6\n4+5+6=15\n'+str(a)+'+'+str(b)+'+'+str(c)+'=',i)
    for i in range(8):
        n=256+i%2;expected=846273+i*239;key='QZ'+str(714+i);fact=tokens('\nRecord '+key+' has numeric value '+str(expected)+'.\n');query=tokens('\nEnd archive. What is the numeric value of record '+key+'? Answer:')
        left=n-len(header)-len(fact)-len(query);padding=(filler*((left+len(filler)-1)//len(filler)))[:left];split=left//(4 if i<4 else 2);ids=header+padding[:split]+fact+padding[split:]+query
        assert len(ids)==n;cases.append({'case_id':'retrieve-'+key,'expected':expected,'input_tokens':n,'prompt_token_ids':ids,'prompt_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
    root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32/adopt-cases-v1');root.mkdir(exist_ok=False)
    report={'cases':cases,'scope':'24 fixed numerical/extraction fixtures at N256/257; full accuracy not claimed'}
    (root/'cases.json').write_text(json.dumps(report,indent=2)+'\n')
    report['host_cases_file']=str(root/'cases.json')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END')


if __name__=='__main__':main()
