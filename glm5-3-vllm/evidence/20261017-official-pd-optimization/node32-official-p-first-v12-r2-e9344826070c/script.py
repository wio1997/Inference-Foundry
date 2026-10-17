#!/usr/bin/env python3
"""Measure stock official-PD streaming C1/C4, cold/hot prefixes, then task checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import threading
import time
import urllib.request
import uuid

URL={'P':'http://172.16.12.32:29481','D':'http://172.16.12.33:29482','proxy':'http://172.16.12.32:29880'}
BASE=Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')


def http(role,path,payload=None,request_id=None,timeout=180):
    headers={'Content-Type':'application/json'}
    if request_id:headers['X-Request-Id']=request_id
    request=urllib.request.Request(URL[role]+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers)
    return urllib.request.urlopen(request,timeout=timeout)


def distribution(values):
    values=sorted(values)
    def pct(q):
        x=(len(values)-1)*q;i=math.floor(x);j=math.ceil(x)
        return values[i]+(values[j]-values[i])*(x-i)
    return {'n':len(values),'p50':pct(.5),'p95':pct(.95),'min':values[0],'max':values[-1]} if values else {}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--stage',choices=('baseline','mtp1','shared-overlap','mtp5','early-output'),required=True)
    parser.add_argument('--waves',type=int,default=6);parser.add_argument('--revision',type=int,default=2);parser.add_argument('--ready-seconds',type=int,default=1200);parser.add_argument('--after-run');parser.add_argument('--prefill-revision',type=int);parser.add_argument('--after-profile');args=parser.parse_args()
    prefill_revision=args.prefill_revision if args.prefill_revision is not None else args.revision
    assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32' and re.fullmatch(r'[a-z0-9-]+',args.run_id) and 3<=args.waves<=12
    root=BASE/'tests'/args.run_id;root.mkdir(parents=True,exist_ok=False)
    fixtures={}
    source=Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/native-eager-c1-v1')
    for n in (256,1024):
        original=json.loads((source/f'cold-eager-reference-{n}.json').read_text());assert len(original['request']['prompt'])==n
        fixtures[n]=original['request']['prompt']
    report={'completed':False,'passed':False,'stage':args.stage,'run_id':args.run_id,'revision':args.revision,'prefill_revision':prefill_revision,
        'started_host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_ready','cases':[],
        'scope':'Synthetic fixed256/64 and1024/64, C1/C4 closed-loop stock official proxy. Cold prefix uses unique salts; hot prefix reuses one salt per client after warmup. Initial connection/compile warmups separate. No profiler or payload audit. Task/numerical/exact-format checks separate; no production SLO/capacity or full model accuracy claim.'}
    def save(): (root/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    def metrics(label):
        out={}
        for role in ('P','D'):
            with http(role,'/metrics',timeout=10) as response:raw=response.read().decode()
            (root/f'{label}-{role}.txt').write_text(raw);v={}
            for line in raw.splitlines():
                if not line or line.startswith('#'):continue
                k,value=line.rsplit(' ',1);k=k.split('{',1)[0];v[k]=v.get(k,0)+float(value)
            out[role]=v
        return out
    def invoke(n,salt,barrier=None,output=64):
        payload={'model':'glm5','prompt':fixtures[n],'max_tokens':output,'temperature':0,'top_p':1,'seed':1024,
            'ignore_eos':True,'return_token_ids':True,'cache_salt':salt,'stream':True,'stream_options':{'include_usage':True}}
        row={'request_id':'bench-'+args.run_id+'-'+uuid.uuid4().hex,'input_tokens':n,'requested_output':output,'cache_salt':salt,'token_ids':[],'text':'','arrivals_ns':[],'chunks':[]}
        if barrier:barrier.wait(timeout=30)
        row['start_ns']=time.perf_counter_ns()
        try:
            with http('proxy','/v1/completions',payload,row['request_id']) as response:
                row['http_status']=response.status
                for line in response:
                    if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                    now=time.perf_counter_ns();chunk=json.loads(line[6:]);row['chunks'].append(chunk)
                    if chunk.get('error'):raise RuntimeError(str(chunk['error']))
                    for choice in chunk.get('choices',[]):
                        ids=choice.get('token_ids') or [];row['token_ids'].extend(ids);row['arrivals_ns'].extend([now]*len(ids));row['text']+=choice.get('text','')
                    if chunk.get('usage'):row['usage']=chunk['usage']
            row['http_success']=row['http_status']==200 and len(row['token_ids'])==output
            number=re.match(r'\s*([-+]?\d+(?:\.\d+)?)',row['text'])
            row['functional_success']=bool(number and Decimal(number.group(1))==5)
        except Exception as exc:row.update(http_success=False,functional_success=False,error=type(exc).__name__+': '+str(exc))
        row['closed_ns']=time.perf_counter_ns();a=row['arrivals_ns']
        if a:row.update(ttft_ms=(a[0]-row['start_ns'])/1e6,tpot_ms=(a[-1]-a[0])/1e6/max(1,len(a)-1),e2e_ms=(row['closed_ns']-row['start_ns'])/1e6)
        row['success']=row['http_success'] and row['functional_success'];return row
    try:
        save();deadline=time.monotonic()+args.ready_seconds
        if args.after_run:
            assert re.fullmatch(r'[a-z0-9-]+',args.after_run)
            report['state']='waiting_predecessor';save()
            previous=BASE/'tests'/args.after_run/'result.json'
            while not previous.exists():
                assert time.monotonic()<deadline,'Predecessor wait deadline exceeded';time.sleep(5)
            prior=json.loads(previous.read_text());assert prior['completed'] and prior['passed'],'Predecessor failed; do not overlap or hide its failure'
            deadline=time.monotonic()+args.ready_seconds
            report['state']='waiting_ready';save()
        while True:
            ready=[]
            for role in ('P','D'):
                try:
                    with http(role,'/v1/models',timeout=2) as response:data=json.load(response)
                    ready.append(any(m['id']=='glm5' for m in data['data']))
                except Exception:ready.append(False)
            if all(ready):break
            assert time.monotonic()<deadline,'Official pair readiness deadline exceeded';time.sleep(5)
        prefill_root=BASE/f'prefill-node32-v{prefill_revision}'
        assert (prefill_root/'launch.json').is_file(),'Missing formal prefill launch receipt'
        if args.after_profile:
            assert re.fullmatch(r'[a-z0-9-]+', args.after_profile)
            prior_profile_path = BASE / 'tests' / args.after_profile / 'result.json'
            prior_profile_raw = prior_profile_path.read_bytes()
            prior_profile = json.loads(prior_profile_raw)
            assert prior_profile['completed'] and prior_profile['prefill_revision'] == prefill_revision
            assert any(x['role'] == 'P' for x in prior_profile['profiles'])
            assert all(x.get('stopped') for x in prior_profile['profiles'])
            report['prior_stopped_profile'] = {'run_id': args.after_profile,
                'sha256': hashlib.sha256(prior_profile_raw).hexdigest(),
                'scope': 'Historical profile fully stopped before this new timing run on retained P.'}
        else:
            assert not any(p.is_file() for p in (prefill_root / 'profile').rglob('*')), 'Profiler files require explicit completed/stopped profile gate'
        report['state']='connection_and_compile_warmup';save()
        report['initial_warmups']=[invoke(n,'connect-'+uuid.uuid4().hex) for n in (256,1024)]
        assert all(r['success'] for r in report['initial_warmups']),'Initial proxy functional check failed'
        save()
        for concurrency in (1,4):
            for cache in ('cold','hot'):
                for n in (256,1024):
                    case_id=f'{n}-64-c{concurrency}-{cache}';report['state']='measuring-'+case_id;save()
                    salts=['warm-'+uuid.uuid4().hex for unused in range(concurrency)]
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        def wave(s):
                            barrier=threading.Barrier(concurrency)
                            futures=[pool.submit(invoke,n,s[i],barrier) for i in range(concurrency)]
                            return [f.result() for f in futures]
                        warm=wave(salts);assert all(r['success'] for r in warm),'Case warmup failed: '+case_id
                        (root/f'{case_id}-warmup.json').write_text(json.dumps(warm,indent=2)+'\n')
                        before=metrics(case_id+'-before');rows=[]
                        for index in range(args.waves):
                            measured=wave(salts if cache=='hot' else ['cold-'+uuid.uuid4().hex for unused in range(concurrency)])
                            for row in measured:row['wave']=index
                            rows.extend(measured)
                        after=metrics(case_id+'-after')
                    (root/f'{case_id}-requests.json').write_text(json.dumps(rows,indent=2)+'\n')
                    duration=(max(r['closed_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
                    good=[r for r in rows if r['success']]
                    case={'case':case_id,'input_tokens':n,'output_tokens':64,'concurrency':concurrency,'prefix':cache,
                        'waves':args.waves,'requests':len(rows),'http_successes':sum(r['http_success'] for r in rows),'functional_successes':len(good),
                        'duration_s':duration,'effective_output_tok_s':sum(len(r['token_ids']) for r in good)/duration,
                        'ttft_ms':distribution([r['ttft_ms'] for r in good]),'tpot_ms':distribution([r['tpot_ms'] for r in good]),
                        'prefix_hit_delta':{role:after[role].get('vllm:prefix_cache_hits_total',0)-before[role].get('vllm:prefix_cache_hits_total',0) for role in ('P','D')},
                        'spec_metrics_delta':{k:after['D'].get(k,0)-before['D'].get(k,0) for k in after['D'] if 'spec_decode' in k and not k.endswith('_created')},
                        'output_variants':len({tuple(r['token_ids']) for r in good}),
                        'non_successes':[{k:r.get(k) for k in ('request_id','http_status','http_success','functional_success','error','text')} for r in rows if not r['success']]}
                    report['cases'].append(case);save();print(json.dumps(case),flush=True)
                    assert len(good)==len(rows),'A formal case failed its HTTP/length/task gate'
                    if cache=='cold':assert all(v==0 for v in case['prefix_hit_delta'].values()),'Unexpected local hits in cold-prefix run'
        report['state']='task_checks';save();checks=[]
        for expr,expected in [('2+3',5),('17+25',42),('37-19',18),('6*7',42),('81/9',9),('16+16',32),('14+29',43),('90-47',43),('8*8',64),('100/4',25),('23+19',42),('9*7',63)]:
            payload={'model':'glm5','prompt':'Complete arithmetic expressions. Return only the final number.\n1+1=2\n3*4=12\n'+expr+'=',
                'max_tokens':16,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'stream':False,'stop':['\n'],'cache_salt':'task-'+uuid.uuid4().hex}
            with http('proxy','/v1/completions',payload) as response:value=json.load(response)
            text=value['choices'][0]['text'];match=re.match(r'\s*([-+]?\d+(?:\.\d+)?)',text)
            checks.append({'expression':expr,'expected':expected,'response':value,'numeric_pass':bool(match and Decimal(match.group(1))==expected)})
        report['task_checks']=checks;report['task_numeric_passes']=sum(x['numeric_pass'] for x in checks)
        report['completed']=True;report['passed']=all(c['functional_successes']==c['requests'] for c in report['cases'])
        report['state']='baseline_collected_continue_optimization' if args.stage=='baseline' else 'candidate_collected_continue_validation'
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc);report['state']='needs_evidence_based_followup'
    report['ended_host_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n');save()
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END',flush=True)
    assert report['completed'] and report['passed'],report.get('failure')


if __name__=='__main__':main()
