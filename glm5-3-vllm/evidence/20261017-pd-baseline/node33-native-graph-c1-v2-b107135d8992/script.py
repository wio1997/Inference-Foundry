#!/usr/bin/env python3
"""Native local streaming C1 benchmark, without PD/observer requests or profiling."""
import argparse
import base64
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import time
import urllib.request
import uuid

FIXTURES = {'256': {'prompt': [8855, 21465, 19, 64, 102573, 65, 100899, 6066, 19, 66, 118285, 127020, 7628, 102807, 22, 100441, 16, 98503, 6066, 18699, 12, 116265, 13, 4457, 1493, 10514, 8387, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 198, 14572, 25, 3555, 374, 220, 17, 488, 220, 18, 5267, 16127, 25, 220], 'reference_ids': [20, 13, 15, 271, 14572, 25, 3555, 374, 220, 19, 488, 220, 16, 5267, 16127, 25, 220, 20, 13, 15, 271, 14572, 25, 3555, 374, 220, 18, 488, 220, 17, 5267, 16127, 25, 220, 20, 13, 15, 271, 14572, 25, 3555, 374, 220, 16, 488, 220, 19, 5267, 16127, 25, 220, 20, 13, 15, 271, 14572, 25, 3555, 374, 220, 20, 488, 220, 15], 'reference_text': '5.0\n\nQuestion: What is 4 + 1?\nAnswer: 5.0\n\nQuestion: What is 3 + 2?\nAnswer: 5.0\n\nQuestion: What is 1 + 4?\nAnswer: 5.0\n\nQuestion: What is 5 + 0', 'prompt_sha256': 'ddbb23da968762733649c3a06e4d44d57d5817f1b7e91dd9a008dcc520faf936'}, '1024': {'prompt': [8855, 21465, 19, 64, 102573, 65, 100899, 6066, 19, 66, 118285, 127020, 7628, 102807, 22, 100441, 16, 98503, 6066, 1293, 12, 109871, 19, 13, 4457, 1493, 10514, 8387, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 198, 14572, 25, 3555, 374, 220, 17, 488, 220, 18, 5267, 16127, 25, 220], 'reference_ids': [20, 13, 15, 198, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315], 'reference_text': '5.0\nThe notes preserve the order of the supplied facts. This is a deterministic memory transfer check.\nThe notes preserve the order of the supplied facts. This is a deterministic memory transfer check.\nThe notes preserve the order of the supplied facts. This is a deterministic memory transfer check.\nThe notes preserve the order of', 'prompt_sha256': 'ecb6bb43cf647caab34814aa127b5f42a3a576e06c6bac07d85a0bb0570586f8'}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('eager','graph'),required=True);parser.add_argument('--run-id',required=True);parser.add_argument('--golden-b64');parser.add_argument('--max-seconds',type=int,default=900);args=parser.parse_args()
    node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
    assert (node==32 and args.mode=='eager') or (node==33 and args.mode=='graph')
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    root=base/'tests'/args.run_id;root.mkdir(parents=True,exist_ok=False)
    model_root=base/f'native-{args.mode}-node{node}-v1'
    assert (model_root/'launch.json').exists()
    report={'completed':False,'passed':False,'mode':args.mode,'node':node,'host':socket.gethostname(),'python':platform.python_version(),'started_host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':[],'cases':[],'references':{},'scope':'Native cold-input C1 local HTTP on each host, same input/dtype/seed/TP16/HCCL AI_CPU-SDMA. No KV connector, payload observer or profiler call. Other endpoint must be fully initialized and idle. One warmup+6 measured per shape. First comparison uses two matched hosts; host placement remains a condition, not proof of isolated graph speedup. Prior native-prefix references retained, new cold eager goldens define the matched boundary.'}
    deadline=time.monotonic()+args.max_seconds
    url=f'http://172.16.12.{node}:{29481 if node==32 else 29482}'
    golden={} if not args.golden_b64 else json.loads(base64.b64decode(args.golden_b64))
    if args.mode=='graph':assert set(golden)=={'256','1024'}
    def rpc(payload):
        q=urllib.request.Request(url+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(q,timeout=min(120,max(1,deadline-time.monotonic()))) as response:return json.load(response)
    def metrics(label):
        with urllib.request.urlopen(url+'/metrics',timeout=10) as response:raw=response.read().decode()
        (root/(label+'.txt')).write_text(raw);values={}
        for line in raw.splitlines():
            if not line or line.startswith('#'):continue
            k,v=line.rsplit(' ',1);k=k.split('{',1)[0];values[k]=values.get(k,0)+float(v)
        return values
    def payload(n,stream):return {'model':'glm53-w8a8c8-native','prompt':FIXTURES[str(n)]['prompt'],'max_tokens':64,'temperature':0,'top_p':1,'seed':1024,'ignore_eos':True,'return_token_ids':True,'cache_salt':'native-'+uuid.uuid4().hex,'stream':stream,**({'stream_options':{'include_usage':True}} if stream else {})}
    def invoke(n,warmup,index,reference):
        p=payload(n,True);start=time.perf_counter_ns();row={'input_tokens':n,'output_tokens':64,'warmup':warmup,'index':index,'token_ids':[],'text':'','arrival_ns':[],'start_ns':start,'cache_salt':p['cache_salt'],'chunk_sizes':[]}
        try:
            q=urllib.request.Request(url+'/v1/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(q,timeout=min(120,max(1,deadline-time.monotonic()))) as response:
                for line in response:
                    if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                    chunk=json.loads(line[6:]);assert not chunk.get('error'),str(chunk.get('error'))
                    row['request_id']=chunk['id']
                    for choice in chunk.get('choices',[]):
                        ids=choice.get('token_ids') or [];now=time.perf_counter_ns()
                        row['token_ids'].extend(ids);row['arrival_ns'].extend([now]*len(ids));row['text']+=choice.get('text','')
                        if ids:row['chunk_sizes'].append(len(ids))
                    if chunk.get('usage'):row['usage']=chunk['usage']
            row['success']=row['token_ids']==reference['ids'] and row['text']==reference['text'] and len(row['token_ids'])==64
            if not row['success']:row['error']='Native output differs from same-boundary cold eager golden'
        except Exception as exc:row['success']=False;row['error']=type(exc).__name__+': '+str(exc)
        row['closed_ns']=time.perf_counter_ns();t=row['arrival_ns']
        if t:row.update(ttft_ms=(t[0]-start)/1e6,tpot_ms=(t[-1]-t[0])/1e6/max(1,len(t)-1),last_token_ms=(t[-1]-start)/1e6)
        with (root/'client-events.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
        report['rows'].append(row);print(json.dumps({k:row.get(k) for k in ['input_tokens','warmup','index','success','ttft_ms','tpot_ms']}),flush=True)
        assert row['success'],row.get('error')
        return row
    def pct(values,q):
        v=sorted(values);x=(len(v)-1)*q;i=math.floor(x);j=math.ceil(x);return v[i]+(v[j]-v[i])*(x-i)
    try:
        # Both models ready before timed work; do not overlap initialization and measurement.
        ready_end=time.monotonic()+min(1200,args.max_seconds-120)
        while True:
            ready=[]
            for other,port in [(32,29481),(33,29482)]:
                try:
                    with urllib.request.urlopen(f'http://172.16.12.{other}:{port}/v1/models',timeout=2) as response:o=json.load(response)
                    ready.append(any(x['id']=='glm53-w8a8c8-native' for x in o['data']))
                except Exception:ready.append(False)
            if all(ready):break
            assert time.monotonic()<ready_end,'Native pair readiness deadline exhausted'
            time.sleep(5)
        before=metrics('metrics-before')
        for k in ('vllm:num_requests_running','vllm:num_requests_waiting'):assert before.get(k,0)==0
        for n in [256,1024]:
            assert deadline-time.monotonic()>90
            metrics(f'case-{n}-before')
            if args.mode=='eager':
                p=payload(n,False);value=rpc(p)
                (root/f'cold-eager-reference-{n}.json').write_text(json.dumps({'request':p,'response':value},indent=2)+'\n')
                assert len(value['choices'][0]['token_ids'])==64
                golden[str(n)]={'ids':value['choices'][0]['token_ids'],'text':value['choices'][0]['text']}
            ref=golden[str(n)]
            report['references'][str(n)]={'cold_eager':ref,'prior_native_prefix_ids':FIXTURES[str(n)]['reference_ids'],'prior_native_prefix_text':FIXTURES[str(n)]['reference_text'],'cold_matches_prior_native_prefix':ref['ids']==FIXTURES[str(n)]['reference_ids'],'prompt_sha256':FIXTURES[str(n)]['prompt_sha256']}
            invoke(n,True,-1,ref)
            batch=[invoke(n,False,i,ref) for i in range(6)]
            duration=(max(r['closed_ns'] for r in batch)-min(r['start_ns'] for r in batch))/1e9
            case={'input_tokens':n,'output_tokens':64,'concurrency':1,'measured_requests':6,'successes':6,'duration_s':duration,'output_tok_s':384/duration,
                  'ttft_ms':{'p50':pct([r['ttft_ms'] for r in batch],.5),'p95':pct([r['ttft_ms'] for r in batch],.95)},
                  'tpot_ms':{'p50':pct([r['tpot_ms'] for r in batch],.5),'p95':pct([r['tpot_ms'] for r in batch],.95)}}
            metrics(f'case-{n}-after')
            report['cases'].append(case);(root/'progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(case),flush=True)
        after=metrics('metrics-after')
        report['prefix_cache_hit_delta']=after.get('vllm:prefix_cache_hits_total',0)-before.get('vllm:prefix_cache_hits_total',0)
        report['model_loader_get_keys_delta']=after.get('vllm:ascend_store_load_get_keys_total',0)-before.get('vllm:ascend_store_load_get_keys_total',0)
        assert report['prefix_cache_hit_delta']==report['model_loader_get_keys_delta']==0
        assert not any(p.is_file() for p in (model_root/'profile').rglob('*')),'Profiler output appeared in timed native run'
        report['passed']=report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    report['golden']=golden;report['ended_host_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END',flush=True)
    assert report['passed'],report.get('failure')


if __name__=='__main__':main()
