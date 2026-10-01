"""Real streaming GLM requests, explicit arrival offsets and end-to-end evidence."""
import argparse, concurrent.futures, hashlib, json, threading, time, urllib.request
from pathlib import Path
from phase_runner import atomic_json, utc

def main(spec_path):
    path=Path(spec_path).resolve();spec=json.loads(path.read_text());out=path.parent
    prompts=[json.loads(l)['question'] for l in Path(spec['dataset']).read_text().splitlines() if l.strip()]
    started=time.monotonic();lock=threading.Lock();rows=[]
    def request(item):
        target=started+item.get('arrival_s',0);time.sleep(max(0,target-time.monotonic()))
        t0=time.monotonic();row={'id':item['id'],'prompt_id':item['prompt_id'],'planned_arrival_s':item.get('arrival_s',0),'arrived_s':t0-started,'started_at':utc(),'max_tokens':item['max_tokens'],'first_content_s':None,'usage':None,'finish_reason':None,'done':False,'error':None}
        body={'model':'glm-52','messages':[{'role':'user','content':prompts[item['prompt_id']]}],'stream':True,'stream_options':{'include_usage':True},'temperature':0,'ignore_eos':True,'max_tokens':item['max_tokens']}
        digest=hashlib.sha256();chunks=0
        try:
            req=urllib.request.Request(spec['endpoint'],data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=spec.get('socket_timeout_s',180)) as response, (out/(str(item['id'])+'.sse.jsonl')).open('x') as raw:
                for line in response:
                    text=line.decode().strip()
                    if not text.startswith('data:'):continue
                    value=text[5:].strip();ts=time.monotonic()-t0
                    raw.write(json.dumps({'elapsed_s':ts,'data':value})+'\n')
                    if value=='[DONE]':row['done']=True;break
                    event=json.loads(value)
                    if event.get('usage'):row['usage']=event['usage']
                    for choice in event.get('choices',[]):
                        content=choice.get('delta',{}).get('content') or choice.get('delta',{}).get('reasoning_content') or choice.get('delta',{}).get('reasoning') or ''
                        if content:
                            if row['first_content_s'] is None:row['first_content_s']=ts
                            digest.update(content.encode());chunks+=1
                        if choice.get('finish_reason'):row['finish_reason']=choice['finish_reason']
        except Exception as e:row['error']=type(e).__name__+': '+str(e)
        row['elapsed_s']=time.monotonic()-t0;row['content_sha256']=digest.hexdigest();row['content_chunks']=chunks
        count=(row['usage'] or {}).get('completion_tokens');row['valid']=row['done'] and row['error'] is None and count==item['max_tokens'] and row['finish_reason']=='length' and chunks>0
        row['tpot_s']=(row['elapsed_s']-row['first_content_s'])/(count-1) if count and count>1 and row['first_content_s'] is not None else None
        atomic_json(out/(str(item['id'])+'.request.json'),row)
        with lock:rows.append(row)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=spec['concurrency']) as pool:
        list(pool.map(request,spec['requests']))
    elapsed=time.monotonic()-started
    result={'kind':spec.get('kind','diagnostic'),'cache_condition':spec['cache_condition'],'endpoint':spec['endpoint'],'requests':sorted(rows,key=lambda r:r['id']),'elapsed_s':elapsed,'valid':len(rows)==len(spec['requests']) and all(r['valid'] for r in rows),'effective_output_tokens':sum((r['usage'] or {}).get('completion_tokens',0) for r in rows if r['valid'])}
    result['effective_tps']=result['effective_output_tokens']/elapsed
    atomic_json(out/'probe_result.json',result)
    print(json.dumps({'valid':result['valid'],'elapsed_s':elapsed,'effective_tps':result['effective_tps'],'requests':len(rows)}))
    return 0 if result['valid'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();raise SystemExit(main(a.spec))
