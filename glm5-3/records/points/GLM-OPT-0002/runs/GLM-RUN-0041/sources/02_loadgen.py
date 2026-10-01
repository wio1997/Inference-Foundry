"""Open-arrival GLM E2E measurement with unchanged request bodies.

This is a finite workload runner, not a stable-capacity certificate. All arrivals
get independent async tasks; connection/client dispatch delay remains in latency.
"""
import argparse,asyncio,base64,hashlib,json,math,time
from pathlib import Path
from phase_runner import atomic_json,utc
from sse_observer import native_error

async def execute(plan,root,transport=None):
    import httpx
    root=Path(root);requests=plan['requests']
    if not requests:raise ValueError('nonempty workload required')
    ids=[r['id'] for r in requests]
    if len(set(ids))!=len(ids):raise ValueError('request ids must be unique')
    for r in requests:
        if not isinstance(r['arrival_s'],(int,float)) or not math.isfinite(r['arrival_s']) or r['arrival_s']<0:raise ValueError('arrival must be finite nonnegative')
    client=httpx.AsyncClient(trust_env=False,transport=transport,timeout=httpx.Timeout(connect=10,read=None,write=30,pool=None),limits=httpx.Limits(max_connections=len(requests) or 1,max_keepalive_connections=len(requests) or 1))
    # Bodies load before the epoch so disk setup is not silently charged as arrival delay.
    bodies={r['id']:Path(r['body_path']).read_bytes() for r in requests}
    epoch_ns=time.monotonic_ns();epoch=epoch_ns/1e9;started_at=utc();records=[]
    async def one(r):
        target=epoch+r['arrival_s'];await asyncio.sleep(max(0,target-time.monotonic()));body=bodies[r['id']];dispatch=time.monotonic()
        row={'id':r['id'],'planned_arrival_s':r['arrival_s'],'dispatch_delay_s':dispatch-target,'body_path':r['body_path'],'body_sha256':hashlib.sha256(body).hexdigest(),'dispatch_at':utc(),'http_status':None,'done':False,'usage':None,'finish_reasons':{},'ttft_s':None,'elapsed_s':None,'error':None,'outcome':None,'expected':r.get('expected',{})}
        content=hashlib.sha256();events=0
        async def consume():
            nonlocal events
            async with client.stream('POST',r.get('endpoint',plan['endpoint']),content=body,headers={'Content-Type':'application/json',**({'x-request-id':r['request_header_id']} if r.get('request_header_id') else {})}) as response:
                row['http_status']=response.status_code
                with (root/(str(r['id'])+'.response.jsonl')).open('x') as raw:
                    if response.status_code!=200:
                        payload=await response.aread();raw.write(json.dumps({'elapsed_s':time.monotonic()-target,'error_body_sha256':hashlib.sha256(payload).hexdigest(),'error_body_bytes':len(payload),'error_body_base64':base64.b64encode(payload).decode('ascii')})+'\n');row['outcome']='http_error';return
                    if json.loads(body).get('stream',False):
                        async for line in response.aiter_lines():
                            if not line.startswith('data:'):continue
                            value=line[5:].strip();ts=time.monotonic()-target;raw.write(json.dumps({'elapsed_s':ts,'data':value})+'\n')
                            if value=='[DONE]':row['done']=True;break
                            event=json.loads(value);events+=1
                            error=native_error(event)
                            if error is not None:row['outcome']='native_error';row['error']=error
                            if event.get('usage'):row['usage']=event['usage']
                            for choice in event.get('choices',[]):
                                delta=choice.get('delta',{});text=delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning') or choice.get('text') or ''
                                if text or delta.get('tool_calls') or delta.get('function_call'):
                                    if row['ttft_s'] is None:row['ttft_s']=ts
                                    content.update(json.dumps(delta,sort_keys=True,ensure_ascii=False).encode())
                                if choice.get('finish_reason'):row['finish_reasons'][str(choice.get('index',0))]=choice['finish_reason']
                    else:
                        payload=await response.aread();value=json.loads(payload);raw.write(json.dumps({'elapsed_s':time.monotonic()-target,'data':value})+'\n');row['usage']=value.get('usage');row['done']=True;row['ttft_s']=None;content.update(payload)
                        error=native_error(value)
                        if error is not None:row['outcome']='native_error';row['error']=error
                        row['finish_reasons']={str(c.get('index',0)):c.get('finish_reason') for c in value.get('choices',[])}
        try:
            timeout=r.get('timeout_s')
            if timeout is not None:
                remaining=timeout-(dispatch-target)
                if remaining<=0:raise asyncio.TimeoutError()
                await asyncio.wait_for(consume(),timeout=remaining)
            else:await consume()
        except asyncio.TimeoutError:row['outcome']='timed_out';row['error']='request lifetime timeout; cleanup uses native client disconnect'
        except Exception as error:row['outcome']='transport_or_protocol_error';row['error']=type(error).__name__+': '+str(error)
        row['elapsed_s']=time.monotonic()-target;row['delta_stream_sha256']=content.hexdigest();row['sse_events']=events
        expected=row['expected'];wanted_status=expected.get('http_status',200)
        if row['outcome'] is None:row['outcome']='completed'
        if wanted_status!=200:
            row['valid']=row['http_status']==wanted_status and row['outcome']=='http_error';row['outcome']='expected_rejection' if row['valid'] else row['outcome']
        else:
            usage=row['usage'] or {};output=usage.get('completion_tokens');reasons=list(row['finish_reasons'].values())
            row['valid']=row['outcome']=='completed' and row['done'] and isinstance(output,int) and output>=0 and bool(reasons) and len(reasons)==json.loads(body).get('n',1) and all(x in ['stop','length','tool_calls','function_call','content_filter'] for x in reasons)
            if 'output_tokens' in expected:row['valid']=row['valid'] and output==expected['output_tokens']
            if 'prompt_tokens' in expected:row['valid']=row['valid'] and usage.get('prompt_tokens')==expected['prompt_tokens']
            row['tpot_s']=(row['elapsed_s']-row['ttft_s'])/(output-1) if row['valid'] and row['ttft_s'] is not None and output>1 and json.loads(body).get('n',1)==1 else None
        atomic_json(root/(str(r['id'])+'.request.json'),row);records.append(row)
    try:await asyncio.gather(*(one(r) for r in requests))
    finally:await client.aclose()
    elapsed=time.monotonic()-epoch;effective=[r for r in records if r['valid'] and r['outcome']=='completed']
    result={'kind':plan.get('kind','diagnostic'),'arrival_model':'open_loop; planned arrivals are latency origin','started_at_utc':started_at,'started_monotonic_ns':epoch_ns,'elapsed_s':elapsed,'request_count':len(records),'requests':sorted(records,key=lambda r:str(r['id'])),'valid':len(records)==len(requests) and all(r['valid'] for r in records),'successful_inference_requests':len(effective),'effective_output_tokens':sum(r['usage']['completion_tokens'] for r in effective),'expected_rejections':sum(r['outcome']=='expected_rejection' for r in records),'limits':['finite load is not proof of stable capacity','TTFT includes dispatch and connection queues; nonstream TTFT unobserved','SSE coalescing/MTP means wall TPOT is not device-step time; usage is authoritative; multi-choice TPOT is unobserved','cancel/timeout does not certify remote engine drain','delta_stream_sha256 includes chunk structure, not a final token/text identity']}
    result['effective_tps']=result['effective_output_tokens']/elapsed if elapsed else None;atomic_json(root/'load_result.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();path=Path(a.plan).resolve();plan=json.loads(path.read_text());result=asyncio.run(execute(plan,path.parent));print(json.dumps({k:result[k] for k in ['valid','elapsed_s','successful_inference_requests','effective_output_tokens','effective_tps']}));raise SystemExit(0 if result['valid'] else 1)
