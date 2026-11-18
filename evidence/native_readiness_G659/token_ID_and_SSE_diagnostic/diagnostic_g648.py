"""No configuration changes: distinguish sampling, transport, and text decoding."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, signal, socket, subprocess, time, urllib.request

ROOT=Path('/PRIVATE_TASK_ROOT')
RUN=ROOT/'vllm_pd_g630'; OUT=RUN/'diagnostic_g648'
P='http://<ROLE_HOST>:19103'; D='http://<ROLE_HOST>:19104'; PROXY='http://<ROLE_HOST>:18100'
assert socket.gethostname()=='<ROLE_HOSTNAME>'
os.chdir(ROOT)
lock=Path('/root/.codex/coord/deepseek-v41-p3d4-workload.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
OUT.mkdir(exist_ok=False); started=time.monotonic()
def save(name,d):
 p=OUT/name; q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def state(phase,**kw):
 d=dict(phase=phase,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,**kw)
 save('state.json',d);print(json.dumps(d),flush=True)
def alarm(*_):raise TimeoutError('G648 diagnostic bound 5400s; retain existing engines')
signal.signal(signal.SIGALRM,alarm);signal.alarm(5400)
def http(url,body=None,tag=None,timeout=30):
 req=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as f:
  raw=f.read();assert f.status==200
 if tag:(OUT/tag).write_bytes(raw)
 return raw
def identity():
 s=json.loads((RUN/'state.json').read_text());assert s['generation']==630
 assert str(RUN/'node_controller.py').encode() in Path('/proc',str(s['pid']),'cmdline').read_bytes()
 a=json.loads(subprocess.check_output(['docker','inspect',s['container_id']],cwd=ROOT))[0]
 assert a['State']['Running'] and a['Id']==s['container_id'] and a['Config']['Labels']['codex.run']==RUN.name
 assert a['Image']=='sha256:d5492ecfdb3071ff2d4bae3f849245014c54b63035d9eefb35243869461ddae1'
 for role,b in [('p',P),('d',D)]:
  http(b+'/health'); m=json.loads(http(b+'/v1/models'));assert m['data'][0]['id']=='ds41flash'
 return a
def metrics(tag,phase):
 for role,b in [('p',P),('d',D)]:http(b+'/metrics',tag=f'{tag}_{role}_{phase}_metrics.txt')
def completion(tag,body,manual=False):
 identity();save(tag+'_request.json',body);metrics(tag,'before');state('request_running',case=tag,manual_PD=manual)
 if manual:
  pre=dict(body,stream=False,max_tokens=1,min_tokens=1)
  pre['kv_transfer_params']={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None}
  save(tag+'_prefill_request.json',pre)
  pr=json.loads(http(P+'/v1/completions',pre,tag+'_prefill_response.json',1200))
  transfer=pr.get('kv_transfer_params');assert transfer,('No native transfer metadata',pr)
  dec=dict(body,kv_transfer_params=transfer);save(tag+'_decode_request.json',dec)
  raw=http(D+'/v1/completions',dec,tag+'_response.json',1200)
 else:raw=http(PROXY+'/v1/completions',body,tag+'_response.json',1200)
 r=json.loads(raw);metrics(tag,'after');c=r['choices'][0];ids=c.get('token_ids')
 d={'usage':r.get('usage'),'text_chars':len(c.get('text','')),'returned_token_ids':ids,'finish_reason':c.get('finish_reason'),'sha256':hashlib.sha256(raw).hexdigest(),'scope':'Diagnostic, not correctness/performance acceptance'}
 save(tag+'_summary.json',d);return r
try:
 a=identity();old=RUN/'functional_PD_g644';s=json.loads((old/'state.json').read_text());assert s['phase']=='failed_requires_review'
 p=Path('/proc',str(s['pid']),'cmdline');assert not p.exists() or b'functional_g644.py' not in p.read_bytes()
 for rel in ['vllm/entrypoints/openai/completion/serving.py','vllm/entrypoints/openai/completion/protocol.py','vllm/v1/engine/detokenizer.py','vllm/tokenizers/hf.py']:
  cmd=['docker','exec',a['Id'],'cat','/vllm-workspace/vllm/'+rel];r=subprocess.run(cmd,cwd=ROOT,capture_output=True)
  name='source/'+rel;p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  if r.returncode==0:p.write_bytes(r.stdout)
  else:save(rel.replace('/','_')+'_source_error.json',{'returncode':r.returncode,'stderr':r.stderr.decode(errors='replace')})
 base=json.loads((old/'tail_129_request.json').read_text());base.update(seed=0,return_token_ids=True,logprobs=5)
 results={}
 for tag,manual in [('proxy129a',False),('proxy129b',False),('manual129a',True),('manual129b',True)]:results[tag]=completion(tag,base,manual)
 long=json.loads((old/'native_32768_32_request.json').read_text());long.update(seed=0,return_token_ids=True,logprobs=1)
 results['manual32768']=completion('manual32768',long,True)
 pairs={}
 for x,y in [('proxy129a','proxy129b'),('manual129a','manual129b'),('proxy129b','manual129a')]:
  l=results[x]['choices'][0];r=results[y]['choices'][0];pairs[x+'__'+y]={'text_equal':l['text']==r['text'],'token_ids_equal':l.get('token_ids')==r.get('token_ids'),'token_ids_present':isinstance(l.get('token_ids'),list) and isinstance(r.get('token_ids'),list)}
 save('comparison.json',pairs)
 # A fresh 32K text SSE is diagnostic only; none of the reserved benchmark inputs.
 request=(RUN/'fresh_usage_request_g648.json').read_bytes();assert hashlib.sha256(request).hexdigest()=='c3e55a36beb6639f13f1ab8d6489ccd140061dcdde1577dff4f3705a07cdfc2d'
 body=json.loads(request);assert body['stream'] and body['stream_options']['include_usage'] and body['max_tokens']==1024
 (OUT/'fresh32768_1024_request.json').write_bytes(request);metrics('fresh32768_1024','before');state('fresh_SSE_running',HTTP_bound_s=1800)
 req=urllib.request.Request(PROXY+'/v1/completions',data=request,headers={'Content-Type':'application/json'})
 events=[];start=time.monotonic();first=None
 with urllib.request.urlopen(req,timeout=1800) as response,(OUT/'fresh32768_1024_response.sse').open('wb') as f:
  assert response.status==200;save('SSE_headers.json',dict(response.headers));done=False
  for line in response:
   f.write(line);f.flush()
   if not line.startswith(b'data: '):continue
   item=line[6:].strip()
   if item==b'[DONE]':done=True;continue
   d=json.loads(item);events.append(d)
   if d.get('choices') and d['choices'][0].get('text') and first is None:first=time.monotonic()-start
 metrics('fresh32768_1024','after')
 usages=[e['usage'] for e in events if e.get('usage')];last=usages[-1] if usages else None
 save('fresh_SSE_summary.json',{'done':done,'events':len(events),'final_usage':last,'elapsed_s':time.monotonic()-start,'first_content_s':first,'scope':'Fresh unscored standard SSE protocol/usage diagnostic; not AISBench or quality/performance acceptance'})
 assert done and last and last['prompt_tokens']==32768 and last['completion_tokens']==1024,(done,last)
 state('diagnostic_complete_requires_parent_review',comparison=pairs)
except BaseException as exc:state('failed_requires_review',error=repr(exc));raise
