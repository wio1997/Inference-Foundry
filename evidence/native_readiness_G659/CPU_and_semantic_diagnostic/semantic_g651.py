"""No configuration changes: distinguish sampling, transport, and text decoding."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, signal, socket, subprocess, time, urllib.request

ROOT=Path('/PRIVATE_TASK_ROOT')
RUN=ROOT/'vllm_pd_g630'; OUT=RUN/'semantic_g651'
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
def alarm(*_):raise TimeoutError('G651 diagnostic bound 5400s; retain existing engines')
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
 a=identity();old=RUN/'diagnostic_g648';st=json.loads((old/'state.json').read_text());assert st['phase']=='diagnostic_complete_requires_parent_review'
 p=Path('/proc',str(st['pid']),'cmdline');assert not p.exists() or b'diagnostic_g648.py' not in p.read_bytes()
 cpu=subprocess.run(['docker','exec',a['Id'],'python3','/task/vllm_pd_g630/CPU_detokenizer_repro_g650.py'],cwd=ROOT,capture_output=True,text=True,timeout=90)
 (OUT/'CPU_reproduction_stdout.txt').write_text(cpu.stdout);(OUT/'CPU_reproduction_stderr.txt').write_text(cpu.stderr);assert cpu.returncode==0,('CPU reproduction failed',cpu.returncode)
 save('CPU_reproduction.json',json.loads(cpu.stdout.strip().splitlines()[-1]))
 answers=[]
 def chat(tag,question,expected):
  identity();body={'model':'ds41flash','messages':[{'role':'user','content':question+' Return only the answer, without explanations.'}],'temperature':0,'seed':0,'max_tokens':256,'stream':False,'chat_template_kwargs':{'thinking':False},'return_token_ids':True}
  save(tag+'_request.json',body);metrics(tag,'before');state('request_running',case=tag);raw=http(PROXY+'/v1/chat/completions',body,tag+'_response.json',1200);metrics(tag,'after');response=json.loads(raw);choice=response['choices'][0];actual=(choice['message'].get('content') or '').strip();d={'expected':expected,'actual':actual,'passed':actual==expected,'usage':response.get('usage'),'returned_token_ids':choice.get('token_ids'),'response_sha256':hashlib.sha256(raw).hexdigest(),'scope':'Selected semantic smoke/cache case, not full quality suite'};save(tag+'_summary.json',d);answers.append(d);return d
 for i,(q,answer) in enumerate([('123+456','579'),('1000-379','621'),('17*19','323'),('144/12','12'),('3 to the fifth power','243'),('Sum the integers from 1 to 100 inclusive','5050'),('12.5 percent of 640','80'),('(8+7)*6','90')]):chat('semantic_'+str(i),q,answer)
 context=('Record G651: Memory, storage, network, attention, and scheduling are background topics. '*768)+'\nIgnore the background. Calculate 19 times 23.'
 first=chat('natural_prefix_first',context,'437');second=chat('natural_prefix_replay',context,'437')
 save('semantic_and_replay_result.json',{'all_semantic_cases_passed':all(x['passed'] for x in answers),'same_long_answer':first['actual']==second['actual'],'long_tokens_equal':first['returned_token_ids']==second['returned_token_ids'],'token_ids_returned':isinstance(first['returned_token_ids'],list),'scope':'10 selected semantic cases only; cache hit efficacy from independently captured counters; not full quality/performance acceptance'})
 state('semantic_CPU_diagnostic_complete_requires_parent_review')
except BaseException as exc:state('failed_requires_review',error=repr(exc));raise
