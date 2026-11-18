"""Complete pinned GSM8K test; raw requests/responses, official strict grader, no retries."""
from pathlib import Path
import ast,concurrent.futures,datetime,hashlib,json,os,re,signal,time,urllib.error,urllib.request
ROOT=Path('/task/client_ds41_g503/quality_g655');OUT=ROOT/'vllm_native630_full_test'
os.chdir(ROOT);OUT.mkdir(exist_ok=False);started=time.monotonic()
URL='http://<ROLE_HOST>:18100';expected='ds41flash'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def state(phase,**kw):
 d={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,**kw};save(OUT/'state.json',d);print(json.dumps(d),flush=True)
def alarm(*_):raise TimeoutError('G655 whole quality evaluation 7200s bound, no automatic restart')
signal.signal(signal.SIGALRM,alarm);signal.alarm(7200)
provenance=json.loads((ROOT/'dataset_provenance.json').read_text())
for e in provenance['files']:assert sha(ROOT/e['path'])==e['sha256']
dataset_path=ROOT/'grade_school_math/data/test.jsonl';data=[json.loads(s) for s in dataset_path.read_text().splitlines()];assert len(data)==1319
source=(ROOT/'grade_school_math/dataset.py').read_text();nodes=[]
for n in ast.parse(source).body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['ANS_RE','INVALID_ANS'] for t in n.targets):nodes.append(n)
 if isinstance(n,ast.FunctionDef) and n.name in ['extract_answer','is_correct']:nodes.append(n)
assert len(nodes)==4
env={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned official GSM8K grader>','exec'),env)
extract=env['extract_answer'];correct=env['is_correct'];assert all(extract(d['answer'])!=env['INVALID_ANS'] for d in data)
with urllib.request.urlopen(URL+'/v1/models',timeout=20) as f:models=json.load(f)
assert models['data'][0]['id']==expected;save(OUT/'actual_models.json',models)
manifest={'dataset_sha256':sha(dataset_path),'dataset_commit':provenance['commit'],'grader_sha256':sha(ROOT/'grade_school_math/dataset.py'),'grader_selection':'Unmodified official ANS_RE/INVALID_ANS/extract_answer/is_correct AST; exact string after comma removal, no fallback','test_rows':1319,'concurrency':16,'max_tokens':2048,'temperature':0,'seed':0,'thinking':False,'few_shot_examples':0,'retries':0,'native_image':'sha256:d5492ecfdb3071ff2d4bae3f849245014c54b63035d9eefb35243869461ddae1','scope':'Complete zero-shot arithmetic quality suite. Not full model-capability certification or AISBench performance. Same dataset/prompt/budget/grader must be used for SGLang.'}
save(OUT/'manifest.json',manifest)
def task(i,row):
 tag=f'{i:04d}';body={'model':expected,'messages':[{'role':'user','content':row['question']+'\nSolve the problem. Show a brief calculation, then end your response with "#### <numeric answer>".'}],'temperature':0,'seed':0,'max_tokens':2048,'stream':False,'chat_template_kwargs':{'thinking':False}}
 save(OUT/(tag+'_request.json'),body);before=time.monotonic();req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=600) as f:raw=f.read();assert f.status==200
  (OUT/(tag+'_response.json')).write_bytes(raw);r=json.loads(raw);assert r['model']==expected;choice=r['choices'][0];usage=r.get('usage') or {};assert usage.get('completion_tokens',0)>0
  text=choice['message'].get('content') or '';return {'index':i,'transport_ok':True,'correct':correct(text,row),'expected':extract(row['answer']),'extracted':extract(text),'finish_reason':choice.get('finish_reason'),'usage':usage,'elapsed_s':time.monotonic()-before,'request_sha256':sha(OUT/(tag+'_request.json')),'response_sha256':sha(OUT/(tag+'_response.json'))}
 except Exception as exc:
  if isinstance(exc,urllib.error.HTTPError):(OUT/(tag+'_HTTP_error.txt')).write_bytes(exc.read())
  return {'index':i,'transport_ok':False,'correct':False,'error':repr(exc),'elapsed_s':time.monotonic()-before,'expected':extract(row['answer'])}
try:
 state('full_test_running',rows=1319,concurrency=16)
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool,(OUT/'results.jsonl').open('x') as log:
  futures={pool.submit(task,i,row):i for i,row in enumerate(data)}
  for f in concurrent.futures.as_completed(futures):
   r=f.result();results.append(r);log.write(json.dumps(r)+'\n');log.flush()
   if len(results)%64==0:state('full_test_running',completed=len(results),correct=sum(x['correct'] for x in results),transport_failures=sum(not x['transport_ok'] for x in results))
 assert len(results)==1319 and {x['index'] for x in results}==set(range(1319))
 right=sum(x['correct'] for x in results);bad=sum(not x['transport_ok'] for x in results);truncated=sum(x.get('finish_reason')=='length' for x in results)
 summary={'completed':1319,'correct':right,'accuracy':right/1319,'transport_failures':bad,'length_limited_responses':truncated,'all_raw_responses_present':all((OUT/f'{i:04d}_response.json').is_file() for i in range(1319)),'dataset_sha256':sha(dataset_path),'grader_sha256':sha(ROOT/'grade_school_math/dataset.py'),'results_jsonl_sha256':sha(OUT/'results.jsonl'),'scope':manifest['scope'],'quality_comparison_accepted':False,'reason':'Requires parent raw/SHA audit and matched SGLang full-set score; no predetermined quality threshold retrofitted to this result'}
 save(OUT/'result.json',summary);state('full_test_complete_requires_parent_review',**summary)
except BaseException as exc:state('failed_requires_review',error=repr(exc));raise
