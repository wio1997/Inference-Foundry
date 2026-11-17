"""Bounded CPU-only exact-token datasets, shared byte-for-byte by both framework arms."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,importlib.metadata,json,os,random,signal,time
from tokenizers import Tokenizer
ROOT=Path('/task/client_ds41_g83');DATA=ROOT/'datasets';TOKENIZER=Path('/models/ds41/tokenizer.json')
LENGTHS=[32768,65536,131072,262144];PHASES=['warmup1','warmup2','warmup3','warmup4','warmup5','formal']
WORDS=('research systems language performance memory attention communication request response document analysis evidence input output model token scheduling balance network storage measurement source consistent implementation calculation sequence state cache expert The experiment compares two software frameworks using the same quantized model For every observation preserve raw data and explain scope of conclusion Several engineers discuss a long technical document about distributed inference A reader asks how each stage consumes time under different sequence lengths').split()
TOK=None

def initialize():
 global TOK
 TOK=Tokenizer.from_file(str(TOKENIZER))
 os.environ['TOKENIZERS_PARALLELISM']='false'

def request(task):
 phase,index=task;seed=int.from_bytes(hashlib.sha256(f'ds41-v1:{phase}:{index}'.encode()).digest()[:8],'big');rng=random.Random(seed)
 text=' '.join(rng.choices(WORDS,k=300000));base=TOK.encode(text,add_special_tokens=False).ids
 while len(base)<max(LENGTHS)+512:
  text+=' '+' '.join(rng.choices(WORDS,k=100000));base=TOK.encode(text,add_special_tokens=False).ids
 result=[]
 for n in LENGTHS:
  nonce='ds41-v1-'+hashlib.sha256(f'{phase}:{index}:{n}'.encode()).hexdigest()
  ids=TOK.encode(nonce+' ',add_special_tokens=False).ids+base
  for attempt in range(8):
   prompt=TOK.decode(ids[:n],skip_special_tokens=False);actual=TOK.encode(prompt,add_special_tokens=False).ids
   if len(actual)==n:break
   ids=actual+base[n:n+256]
  assert len(actual)==n and prompt.startswith(nonce)
  row={'question':prompt,'answer':'','max_out_len':1024,'input_tokens':n,'request_id':f'{phase}_{n}_{index:03d}'}
  result.append((phase,n,json.dumps(row,ensure_ascii=False)+'\n',actual[:128]))
 return result

def main():
 os.chdir(ROOT);started=time.monotonic();DATA.mkdir(exist_ok=False)
 def state(phase,**extra):
  value={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,**extra};p=ROOT/'assets_state.tmp';p.write_text(json.dumps(value,indent=2));p.replace(ROOT/'assets_state.json');print(json.dumps(value),flush=True)
 def alarm(sig,frame):raise TimeoutError('CPU assets 1800s bound')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(1800)
 handles={(p,n):(DATA/f'{p}_{n}.jsonl').open('x') for p in PHASES for n in LENGTHS}
 tasks=[(p,i) for p in PHASES for i in range(32 if p=='formal' else 16)];prefixes={};counts={key:0 for key in handles}
 try:
  with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
   for count,result in enumerate(pool.map(request,tasks,chunksize=1),1):
    for phase,n,line,prefix in result:
     key=tuple(prefix);assert key not in prefixes,('Full-page prefix reused',phase,n,prefixes.get(key));prefixes[key]=(phase,n);handles[phase,n].write(line);counts[phase,n]+=1
    state('generating',finished_requests=count,total_requests=len(tasks),workers=4)
 finally:
  for f in handles.values():f.close()
 manifest={'schema':1,'model':'DeepSeek-V4.1-Flash-1017-w4a8','tokenizer_sha256':hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),'input_lengths':LENGTHS,'output_tokens':1024,'client_concurrency':16,'warmup_requests_per_pass':16,'formal_requests':32,'construction':'Deterministic shuffled English synthetic text; unique first128token page per phase/request/length; exact no-special-token tokenizers roundtrip. Actual server usage still required.','cold':'No formal bytes submitted before first scored cold in each model epoch; independent full-shape warm inputs; no cache flush claimed','hot':'Replay identical formal bytes without resetting prefix state; efficacy comes from actual cache counters','profiles':[],'package_versions':{},'server_usage_validated':False}
 for (phase,n),f in handles.items():
  path=Path(f.name);assert counts[phase,n]==(32 if phase=='formal' else 16)
  manifest['profiles'].append({'phase':phase,'input_tokens':n,'requests':counts[phase,n],'min_actual':n,'max_actual':n,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 for package in ['ais_bench','ais-bench','tokenizers','transformers']:
  try:manifest['package_versions'][package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:pass
 (ROOT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2));(ROOT/'cold_prefix_audit.json').write_text(json.dumps({'verified':True,'prefix_tokens':128,'prefix_count':len(prefixes),'all_phases_lengths_request_prefixes_unique':True,'tokenizer_sha256':manifest['tokenizer_sha256'],'no_service_request_submitted':True},indent=2));state('complete',profiles=len(manifest['profiles']),prefix_count=len(prefixes),server_usage_validated=False)
if __name__=='__main__':main()
