"""Read existing owned P traces and installed cache sources; no inference/imports."""
import ast,base64,gzip,hashlib,importlib.util,json,os,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L11-P-trace-hotspots-readonly-v2'
CAP='20261018T-L11-sglang-C32-P-two-step-native-v1'
PROBE=r'''
import pathlib,json,hashlib,collections,re,csv,sys
out={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'traces':[],'sources':[]}
for item in TRACES:
 p=pathlib.Path(item['path']);assert p.is_file() and not p.is_symlink()
 raw=p.read_bytes();assert len(raw)==item['bytes']<=512*1024**2 and hashlib.sha256(raw).hexdigest()==item['sha256']
 value=json.loads(raw);del raw
 events=value['traceEvents'] if isinstance(value,dict) else value
 assert isinstance(events,list) and len(events)<=2_000_000
 duration=[]
 for e in events:
  if e.get('ph')!='X':continue
  try:ts=float(e['ts']);dur=float(e['dur'])
  except (KeyError,TypeError,ValueError):continue
  if not __import__('math').isfinite(ts+dur) or dur<0:continue
  e['ts']=ts;e['dur']=dur;duration.append(e)
 def compact(e):
  return {k:e[k] for k in ['name','cat','pid','tid','ts','dur','args'] if k in e}
 selected=sorted([e for e in duration if re.search('index_put|transfer_kv_dim_exchange|Nonzero',e.get('name',''),re.I)],key=lambda e:e['dur'],reverse=True)[:24]
 contexts=[]
 for e in selected:
  left=e['ts'];right=left+e['dur']
  same=[v for v in duration if v.get('pid')==e.get('pid') and v.get('tid')==e.get('tid')]
  parents=sorted([v for v in same if v is not e and v['ts']<=left and v['ts']+v['dur']>=right],key=lambda v:v['dur'])[:5]
  children=sorted([v for v in same if v is not e and left<=v['ts'] and v['ts']+v['dur']<=right],key=lambda v:v['dur'],reverse=True)[:10]
  following=sorted([v for v in same if v is not e and right<=v['ts']<right+5000],key=lambda v:v['ts'])[:5]
  contexts.append({'event':compact(e),'parents':[compact(v) for v in parents],'children':[compact(v) for v in children],'following':[compact(v) for v in following]})
 names=collections.Counter(e.get('cat','') for e in events)
 steps=[compact(e) for e in duration if re.match(r'step\[',e.get('name',''))]
 # Narrow full operator summaries disambiguate omitted repeated spans.
 csvpath=p.parent/'operator_details.csv';aggregates={}
 with csvpath.open(newline='') as file:
  reader=csv.DictReader(file);headers=reader.fieldnames
  for row in reader:
   name=row['Name']
   if not re.search('transfer_kv|index_put|step\[|draft_extend|nonzero|copy_|cpu|synchronize|item',name,re.I):continue
   a=aggregates.setdefault(name,{'count':0,'host_self_us':0.,'host_total_us':0.});a['count']+=1
   a['host_self_us']+=float(row['Host Self Duration(us)']);a['host_total_us']+=float(row['Host Total Duration(us)'])
 out['traces'].append({'rank':item['rank'],'namespace_pid':item['namespace_pid'],'path':str(p),'bytes':item['bytes'],'sha256':item['sha256'],'event_categories':dict(names),'steps':steps,'hotspot_contexts':contexts,'operator_csv_headers':headers,'operator_aggregates':aggregates,'raw_trace_transferred':False,'analysis_pass':bool(steps) and bool(contexts),'raw_event_time_field_types':dict(collections.Counter(str((type(e.get('ts')).__name__,type(e.get('dur')).__name__)) for e in events[:20]))})
 del value,events,duration
if READ_SOURCES:
 sr=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt')
 paths=[sr/x for x in ['mem_cache/memory_pool.py','mem_cache/memory_pool_host.py','managers/cache_controller.py','managers/schedule_batch.py','mem_cache/radix_cache.py']]
 paths+=list((sr/'mem_cache/pool_host').rglob('*.py'))
 paths+=list((sr/'mem_cache/allocator').rglob('*.py'))
 total=0
 for p in sorted(set(paths)):
  if not p.is_file():continue
  assert not p.is_symlink() and p.resolve().is_relative_to(sr.resolve())
  raw=p.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=8*1024**2
  out['sources'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})
assert not any(x.startswith(('torch','sglang','memcache_hybrid','memfabric_hybrid')) for x in sys.modules)
'''
HOST=r'''
import pathlib,subprocess,json,hashlib,socket,gzip,base64
assert socket.gethostname()==HOSTNAME
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0];assert d['Id']==CID and d['Image']==IMAGE and d['Name']=='/'+NAME and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED
state=json.loads((pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and 'model_exit_code' not in state
r=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-S','-'],input=PROBE+"\nraw=json.dumps(out).encode();print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'gzip_base64':__import__('base64').b64encode(__import__('gzip').compress(raw,mtime=0)).decode()}))",capture_output=True,text=True,timeout=240)
assert r.returncode==0,r.stderr
print(json.dumps({'project_id':'glm53','node':NODE,'container_id':CID,'probe':json.loads(r.stdout),'stderr':r.stderr,'model_requests':0,'SDK_init_calls':0,'remote_mutations':False}))
'''
def render(src,values):
 import re
 src=re.sub(r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b',lambda m:repr(values[m[0]]),src);ast.parse(src);return src
def plans():
 settings=json.loads((ROOT/'config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v2.json').read_text())
 for n in [34,35]:
  wire=json.loads((ROOT/f'runs/{CAP}/node-{n}.json').read_text());assert wire['rc']==0
  o=json.loads(wire['stdout']);p=o['native_profile'];assert p['analysis_pass']
  chosen=[w for w in p['workers'].values() if w['TP_rank'] in [0,8,16,24]];assert len(chosen)==2
  traces=[]
  for w in chosen:
   folder=next(name for name,v in p['workers'].items() if v['TP_rank']==w['TP_rank'])
   row=next(x for x in p['files'] if x['relative']==folder+'/ASCEND_PROFILER_OUTPUT/trace_view.json')
   traces.append({'rank':w['TP_rank'],'namespace_pid':w['namespace_pid'],'path':'/run/glm53/profile/'+settings['run_id']+'/'+row['relative'],'bytes':row['bytes'],'sha256':row['sha256']})
  probe=render(PROBE,{'TRACES':traces,'READ_SOURCES':n==34})
  yield n,settings,probe
def main():
 proof=json.loads((ROOT/'results/C32-L11-P-two-step-evidence-v1.json').read_text());assert proof['diagnostic_evidence_pass']
 path=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
 spec=importlib.util.spec_from_file_location('inventory',path);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','nodes':{},'model_requests':0,'SDK_init_calls':0,'remote_mutations':False}
 try:
  for n,settings,probe in plans():
   item=settings['runtime_containers'][str(n)]
   code=render(HOST,{'HOSTNAME':b.NODES[n][1],'NODE':n,'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'SHARED':settings['shared_run_id'],'PROBE':probe});(run/f'host-{n}.py').write_text(code)
   pod=b.NODES[n][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']==f'172.16.12.{n}'
   command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);out=b.invoke([command],code,timeout=300);(run/f'node-{n}.json').write_text(json.dumps(out,indent=2)+'\n');receipt['nodes'][str(n)]=out['rc'];assert out['rc']==0
   obj=json.loads(out['stdout']);packet=obj['probe'];raw=gzip.decompress(base64.b64decode(packet['gzip_base64'],validate=True));assert len(raw)==packet['bytes'] and hashlib.sha256(raw).hexdigest()==packet['sha256'];(run/f'analysis-{n}.json').write_bytes(raw)
   print(json.dumps({'node':n,'rc':0,'analysis_bytes':len(raw)}),flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
