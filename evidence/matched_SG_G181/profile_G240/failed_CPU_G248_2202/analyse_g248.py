"""Read-only CPU analysis after native SG capture completion; no SDK mutations."""
from pathlib import Path
import collections,concurrent.futures,csv,datetime,fcntl,hashlib,json,math,os,re,signal,socket,subprocess,time

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def scan(root):
 return {str(p.relative_to(root)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()}
def union(rows):
 total=0.;end=None
 for a,b in sorted(rows):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total

def analyse(entry):
 root=Path(entry['capture']);trace=root/'ASCEND_PROFILER_OUTPUT/trace_view.json';op=root/'ASCEND_PROFILER_OUTPUT/op_statistic.csv'
 assert trace.is_file() and op.is_file();raw=trace.read_bytes();data=json.loads(raw);events=data.get('traceEvents',[]) if isinstance(data,dict) else data
 names={str(e['pid']):e.get('args',{}).get('name','') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
 hw={k for k,v in names.items() if v in ['Ascend Hardware','NPU Hardware']};assert hw,'Missing hardware process metadata'
 labels={str(e['pid']):e.get('args',{}).get('labels','') for e in events if e.get('ph')=='M' and e.get('name')=='process_labels'}
 for p in hw:
  m=re.search(r'\bNPU\s+(\d+)\b',labels.get(p,''));assert m and int(m[1])==entry['rank'],labels
 kernels=[e for e in events if e.get('ph')=='X' and str(e.get('pid')) in hw and isinstance(e.get('dur'),(int,float)) and math.isfinite(e['dur']) and e['dur']>0];assert kernels
 cpu=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cpu_op'];assert cpu
 models=[e for e in kernels if re.search(r'matmul|gemm|rmsnorm|attention|gateddelta|groupedmat|sinkhorn|moe',str(e.get('name','')),re.I)];assert models,'No explicit model compute kernel; memcpy metadata insufficient'
 stats=list(csv.DictReader(op.open()));assert stats and all(int(x['Device_id'])==entry['rank'] for x in stats)
 groups=collections.defaultdict(lambda:{'count':0,'sum_us':0.})
 for e in kernels:g=groups[str(e.get('name',''))];g['count']+=1;g['sum_us']+=e['dur']
 top=sorted([dict(name=k,**v) for k,v in groups.items()],key=lambda x:x['sum_us'],reverse=True)[:50]
 steps=[{'name':e['name'],'ts':e['ts'],'duration_us':e['dur']} for e in cpu if str(e.get('name','')).startswith('step[')]
 return {'rank':entry['rank'],'namespace_pid':entry['namespace_pid'],'physical_npu':entry['physical_npu'],'raw_capture':root.name,'trace_sha256':hashlib.sha256(raw).hexdigest(),'op_statistic_sha256':sha(op),'cpu_X_events':len(cpu),'hardware_X_events':len(kernels),'explicit_model_X_events':len(models),'model_examples':sorted({str(x['name']) for x in models})[:20],'hardware_busy_union_us':union([(e['ts'],e['ts']+e['dur']) for e in kernels]),'hardware_span_us':max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels),'hardware_sum_us':sum(e['dur'] for e in kernels),'top50_hardware':top,'CPU_step_windows':steps,'device_activity_verified':True,'scope':'Within-rank device evidence; sums overlap and are not wallclock shares; no cross-host clock calibration or causal attribution'}

def main():
 host=socket.gethostname();assert host in ['<ROLE_NODE>','<ROLE_NODE>'];role='p' if host.endswith('1') else 'd';root=Path('/data' if role=='p' else '/data1')/'tiankuan/yxy/glm53_pd_align_910b_20261116';base=root/'sg_pd_g181';profile=base/f'profile_g240/traces/{role}';out=base/'profile_g240/review_g248';out.mkdir(exist_ok=False)
 os.sched_setaffinity(0,{188,189});os.nice(10)
 lock=(out/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(phase,**k):
  d=dict(phase=phase,pid=os.getpid(),role=role,at=datetime.datetime.now(datetime.timezone.utc).isoformat());d.update(k);atomic(out/'state.json',d)
 status('identity_and_raw_freeze')
 try:
  info=json.loads(subprocess.check_output(['docker','inspect',f'yxy-glm53-sg-{role}-g181']))[0]
  expected={'p':'b5e25e930d53701e2347ef3dc0b229a10428f1bf072fb467d393f344ac4bdf30','d':'ebf0f45cff870828df7b81496c74adfe48cb6c0819706c99ab5a24c407885900'}[role];assert info['Id']==expected and info['State']['Running']
  npu=subprocess.check_output(['npu-smi','info'],text=True);workers=re.findall(r'\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|\s*sglangschedul',npu);assert len(workers)==8
  mapping={}
  for dev,pid in workers:
   text=Path('/proc',pid,'status').read_text();nspid=next(int(line.split()[-1]) for line in text.splitlines() if line.startswith('NSpid:'));mapping[nspid]={'host_pid':int(pid),'physical_npu':int(dev)}
  entries=[]
  for capture in profile.glob('*_ascend_pt'):
   infos=list(capture.glob('profiler_info_*.json'));assert len(infos)==1;rank=int(re.search(r'profiler_info_(\d+)\.json',infos[0].name)[1]);m=re.search(r'_(\d+)_\d+_ascend_pt$',capture.name);assert m;pid=int(m[1]);assert pid in mapping and mapping[pid]['physical_npu']==rank
   entries.append(dict(rank=rank,namespace_pid=pid,capture=str(capture),**mapping[pid]))
  assert len(entries)==8 and {e['rank'] for e in entries}==set(range(8));entries.sort(key=lambda x:x['rank'])
  identity={'role':role,'container_id':info['Id'],'image_id':info['Image'],'started_at':info['State']['StartedAt'],'mapping':entries};atomic(out/'identity.json',identity)
  first=scan(profile);time.sleep(2);second=scan(profile);assert first==second and first,'Rawprofilechangedafter DONE; preserve and wait, do not recapture'
  atomic(out/'raw_stability.json',{'role':role,'files':second,'two_SHA_scans_equal':True,'scope':'Originalraw read-only; no SDK or trace rewrite in this analysis job'})
  status('CPU_rank_analysis',raw_bytes=sum(x['bytes'] for x in second.values()))
  with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
   reports=list(pool.map(analyse,entries))
  assert scan(profile)==second,'Rawprofilechangedwhile reading'
  for report in reports:atomic(out/f"rank_{report['rank']}.json",report)
  result={'role':role,'role_complete':all(x['device_activity_verified'] for x in reports),'reports':[{'rank':x['rank'],'hardware_events':x['hardware_X_events'],'model_events':x['explicit_model_X_events']} for x in reports],'raw_bytes':sum(x['bytes'] for x in second.values()),'raw_files':len(second),'raw_unchanged_after_analysis':True,'scope':'MatchedG138input nativeG181 capture only; not full optimization map or model accuracy','cross_host_clock_calibration':False};atomic(out/'role.json',result);status('passed',**result)
 except BaseException as e:status('failed_requires_review',error=repr(e));raise
if __name__=='__main__':main()
