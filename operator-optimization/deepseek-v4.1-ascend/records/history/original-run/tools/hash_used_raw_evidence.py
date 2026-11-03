"""Hash only retained raw data actually used; no NPU or inference."""
from pathlib import Path
import hashlib,json
R=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
paths=[]
for rp in sorted(R.glob('*_ascend_pt')):
 db=list((rp/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'));meta=list(rp.glob('profiler_info_*.json'));assert len(db)==len(meta)==1
 paths+=db+meta
 for pat in ['PROF_*/host/start_info','PROF_*/host/host_start.log','PROF_*/device_*/start_info.*','PROF_*/device_*/info.json.*']:
  v=[x for x in rp.glob(pat) if not x.name.endswith('.done')];assert len(v)==1,(rp.name,pat,len(v));paths+=v
 if '_ep0_' in rp.name:paths.append(rp/'ASCEND_PROFILER_OUTPUT/trace_view.json')
assert len(paths)==97,len(paths)
records=[]
for p in paths:
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 after=p.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,p
 records.append({'relative_path':str(p.relative_to(R)),'size_bytes':before.st_size,'sha256':h.hexdigest(),'mtime_ns':before.st_mtime_ns})
o={'status':'USED_RAW_SOURCES_HASHED','trace_root':str(R),'records':records,'count':len(records),'kernel_csv_hashes':'evidence/TRACE-EXPORT-SHA256.json','limits':['Hashes verify file content for retained snapshot, not correctness of semantics or zero profiler overhead.','Other exported artifacts not used by this analysis may remain unhashed.'],'no_device_work':True}
p=T/'evidence/used-raw-source-sha256.json';assert not p.exists();p.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({'status':o['status'],'count':len(records),'total_bytes':sum(x['size_bytes'] for x in records)}))
