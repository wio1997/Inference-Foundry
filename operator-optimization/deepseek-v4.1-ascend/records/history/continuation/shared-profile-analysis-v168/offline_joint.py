"""Vendor export and all-source connection census after collection is terminal."""
import collections,hashlib,json,os,resource,sqlite3,time
from pathlib import Path
import torch
from torch_npu.profiler.profiler import analyse
ROOT=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/joint168')
OUT=ROOT/'derived';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1);resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
status=json.loads((ROOT/'STATUS.json').read_text());assert status['state'] in ['ORIGINAL_SERVICE_RESTORED_REVIEW_PENDING','ACQUISITION_FAILED_ORIGINAL_SERVICE_RESTORED']
profiles=[p for sub in ['startup','runtime'] for p in sorted((ROOT/sub).glob('*_ascend_pt'))]
inputs=[p for base in profiles for p in base.rglob('*') if p.is_file()]
manifest=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
(OUT/'PROFILE-INPUT-MANIFEST.json').write_text(json.dumps({'profiles':len(profiles),'files':manifest},indent=2)+'\n')
for sub in ['startup','runtime']:
 if any((ROOT/sub).glob('*_ascend_pt')):analyse(str(ROOT/sub),max_process_number=4,export_type='db')
changed=[]
for a in manifest:
 p=ROOT/a['path']
 if not p.exists() or p.stat().st_size!=a['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=a['sha256']:changed.append(a['path'])
(OUT/'RAW-POSTCHECK.json').write_text(json.dumps({'raw_inputs_unchanged':not changed,'changed':changed,'input_files':len(manifest)},indent=2)+'\n')

def one(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 tables={a[0] for a in c.execute("select name from sqlite_master where type='table'")};s=dict(c.execute('select id,value from STRING_IDS'));summary={'database':str(db),'sha256':hashlib.sha256(db.read_bytes()).hexdigest(),'tables':sorted(tables),'phase':db.relative_to(ROOT).parts[0]}
 summary['rows']={n:c.execute('select count(*) from '+n).fetchone()[0] if n in tables else 0 for n in ['TASK','COMPUTE_TASK_INFO','CANN_API','MSTX_EVENTS','PYTORCH_API','TASK_PMU_INFO']}
 pidrows=[r[0] for r in c.execute('select distinct globalPid from TASK')];summary['native_pids']=pidrows
 if not {'TASK','COMPUTE_TASK_INFO','PYTORCH_API','CONNECTION_IDS'}<=tables:c.close();return summary
 # PYTORCH_API.connectionId is a CONNECTION_IDS set ID,not the native flow ID.
 cpu_by_native=collections.defaultdict(list)
 for row in c.execute('select p.rowid CPU_ROW,p.*,k.connectionId native_connection from PYTORCH_API p join CONNECTION_IDS k on p.connectionId=k.id where p.type=50001'):
  cpu_by_native[row['native_connection']].append(dict(row))
 chain=collections.defaultdict(list)
 if 'PYTORCH_CALLCHAINS' in tables:
  for row in c.execute('select * from PYTORCH_CALLCHAINS'):chain[row['id']].append({'depth':row['stackDepth'],'text':s.get(row['stack'])})
 templates={};assignment=collections.Counter();captured_instances=0
 for row in c.execute('select t.rowid TASK_ROW,t.*,x.name,x.inputShapes,x.outputShapes from TASK t join COMPUTE_TASK_INFO x on t.globalTaskId=x.globalTaskId where t.modelId!=4294967295'):
  r=dict(row);captured_instances+=1;cpus=cpu_by_native.get(r['connectionId'],[]);assignment['unique' if len(cpus)==1 else ('none' if not cpus else 'ambiguous')]+=1
  key=(r['globalPid'],r['modelId'],r['streamId'],r['taskId'],r['name'],r['inputShapes'],r['outputShapes'])
  if key not in templates:
   templates[key]={'native_pid':r['globalPid'],'native_model_id':r['modelId'],'stream_id':r['streamId'],'task_id':r['taskId'],'native_name':s.get(r['name']),'input_shapes':s.get(r['inputShapes']),'output_shapes':s.get(r['outputShapes']),'instances':0,'CPU_creator_candidates':[]}
  t=templates[key];t['instances']+=1
  for p in cpus:
   if any(a['CPU_ROW']==p['CPU_ROW'] for a in t['CPU_creator_candidates']):continue
   t['CPU_creator_candidates'].append({'CPU_ROW':p['CPU_ROW'],'name':s.get(p['name']),'callchain':sorted(chain.get(p['callchainId'],[]),key=lambda a:a['depth']),'official_native_connection':r['connectionId'],'official_connection_set_id':p['connectionId']})
 name=db.parent.parent.name;out=OUT/(name+'-creation-connections.jsonl')
 with out.open('w') as f:
  for value in templates.values():f.write(json.dumps(value,separators=(',',':'))+'\n')
 summary.update(captured_instances=captured_instances,captured_instance_CPU_assignment=dict(assignment),captured_native_templates=len(templates),source_bindings_path=str(out));c.close();return summary
# Explicit two roots avoids matching derived or an unrelated old profile.
rows=[]
for phase in ['startup','runtime']:
 for p in sorted((ROOT/phase).glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler.db')):rows.append(one(p))
result={'status':'VENDOR_EXPORT_AND_CREATION_SOURCE_CENSUS_RETURNED','generated_unix':time.time(),'profiles':rows,'raw_inputs_unchanged':not changed,'scope':'Official connection-set source census in this exact cohort. Runtime graph-token/native-template and operand/reader identities require their own checks. No nearest-time or native-name source guess.','historical_native_or_phase_promotions':0,'overall_complete':False}
(OUT/'PROFILE-EXPORT-REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'databases':len(rows),'raw_inputs_unchanged':not changed,'startup_captured_templates':sum(r.get('captured_native_templates',0) for r in rows if r['phase']=='startup')}))
