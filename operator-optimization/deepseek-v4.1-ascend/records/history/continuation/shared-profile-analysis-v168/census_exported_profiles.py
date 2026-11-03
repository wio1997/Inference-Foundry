import collections,hashlib,json,sqlite3,time
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168')
OUT=ROOT/'derived'
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
 name=db.parent.parent.name;creator_path=OUT/(name+'-CPU-launch-creators.jsonl')
 with creator_path.open('w') as f:
  for connection,ps in cpu_by_native.items():
   f.write(json.dumps({'native_connection':connection,'CPU_candidates':[{'CPU_ROW':p['CPU_ROW'],'name':s.get(p['name']),'set_id':p['connectionId'],'callchain':sorted(chain.get(p['callchainId'],[]),key=lambda a:a['depth'])} for p in ps]},separators=(',',':'))+'\n')
 summary['CPU_launch_creator_map']=str(creator_path)
 summary['CPU_native_launch_connections']=len(cpu_by_native)
 name=db.parent.parent.name;out=OUT/(name+'-creation-connections.jsonl')
 with out.open('w') as f:
  for value in templates.values():f.write(json.dumps(value,separators=(',',':'))+'\n')
 summary.update(captured_instances=captured_instances,captured_instance_CPU_assignment=dict(assignment),captured_native_templates=len(templates),source_bindings_path=str(out));c.close();return summary

rows=[]
for phase in ['startup','runtime']:
 for p in sorted((ROOT/phase).glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db')):rows.append(one(p))
assert len(rows)==32,len(rows)
result={'status':'CORRECTED_32_DB_SOURCE_CENSUS','profiles':rows,'raw_postcheck':json.loads((OUT/'RAW-POSTCHECK.json').read_text()),'discovery_correction':'Distributedprofiler usesascend_pytorch_profiler_<rank>.db;firstexact-filenamecensusfound0. Officialexportwasvalid;not rerun.','historical_native_or_phase_promotions':0,'overall_complete':False}
(OUT/'CORRECTED-PROFILE-CENSUS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'databases':len(rows),'startup_CPU_launch_connections':sum(r.get('CPU_native_launch_connections',0) for r in rows),'compute_metadata_rows':sum(r['rows']['COMPUTE_TASK_INFO'] for r in rows)}))
