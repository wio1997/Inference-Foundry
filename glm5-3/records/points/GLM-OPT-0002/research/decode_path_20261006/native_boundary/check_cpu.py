import ctypes,json,os,statistics,sys,time
from pathlib import Path
from perf_boundary import Observer,function_offsets
root=Path(__file__).resolve().parent;path=str(root/'fixture.so');lib=ctypes.CDLL(path)
names=['fixture_outer','fixture_predicate','fixture_workspace'];offsets,sha=function_offsets(path,names)
probes={name+'_'+side:dict(path=path,offset=offsets[name]['offset'],return_=side=='exit') for name in names for side in ['enter','exit']}
for x in probes.values():x['return']=x.pop('return_')
out=(ctypes.c_uint64*4)();lib.fixture_outer.argtypes=[ctypes.POINTER(ctypes.c_uint64)];lib.fixture_outer.restype=ctypes.c_uint64
obs=Observer(probes,[os.getpid()],path,'glm53_cpu_boundary');ref=[]
try:
 obs.enable()
 for i in range(20):lib.fixture_outer(out);ref.append(list(out));obs.drain()
 obs.disable();events=sorted(obs.events,key=lambda x:x.get('wall_ns',0));assert not any('lost' in x or 'other_record_type' in x for x in events)
 samples={name:[] for name in names};active={}
 for x in events:
  name,side=x['event'].rsplit('_',1)
  if side=='enter':assert name not in active;active[name]=x
  else:
   a=active.pop(name);assert x['task_cpu_ns']>=a['task_cpu_ns']
   samples[name].append(dict(wall_ns=x['wall_ns']-a['wall_ns'],cpu_ns=x['task_cpu_ns']-a['task_cpu_ns']))
 assert not active and all(len(x)==20 for x in samples.values())
 errors=[dict(wall_delta_ns=x['wall_ns']-(r[3]-r[0]),cpu_delta_ns=x['cpu_ns']-(r[2]-r[1])) for x,r in zip(samples['fixture_outer'],ref)]
 assert all(abs(x['wall_delta_ns'])<300000 and abs(x['cpu_delta_ns'])<300000 for x in errors),errors
 assert statistics.median(x['wall_ns']-x['cpu_ns'] for x in samples['fixture_predicate'])>1800000
 assert statistics.median(x['cpu_ns'] for x in samples['fixture_workspace'])>950000
 result=dict(passed=True,model_requests=0,NPU_import=False,probe_source_sha256=sha,events=len(events),lost=0,samples=samples,reference=ref,errors=errors,group_counter='PERF_COUNT_SW_TASK_CLOCK + PERF_SAMPLE_READ/PERF_FORMAT_GROUP',limits=['Probe CPU includes instrumentation/kernel execution; no performance claim.','Fixture demonstrates sleeping wall vs scheduled task CPU at matching entry/return events.'])
 (root/'CPU_result_A.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['samples','reference','errors']}))
finally:
 obs.close()
 assert obs.group+'/' not in Path('/sys/kernel/tracing/uprobe_events').read_text()
