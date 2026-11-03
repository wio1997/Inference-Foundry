#!/usr/bin/env python3
"""Read frozen EP3 DB/trace and current stage scatter metadata; no device runtime calls."""
import hashlib, json, pathlib, sqlite3
import ijson
p = pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT')
k = pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/scatter_nd_update_sk')
c = sqlite3.connect((p/'ascend_pytorch_profiler_3.db').as_uri()+'?mode=ro', uri=True)
c.row_factory = sqlite3.Row
schemas = {t: [r[1] for r in c.execute('pragma table_info('+t+')')] for t in ('TASK','COMPUTE_TASK_INFO')}
cases = []
for gid, tid in ((2078,13226),(2942,13468)):
 task = [dict(x) for x in c.execute('select * from TASK where globalTaskId=?',(gid,))]
 compute = [dict(x) for x in c.execute('select * from COMPUTE_TASK_INFO where globalTaskId=?',(gid,))]
 assert len(task)==len(compute)==1 and task[0]['streamId']==35 and task[0]['taskId']==tid
 vals = {}
 for field in ('name','opType','inputFormats','inputDataTypes','inputShapes','outputFormats','outputDataTypes','outputShapes','attrInfo','opState','hf32Eligible','gridDim','blockDim'):
  row = c.execute('select value from STRING_IDS where id=?',(compute[0][field],)).fetchone()
  vals[field] = row[0] if row else None
 cases.append({'globalTaskId':gid, 'taskId':tid, 'task':task[0], 'compute':compute[0], 'resolved':vals})
for case in cases: case['trace_events'] = []
with (p/'trace_view.json').open('rb') as fp:
 for e in ijson.items(fp,'item'):
  if not isinstance(e,dict) or e.get('ph')!='X' or str(e.get('pid'))!='2179523' or str(e.get('tid'))!='35': continue
  for case in cases:
   if (e.get('args') or {}).get('Task Id') == case['taskId']:
    case['trace_events'].append({'name':e.get('name'),'args':e.get('args'),'ts':str(e.get('ts')),'dur':str(e.get('dur'))})
for case in cases: assert len(case['trace_events']) == 1
candidates=[]; json_files=sorted(k.glob('*.json'))
for f in json_files:
 d=json.loads(f.read_text()); s=d['supportInfo']
 ins=[i.get('dtype') for i in s.get('inputs',[])]; outs=[i.get('dtype') for i in s.get('outputs',[])]
 if ins==['bfloat16','int32','bfloat16'] and outs==['bfloat16']:
  binary=k/(d['binFileName']+d['binFileSuffix']); actual=hashlib.sha256(binary.read_bytes()).hexdigest()
  candidates.append({'descriptor':f.name,'descriptor_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'binary_sha256':actual,'declared_sha256':d.get('sha256'),'hash_matches':actual==d.get('sha256'),'inputs':ins,'outputs':outs,'kernel_list':[x['kernelName'] for x in d['kernelList']],'attrs':s.get('attrs'),'opMode':s.get('opMode'),'simplifiedKey':s.get('simplifiedKey')})
print(json.dumps({'scope':'frozen EP3 export + stage current kernel metadata; no runtime binding', 'schemas':schemas,'cases':cases,'installed_descriptor_count':len(json_files),'dtype_candidates':candidates,'limits':['dtype filter does not prove historical binary identity, runtime candidate selection, tiling key or tiling data','logical trace name is not executable kernel name','empty attrInfo does not prove an empty runtime strides value','raw device slices are not decoded here']},ensure_ascii=False,indent=2))
