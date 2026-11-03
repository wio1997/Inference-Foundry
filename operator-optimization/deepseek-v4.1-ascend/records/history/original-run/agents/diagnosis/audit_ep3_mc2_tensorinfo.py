#!/usr/bin/env python3
"""Aggregate saved EP3 CANN tensor-info metadata; never emit tensor contents or IDs."""
import base64,hashlib,json,pathlib,shlex,subprocess
root=pathlib.Path(__file__).resolve().parents[2]
remote=r'''import collections,hashlib,json,pathlib,struct,sys
r=pathlib.Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis');sys.path.insert(0,str(r))
from msparser.add_info.tensor_add_info_bean import TensorAddInfoBean
from msparser.data_struct_size_constant import StructFmt
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_constant.level_type_constant import LevelDataType
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt')
files=sorted(p for p in base.rglob('*.additional.tensor_info.slice_*') if not p.name.endswith('.done'))
if len(files)!=6:raise RuntimeError('Unexpected number of files')
size=StructFmt.TENSOR_ADD_INFO_SIZE
if size!=256:raise RuntimeError('Unexpected record size')
expected_in='1023,5120;24,5120,4608;24,2304,5120;1023,6;110592;122880;1023,6;1023'
expected_out='1023,5120;24'
groups={};stats=[]
for p in files:
 before=p.stat();b=p.read_bytes();after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(b)%size:raise RuntimeError('Unstable or misaligned file')
 kind=p.name.split('.')[0];magic_bad=level_bad=count_bad=decode_bad=0
 for i in range(0,len(b),size):
  chunk=b[i:i+size];magic,level=struct.unpack('=HH',chunk[:4])
  if magic!=NumberConstant.MAGIC_NUM:magic_bad+=1;continue
  try:LevelDataType(level)
  except ValueError:level_bad+=1;continue
  try:
   bean=TensorAddInfoBean.decode(chunk)
   if not 0<=bean.tensor_num<=5:count_bad+=1;continue
   # parser's aggregation key is timestamp, thread, node; do not output it
   key=(kind,bean.timestamp,bean.thread_id,bean.node_id)
   g=groups.setdefault(key,{'in':[],'out':[],'dtype_out':[],'levels':set()})
   if bean.input_shape:g['in'].append(bean.input_shape)
   if bean.output_shape:g['out'].append(bean.output_shape)
   if bean.output_data_type:g['dtype_out'].append(bean.output_data_type)
   g['levels'].add(bean.level)
  except Exception:decode_bad+=1
 stats.append({'name':p.name,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'records':len(b)//size,'bad_magic':magic_bad,'bad_level':level_bad,'bad_count':count_bad,'decode_fail':decode_bad})
exact=[g for g in groups.values() if ';'.join(g['in'])==expected_in and ';'.join(g['out'])==expected_out]
out_only=[g for g in groups.values() if ';'.join(g['out'])==expected_out]
print(json.dumps({'scope':'Saved EP3 profiling slices; metadata aggregate only; no IDs, timestamps, shapes, tensor payloads, or requests emitted','files':stats,'group_count':len(groups),'matching_exact_shape_groups':len(exact),'matching_output_shape_groups':len(out_only),'matching_exact_output_dtype_histogram':dict(collections.Counter(';'.join(g['dtype_out']) for g in exact)),'matching_exact_level_histogram':dict(collections.Counter(','.join(sorted(g['levels'])) for g in exact)),'limitations':['Grouping follows parser timestamp/thread/node key separately by aging type but does not prove producer version, completeness, task identity or timestamp clock mapping.','Bean exposes descriptor data only, no values; shape signature is not unique task identification.']},sort_keys=True))
'''
encoded=base64.b64encode(remote.encode()).decode()
inner='import base64;exec(compile(base64.b64decode('+repr(encoded)+'),"<safe-tensor-info>","exec"))'
outer=shlex.join(['docker','exec','-e','PYTHONDONTWRITEBYTECODE=1','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',outer],capture_output=True,text=True,timeout=120)
if r.returncode: raise RuntimeError('Container metadata audit failed; details intentionally suppressed')
v=json.loads(r.stdout); dest=root/'agents/diagnosis/ep3-mc2-tensorinfo-aggregate.json';dest.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
print('Records:',sum(f['records'] for f in v['files']),'failed:',sum(f['bad_magic']+f['bad_level']+f['bad_count']+f['decode_fail'] for f in v['files']),'groups:',v['group_count'],'exact shape groups:',v['matching_exact_shape_groups'],'output shape groups:',v['matching_output_shape_groups'])
