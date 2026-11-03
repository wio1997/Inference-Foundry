#!/usr/bin/env python3
"""Inspect already captured EP3 host API trace; no NPU operations."""
import sqlite3,json,pathlib
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
TID=9139690412736
samples={'short_norm':(1793241328962919370,1793241328964225290,'aclnnRmsNorm'), 'short_rope':(1793241328964231550,1793241328965916870,'aclnnInplacePartialRotaryMul'), 'long_norm':(1793241329712288300,1793241329712680350,'aclnnRmsNorm'), 'long_rope':(1793241329712685940,1793241329713116470,'aclnnInplacePartialRotaryMul')}
def union(intervals):
 if not intervals:return 0
 x=sorted(intervals); a,b=x[0];total=0
 for l,r in x[1:]:
  if l<=b:b=max(b,r)
  else:total+=b-a;a,b=l,r
 return total+b-a
out={'scope':'Four selected API intervals on one captured EP3 host thread; nested records do not classify uncovered time.', 'samples':{}}
for key,(start,end,name) in samples.items():
 outer=list(c.execute('select a.startNs,a.endNs,s.value name from CANN_API a join STRING_IDS s on s.id=a.name where a.globalTid=? and a.startNs=? and a.endNs=? and s.value=?',(TID,start,end,name)));assert len(outer)==1
 child=[dict(x) for x in c.execute('select a.startNs,a.endNs,s.value name,a.type from CANN_API a join STRING_IDS s on s.id=a.name where a.globalTid=? and a.startNs>=? and a.endNs<=? and not (a.startNs=? and a.endNs=? and s.value=?) order by a.startNs,a.endNs desc',(TID,start,end,start,end,name))]
 ancestors=[dict(x) for x in c.execute('select a.startNs,a.endNs,s.value name,a.type from PYTORCH_API a join STRING_IDS s on s.id=a.name where a.globalTid=? and CAST(a.startNs AS INTEGER)<=? and CAST(a.endNs AS INTEGER)>=? order by CAST(a.startNs AS INTEGER) desc limit 15',(TID,start,end))]
 in_pt=[dict(x) for x in c.execute('select a.startNs,a.endNs,s.value name,a.type from PYTORCH_API a join STRING_IDS s on s.id=a.name where a.globalTid=? and CAST(a.startNs AS INTEGER)>=? and CAST(a.endNs AS INTEGER)<=? order by CAST(a.startNs AS INTEGER) limit 100',(TID,start,end))]
 covered=union([(x['startNs'],x['endNs']) for x in child]);dur=end-start
 out['samples'][key]={'api':name,'start_ns':start,'end_ns':end,'duration_ns':dur,'api_end_minus_launch_end_ns':end-next(x['endNs'] for x in child if x['name']=='launch'),'same_thread_child_cann_union_ns':covered,'remainder_outside_recorded_child_cann_intervals_ns':dur-covered,'child_cann':child,'pytorch_enclosing':ancestors,'pytorch_inside':in_pt}
trace_root=p.parent.parent
info=json.loads((trace_root/'profiler_info_3.json').read_text())
host=json.loads(next(trace_root.glob('PROF_*/host/sample.json')).read_text())
out['profile_configuration']={'activities':info['config']['common_config']['activities'],'experimental_host_sys':info['config']['experimental_config']['_host_sys'],'cann_version':info['cann_version'],'torch_npu_version':info['torch_npu_version'],'host_options':{k:host.get(k) for k in ('host_cpu_profiling','cpu_profiling','host_osrt_profiling','hostAllPidCpuProfiling','host_sys','hostProfiling')}}
out['tables']=[x[0] for x in c.execute("select name from sqlite_master where type='table' order by name")]
out['meta']=[dict(x) for x in c.execute('select * from META_DATA')]
print(json.dumps(out,indent=2))
