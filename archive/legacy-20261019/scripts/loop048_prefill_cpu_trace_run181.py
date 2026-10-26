#!/usr/bin/env python3
"""Offline direct-child CPU scope census for Run165 single profiled prefill."""
import collections,glob,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=Path(glob.glob(str(root/'evidence/20260925_loop045_boundary/run165/profiler/rank0/*/ASCEND_PROFILER_OUTPUT/trace_view.json'))[0])
a=json.loads(p.read_text())
ops=[{'name':x['name'],'start':float(x['ts']),'end':float(x['ts'])+float(x['dur']),'dur':float(x['dur']),'tid':x['tid']} for x in a if x.get('cat')=='cpu_op' and x.get('pid')==752 and x.get('tid')==752 and x.get('ph')=='X']
ops.sort(key=lambda x:(x['start'],-x['end']))
for x in ops:x['children']=[]
stack=[]
for x in ops:
 while stack and stack[-1]['end']<=x['start']+0.00001:stack.pop()
 while stack and x['end']>stack[-1]['end']+0.00001:stack.pop()
 if stack:stack[-1]['children'].append(x)
 stack.append(x)
def union_span(rows):
 intervals=sorted((max(x['start'],parent['start']),min(x['end'],parent['end'])) for x in rows)
 merged=[]
 for lo,hi in intervals:
  if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
  else:merged.append((lo,hi))
 return sum(hi-lo for lo,hi in merged)
summary={}
for name in ['vllm::dsa_forward','vllm::moe_forward_shared','vllm::matmul_and_reduce','vllm::maybe_all_gather_and_maybe_unpad']:
 scopes=[x for x in ops if x['name']==name]
 by_child=collections.defaultdict(list);self_us=[]
 for parent in scopes:
  children=parent['children']
  for child in children:by_child[child['name']].append(child['dur'])
  self_us.append(parent['dur']-union_span(children))
 summary[name]={'scope_count':len(scopes),'inclusive_total_ms':sum(x['dur'] for x in scopes)/1000,'inclusive_median_ms':statistics.median(x['dur'] for x in scopes)/1000,'direct_child_uncovered_total_ms':sum(self_us)/1000,'direct_child_uncovered_median_ms':statistics.median(self_us)/1000,'top_direct_children':[{'name':k,'count':len(v),'total_ms':sum(v)/1000,'median_us':statistics.median(v)} for k,v in sorted(by_child.items(),key=lambda kv:-sum(kv[1]))[:18]]}
# The high-level DSA/MoE scopes do not overlap on the traced Python thread.
major=[x for x in ops if x['name'] in ['vllm::dsa_forward','vllm::moe_forward_shared']]
major.sort(key=lambda x:x['start'])
overlap=sum(max(0,major[i]['end']-major[i+1]['start']) for i in range(len(major)-1))
assert overlap<0.01,overlap
out={'run':'run181','source_trace':str(p.relative_to(root)),'scope':'Run165 one rank0, one 83-token warmed prefill forward; torch_npu profiler and explicit synchronization perturb execution','cpu_thread':752,'major_scope_summary':summary,'dsa_moe_nonoverlap_ms':overlap/1000,'dsa_moe_inclusive_union_ms':sum(x['dur'] for x in major)/1000,'profiled_forward_stage_ms':444.17975,'device_free_ms':401.721173,'host_flow_before_next_submit_ms':357.603036,'limits':['These are CPU profiler scopes; direct-child-uncovered time includes uninstrumented Python/C++ and profiling overhead, not automatically removable.','DSA and MoE scope durations are inclusive; nested child sums and device timing are not additive.','Only one rank0 forward was profiled; unprofiled matched Run161 83-token _model_forward was about 0.320s.','Cannot infer latest-rank warmed cohort or formal E2E gain.']}
q=root/'evidence/20260925_loop048_prefill/run181/analysis.json';q.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'dsa_moe_inclusive_union_ms':out['dsa_moe_inclusive_union_ms'],'scope_summary':{k:{x:v[x] for x in ['scope_count','inclusive_total_ms','direct_child_uncovered_total_ms']} for k,v in summary.items()}},indent=2))
