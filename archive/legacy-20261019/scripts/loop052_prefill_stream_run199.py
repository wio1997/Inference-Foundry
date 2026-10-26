#!/usr/bin/env python3
"""Read-only Run165 one-forward device stream activity/overlap screen."""
import glob,json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=glob.glob(str(root/'evidence/20260925_loop045_boundary/run165/profiler/rank0/*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
assert len(p)==1
e=json.loads(Path(p[0]).read_text())
sc=[x for x in e if x.get('cat')=='cpu_op' and x.get('name') in ('vllm::dsa_forward','vllm::moe_forward_shared')]
assert len(sc)==86
a=min(float(x['ts']) for x in sc);b=max(float(x['ts'])+float(x['dur']) for x in sc)
ks=[x for x in e if a<=float(x.get('ts',-1))<b and x.get('args',{}).get('Task Type','').startswith('KERNEL_')]
by_tid=collections.defaultdict(list)
for x in ks:by_tid[str(x['tid'])].append((float(x['ts']),float(x['ts'])+float(x['dur']),x['name']))
def covered(intervals):
 seq=sorted((s,t) for s,t,*_ in intervals);z=[]
 for s,t in seq:
  if z and s<=z[-1][1]:z[-1]=(z[-1][0],max(t,z[-1][1]))
  else:z.append((s,t))
 return z
def overlap(a,b):
 i=j=0;total=0
 while i<len(a) and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return total
main=covered(by_tid['47']);shared=covered(by_tid['36']);comm=covered(by_tid['38'])
result={'run':'run199','source':'Run165 one rank0 profiler/synchronized 83-token warmed prefill forward','dsa_moe_cpu_scope_envelope_ms':(b-a)/1000,'kernel_counts_by_stream':{k:len(v) for k,v in by_tid.items()},'kernel_sum_ms_by_stream':{k:sum(t-s for s,t,_ in v)/1000 for k,v in by_tid.items()},'kernel_union_ms_by_stream':{k:sum(t-s for s,t in covered(v))/1000 for k,v in by_tid.items()},'stream36_names':dict(collections.Counter(n for _,_,n in by_tid['36'])),'stream38_names_top':collections.Counter(n for _,_,n in by_tid['38']).most_common(5),'stream36_overlap_main47_ms':overlap(shared,main)/1000,'stream36_overlap_comm38_ms':overlap(shared,comm)/1000,'interpretation':'Stream36 carries four shared-expert kernels per43 MoE layers (dynamic quant, gate quant matmul, dequant SwiGLU quant, down quant matmul). Its ~2.1ms recorded device activity is small relative to the profiled ~437ms Host submission envelope; this does not measure cost of switching to same stream or the real unprofiled Host event overhead. Stream38 AivKernel is separate and not assumed removable.','limits':['Single profiled/synchronized rank0 forward, not unprofiled 8-rank product.','Stream IDs inferred from name pattern; no explicit source-to-stream marker in trace.','Duration overlap is actual trace activity, not a throughput saving prediction.']}
out=root/'evidence/20260925_loop052_prefill_submission/run199/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('stream36_names','stream38_names_top','interpretation','limits')},indent=2))
