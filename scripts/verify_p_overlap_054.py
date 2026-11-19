#!/usr/bin/env python3
"""Validate selected P trace-coordinate unions and pair overlaps, not busy time."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={'evidence/p_overlap_054/interval.json':'b4b8fffa30f5036b84338db7749723a003ee709b0cd27ff76d61d02e7598fcae','evidence/p_overlap_054/pair.json':'39325b32c836d8803d7902b11d839dbdd6000d3cc301173ed3acfd6d77aac23c','evidence/p_heavy_052/step_detail.json':'363153a544c94cee7bf5d7ba4495ef796bacf4244d64e29edbc94cee9c6ef6cd'}


def need(v,m):
 if not v:raise ValueError(m)


def load(path):
 b=(ROOT/path).read_bytes();need(hashlib.sha256(b).hexdigest()==FILES[path],'SHA mismatch');return json.loads(b)


def verify():
 a=load('evidence/p_overlap_054/interval.json');b=load('evidence/p_overlap_054/pair.json');c=load('evidence/p_heavy_052/step_detail.json');prior={(x['framework'],x['rank']):x for x in a['ranks']};heavy={(x['framework'],x['rank']):x for x in c['ranks']};now={(x['framework'],x['rank']):x for x in b['ranks']};need(set(prior)==set(heavy)==set(now)=={('sglang',0),('vllm',0)},'rank coverage');out={}
 for key,r in now.items():
  p=prior[key];h=heavy[key];need(all(r[f]==p[f]==h[f] for f in ('source_csv_sha256','trace_sha256','selected_kernel_count','step_markers_total')),'source binding');need(r['cpu_range_presence']==p['cpu_range_presence'],'range binding')
  oldkern={x['kernel_name']:x for x in h['kernels']};pairs={x['kernel_name']:x for x in r['kernels']};need(set(oldkern)==set(pairs),'selected kernel coverage')
  for n,x in pairs.items():
   need(x['counts']==oldkern[n]['counts'] and x['cpu_step_device_event_stats']==oldkern[n]['cpu_step_device_event_stats'],'candidate consistency')
   q=x['cpu_range_candidate_counts'];total=x['counts'].get('typed_flow_unique',0)
   for f in ('profiler_step','execute_context','model_runner_exec','npu_model_forward','npu_execute_model','vllm_execute_model'):need(sum(q.get(f+'_'+k,0) for k in ('zero','one','multiple'))==total,'range denominator')
  need(len(r['step_details'])==len(p['step_details'])==len(h['step_details'])==r['step_markers_total'],'marker coverage')
  for x,y,z in zip(r['step_details'],p['step_details'],h['step_details']):
   need(all(x[f]==y[f]==z[f] for f in ('ordinal','stage','cpu_marker_duration_ms','context_kv_tokens','step_tokens','batch_size','forward_id_recorded')),'marker metadata')
   u=x['selected_device_trace_intervals'];v=y['selected_device_trace_intervals'];need(all(u[f]==v[f] for f in ('event_count','event_sum_us','span_us','selected_union_us','different_names_overlap_us','single_name_us')),'interval reread')
   need(u['event_count']==sum(k['count'] for k in x['kernels']) and 0<=u['selected_union_us']<=min(u['event_sum_us'],u['span_us'])+0.001,'union bound')
   need(abs(u['selected_union_us']-u['different_names_overlap_us']-sum(u['single_name_us'].values()))<0.01,'union partition')
   q={x['kernel_name']:x for x in x['kernels']};seen=set()
   for pair in u['pair_overlap_us']:
    l=pair['left'];rr=pair['right'];need(l in q and rr in q and l<rr and (l,rr) not in seen,'pair identity');seen.add((l,rr));need(0<=pair['duration_us']<=min(q[l]['device_trace_union_us'],q[rr]['device_trace_union_us'],u['different_names_overlap_us'])+0.001,'pair bound')
   if u['peak_distinct_names']<=2:need(abs(sum(t['duration_us'] for t in u['pair_overlap_us'])-u['different_names_overlap_us'])<0.01,'pair partition when peak <=2')
  need(all(v==0 for v in r['cpu_range_presence'].values()),'named range inventory changed')
  out[key[0]]={'kernels':len(r['kernels']),'step_markers':r['step_markers_total'],'named_cpu_range_observations':0}
 sg=now[('sglang',0)];ext=[x for x in sg['step_details'] if x['stage']=='EXTEND'];need(len(ext)==8 and all(x['selected_device_trace_intervals']['event_count']==596 for x in ext),'SG extend coverage');
 def pairval(s,a,b):
  return next(x['duration_us'] for x in s['selected_device_trace_intervals']['pair_overlap_us'] if {x['left'],x['right']}=={a,b})
 a='NotifyDispatchMc2AicpuKernel';b='SparseFlashAttention';v=[pairval(s,a,b)/1000 for s in ext];need(min(v)>194 and max(v)<205,'notified attention interval range')
 out['sglang']['notify_sparse_pair_trace_coordinate_ms']=[round(min(v),3),round(max(v),3)]
 return {'rank0':out,'scope':'selected event trace-coordinate overlap only; not total busy, causal overlap, calibrated CPU/device clock, request critical path or optimization benefit'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,StopIteration) as e:raise SystemExit('FAILED: '+str(e))
