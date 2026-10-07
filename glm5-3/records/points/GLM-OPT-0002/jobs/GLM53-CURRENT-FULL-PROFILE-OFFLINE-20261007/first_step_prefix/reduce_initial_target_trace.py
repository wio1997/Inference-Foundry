from pathlib import Path
from decimal import Decimal as D
import json,gzip,collections,hashlib
j=Path('/Users/wio/work/Inference-Foundry-glm5-3/glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007/first_step_prefix')
def bounds(x):
 e=x['event'];s=D(str(e['ts']));return s,s+D(str(e.get('dur',0)))
def merge(iv):
 out=[]
 for s,e in sorted(iv):
  if e<=s:continue
  if out and s<=out[-1][1]:out[-1][1]=max(out[-1][1],e)
  else:out.append([s,e])
 return out
def union(a,lo,hi):return merge([(max(lo,bounds(x)[0]),min(hi,bounds(x)[1])) for x in a])
def length(iv):return sum((b-a for a,b in iv),D(0))
reports=[]
for rank in (0,13,15):
 f=j/('rank%d_trace_events.json.gz'%rank);z=json.loads(gzip.decompress(f.read_bytes()));meta=z['process_names'];xs=[x for x in z['events'] if x['event']['ph']=='X'];hw=sorted([x for x in xs if meta[str(x['event']['pid'])]=='Ascend Hardware'],key=lambda x:bounds(x)[0]);cann=[x for x in xs if meta[str(x['event']['pid'])]=='CANN'];cpu=[x for x in xs if meta[str(x['event']['pid'])]=='Python' and x['event'].get('cat')=='cpu_op'];enqs=[x for x in xs if x['event'].get('cat')=='enqueue'];deqs=[x for x in xs if x['event'].get('cat')=='dequeue']
 emb=[x for x in hw if x['event']['name']=='aclnnEmbedding_GatherV2AiCore_GatherV2'];arg=[x for x in hw if x['event']['name']=='aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2'];assert len(emb)==len(arg)==2
 lo=bounds(emb[0])[0];hi=bounds(arg[0])[1];h=[x for x in hw if bounds(x)[0]<hi and bounds(x)[1]>lo];nonwait_types=('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','COMMUNICATION','SDMA_SQE','PCIE_DMA_SQE','AI_CPU');nonwait=[x for x in h if x['event'].get('args',{}).get('Task Type') in nonwait_types];iv=union(nonwait,lo,hi);gaps=[];last=lo
 for a,b in iv:
  if a>last:gaps.append((last,a))
  last=b
 if last<hi:gaps.append((last,hi))
 contexts=[]
 for a,b in sorted(gaps,key=lambda v:v[1]-v[0],reverse=True)[:12]:
  following=sorted([x for x in nonwait if bounds(x)[0]>=b],key=lambda x:bounds(x)[0])[:3]
  core=next((x for x in hw if x['event'].get('args',{}).get('Physic Stream Id')==47 and x['event'].get('args',{}).get('Task Type') in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV') and bounds(x)[0]>=b),None)
  context=dict(lo_us=str(a),hi_us=str(b),duration_us=str(b-a),following_nonwait=following,wait_events=[x for x in h if x['event'].get('args',{}).get('Task Type') in ('EVENT_WAIT','NOTIFY_WAIT') and bounds(x)[0]<b and bounds(x)[1]>a],CPU_vllm_scopes=[x for x in cpu if x['event']['name'].startswith('vllm::') and bounds(x)[0]<b and bounds(x)[1]>a])
  if core:
   con=core['event']['args']['connection_id'];nodes=[x for x in cann if x['event']['name']=='Node@launch' and x['event'].get('args',{}).get('connection_id')==con];context['next_core']=core;context['nodes']=nodes
   if len(nodes)==1:
    node=nodes[0];s=bounds(node)[0];deq=[x for x in deqs if x['event']['tid']==node['event']['tid'] and bounds(x)[0]<=s<bounds(x)[1]];context['dequeues']=deq
    if len(deq)==1:
     corr=deq[0]['event']['args']['correlation_id'];enq=[x for x in enqs if x['event']['args']['correlation_id']==corr];context['enqueues']=enq
     if len(enq)==1:
      et=bounds(enq[0])[0];scopes=[x for x in cpu if x['event']['tid']==enq[0]['event']['tid'] and bounds(x)[0]<=et<bounds(x)[1]];context['CPU_enclosing_enqueue']=sorted(scopes,key=lambda x:bounds(x)[1]-bounds(x)[0]);context['enqueue_start_relative_gap_end_us']=str(et-b);context['core_start_minus_node_end_us']=str(bounds(core)[0]-bounds(node)[1])
  contexts.append(context)
 types=collections.Counter(x['event'].get('args',{}).get('Task Type','?') for x in h)
 type_unions={k:str(length(union([x for x in h if x['event'].get('args',{}).get('Task Type')==k],lo,hi))) for k in types}
 report=dict(rank=rank,trace_sha256=z['trace_sha256'],event_export_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),target_embedding=emb[0],target_argmax=arg[0],draft_embedding=emb[1],draft_argmax=arg[1],target_extent_us=str(hi-lo),target_nonwait_union_us=str(length(iv)),target_nonwait_complement_us=str(hi-lo-length(iv)),hardware_types=dict(types),type_union_us=type_unions,largest_nonwait_complements=contexts,NPU_requests=0,new_profile=0,parser_rerun=0,limits='Task union includes communication peer waiting; wait/complement/cpu scope are not proven removable. Initial target/source grouping supported but not API publication timestamp or savings. Indexed joins preserve raw events; one rank can delay peers, need cross-rank joins before causal claim.')
 reports.append(report)
p=j/'initial_target_causal_reduced.json';assert not p.exists();p.write_text(json.dumps(reports,indent=2)+'\n');(j/'reduce_initial_target_trace.py').write_text(Path(__file__).read_text())
for r in reports:
 print({k:r[k] for k in ['rank','target_extent_us','target_nonwait_union_us','target_nonwait_complement_us','hardware_types']})
 for c in r['largest_nonwait_complements'][:4]:
  print({k:c.get(k) for k in ['duration_us','enqueue_start_relative_gap_end_us','core_start_minus_node_end_us']},[(x['event']['name'],x['event']['dur']) for x in c.get('CPU_enclosing_enqueue',[])[:3]], [(x['event']['name'],x['event']['dur']) for x in c['wait_events'][:3]])
