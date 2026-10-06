"""Run249 only: disjoint post-combine stages from actual Ascend 0.27.1 path.

No model requests/import Torch. No exclusive CPU runtime claim from op scopes.
"""
import bisect,collections,gzip,hashlib,json
from pathlib import Path
from reduce_gap_owners import segments
from reduce_prepare_criticality import duration,intersection


def reduce(data,queue):
 cpu=sorted((int(x[0]),int(x[1]),x[2]) for x in data['cpu']);starts=[x[0] for x in cpu]
 moes=[x for x in cpu if x[2]=='vllm::moe_forward_shared'];assert len(moes)==380
 phases=collections.defaultdict(list);events=[];copies=[];examples=[]
 for i,(a,b,_) in enumerate(moes):
  xs=[x for x in cpu[bisect.bisect_left(starts,a):bisect.bisect_right(starts,b)] if x[1]<=b]
  combine=[x for x in xs if x[2]=='npu::npu_moe_distribute_combine_v2'];assert len(combine)==1
  begin=combine[0][1];xs=[x for x in xs if x[0]>=begin]
  def one(name):
   out=[x for x in xs if x[2]==name];assert len(out)==1,(i,name,len(out));return out[0]
  gather=one('c10d::allgather_');quant=one('npu::npu_dynamic_quant');activation=one('_C_ascend::npu_dequant_swiglu_quant')
  mm=[x for x in xs if x[2]=='npu::npu_quant_matmul'];wait=[x for x in xs if x[2]=='Event::wait'];assert len(mm)==2 and len(wait)==3
  first_wait=wait[0][0]
  bounds=[(begin,gather[0],'finalize_allocation_and_split_before_gather'),(gather[0],gather[1],'finalize_list_gather_and_materialization'),(gather[1],first_wait,'finalize_tail_and_shared_entry'),(first_wait,quant[0],'shared_first_same_stream_event_wait'),(quant[0],mm[1][1],'shared_dynamic_quant_gate_activation_down'),(mm[1][1],b,'shared_tail_and_event_cleanup')]
  assert all(x[1]==y[0] for x,y in zip(bounds,bounds[1:])) and all(x[1]>=x[0] for x in bounds)
  assert quant[1]<=mm[0][0]<mm[0][1]<=wait[1][0]<wait[1][1]<=activation[0]<activation[1]<=wait[2][0]<wait[2][1]<=mm[1][0]
  for s,e,n in bounds:phases[n].append((s,e))
  events.extend((s,e) for s,e,n in xs if n=='Event::wait')
  current_copies=[(s,e) for s,e,n in xs if n=='aten::copy_' and gather[0]<=s and e<=gather[1]];assert len(current_copies)==16;copies+=current_copies
  if len(examples)<2:examples.append(dict(MoE_ordinal=i,post_combine_ns=[begin,b],stages=bounds,gather=gather,shared_quant=quant,shared_matmuls=mm,shared_activation=activation,shared_event_waits=wait))
 _,_,gaps=segments(data,queue)
 summary={n:dict(count=len(xs),wall_ms=duration(xs),local_pre_enqueue_exposure_ms=intersection(xs,gaps)) for n,xs in phases.items()}
 total=duration([x for xs in phases.values() for x in xs]);exposure=intersection([x for xs in phases.values() for x in xs],gaps)
 assert abs(sum(x['wall_ms'] for x in summary.values())-total)<1e-6 and abs(sum(x['local_pre_enqueue_exposure_ms'] for x in summary.values())-exposure)<1e-6
 return dict(rank=data['rank'],db_sha256=data['db_sha256'],MoE_calls=380,post_combine_wall_ms=total,post_combine_local_exposure_ms=exposure,stages=summary,nested_subsets=dict(H4_gather_output_copies=dict(count=len(copies),wall_ms=duration(copies),local_exposure_ms=intersection(copies,gaps)),H5_shared_same_stream_waits=dict(count=len(events),wall_ms=duration(events),local_exposure_ms=intersection(events,gaps))),examples=examples,limits=['Actual SP-disabled shared-overlap-disabled W8A8 path: routed combine/finalize first, then shared dynamic quant, gate-up, activation, down; shared TP sum occurs after outer moe_forward_shared returns.','All 380 source sequences and 16 copies/3 shared waits per layer asserted. Nested subsets overlap stage columns; never add them.','Scope wall and local launch exposure are profiler-ON, not CPU clock or OFF gain budgets. Work between markers mixes Python/native teardown/preparation.','Shared W8A8 operators and separate shared TP sum have real mathematical consumers; their complete durations are not removable. H4/H5 already have independent interventions, not an automatically composable stack.'])


if __name__=='__main__':
 here=Path(__file__).resolve().parent;point=here.parents[1];rows=[]
 for rank in [13,15]:
  paths=[point/f'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank{rank}.json.gz',point/f'jobs/GLM53-DECODE-QUEUES-20261006/queues_rank{rank}.json.gz']
  out=reduce(*(json.loads(gzip.decompress(p.read_bytes())) for p in paths));out['input_sha256']={str(p.relative_to(point)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};rows.append(out);print(rank,out['stages'],out['nested_subsets'])
 (here/'post_combine.json').write_text(json.dumps(rows,indent=2)+'\n')
