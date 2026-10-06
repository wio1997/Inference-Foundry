"""Existing Run249 source-contained inter-MC2 supply region partition only."""
import bisect,collections,gzip,hashlib,heapq,json
from pathlib import Path
from reduce_gap_owners import segments
from reduce_prepare_criticality import merged,intersection


def read(path):return json.loads(gzip.decompress(path.read_bytes()))
def ms(xs):return sum(b-a for a,b in merged(xs))/1e6


def reduce(data,queue):
 cpu=sorted((int(x[0]),int(x[1]),x[2]) for x in data['cpu']);starts=[x[0] for x in cpu]
 moe=[x for x in cpu if x[2]=='vllm::moe_forward_shared'];mla=[x for x in cpu if x[2]=='vllm::mla_forward'];assert len(moe)==380 and len(mla)==395
 natives={k:[x for x in cpu if x[2]=='npu::npu_moe_distribute_'+k+'_v2'] for k in ['dispatch','combine']};assert all(len(xs)==380 for xs in natives.values())
 regions=[]
 for i,scope in enumerate(moe):
  a,b,_=scope;s,e,_=natives['dispatch'][i];u,v,_=natives['combine'][i];assert a<=s<e<=u<v<=b
  if i>0:
   old=moe[i-1];assert old[1]<=a
   regions.append((natives['combine'][i-1][1],old[1],'previous_MoE_finalize_and_shared_output'))
   # These source scopes have a common main thread. MLA scopes lie fully in
   # this between-MoE interval. Keep other residual/norm/control work separate.
   children=[x for x in mla if old[1]<=x[0] and x[1]<=a]
   pos=old[1]
   for x,y,_ in children:
    assert pos<=x<y<=a;regions.extend([(pos,x,'between_MoE_outside_MLA'),(x,y,'MLA_attention')]);pos=y
   regions.append((pos,a,'between_MoE_outside_MLA'))
   regions.append((a,s,'next_MoE_prepare_route_and_gate'))
  regions.extend([(s,e,'native_dispatch'),(e,u,'routed_expert_MLP_between_EP'),(u,v,'native_combine')])
 regions=sorted((a,b,n) for a,b,n in regions if b>a)
 assert all(x[1]==y[0] for x,y in zip(regions,regions[1:]))
 lo,hi=regions[0][0],regions[-1][1];assert lo==natives['dispatch'][0][0] and hi==natives['combine'][-1][1]
 _,_,gaps=segments(data,queue);gaps=[(max(a,lo),min(b,hi)) for a,b in gaps if b>lo and a<hi]
 by_phase=collections.defaultdict(list)
 for a,b,n in regions:by_phase[n].append((a,b))
 phase_summary={n:dict(count=len(xs),scope_wall_ms=ms(xs),local_pre_enqueue_exposure_ms=intersection(xs,gaps)) for n,xs in by_phase.items()}
 assert abs(sum(x['scope_wall_ms'] for x in phase_summary.values())-(hi-lo)/1e6)<1e-6
 assert abs(sum(x['local_pre_enqueue_exposure_ms'] for x in phase_summary.values())-ms(gaps))<1e-6
 # Shortest active exported CPU-op convention only, on exposed gaps, partition
 # separately for each source region. This is not a Python stack or CPU clock.
 events=[]
 for i,(a,b,n) in enumerate(cpu):
  if b>lo and a<hi and b>a:events.extend([(max(a,lo),1,i),(min(b,hi),-1,i)])
 for a,b in gaps:events.extend([(a,2,-1),(b,-2,-1)])
 for i,(a,b,n) in enumerate(regions):events.extend([(a,3,i),(b,-3,i)])
 events.sort();active=set();leaf=[];inside=0;region=None;previous=lo;labels=collections.defaultdict(collections.Counter)
 for t,kind,i in events:
  if t>previous and inside:
   assert region is not None
   while leaf and leaf[0][2] not in active:heapq.heappop(leaf)
   name=leaf[0][3] if leaf else 'NO_EXPORTED_CPU_SCOPE';labels[regions[region][2]][name]+=(t-previous)/1e6
  previous=t
  if kind==1:
   active.add(i);a,b,n=cpu[i];heapq.heappush(leaf,(b-a,-a,i,n))
  elif kind==-1:active.remove(i)
  elif kind==2:inside+=1
  elif kind==-2:inside-=1
  elif kind==3:assert region is None;region=i
  else:assert region==i;region=None
  assert inside in (0,1)
 assert not inside and region is None
 for n,row in phase_summary.items():assert abs(sum(labels[n].values())-row['local_pre_enqueue_exposure_ms'])<1e-6
 return dict(rank=data['rank'],db_sha256=data['db_sha256'],native_first_to_last_ns=[lo,hi],window_ms=(hi-lo)/1e6,local_pre_enqueue_exposure_ms=ms(gaps),phases=phase_summary,shortest_scope_local_exposure_ms={n:dict(v.most_common(25)) for n,v in labels.items()},limits=['Run249 profiler-ON scope wall and exposed gaps only, not CPU runtime/OFF budget.','Regions follow exact source-contained MoE/native boundaries. MLA includes its host submission and waits, not solely necessary math.','Actual Ascend 0.27.1 runs routed finalize then shared expert; post-combine includes necessary shared quant/MLP, not just finalize. Pre-dispatch is prepare/routing/gate, not shared MLP. Whole region durations are not deletable.','Four inter-round gaps are included among between-MoE outside-MLA work; exact Scheduler/Core/Executor/ModelRunner entry/exit absent.','The initial prefix before first MC2 and final suffix after last combine are outside this envelope.'])


if __name__=='__main__':
 here=Path(__file__).resolve().parent;point=here.parents[1];out=[]
 for rank in [13,15]:
  paths=[point/f'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank{rank}.json.gz',point/f'jobs/GLM53-DECODE-QUEUES-20261006/queues_rank{rank}.json.gz'];r=reduce(*(read(p) for p in paths));r['input_sha256']={str(p.relative_to(point)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};out.append(r);print(rank,r['window_ms'],r['local_pre_enqueue_exposure_ms'],r['phases'])
 (here/'interlayer_supply.json').write_text(json.dumps(out,indent=2)+'\n')
