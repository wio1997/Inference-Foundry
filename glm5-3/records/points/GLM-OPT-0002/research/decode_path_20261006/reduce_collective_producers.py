"""Run249 producer-side classification via CANN->Dequeue->flow->Enqueue.

Never classify a Python source phase using an asynchronous consumer launch time.
"""
import bisect,collections,gzip,hashlib,json
from pathlib import Path


def reduce(data,queue):
 cpu=sorted((int(x[0]),int(x[1]),x[2]) for x in data['cpu']);scopes=[x for x in cpu if x[2] in ['vllm::mla_forward','vllm::moe_forward_shared','aten::embedding']]
 flow=collections.defaultdict(set)
 for group,connection in queue['connection_ids']:flow[group].add(connection)
 enqueues=collections.defaultdict(list)
 for q in queue['queue']:
  if q[2].startswith('Enqueue@'):
   for c in flow[q[3]]:enqueues[c].append(q)
 deqs=sorted((q for q in queue['queue'] if q[2].startswith('Dequeue@')),key=lambda q:int(q[0]));starts=[int(q[0]) for q in deqs];counts=collections.Counter();examples={};gather_sizes=collections.Counter()
 for c in sorted((c for c in data['comm'] if c[0].startswith('hcom_')),key=lambda c:c[2]):
  assert c[8] is not None and c[9] is not None
  d=deqs[bisect.bisect_right(starts,c[8])-1];assert int(d[0])<=c[8] and c[9]<=int(d[1]) and d[4]==c[11]
  matches=[q for z in flow[d[3]] for q in enqueues[z]];assert len(matches)==1;producer=matches[0];t=int(producer[0]);assert producer[2].removeprefix('Enqueue@')==d[2].removeprefix('Dequeue@')
  enclosing=[x for x in scopes if x[0]<=t<=x[1]];assert len(enclosing)<=1
  if 'allReduce' in c[0]:
   assert producer[2]=='Enqueue@HcclAllreduce'
   if enclosing:assert enclosing[0][2]=='vllm::mla_forward';kind='attention_output_TP_allreduce'
   else:
    previous=max((x for x in scopes if x[1]<=t),key=lambda x:x[1]);kind={'vllm::moe_forward_shared':'shared_expert_output_TP_allreduce','vllm::mla_forward':'dense_MLP_output_TP_allreduce','aten::embedding':'embedding_TP_allreduce'}[previous[2]]
  elif 'allGather' in c[0]:
   assert producer[2] in ['Enqueue@HcclAllgather','Enqueue@HcclAllGather']
   if enclosing:
    kind={'vllm::mla_forward':'attention_DCP_allgather','vllm::moe_forward_shared':'routed_finalize_TP_allgather'}[enclosing[0][2]]
    if kind=='attention_DCP_allgather':gather_sizes[str(c[6])]+=1
   else:kind='target_or_MTP_logits_TP_allgather'
  elif 'alltoall' in c[0].lower():
   assert enclosing and enclosing[0][2]=='vllm::mla_forward';kind='attention_DCP_output_LSE_alltoall'
  else:raise AssertionError(c[0])
  counts[kind]+=1
  if kind not in examples:examples[kind]=dict(comm_name=c[0],group=c[1],CANN_launch_ns=c[8:10],dequeue=d,producer_enqueue=producer,enclosing_source_scope=enclosing,preceding_source_scope=None if enclosing else max((x for x in scopes if x[1]<=t),key=lambda x:x[1]))
 expected=dict(attention_output_TP_allreduce=395,shared_expert_output_TP_allreduce=380,dense_MLP_output_TP_allreduce=15,embedding_TP_allreduce=10,attention_DCP_allgather=395,routed_finalize_TP_allgather=380,target_or_MTP_logits_TP_allgather=10,attention_DCP_output_LSE_alltoall=316)
 assert dict(counts)==expected,(counts,expected);assert dict(gather_sizes)=={'221184':79,'4608':316}
 return dict(rank=data['rank'],db_sha256=data['db_sha256'],records=sum(counts.values()),producer_source_counts=dict(counts),DCP_gather_element_counts=dict(gather_sizes),examples=examples,limits=['Same host167 clock, CANN containment and consumer TID plus unique queue connection-flow joins; source containment uses producer enqueue, not asynchronous CANN host launch.','Source semantic role checked against actual Ascend/vLLM output consumers. Interval membership is not an exclusive CPU cost or a collective-transfer-only duration.','Routed output gathered across token shards and TP-partitioned shared output reduced separately. Shared TP reduction and final combined-output reduction are mutually exclusive in actual MoERunner; this does not prove globally minimal communication over other execution designs.'])


if __name__=='__main__':
 here=Path(__file__).resolve().parent;point=here.parents[1];rows=[]
 for rank in [13,15]:
  paths=[point/f'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank{rank}.json.gz',point/f'jobs/GLM53-DECODE-QUEUES-20261006/queues_rank{rank}.json.gz'];out=reduce(*(json.loads(gzip.decompress(p.read_bytes())) for p in paths));out['input_sha256']={str(p.relative_to(point)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};rows.append(out);print(rank,out['records'],out['producer_source_counts'])
 (here/'collective_producers.json').write_text(json.dumps(rows,indent=2)+'\n')
