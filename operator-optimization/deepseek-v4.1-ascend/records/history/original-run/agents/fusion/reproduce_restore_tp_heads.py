#!/usr/bin/env python3
"""Pure Python check of restore_tp_heads index permutation, no NPU or libraries."""
from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parents[2]
paths=['sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py',
       'sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py',
       'sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py',
       'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py']
hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}

def simulate(p,tokens,heads,width):
 assert p>1 and heads%p==0
 local_heads=heads//p
 # Original tensor output[source_rank][token][global_head][channel].
 def value(s,t,h,c):return (((s*tokens+t)*heads+h)*width+c)
 # Equivalent to view(T,P,Hlocal,D).permute(1,0,2,3).contiguous().
 send=[[[value(s,t,r*local_heads+h,c) for t in range(tokens)
         for h in range(local_heads) for c in range(width)] for r in range(p)] for s in range(p)]
 # torch.distributed.all_to_all_single without split arrays is assumed equal-split
 # and concatenates received source ranks in rank order.
 recv=[sum((send[s][r] for s in range(p)),[]) for r in range(p)]
 expected=[[value(s,t,r*local_heads+h,c) for s in range(p) for t in range(tokens)
            for h in range(local_heads) for c in range(width)] for r in range(p)]
 assert recv==expected
 return {'world_size':p,'tokens_per_source':tokens,'global_heads':heads,'local_heads':local_heads,
         'width':width,'each_source_chunk_elements':len(send[0][0]),'received_elements':len(recv[0]),
         'all_output_elements_checked':sum(map(len,recv)),
         'mapping': 'recv[receiver][source*tokens+token][local_head][channel] = input[source][token][receiver*local_heads+local_head][channel]'}
print(json.dumps({'source_sha256':hashes,'cases':[simulate(2,2,4,2),simulate(8,3,64,4)],
  'limitations':['Model assumes equal-sized all_to_all_single chunks and rank-order concatenation.',
                 'Abstract index example does not execute collective, measure throughput, or prove runtime resolved config.',
                 'Actual head_dim=512; width 4 in second case abbreviates index domain.']},indent=2))
