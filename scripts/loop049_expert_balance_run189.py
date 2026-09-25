#!/usr/bin/env python3
"""Offline optimistic active-weight placement screen from Run121 and Run116."""
import json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
outdir=root/'evidence/20260925_loop049_expert_balance/run189'
audit=json.loads((root/'evidence/20260925_loop039_gmm/run116/priority_audit.json').read_text())
weights_bytes_per_expert=12*1024*1024 # W1 8 MiB + W2 4 MiB, packed
cycles=[]
for cycle in [64,65]:
 rows=[json.loads((root/f'evidence/20260925_loop039_gmm/run121/counts/rank{rank}_cycle{cycle}.json').read_text()) for rank in range(8)]
 assert all(x['rank']==rank and x['cycle']==cycle and x['ref_count']==86 for rank,x in enumerate(rows))
 active=[]
 for ordinal in range(43,86):
  counts=[[int(v>0) for v in rows[rank]['rows'][ordinal]['counts']] for rank in range(8)]
  assert all(len(z)==32 for z in counts)
  active.append([sum(z) for z in counts])
 rank_sum=[sum(layer[rank] for layer in active) for rank in range(8)]
 maxsum=sum(max(layer) for layer in active)
 mean_sum=sum(sum(layer)/8 for layer in active)
 ceilsum=sum(math.ceil(sum(layer)/8) for layer in active)
 prof=[w for w in audit['windows'] if w['cycle']==cycle]
 observed_gmm=[w['gmm_kernel_sum_ms'] for w in prof]
 cycles.append({'cycle':cycle,'source_run121_rank_active_expert_layer_sums':rank_sum,'source_run121_total_active_pairs':sum(rank_sum),'sum_per_layer_max_active_experts':maxsum,'sum_per_layer_mean_active_experts':mean_sum,'sum_per_layer_ceil_lower_bound':ceilsum,'ideal_continuous_balance_gap_active_expert_reads':maxsum-mean_sum,'ideal_integer_balance_gap_active_expert_reads':maxsum-ceilsum,'ideal_continuous_balance_gap_packed_bytes_gib':(maxsum-mean_sum)*weights_bytes_per_expert/(1024**3),'ideal_continuous_balance_gap_time_at_1tb_s_ms':(maxsum-mean_sum)*weights_bytes_per_expert/1e12*1000,'profile_run107_gmm_sum_ms_min':min(observed_gmm),'profile_run107_gmm_sum_ms_max':max(observed_gmm),'profile_run107_gmm_sum_ms_spread':max(observed_gmm)-min(observed_gmm),'profile_run107_valid_rank_count':len(prof),'active_per_layer_rank_counts':active})
result={'run':'run189','sources':['Run121 independent service live expert counts at cycles64/65','Run116 uses Run107 independent service synchronized profiled target windows','Run148/150 product-shape one-card GMM weight traffic near packed weight bytes'],'cycles':cycles,'decision':'The perfect per-layer active-weight balance arithmetic screens only about2.0-2.2ms/cycle at 1TB/s before routing, remap, HCCL and implementation costs. This is not an achievable bound, and Run107 GMM rank-sum spreads <0.52ms do not reveal a persistent hot rank. Static expert remapping is currently lower priority; a genuine same-state per-layer timing/count pairing would be required before any code change.','limits':['Run121 live counts and Run107 timing are separate services and not same-state, despite shared cycle numbers. No causal correlation is claimed.','Max active expert count is an optimistic serial-layer traffic proxy; GMM kernel duration need not scale linearly with packed bytes and layer work can overlap.','1TB/s is representative one-card counter order from Run148/150, not a guaranteed capacity or attainable latency bound.','Perfect per-layer balancing may be impossible for one fixed expert placement across all layers and cycles; routing/weight movement changes collectives and semantics.']}
(outdir/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{'cycle':x['cycle'],'rank_sum':x['source_run121_rank_active_expert_layer_sums'],'ideal_gap_pairs':x['ideal_continuous_balance_gap_active_expert_reads'],'ideal_gap_ms_at_1tb_s':x['ideal_continuous_balance_gap_time_at_1tb_s_ms'],'profile_rank_gmm_spread_ms':x['profile_run107_gmm_sum_ms_spread']} for x in cycles],indent=2))
