#!/usr/bin/env python3
"""Cross-check live routing, profiled GMM families and W4A8 kernel zero handling."""
import hashlib
import json
import statistics
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
trace=json.loads((ROOT/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
counts=json.loads((ROOT/'evidence/20260925_loop039_gmm/run121/counts_summary.json').read_text())
excluded={(x['rank'],x['cycle']) for x in trace['excluded_windows']}
names={
 'gmm1':'aclnnGroupedMatmulSwigluQuantWeightNzV2_GroupedMatmulSwigluQuantV2_GroupedMatmulSwigluQuantV2',
 'gmm2':'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul',
}
rows=[]
for rank in trace['ranks']:
 for cycle in rank['cycles']:
  if (rank['rank'],cycle['cycle']) in excluded: continue
  sums=dict(cycle['top_kernel_sums_ms'])
  assert all(k in sums for k in names.values())
  rows.append({'rank':rank['rank'],'cycle':cycle['cycle'],
               'gmm1_ms':sums[names['gmm1']], 'gmm2_ms':sums[names['gmm2']],
               'gmm_total_ms':cycle['grouped_matmul_sum_ms'],
               'family_sum_ms':sums[names['gmm1']]+sums[names['gmm2']]})
assert len(rows)==15
assert all(abs(r['gmm_total_ms']-r['family_sum_ms'])<1e-8 for r in rows)
src=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h')
s=src.read_text()
assert 'if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0)' in s
assert 'int32_t splitValue = (currSplitValue - prevSplitValue) * 2' in s
out={
 'run':'run122','kind':'source_and_trace_audit',
 'source_trace':'Run107 synchronized 8-rank target windows, 15 valid rank-cycle windows',
 'source_counts':'Run121 live 8-rank cycles64/65, 43 layers',
 'gmm1_family':names['gmm1'],'gmm2_family':names['gmm2'],
 'gmm1_ms':{'median':statistics.median(r['gmm1_ms'] for r in rows),
            'min':min(r['gmm1_ms'] for r in rows),'max':max(r['gmm1_ms'] for r in rows)},
 'gmm2_ms':{'median':statistics.median(r['gmm2_ms'] for r in rows),
            'min':min(r['gmm2_ms'] for r in rows),'max':max(r['gmm2_ms'] for r in rows)},
 'gmm_total_median_ms':statistics.median(r['gmm_total_ms'] for r in rows),
 'active_experts_mean_of_32':counts['active_experts_per_rank_layer']['mean'],
 'active_experts_median_of_32':counts['active_experts_per_rank_layer']['median'],
 'gmm1_source':str(src),'gmm1_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
 'gmm1_zero_handling':'A8W4 MSD mid kernel computes splitValue from live group_list and continues before MatMul when mnConfig.m<=0; empty experts do not execute that MatMul path.',
 'inference':'Run121 sparsity cannot be counted as new 50% removable GMM1 work. A >=5ms target-stage improvement would require a measured operator substitution/fusion on real shapes, not merely skipping zero experts.',
 'limits':['Run107 synchronized diagnostic; kernel sums are not stage reduction.','Run121 routes are different cycles/run from Run107 traces.','GMM2 zero handling not proven by this source audit.'],
 'windows':rows}
p=ROOT/'evidence/20260925_loop039_gmm/run122'
p.mkdir(parents=True,exist_ok=True)
(p/'path_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('gmm1_ms','gmm2_ms','gmm_total_median_ms','active_experts_mean_of_32')},indent=2))
