#!/usr/bin/env python3
"""Product GMM live-route weight footprint and source-path audit."""
import hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
shape=json.loads((root/'evidence/20260925_loop039_gmm/run115/shape_summary.json').read_text())
counts=json.loads((root/'evidence/20260925_loop039_gmm/run121/counts_summary.json').read_text())
trace=json.loads((root/'evidence/20260925_loop044_target/run145/comm_gmm_audit.json').read_text())
sample=shape['rank_records'][0]['target_96x6_shape']
assert sample['hidden_states']['shape']==[576,4096] and sample['w1'][0]['shape']==[32,4096,512] and sample['w2'][0]['shape']==[32,2048,512]
assert len(counts['layers'])==86
w1_per_expert=4096*512*4
w2_per_expert=2048*512*4
packed_per_expert=w1_per_expert+w2_per_expert
routes=[]
for cycle in (64,65):
 for rank in range(8):
  layers=[x for x in counts['layers'] if x['cycle']==cycle]
  assert len(layers)==43
  active=sum(x['active_experts_per_rank'][rank] for x in layers)
  tokens=sum(x['tokens_per_rank'][rank] for x in layers)
  routes.append({'cycle':cycle,'rank':rank,'active_expert_layer_pairs':active,'routed_tokens':tokens,'minimum_packed_weight_bytes_if_loaded_once_per_active_expert':active*packed_per_expert})
def desc(v):return {'median':statistics.median(v),'min':min(v),'max':max(v)}
source=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/device/device_op.py')
src=source.read_text()
assert 'def npu_grouped_matmul_gmm2(' in src and 'return torch_npu.npu_grouped_matmul(' in src
gmm1=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h')
assert 'if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0)' in gmm1.read_text()
out={'run':'run146','source':'Run115 product GMM shape, Run121 live route counts, Run107 profiled GMM duration (different diagnostic cohorts)',
     'product_shape':sample,'w1_packed_bytes_per_expert':w1_per_expert,'w2_packed_bytes_per_expert':w2_per_expert,
     'packed_bytes_per_expert':packed_per_expert,
     'active_expert_layer_pairs_per_rank_cycle':desc([r['active_expert_layer_pairs'] for r in routes]),
     'routed_tokens_per_rank_cycle':desc([r['routed_tokens'] for r in routes]),
     'active_packed_weight_bytes_per_rank_cycle':desc([r['minimum_packed_weight_bytes_if_loaded_once_per_active_expert'] for r in routes]),
     'active_packed_weight_GiB_median':statistics.median(r['minimum_packed_weight_bytes_if_loaded_once_per_active_expert'] for r in routes)/2**30,
     'profiled_gmm_ms_median':trace['per_rank_gmm_duration_ms']['median'],
     'implied_weight_read_GBps_if_each_active_weight_loaded_once':statistics.median(r['minimum_packed_weight_bytes_if_loaded_once_per_active_expert'] for r in routes)/1e9/(trace['per_rank_gmm_duration_ms']['median']/1000),
     'gmm1_source':str(gmm1),'gmm1_sha256':hashlib.sha256(gmm1.read_bytes()).hexdigest(),
     'gmm2_source':str(source),'gmm2_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
     'gmm1_path':'custom fused grouped matmul + SwiGLU + quant; zero-M experts skipped in source',
     'gmm2_path':'DeviceOperator.npu_grouped_matmul_gmm2 -> BaseDeviceAdaptor.npu_grouped_matmul_gmm2 -> torch_npu.npu_grouped_matmul; vendor kernel internals unavailable here',
     'routes':routes,
     'interpretation':'Packed active-weight bytes are a conditional traffic estimate, not measured HBM bytes. Different runs supply counts and duration. Kernel could reread weights, use cache, or load unused data. No hardware bandwidth/compute counter is present in Run107 kernel_details.csv; achievable bound cannot be inferred from this estimate alone.'}
p=root/'evidence/20260925_loop044_target/run146/gmm_bound_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k in ('active_expert_layer_pairs_per_rank_cycle','routed_tokens_per_rank_cycle','active_packed_weight_bytes_per_rank_cycle','active_packed_weight_GiB_median','profiled_gmm_ms_median','implied_weight_read_GBps_if_each_active_weight_loaded_once')},indent=2))
