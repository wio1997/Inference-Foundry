#!/usr/bin/env python3
"""Audit DSA CP quantization reuse against Run131/132 call evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
dsa=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py')
model=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py')
linear=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py')
serve=ROOT/'scripts/serve.sh'
text=dsa.read_text();ms=model.read_text();ls=linear.read_text();ss=serve.read_text()
need=[
 'qr_local, qr_pertoken_scale_local = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(',
 'self.wq_b.weight_1,','self.wq_b.weight_2,',
 'self.inderxer_wq_b.weight,','qr_pertoken_scale_local,',
 'q_a = self.wq_a(hidden_states_local)','kv = self.wkv(hidden_states_cache)',
 'weights = self.weights_proj(x)',
]
assert all(x in text for x in need)
assert 'quant_config=None,' in ms[ms.index('self.weights_proj = ReplicatedLinear('):ms.index('self.weights_proj = ReplicatedLinear(')+220]
assert 'npu_dynamic_quant(x, dst_type=self.act_quant_type)' in ls
assert 'FLASHCOMM1_ENABLED:-true' in ss and 'DSA_CP_ENABLED:-true' in ss
trace=json.loads((ROOT/'evidence/20260925_loop040_quant/run131/trace_map.json').read_text())
calls=json.loads((ROOT/'evidence/20260925_loop040_quant/run132/call_summary.json').read_text())
assert trace['quant_kernels_per_target']==236 and calls['rank_count']==8
out={
 'run':'run133','kind':'source_and_call_accounting',
 'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (dsa,model,linear,serve)},
 'frozen_serve_flags':{'enable_dsa_cp':True,'enable_flashcomm1':True},
 'facts':[
  'wq_a consumes hidden_states_local; wkv consumes hidden_states_cache after maybe_all_gather_and_maybe_unpad, so they are not an obvious same tensor reuse pair under TP8 FlashComm1.',
  'DSA CP quantizes q_a with npu_rms_norm_dynamic_quant once; both wq_b weight_1/weight_2 quant matmuls consume qr_local and qr_pertoken_scale_local.',
  'For c4 indexer, indexer wq_b direct quant matmul also consumes qr and qr_pertoken_scale_local.',
  'weights_proj is declared quant_config=None and is not a W8A8 dynamic-quant reuse partner.',
  'Run132 96-token W8A8 apply probe logs one wkv module per target layer during graph setup; it excludes local-token wq_a and direct quant matmul calls.'
 ],
 'quant_kernel_pattern':'Run131 236/target: c4 adds one quant-matmul relative to c128, consistent with indexer wq_b; exact mapping of every kernel to call still lacks direct runtime call tags.',
 'decision':'Reject the specific same-input repeated dynamic quantization hypothesis for the obvious DSA CP pairs. Do not undertake an 8-rank fusion experiment without a distinct shared-input pair. Prioritize a bounded trace/source audit of HC pre, residual clone, normalization and cache-write path.',
 'limits':['Source proves call dataflow under stated flags but does not give runtime tensor identity for every branch.','Run132 diagnostic logging did not cover local-token or direct quant matmul calls.','No candidate implementation or formal E2E.']
}
p=ROOT/'evidence/20260925_loop040_quant/run133'
p.mkdir(parents=True,exist_ok=True)
(p/'source_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'facts':out['facts'],'decision':out['decision']},indent=2))
