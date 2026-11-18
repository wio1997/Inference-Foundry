"""DSV4.1/NPU factory-scoped route; preserves native NPU finalization."""
from pathlib import Path
import hashlib,json,os
_INSTALLED=False
def install(root):
 global _INSTALLED
 if _INSTALLED:return
 from route_compat import get_gate
 gate=get_gate(root)
 import sglang.srt.hardware_backend.npu.moe.topk as native
 prior=native.fused_topk_npu
 def fused(hidden_states,router_logits,topk_config,num_token_non_padded=None,expert_location_dispatch_info=None,layer_id=None):
  if router_logits.device.type!='npu' or topk_config.scoring_func!='sqrtsoftplus':
   return prior(hidden_states,router_logits,topk_config,num_token_non_padded,expert_location_dispatch_info,layer_id)
  n=router_logits.shape[-1];routed={384:6,128:3}.get(n);shared=topk_config.num_shared_topk_slots
  assert routed is not None and shared in (0,1) and topk_config.top_k==routed+shared,(n,topk_config.top_k,shared)
  assert not topk_config.use_grouped_topk and topk_config.renormalize and topk_config.sqrtsoftplus_log1p
  bias=topk_config.correction_bias;assert bias is not None and bias.ndim==1 and bias.shape[0]==n
  w,ids=gate(router_logits.to(__import__('torch').float32),bias,topk_config.top_k,scoring_func='sqrtsoftplus',num_fused_shared_experts=shared,renormalize=True,routed_scaling_factor=float(topk_config.routed_scaling_factor),apply_routed_scaling_factor_on_output=bool(topk_config.apply_routed_scaling_factor_on_output),sqrtsoftplus_log1p=True)
  # Preserve valid padded IDs, logical-to-physical EP mapping, routed-expert
  # capture, zero padded weights and recorder policy from the fixed NPU backend.
  return native._finalize_topk_npu(topk_ids=ids,topk_weights=w,router_logits=router_logits,topk_config=topk_config,num_token_non_padded=num_token_non_padded,expert_location_dispatch_info=expert_location_dispatch_info,layer_id=layer_id)
 native.fused_topk_npu=fused;_INSTALLED=True
 record={'pid':os.getpid(),'scope':'DSV4.1 NPU worker factory only, sqrtsoftplus384/top6 or128/top3, nativefinalizer retained','native_source_pins':json.loads((root/'native_route_sha256.json').read_text()),'route_kernel_sha256':hashlib.sha256((root/'route_kernel.py').read_bytes()).hexdigest(),'full_model_PD_quality_performance_accepted':False}
 (root/f'route_install.pid{os.getpid()}.json').write_text(json.dumps(record,indent=2))
