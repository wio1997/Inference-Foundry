"""Real16rank TP4 layout, checkpoint HC and stock AscendOps component contracts.

No model score. All communicator lookups below are injected only in this
isolated process; the production GroupCoordinator/custom_op and stock routing
implementations themselves are exercised unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import datetime,hashlib,json,os,traceback
import torch,torch_npu,torch.distributed as dist

R=Path('/task');O=R/'probes/hc_rows_g43'
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
result={'rank':rank,'status':'running','checks':{},'HC_cases':[]}
def stamp(stage):
 result.update(stage=stage,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 p=O/f'rank_{rank}.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(O/f'rank_{rank}.json')
 print(json.dumps({'rank':rank,'stage':stage}),flush=True)
try:
 torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.npu.set_device(local)
 from sglang.srt.hardware_backend.npu.utils import init_npu_backend
 init_npu_backend()
 from sglang.srt.distributed.parallel_state import GroupCoordinator
 from sglang.srt.layers import dp_attention as dp
 from sglang.srt.layers.moe.token_dispatcher import ascend_ops as ao
 from sglang.srt.layers.moe.moe_runner import ascend as runner
 from sglang.srt.layers.moe.utils import AscendOpsConfig
 from sglang.srt.layers.layernorm import RMSNorm
 from sglang.srt.models import deepseek_v4_mhc as mhc
 from safetensors import safe_open
 from hc_row_shard import RowLayout
 stamp('HCCL_init');dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=180))
 assert dist.get_world_size()==16
 groups=[list(range(i,i+4)) for i in range(0,16,4)]
 attn=GroupCoordinator(groups,local,'hccl',use_pynccl=False,use_mscclpp=False,
                       use_custom_allreduce=False,use_torch_symm_mem_all_reduce=False,
                       use_hpu_communicator=False,use_xpu_communicator=False,
                       group_name='hc43attn')
 ep=dist.new_group(list(range(16)),backend='hccl')
 parallel=SimpleNamespace(attn_tp_group=attn,attn_tp_size=4,attn_tp_rank=rank%4,
  tp_group=SimpleNamespace(device_group=dist.group.WORLD),tp_size=16,
  moe_ep_group=SimpleNamespace(device_group=ep,world_size=16,rank_in_group=rank),moe_ep_size=16)
 dp.get_parallel=lambda:parallel;ao.get_parallel=lambda:parallel
 result['source_hashes']={n:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
  for n,m in [('HC',mhc),('dispatcher',ao),('runner',runner)]}
 result['helper_sha256']=hashlib.sha256((R/'compat/runtime/g43/hc_row_shard.py').read_bytes()).hexdigest()
 result['group']={'TP':4,'EP':16,'DP_groups':4,'actual_GroupCoordinator':True,'rank_in_group':attn.rank_in_group}
 assert attn.rank_in_group==rank%4
 model=Path('/model');cfg=json.loads((model/'config.json').read_text());cfg=cfg.get('text_config',cfg)
 index=json.loads((model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
 hc_cfg=mhc.HcConfig(4,cfg['hc_sinkhorn_iters'],cfg['hc_eps'],cfg['rms_norm_eps'],5120,True,False)
 def load(name):
  with safe_open(model/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name).contiguous().to('npu')
 def hc(layer,kind):
  values=[load(f'layers.{layer}.hc_{kind}_{suffix}').float() for suffix in ['fn','scale','base']]
  norm=RMSNorm(5120,eps=cfg['rms_norm_eps']).to(device='npu',dtype=torch.bfloat16)
  norm.weight.data.copy_(load(f'layers.{layer}.{kind}_norm.weight').bfloat16())
  return mhc.HcSubLayer(hc_cfg,*values,norm=norm)
 gen=torch.Generator(device='cpu').manual_seed(1664301+rank//4)
 graphs=[]
 with torch.inference_mode():
  for n in [0,1,6,12,18,24,4097,8192]:
   stamp('row_layout_'+str(n));layout=RowLayout(n,4,rank%4)
   full=(torch.arange(n,dtype=torch.int64,device='npu')+100000*(rank//4)).view(n,1)
   shard=layout.shard(full);gathered=layout.gather(shard)
   assert torch.equal(gathered,full)
   if n:assert shard.untyped_storage().nbytes()==shard.numel()*shard.element_size()
   result['checks'][f'row_order_T{n}_global_tail_pad_exact']=True
  for layer_id,kind in [(0,'attn'),(39,'ffn')]:
   h=hc(layer_id,kind)
   for n in [0,1,6,12,18,24,4097,8192]:
    stamp(f'HC_{layer_id}_{kind}_{n}');layout=RowLayout(n,4,rank%4)
    full=torch.randn((n,4,5120),generator=gen).bfloat16().to('npu')
    previous=torch.rand((n,4),generator=gen).float().to('npu')
    state=mhc.HcState(full,previous);local_state=layout.shard_state(state)
    back=layout.gather_state(local_state)
    assert torch.equal(back.residual,full) and torch.equal(back.pre,previous)
    original_x=mhc.combine(h,state);local_x=layout.gather(mhc.combine(h,local_state))
    assert torch.equal(original_x,local_x), 'Combine/norm row layout changed BF16 values'
    original=mhc._mix_stats_impl(h,full);local_coeff=mhc._mix_stats_impl(h,local_state.residual)
    assembled=tuple(layout.gather(x) for x in local_coeff)
    errors=[float((a-b).abs().max()) if a.numel() else 0. for a,b in zip(original,assembled)]
    assert all(torch.allclose(a,b,atol=2e-5,rtol=2e-5) for a,b in zip(original,assembled)),errors
    # Compare post kernels with identical coefficients to isolate row semantics;
    # coefficient differences above retain the unchanged G32/G42 strict gate.
    y=torch.randn((n,5120),generator=gen).bfloat16().to('npu')
    reference_post=mhc.post(h.cfg,y,full,assembled[1],assembled[2])
    local_post=layout.gather(mhc.post(h.cfg,layout.shard(y),local_state.residual,local_coeff[1],local_coeff[2]))
    assert torch.equal(reference_post,local_post), 'HC post row layout changed BF16 values'
    # The full state readers and late-tail narrowing preserve row/pre identity.
    narrowed=back.take_rows(lambda x:x[::2]);small=layout.resized(narrowed.residual.shape[0])
    rebuilt=small.gather_state(small.shard_state(narrowed))
    assert torch.equal(rebuilt.residual,full[::2]) and torch.equal(rebuilt.pre,previous[::2])
    result['HC_cases'].append({'layer':layer_id,'kind':kind,'rows':n,'coefficient_max_abs_errors':errors,
     'coefficient_gate_atol_rtol':[2e-5,2e-5],'combine_norm_bitwise':True,'post_given_same_coefficients_bitwise':True,
     'tail_row_selection_bitwise':True})
    del full,previous,state,local_state,back,original_x,local_x,original,local_coeff,assembled,y,reference_post,local_post,narrowed,rebuilt
    torch.npu.empty_cache()
  result['checks']['actual_checkpoint_HC_and_reader_layout']=True
  result['all_HC_groups']=[]
  for layer_id in range(40):
   for kind in ['attn','ffn']:
    stamp(f'all_HC_groups_{layer_id}_{kind}');h=hc(layer_id,kind);layout=RowLayout(24,4,rank%4)
    full=torch.randn((24,4,5120),generator=gen).bfloat16().to('npu')
    original=mhc._mix_stats_impl(h,full)
    candidate=tuple(layout.gather(x) for x in mhc._mix_stats_impl(h,layout.shard(full)))
    errors=[float((x-y).abs().max()) for x,y in zip(original,candidate)]
    assert all(torch.allclose(x,y,atol=2e-5,rtol=2e-5) for x,y in zip(original,candidate)),(layer_id,kind,errors)
    result['all_HC_groups'].append({'layer':layer_id,'kind':kind,'rows':24,'coefficient_max_abs_errors':errors,'coefficient_gate_atol_rtol':[2e-5,2e-5]})
    del full,original,candidate
  result['checks']['all80_actual_HC_groups_rows24_strict_coefficients']=True
  # Keep HC graph inputs and both captured graphs alive across changed replays.
  hc_graphs=[];h=hc(0,'attn')
  def hc_step(full,previous,y,layout):
   state=layout.shard_state(mhc.HcState(full,previous))
   collapsed=layout.gather(mhc.combine(h,state))
   pre,post,comb=mhc._mix_stats_impl(h,state.residual)
   updated=layout.gather(mhc.post(h.cfg,layout.shard(y),state.residual,post,comb))
   return collapsed,updated
  for n in [6,18,24]:
   stamp('HC_changed_graph_'+str(n));layout=RowLayout(n,4,rank%4)
   full=torch.randn((n,4,5120),generator=gen).bfloat16().to('npu')
   previous=torch.rand((n,4),generator=gen).float().to('npu')
   y=torch.randn((n,5120),generator=gen).bfloat16().to('npu')
   for _ in range(3):hc_step(full,previous,y,layout)
   torch.npu.synchronize();dist.barrier();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):captured=hc_step(full,previous,y,layout)
   hc_graphs.append((full,previous,y,layout,g,captured))
  for full,previous,y,layout,g,captured in hc_graphs+list(reversed(hc_graphs)):
   full.mul_(.91);previous.mul_(.93);y.add_(.03)
   eager=hc_step(full,previous,y,layout);g.replay();torch.npu.synchronize()
   assert all(torch.equal(a,b) for a,b in zip(eager,captured))
  result['checks']['HC_T6_T18_T24_changed_input_alternating_graph_bitwise']=True
  route_cfg=SimpleNamespace(num_experts=384,num_local_experts=24,top_k=6,hidden_size=5120,num_fused_shared_experts=0,moe_ep_rank=rank)
  dispatcher=ao.AscendOpsDispatcher(route_cfg,AscendOpsConfig(prefill='alltoallv',decode='low_latency'))
  dispatcher.set_quant_config({'dispatcher_output_dtype':'int8'})
  def identity(dispatch):
   context={};entry=runner.pre_permute_ascend_ops_to_ascend(dispatch,None,route_cfg,context)
   values=(entry.hidden_states.float()*entry.hidden_states_scale.reshape(-1,1)).bfloat16()
   combined=runner.post_permute_ascend_to_ascend_ops(runner.AscendRunnerOutput(values),None,route_cfg,context)
   return dispatcher.combine(combined)
  def mc2(local_x,ids,weights):return identity(dispatcher._dispatch_low_latency(local_x,weights,ids))
  for n in [1,6,18,24]:
   stamp('MC2_HCshard_'+str(n));layout=RowLayout(n,4,rank%4)
   full=torch.full((n,5120),rank//4+1,dtype=torch.bfloat16,device='npu');x=layout.shard(full)
   ids=((torch.arange(layout.chunk*6,device='npu').reshape(layout.chunk,6)*17+rank*6)%384).long()
   weights=torch.full((layout.chunk,6),1/6,dtype=torch.float32,device='npu')
   eager=layout.gather(mc2(x,ids,weights));assert torch.equal(eager,full)
   for _ in range(2):layout.gather(mc2(x,ids,weights))
   torch.npu.synchronize();dist.barrier();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):captured=layout.gather(mc2(x,ids,weights))
   graphs.append((layout,x,ids,weights,g,captured))
   for scale in [2.,3.]:
    x.mul_(scale);g.replay();torch.npu.synchronize()
    expected=layout.gather(x);assert torch.equal(captured,expected)
   result['checks'][f'INT8_MC2_sharded_T{n}_padding_changed_graph_exact']=True
  for uneven in [False,True]:
   n=8192 if not uneven or rank//4==0 else 0
   stamp('prefill_HCshard_'+('uneven' if uneven else 'all'))
   layout=RowLayout(n,4,rank%4);full=torch.full((n,5120),rank//4+1,dtype=torch.bfloat16,device='npu');x=layout.shard(full)
   ids=((torch.arange(layout.chunk*6,device='npu').reshape(layout.chunk,6)*17+rank*6)%384).long()
   weights=torch.full((layout.chunk,6),1/6,dtype=torch.float32,device='npu')
   y=identity(dispatcher._dispatch_alltoallv(x,weights,ids,dist.group.WORLD))
   assert torch.equal(layout.gather(y),full)
   result['checks']['INT8_alltoallv_T8192_'+str(uneven)+'_exact']=True
  for layout,x,ids,weights,g,captured in reversed(graphs):
   x.add_(1);g.replay();torch.npu.synchronize();assert torch.equal(captured,layout.gather(x))
  result['checks']['captured_graphs_after_prefill_exact']=True
 result['status']='passed_real16rank_HC_row_layout_and_stock_transport_component'
 result['scope']='Real NPU TP4/EP16 row layout, actual checkpoint HC/norm strict coefficients, rowwise BF16 post, tail selection and identity-expert stock INT8 transport/graphs. Not a real expert-GMM/shared-expert, fullmodel quality, fit or performance acceptance.'
 stamp('terminal_passed')
except BaseException:
 result['stage_before_error']=result.get('stage');result['status']='failed';result['error']=traceback.format_exc();stamp('terminal_failed');raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
