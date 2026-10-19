import hashlib,json,statistics,time
from pathlib import Path
import torch,torch_npu
torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=True;torch.manual_seed(1903)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
from vllm_ascend.ops.fused_moe import moe_utils
assert hashlib.sha256(Path(moe_utils.__file__).read_bytes()).hexdigest()=='2c3dc13f2f7955f0e077df07adf8538929404a8f0af35a0ea9da42d42bcf5b5a'
convert=moe_utils.cumsum_group_list
stream=torch.npu.Stream();torch.npu.set_stream(stream)
# Exact observed logical W1/W2 dimensions and native NZ preparation.
w1=torch_npu.npu_format_cast(torch.randint(-3,4,(256,6144,256),dtype=torch.int8,device='npu'),29)
w2=torch_npu.npu_format_cast(torch.randint(-3,4,(256,128,6144),dtype=torch.int8,device='npu'),29)
s1=(torch.rand((256,256),device='npu')*.01+.005).float()
s2=(torch.rand((256,6144),device='npu')*.01+.005).bfloat16()
rows=[];timings=[]
def compare(a,b):
 v=(a.float()-b.float()).abs()
 return {'bitwise_equal':bool(torch.equal(a,b)),'different_elements':int((a!=b).sum().cpu()),'max_abs':float(v.max().cpu())}
for n in [6,24]:
 for pattern in ['scatter','head','tail','mixed']:
  x=torch.randn((n,6144),dtype=torch.bfloat16,device='npu')
  if pattern=='scatter':ids_cpu=torch.stack([torch.randperm(256)[:8] for _ in range(n)]).to(torch.int32)
  else:
   cols={'head':list(range(8)),'tail':list(range(248,256)),'mixed':[0,1,7,17,33,63,127,255]}[pattern]
   ids_cpu=torch.tensor([cols]*n,dtype=torch.int32)
  ids=ids_cpu.npu();probs=torch.rand((n,8),device='npu');probs=(probs/probs.sum(dim=1,keepdim=True)).bfloat16()
  hist=torch.bincount(ids_cpu.flatten().long(),minlength=256);prefix_cpu=hist.cumsum(0)
  def one(mode,debug=False):
   typ=1 if mode=='native' else 0  # Only native routing prefix attribute differs
   before=torch.npu.current_stream().record_event()
   q,idx,g,scale=torch_npu.npu_moe_init_routing_v2(x,ids,active_num=n*8,expert_num=256,expert_tokens_num_type=typ,expert_tokens_num_flag=True,active_expert_range=[0,256],quant_mode=1)
   g=g.to(torch.int64);cum=convert(g,typ,0)
   z,sz,_=torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(x=q,weight=w1,weight_scale=s1,x_scale=scale,group_list=cum,bias=None,offset=None,swiglu_limit=0.)
   dbg=[z,sz,idx,cum] if debug else None
   q.set_(torch.empty((0,),device=q.device,dtype=q.dtype))
   before2=torch.npu.current_stream().record_event()
   count=convert(g,typ,1)  # Exact native utility at the GMM2 consumer
   z=torch_npu.npu_grouped_matmul(x=[z],weight=[w2],scale=[s2],bias=None,per_token_scale=[sz],split_item=2,group_list_type=1,group_type=0,group_list=count,output_dtype=torch.bfloat16)[0]
   if debug:dbg.append(z)
   z=torch_npu.npu_moe_token_unpermute(permuted_tokens=z,sorted_indices=torch.abs(idx),probs=probs)
   return (z,count,dbg) if debug else (z,count)
  # No equality, host copy or synchronization between routing, GMM1, GMM2 and unpermute.
  a=one('native');b=one('prefix');torch.npu.synchronize()
  eager=compare(a[0],b[0]);hist_ok=torch.equal(a[1].cpu(),hist) and torch.equal(b[1].cpu(),hist)
  graphs={};outs={}
  for mode in ['native','prefix']:
   for _ in range(3):one(mode)
   torch.npu.synchronize();gph=torch.npu.NPUGraph();pool=torch.npu.graph_pool_handle()
   with torch.npu.graph(gph,pool=pool,stream=stream):outp=one(mode)
   graphs[mode]=(gph,pool);outs[mode]=outp
  # Interleave two independently owned graph families with changing real inputs.
  captured=[]
  for iteration in range(3):
   x.add_(.125);ids.copy_((ids+13)%256)
   reference=one('native');torch.npu.synchronize()
   for mode in ['native','prefix','prefix','native']:
    graphs[mode][0].replay();torch.npu.synchronize()
    captured.append({'iteration':iteration,'mode':mode,'comparison_to_current_native_eager':compare(reference[0],outs[mode][0]),'recovered_histogram_correct':bool(torch.equal(reference[1],outs[mode][1]))})
  for cycle in range(3):
   for mode in ['native','prefix','prefix','native']:
    torch.npu.synchronize();begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);start=time.perf_counter();begin.record()
    for _ in range(30):graphs[mode][0].replay()
    end.record();end.synchronize()
    timings.append({'N':n,'pattern':pattern,'cycle':cycle,'mode':mode,'replays':30,'device_ms_per_layer':begin.elapsed_time(end)/30,'wall_ms_per_layer':(time.perf_counter()-start)*1000/30})
  row={'N':n,'pattern':pattern,'eager_native_vs_prefix':eager,'routing_histogram_after_consumers_correct':bool(hist_ok),'captured':captured}
  if not eager['bitwise_equal'] or any(not z['comparison_to_current_native_eager']['bitwise_equal'] for z in captured):
   aa=one('native',True);bb=one('prefix',True);torch.npu.synchronize()
   row['debug_retained_intermediates']={'final':compare(aa[0],bb[0]),'stages':[compare(u,v) for u,v in zip(aa[2],bb[2])],'stage_names':['GMM1_int8','GMM1_scale','row_idx','GMM1_prefix_input','GMM2_BF16']}
  rows.append(row)
  del graphs,outs,a,b,reference
  torch.npu.synchronize();torch.npu.empty_cache()
print('FULL_PREFIX_RESULT_BEGIN\n'+json.dumps({'completed':True,'rows':rows,'all_eager_bitwise_equal':all(z['eager_native_vs_prefix']['bitwise_equal'] for z in rows),'all_captured_bitwise_equal':all(t['comparison_to_current_native_eager']['bitwise_equal'] for z in rows for t in z['captured']),'all_histograms_preserved':all(z['routing_histogram_after_consumers_correct'] and all(c['recovered_histogram_correct'] for c in z['captured']) for z in rows),'timings':timings,'timing_summary':[{'N':n,'pattern':pattern,'median_device_ms':{mode:statistics.median(z['device_ms_per_layer'] for z in timings if z['N']==n and z['pattern']==pattern and z['mode']==mode) for mode in ['native','prefix']}} for n in [6,24] for pattern in ['scatter','head','tail','mixed']],'peak_bytes':torch.npu.max_memory_allocated(),'scope':'Single-NPU synthetic exact dimensions and native NZ W8A8 routing count1->int64 Cumsum versus routing prefix0->native diff+cat recovery at original GMM2 count1; fusedGMM1/act/quant and unpermute unchanged, native disposal/event ordering, no intermediate host synchronization. Independent raw ACL graph families, not full-model FX graph/attention/KV/TP validation and only full-layer fragment timing, not service evidence.'})+'\nFULL_PREFIX_RESULT_END')
