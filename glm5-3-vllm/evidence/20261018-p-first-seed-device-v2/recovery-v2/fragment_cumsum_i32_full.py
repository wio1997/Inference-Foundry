"""Real-size native MoE producer/consumer chain on one freed D NPU."""
from pathlib import Path
import json,socket,subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-target-live1-20261019/node33')
plan=json.loads((root/'plan.json').read_text());cid='4d620dfbdf2981ee4c95e029fd1cf3ee297bdb153a548bbb39027ac0e9252041'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert not d['State']['Running'] and d['Id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
assert plan['reservation']['expected_uid']=='736f2c60-59b7-438d-98cf-a7b3a932ce18'
assert subprocess.check_output(['npu-smi','info'],text=True,timeout=30).count('No running processes found in NPU')==8
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
code=r'''import json,time
import torch,torch_npu
torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=True;torch.manual_seed(1903)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
stream=torch.npu.Stream();torch.npu.set_stream(stream)
# Exact observed logical W1/W2 dimensions and native NZ preparation.
w1=torch_npu.npu_format_cast(torch.randint(-3,4,(256,6144,256),dtype=torch.int8,device='npu'),29)
w2=torch_npu.npu_format_cast(torch.randint(-3,4,(256,128,6144),dtype=torch.int8,device='npu'),29)
s1=(torch.rand((256,256),device='npu')*.01+.005).float()
s2=(torch.rand((256,6144),device='npu')*.01+.005).bfloat16()
rows=[]
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
   typ=1  # Native routing and native GMM2 count attributes stay identical
   before=torch.npu.current_stream().record_event()
   q,idx,g,scale=torch_npu.npu_moe_init_routing_v2(x,ids,active_num=n*8,expert_num=256,expert_tokens_num_type=typ,expert_tokens_num_flag=True,active_expert_range=[0,256],quant_mode=1)
   g=g.to(torch.int64);cum=g.cumsum(0) if mode=='native' else g.to(torch.int32).cumsum(0,dtype=torch.int32).to(torch.int64)
   z,sz,_=torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(x=q,weight=w1,weight_scale=s1,x_scale=scale,group_list=cum,bias=None,offset=None,swiglu_limit=0.)
   dbg=[z,sz,idx,cum] if debug else None
   q.set_(torch.empty((0,),device=q.device,dtype=q.dtype))
   before2=torch.npu.current_stream().record_event()
   z=torch_npu.npu_grouped_matmul(x=[z],weight=[w2],scale=[s2],bias=None,per_token_scale=[sz],split_item=2,group_list_type=typ,group_type=0,group_list=g,output_dtype=torch.bfloat16)[0]
   if debug:dbg.append(z)
   z=torch_npu.npu_moe_token_unpermute(permuted_tokens=z,sorted_indices=torch.abs(idx),probs=probs)
   return (z,g,dbg) if debug else (z,g)
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
   x.add_(.125)
   reference=one('native');torch.npu.synchronize()
   for mode in ['native','prefix','prefix','native']:
    graphs[mode][0].replay();torch.npu.synchronize()
    captured.append({'iteration':iteration,'mode':mode,'comparison_to_current_native_eager':compare(reference[0],outs[mode][0])})
  row={'N':n,'pattern':pattern,'eager_native_vs_prefix':eager,'routing_histogram_after_consumers_correct':bool(hist_ok),'captured':captured}
  if not eager['bitwise_equal'] or any(not z['comparison_to_current_native_eager']['bitwise_equal'] for z in captured):
   aa=one('native',True);bb=one('prefix',True);torch.npu.synchronize()
   row['debug_retained_intermediates']={'final':compare(aa[0],bb[0]),'stages':[compare(u,v) for u,v in zip(aa[2],bb[2])],'stage_names':['GMM1_int8','GMM1_scale','row_idx','GMM1_prefix_input','GMM2_BF16']}
  rows.append(row)
  del graphs,outs,a,b,reference
  torch.npu.synchronize();torch.npu.empty_cache()
print('FULL_PREFIX_RESULT_BEGIN\n'+json.dumps({'completed':True,'rows':rows,'all_eager_bitwise_equal':all(z['eager_native_vs_prefix']['bitwise_equal'] for z in rows),'all_captured_bitwise_equal':all(t['comparison_to_current_native_eager']['bitwise_equal'] for z in rows for t in z['captured']),'all_histograms_preserved':all(z['routing_histogram_after_consumers_correct'] for z in rows),'peak_bytes':torch.npu.max_memory_allocated(),'scope':'Single-NPU synthetic exact dimensions and native NZ W8A8 unchanged routing count1->fusedGMM1+act+quant->unchanged GMM2 count1->unpermute; only prefix conversion uses exact bounded int32 intermediate, native disposal/event ordering, no intermediate host synchronization. Independent raw ACL graph families, not full-model FX graph/attention/KV/TP validation and not performance evidence.'})+'\nFULL_PREFIX_RESULT_END')
'''
(out/'fragment.py').write_text(code)
argv=['docker','run','--rm','--pull','never','--network','host','--name','glm53-cumsum-i32-full-fragment-v1','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','8','--memory','24g','--pids-limit','4096','--shm-size','2g','--tmpfs','/tmp:rw,size=4g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','ASCEND_RT_VISIBLE_DEVICES=0','--device','/dev/davinci0','--device','/dev/davinci_manager','--device','/dev/devmm_svm','--device','/dev/hisi_hdc','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v',str(out)+':/probe:ro','--entrypoint','/usr/bin/timeout',plan['image']['id'],'--signal=TERM','--kill-after=10','360','python3','/probe/fragment.py']
q=subprocess.run(argv,capture_output=True,text=True,timeout=390);(out/'stdout.txt').write_text(q.stdout);(out/'stderr.txt').write_text(q.stderr);assert q.returncode==0,q.stdout[-2500:]+q.stderr[-2500:]
x=json.loads(q.stdout.split('FULL_PREFIX_RESULT_BEGIN\n',1)[1].split('\nFULL_PREFIX_RESULT_END',1)[0]);x.update(argv=argv,physical_devices=[0],node=33,models_loaded=False)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(x)+'\nGLM53_RESULT_JSON_END')
