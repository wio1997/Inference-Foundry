"""One unoccupied NPU: existing routing count+cumsum versus direct prefix output."""
from pathlib import Path
import json,socket,subprocess,time
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-cache-layout-live1-20261019/node32')
plan=json.loads((root/'plan.json').read_text())
cid='0c0487ac798214c855ba2bc9886b86c802fdca215fe362dd2384a925b063740b'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert not d['State']['Running'] and d['Id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
assert plan['reservation']['expected_uid']=='d544002b-38bb-4cf5-abd2-46f68662580f'
assert subprocess.check_output(['npu-smi','info'],text=True,timeout=30).count('No running processes found in NPU')==8
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
code=r'''import json,statistics,time
import torch,torch_npu,torchair
from torchair.configs.compiler_config import CompilerConfig
torch.npu.set_device(0);torch.manual_seed(9812)
rows=[];checks=[];compiled=[]
def route(x,ids,t):
 a,b,g,s=torch_npu.npu_moe_init_routing_v2(x,ids,active_num=x.shape[0]*8,expert_num=256,expert_tokens_num_type=t,expert_tokens_num_flag=True,active_expert_range=[0,256],quant_mode=1)
 g=g.to(torch.int64)
 if t==1:g=g.cumsum(dim=0)
 return a,b,g,s
for n in [6,24]:
 x=torch.randn((n,6144),dtype=torch.bfloat16,device='npu')
 ids=torch.stack([torch.randperm(256)[:8] for _ in range(n)]).to(dtype=torch.int32,device='npu')
 native=route(x,ids,1);prefix=route(x,ids,0)
 assert all(torch.equal(a,b) for a,b in zip(native,prefix))
 assert native[2].shape==(256,) and int(native[2][-1].cpu())==n*8
 checks.append({'N':n,'all_four_outputs_bitwise_equal':True,'sum_routes':n*8})
 functions={}
 for mode,t in [('native',1),('prefix',0)]:
  def fn(x,ids,kind=t):return route(x,ids,kind)
  start=time.perf_counter();conf=CompilerConfig();f=torch.compile(fn,backend=torchair.get_npu_backend(compiler_config=conf),fullgraph=True,dynamic=False)
  result=f(x,ids);torch.npu.synchronize()
  assert all(torch.equal(a,b) for a,b in zip(native,result))
  for _ in range(3):f(x,ids)
  torch.npu.synchronize();graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph):captured=f(x,ids)
  def replay(x,ids,_graph=graph,_out=captured):
   _graph.replay();return _out
  functions[mode]=replay;compiled.append({'N':n,'mode':mode,'compile_first_call_s':time.perf_counter()-start,'outputs_bitwise_equal':True,'same_compiled_call_captured_for_replay':True})
  for _ in range(5):functions[mode](x,ids)
 torch.npu.synchronize()
 for cycle in range(3):
  for mode in ['native','prefix','prefix','native']:
   f=functions[mode];begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
   torch.npu.synchronize();start=time.perf_counter();begin.record()
   for _ in range(100):f(x,ids)
   end.record();end.synchronize()
   rows.append({'N':n,'cycle':cycle,'mode':mode,'repeats':100,'device_ms_per_call':begin.elapsed_time(end)/100,'wall_ms_per_call':(time.perf_counter()-start)*10})
 # Existing int8 grouped-matmul consumes either histogram representation.
 inp=torch.randint(-3,4,(n*8,128),dtype=torch.int8,device='npu')
 w=torch.randint(-3,4,(256,128,64),dtype=torch.int8,device='npu')
 scales=torch.ones((256,64),dtype=torch.bfloat16,device='npu')*.01
 acts=torch.ones((n*8,),dtype=torch.float32,device='npu')*.01
 count=torch.diff(torch.cat([native[2].new_zeros(1),native[2]]))
 yy=[]
 for group,typ in [(count,1),(prefix[2],0)]:
  yy.append(torch_npu.npu_grouped_matmul(x=[inp],weight=[w],scale=[scales],per_token_scale=[acts],split_item=2,group_type=0,group_list=group,group_list_type=typ,output_dtype=torch.bfloat16)[0])
 assert torch.equal(*yy)
 checks[-1]['existing_int8_GMM_count_vs_prefix_bitwise_equal']=True
summary=[]
for n in [6,24]:
 med={m:statistics.median(z['device_ms_per_call'] for z in rows if z['N']==n and z['mode']==m) for m in ['native','prefix']}
 summary.append({'N':n,'median_device_ms':med,'prefix_over_native':med['prefix']/med['native']})
print('PREFIX_RESULT_BEGIN\n'+json.dumps({'completed':True,'checks':checks,'compiled':compiled,'rows':rows,'summary':summary,'peak_allocated_bytes':torch.npu.max_memory_allocated(),'scope':'Synthetic BF16 [N,6144], int32 [N,8], 256 experts, dynamic quant routing; same existing operator attributes plus removal of redundant cumulative sum. Exact all routing outputs and existing small int8 GMM semantic checks. TorchAir compiled then ACL-graph replayed fragment to remove repeated Python model-launch supply gaps; no full-model correctness or service gain claim.'})+'\nPREFIX_RESULT_END')
'''
(out/'fragment.py').write_text(code)
argv=['docker','run','--rm','--pull','never','--network','host','--name','glm53-moe-prefix-fragment-v2','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','8','--memory','24g','--pids-limit','4096','--shm-size','2g','--tmpfs','/tmp:rw,size=4g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','ASCEND_RT_VISIBLE_DEVICES=0','--device','/dev/davinci0','--device','/dev/davinci_manager','--device','/dev/devmm_svm','--device','/dev/hisi_hdc','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v',str(out)+':/probe:ro','--entrypoint','/usr/bin/timeout',plan['image']['id'],'--signal=TERM','--kill-after=10','360','python3','/probe/fragment.py']
q=subprocess.run(argv,capture_output=True,text=True,timeout=390);(out/'stdout.txt').write_text(q.stdout);(out/'stderr.txt').write_text(q.stderr)
assert q.returncode==0,q.stdout[-3000:]+q.stderr[-3000:]
x=json.loads(q.stdout.split('PREFIX_RESULT_BEGIN\n',1)[1].split('\nPREFIX_RESULT_END',1)[0]);x.update(argv=argv,physical_devices=[0],models_loaded=False)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(x)+'\nGLM53_RESULT_JSON_END')
