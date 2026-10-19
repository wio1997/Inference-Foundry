"""Model-free one-NPU fragment of the exact observed RoPE cache layout/gather."""
from pathlib import Path
import hashlib,json,subprocess,time
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/p-bucket-control-v1-20261019/node32');plan=json.loads((root/'plan.json').read_text());cid='ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];assert plan['reservation']['expected_uid']=='86fe97c7-4b7c-43f1-b800-993955faae8b'
import urllib.request
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 rows=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(rows)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in rows)
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
code=r'''import json,statistics,time
import torch,torch_npu
torch.npu.set_device(0);torch.manual_seed(1024)
# Match the interleaved native cache's underlying storage and two non-contiguous views.
base=torch.randn((1048576,128),dtype=torch.bfloat16,device='npu')
cos,sin=base[:,:64],base[:,64:]
assert cos.shape==sin.shape==(1048576,64) and cos.stride()==sin.stride()==(128,1)
start=time.perf_counter();cc,ss=cos.contiguous(),sin.contiguous();torch.npu.synchronize();setup_s=time.perf_counter()-start
assert torch.equal(cos,cc) and torch.equal(sin,ss)
rows=[]
for n in [1,4,6,18,24]:
 positions=torch.tensor(([0,1048575,8191,257,257,1,65537]*4)[:n],dtype=torch.int64,device='npu')
 def one(mode):
  c,s=(cos,sin) if mode=='native' else (cc,ss)
  return c[positions].unsqueeze(1).unsqueeze(2),s[positions].unsqueeze(1).unsqueeze(2)
 a,b=one('native'),one('cached');assert all(torch.equal(x,y) for x,y in zip(a,b))
 for mode in ['native','cached']:
  for i in range(5):one(mode)
 torch.npu.synchronize()
 for cycle in range(3):
  for mode in ['native','cached','cached','native']:
   begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);torch.npu.synchronize();t=time.perf_counter();begin.record()
   for i in range(30):one(mode)
   end.record();end.synchronize();rows.append({'N':n,'cycle':cycle,'mode':mode,'repeats':30,'device_ms_per_pair':begin.elapsed_time(end)/30,'wall_ms_per_pair':(time.perf_counter()-t)*1000/30})
summary=[]
for n in [1,4,6,18,24]:
 rates={m:statistics.median(z['device_ms_per_pair'] for z in rows if z['N']==n and z['mode']==m) for m in ['native','cached']}
 summary.append({'N':n,'median_device_ms':rates,'cached_over_native':rates['cached']/rates['native']})
peak=torch.npu.max_memory_allocated();assert peak<2*1024**3
print(json.dumps({'completed':True,'full_cache_bitwise_equal':True,'selected_outputs_bitwise_equal_all_shapes':True,'dtype':'bfloat16','native_shape':[1048576,64],'native_stride':[128,1],'cached_stride':[64,1],'owned_cached_pair_bytes':cc.numel()*cc.element_size()+ss.numel()*ss.element_size(),'setup_s':setup_s,'peak_allocated_bytes':peak,'rows':rows,'summary':summary,'scope':'Model-free synthetic buffer with exact observed layout/shape/dtype and original advanced-index gather. Reordering the same read-only values, no arithmetic/precision/model change. Fragment gain is not an end-to-end prediction.'}))
'''
(out/'fragment.py').write_text(code)
argv=['docker','run','--rm','--pull','never','--network','none','--name','glm53-rope-cache-fragment-v1','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','4','--memory','8g','--pids-limit','512','--shm-size','512m','--tmpfs','/tmp:rw,size=1g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','ASCEND_RT_VISIBLE_DEVICES=0','--device','/dev/davinci0','--device','/dev/davinci_manager','--device','/dev/devmm_svm','--device','/dev/hisi_hdc','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v',str(out)+':/probe:ro','--entrypoint','/usr/bin/timeout',plan['image']['id'],'--signal=TERM','--kill-after=10','120','python3','/probe/fragment.py']
q=subprocess.run(argv,capture_output=True,text=True,timeout=150);(out/'stdout.txt').write_text(q.stdout);(out/'stderr.txt').write_text(q.stderr);assert q.returncode==0,q.stdout+q.stderr
lines=[z for z in q.stdout.splitlines() if z.startswith('{') and 'selected_outputs_bitwise_equal_all_shapes' in z];assert len(lines)==1;x=json.loads(lines[0]);x.update(argv=argv,physical_devices=[0],model_containers_unchanged=[cid,'084035e84399bd88c3788c80f184590f309fd6730f93cc308f460487c75af213']);print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(x)+'\nGLM53_RESULT_JSON_END')
