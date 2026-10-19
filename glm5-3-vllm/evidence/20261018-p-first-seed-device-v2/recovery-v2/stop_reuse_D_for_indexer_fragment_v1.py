from pathlib import Path
import json,socket,subprocess,time,urllib.request
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33')
cid='163c7646c30ab819514f112e0af510934aa592305f3ec37bd7e3b77a3edc2bdc'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
assert not(root/'rope-step-reuse-probe/enable').exists() and (root/'rope-step-reuse-probe/disabled').exists()
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 lines=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(lines)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in lines)
subprocess.run(['docker','stop','--time','30',cid],check=True,capture_output=True,timeout=45)
for i in range(20):
 s=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
 if s.count('No running processes found in NPU')==8:break
 time.sleep(2)
else:raise RuntimeError('Own D stopped but device processes not empty; no fragment allowed')
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert not d['State']['Running']
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'stopped_exact_D_CID':cid,'P_untouched':True,'empty_physical_cards':8,'scope':'Latest complete source/log/quality snapshot verified locally first; current queues/KV zero before exact owned D stop. No other workload, access pod or holder stopped.'})+'\nGLM53_RESULT_JSON_END')
