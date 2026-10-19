from pathlib import Path
import json,socket,subprocess,urllib.request,time
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live1-20261019/node33');cid='3caaa2a7856b78e27a692a909505d9d4a34969b68217c64869e5f4397bc25b5b'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id'] and d['Name']=='/'+plan['container']['name']
assert not(root/'indexer-k-graph-overlap-probe/enable').exists()
for node,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,port),timeout=5) as f:text=f.read().decode()
 rows=[z for z in text.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(rows)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in rows)
subprocess.run(['docker','stop','--time','30',cid],check=True,capture_output=True,timeout=45)
for _ in range(20):
 if subprocess.check_output(['npu-smi','info'],text=True,timeout=20).count('No running processes found in NPU')==8:break
 time.sleep(2)
else:raise RuntimeError('Not empty after exact D stop')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'stopped_exact_D_CID':cid,'empty_physical_cards':8,'P_and_proxy_retained':True})+'\nGLM53_RESULT_JSON_END')
