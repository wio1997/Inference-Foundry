from pathlib import Path
import json,subprocess,urllib.request
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-histogram-live3-20261019/node33');cid='c7be13e9c4f558e7da2aa395b764cccbd5026bf1dc4f5f3cba52abf4c7dc76db';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and json.loads((root/'launch.json').read_text())['container_id']==cid
triage=json.loads((root/'prefix-mode-triage-v1/result.json').read_text());assert triage['completed'] and all(z['equal'] for z in triage['full64_equal_to_native_before'])
probe=root/'moe-prefix-histogram-probe';assert all(not(probe/z).exists() for z in ['enable','replica-enable','MTP-only','target-only','diagnose-MTP','diagnose-serial'])
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 v=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(v)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in v)
assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists();(probe/'disabled').unlink()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'CID':cid,'prefix_observer_enabled':True,'RoPE_remains_enabled_observer_off':True,'previous_functional_attempt':'assertion before request/outputdir due triage cleanup marker, no inference requests repeated'})+'\nGLM53_RESULT_JSON_END')
