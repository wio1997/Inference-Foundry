from pathlib import Path
import datetime,json,socket,subprocess,urllib.request
node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()];root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/'+('moe-cumsum-i32-live2-20261019' if node==32 else 'moe-cumsum-i32-live1-20261019')+'/node'+str(node));cid={32:'076b19aeee16e31ac9ec10f0f3a73adf1e04c41cdf0abf1248b774affdba8742',33:'416f0dc10e34a115e2537cb62e46e2eb98d30825642ab4ba36d098c072cf11f5'}[node];plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];env=dict(z.split('=',1) for z in d['Config']['Env'] if '=' in z)
for k,v in {'HCCL_INTER_HCCS_DISABLE':'FALSE','HCCL_INTRA_ROCE_ENABLE':'0','HCCL_INTRA_PCIE_ENABLE':'1','HCCL_OP_EXPANSION_MODE':'AI_CPU','GLM53_P_FIRST_ADOPT':'0'}.items():assert env[k]==v
if node==33:
 assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists();p=root/'moe-cumsum-i32-probe';assert (p/'target-only').exists() and (p/'disabled').exists() and not any((p/k).exists() for k in ['enable','replica-enable','MTP-only']);profile=json.loads((root/'cumsum-i32-supply-profile-v1/result.json').read_text());assert profile['profile_stopped'] and profile['ranks']==list(range(16))
with urllib.request.urlopen('http://172.16.12.%d:%d/v1/models'%(node,29481 if node==32 else 29482),timeout=5) as f:models=json.load(f)
assert any(z['id']=='glm5' for z in models['data'])
with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,29481 if node==32 else 29482),timeout=5) as f:data=f.read().decode()
metrics=[z for z in data.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(metrics)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in metrics)
now=datetime.datetime.now(datetime.timezone.utc);remaining=plan['container']['lifetime_seconds']-(now-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert remaining>900
out={'node':node,'CID':cid,'model_ready':True,'remaining_s':remaining,'utc':now.isoformat(),'idle_metrics':metrics,'scope':'Read-only current exact owned service health; no inference/timing work. Rejected Cumsum selector off, accepted RoPE active on D, original SDMA configuration retained.'}
if node==32:
 pl=json.loads((root/'official-proxy-v1/launch.json').read_text());pd=json.loads(subprocess.check_output(['docker','inspect',pl['CID']]))[0];assert pl['parent_P_CID']==cid and pd['State']['Running'] and not pd['HostConfig'].get('Devices') and pl['source_sha256']=='85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469'
 with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:out['proxy_health']=json.load(f)
 out['proxy_CID']=pl['CID']
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(out)+'\nGLM53_RESULT_JSON_END')
