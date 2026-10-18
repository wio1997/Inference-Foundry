import argparse,datetime,json,socket,subprocess,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--disable-P-traces',action='store_true');a=p.parse_args()
host=socket.gethostname();node=32 if host=='B4-33U-Atlas900RCKA3-32' else 33
assert host in ['B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33']
roots={32:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-padding-live1-20261018/node32',33:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33'}
cids={32:'28e687e363317e918ed636a4615f918dda3ea80839626e357e82d77ea87faf56',33:'6584fed1fd2712610c6840a6cc631914e6e4549958055a5b2175f22455c7426a'}
root=Path(roots[node]);plan=json.loads((root/'plan.json').read_text());cid=cids[node]
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Image']==plan['image']['id'] and d['Name']=='/'+plan['container']['name']
env=dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
required={'HCCL_INTER_HCCS_DISABLE':'FALSE','HCCL_INTRA_ROCE_ENABLE':'0','HCCL_INTRA_PCIE_ENABLE':'1','HCCL_OP_EXPANSION_MODE':'AI_CPU'}
assert all(env[k]==v for k,v in required.items())
if node==32:
 paths=[root/'prefill-padding-probe/disabled',root/'adopt-trace/disabled']
 assert not (root/'prefill-padding-probe/enable').exists()
 if a.disable_P_traces:
  for p in paths:p.touch()
else:
 assert not a.disable_P_traces
 paths=[root/'divergence-probe'/name for name in ['disabled','overlap-probe-disabled','unused-draft-trace-disabled']]+[root/'adopt-trace/disabled']
 assert not (root/'divergence-probe/skip-unused-single-token-draft').exists()
 assert not (root/'divergence-probe/shared-overlap-disabled').exists()
assert all(p.exists() for p in paths)
now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00');left=plan['container']['lifetime_seconds']-(now-start).total_seconds();assert left>1200
procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();busy=[s for s in procs if any(t in s for t in ['msprof','msprof-export','export_profile','backup_artifacts.py','rsync ']) and 'ps -eo' not in s];assert not busy,busy
metrics={}
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:metrics[str(n)]=[s for s in f.read().decode().splitlines() if s.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
 assert len(metrics[str(n)])>=3 and all(float(s.rsplit(' ',1)[1])==0 for s in metrics[str(n)])
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'node':node,'CID':cid,'image':d['Image'],'remaining_s':left,'diag_disabled':[str(p) for p in paths],'env_verified':required,'profile_export_transfers':busy,'idle_metrics':metrics,'utc':now.isoformat()})+'\nGLM53_RESULT_JSON_END')
