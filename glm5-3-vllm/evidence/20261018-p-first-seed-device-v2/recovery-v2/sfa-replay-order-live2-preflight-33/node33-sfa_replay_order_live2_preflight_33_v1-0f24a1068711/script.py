from pathlib import Path
import argparse,datetime,json,socket,subprocess,urllib.request
p=argparse.ArgumentParser();p.add_argument('--disable-D-trace',action='store_true');a=p.parse_args();host=socket.gethostname();node=32 if host=='B4-33U-Atlas900RCKA3-32' else 33;assert host in ['B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33'];root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live2-20261018/node'+str(node));plan=json.loads((root/'plan.json').read_text());cid={32:'d9762493f4e60c450b06d964bf6c7b9d5fbcfd5261e44086a86f2031ff7ec33f',33:'05e98c8425d8bb0db0c30425ae054f9e8ee7b747914a18defaa2f18d8aee8f2a'}[node];d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];env=dict(s.split('=',1) for s in d['Config']['Env'] if '=' in s);required={'HCCL_INTER_HCCS_DISABLE':'FALSE','HCCL_INTRA_ROCE_ENABLE':'0','HCCL_INTRA_PCIE_ENABLE':'1','HCCL_OP_EXPANSION_MODE':'AI_CPU','GLM53_P_FIRST_ADOPT_TRACE':'0','GLM53_P_FIRST_ADOPT':'0'};assert all(env[k]==v for k,v in required.items())
assert env.get('VLLM_USE_BREAKABLE_CUDAGRAPH','0')=='0' and env.get('ENPU_ENABLE','false').lower()!='true'
if node==33:
 assert env['GLM53_SFA_REPLAY_ORDER_PROBE']=='1' and not (root/'sfa-replay-order-probe/enable').exists()
 if a.disable_D_trace:(root/'sfa-replay-order-probe/disabled').touch()
 assert (root/'sfa-replay-order-probe/disabled').exists()
else:assert not a.disable_D_trace and env['GLM53_SFA_REPLAY_ORDER_PROBE']=='0' and env['GLM53_PREFILL_ATTENTION_PROBE']=='0'
now=datetime.datetime.now(datetime.timezone.utc);left=plan['container']['lifetime_seconds']-(now-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>600
busy=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if any(z in s for z in ['msprof','msprof-export','export_profile','backup_artifacts.py','rsync ']) and 'ps -eo' not in s];assert not busy,busy
metrics={}
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 metrics[str(n)]=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(metrics[str(n)])>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in metrics[str(n)])
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'node':node,'CID':cid,'remaining_s':left,'env_verified':required,'profile_export_transfers':busy,'idle_metrics':metrics,'utc':now.isoformat()})+'\nGLM53_RESULT_JSON_END')
