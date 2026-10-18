from pathlib import Path
import argparse,datetime,json,socket,subprocess,urllib.request
p=argparse.ArgumentParser();p.add_argument('--disable-D-trace',action='store_true');a=p.parse_args();host=socket.gethostname();node=32 if host=='B4-33U-Atlas900RCKA3-32' else 33;assert host in ['B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33'];root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/'+'moe-prefix-target-live1-20261019'+'/node'+str(node));plan=json.loads((root/'plan.json').read_text());cid={32:'13c520313d3369bafc2db811dca35bc93afdfc2ae7ed3572fc1518366c38aceb',33:'4d620dfbdf2981ee4c95e029fd1cf3ee297bdb153a548bbb39027ac0e9252041'}[node];d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];env=dict(s.split('=',1) for s in d['Config']['Env'] if '=' in s);required={'HCCL_INTER_HCCS_DISABLE':'FALSE','HCCL_INTRA_ROCE_ENABLE':'0','HCCL_INTRA_PCIE_ENABLE':'1','HCCL_OP_EXPANSION_MODE':'AI_CPU','GLM53_P_FIRST_ADOPT_TRACE':'0','GLM53_P_FIRST_ADOPT':'0'};assert all(env[k]==v for k,v in required.items())
assert env.get('VLLM_USE_BREAKABLE_CUDAGRAPH','0')=='0' and env.get('ENPU_ENABLE','false').lower()!='true'
if node==33:
 assert env['GLM53_ROPE_CACHE_LAYOUT_PROBE']=='1' and env['GLM53_MTP_MOE_GRAPH_PROBE']=='0' and env['GLM53_SFA_REPLAY_ORDER_PROBE']=='0' and (root/'rope-cache-layout-probe/enable').exists() and not (root/'moe-prefix-histogram-probe/enable').exists() and env['GLM53_MOE_PREFIX_HISTOGRAM_PROBE']=='1'
 assert (root/'moe-prefix-histogram-probe/target-only').exists() and not(root/'moe-prefix-histogram-probe/MTP-only').exists() and not(root/'moe-prefix-histogram-probe/replica-enable').exists()
 if a.disable_D_trace:(root/'moe-prefix-histogram-probe/disabled').touch()
 assert (root/'rope-cache-layout-probe/disabled').exists() and (root/'moe-prefix-histogram-probe/disabled').exists()
else:assert env['GLM53_ROPE_CACHE_LAYOUT_PROBE']=='0' and not a.disable_D_trace and env['GLM53_SFA_REPLAY_ORDER_PROBE']=='0' and env['GLM53_PREFILL_ATTENTION_PROBE']=='0'
now=datetime.datetime.now(datetime.timezone.utc);left=plan['container']['lifetime_seconds']-(now-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>600
busy=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if any(z in s for z in ['msprof','msprof-export','export_profile','backup_artifacts.py','rsync ']) and 'ps -eo' not in s];assert not busy,busy
metrics={}
spec_metrics={}
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 spec_metrics[str(n)]=[z for z in s.splitlines() if not z.startswith('#') and any(k in z.lower() for k in ['spec_decode','draft','accepted','generation_tokens_total','prompt_tokens_total','preemptions'])]
 metrics[str(n)]=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(metrics[str(n)])>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in metrics[str(n)])
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'node':node,'CID':cid,'remaining_s':left,'env_verified':required,'profile_export_transfers':busy,'idle_metrics':metrics,'native_speculation_counters':spec_metrics,'utc':now.isoformat()})+'\nGLM53_RESULT_JSON_END')
