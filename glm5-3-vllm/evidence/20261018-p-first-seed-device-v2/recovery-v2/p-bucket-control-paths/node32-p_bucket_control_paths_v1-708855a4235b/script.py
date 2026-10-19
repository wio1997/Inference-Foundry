from pathlib import Path
import argparse,datetime,json,socket,subprocess,urllib.request
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
p=argparse.ArgumentParser();p.add_argument('--disable',action='store_true');a=p.parse_args()
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/p-bucket-control-v1-20261019/node32');cid='ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];plan=json.loads((root/'plan.json').read_text());assert d['State']['Running'] and d['Id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];env=dict(z.split('=',1) for z in d['Config']['Env'] if '=' in z);assert env['GLM53_PREFILL_PADDING_PROBE']=='1' and env['GLM53_MTP_MOE_GRAPH_PROBE']=='0';cfg=json.loads(plan['serve_argv'][plan['serve_argv'].index('--compilation-config')+1]);assert cfg=={'cudagraph_mode':'PIECEWISE','cudagraph_capture_sizes':[256,512,1024]}
for k,v in {'HCCL_INTER_HCCS_DISABLE':'FALSE','HCCL_INTRA_ROCE_ENABLE':'0','HCCL_INTRA_PCIE_ENABLE':'1','HCCL_OP_EXPANSION_MODE':'AI_CPU'}.items():assert env[k]==v
probe=root/'prefill-padding-probe';assert not (probe/'enable').exists()
rows=[]
for path in sorted(probe.glob('rank-*.jsonl')):
 for line in path.read_text().splitlines():
  z=json.loads(line);assert not z['selected'] and not z['requested'];z['source']=path.name;rows.append(z)
assert sorted(set(z['tp_rank'] for z in rows))==list(range(16))
if a.disable:
 for n,port in [(32,29481),(33,29482)]:
  with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
  idle=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(idle)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in idle)
 busy=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if any(z in s for z in ['msprof','msprof-export','export_profile','backup_artifacts.py','rsync ']) and 'ps -eo' not in s];assert not busy,busy
 (probe/'disabled').touch()
left=plan['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>600
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'P_CID':cid,'config':cfg,'ranks':list(range(16)),'rows':rows,'observer_disabled':(probe/'disabled').exists(),'native_marker_absent':not (probe/'enable').exists(),'remaining_s':left,'scope':'Native P dispatcher metadata, unchanged original determine method; no custom P path selected. SDMA transport configuration unchanged, no new data API.'})+'\nGLM53_RESULT_JSON_END')
