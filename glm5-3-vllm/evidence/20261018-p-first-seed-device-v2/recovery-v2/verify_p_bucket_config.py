from pathlib import Path
import datetime,hashlib,json,re,socket,subprocess,urllib.request
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/p-bucket-control-v1-20261019/node32');cid='ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];p=json.loads((root/'plan.json').read_text());assert d['State']['Running'] and d['Id']==cid and d['Name']=='/'+p['container']['name'] and d['Image']==p['image']['id'];env=dict(s.split('=',1) for s in d['Config']['Env'] if '=' in s)
actual=d['Config']['Cmd'];i=actual.index('--compilation-config')+1;cfg=json.loads(actual[i]);assert cfg=={'cudagraph_mode':'PIECEWISE','cudagraph_capture_sizes':[256,512,1024]}
patch=next(z for z in p['runtime_patches'] if z['target'].endswith('/worker/model_runner_v1.py'));source=Path(patch['source']).read_bytes();assert hashlib.sha256(source).hexdigest()==patch['sha256'];assert b'GLM53_PREFILL_PADDING_PROBE' not in source
assert env['GLM53_MTP_MOE_GRAPH_PROBE']=='0' and env['GLM53_P_FIRST_ADOPT']=='0'
for k,v in {'HCCL_INTER_HCCS_DISABLE':'FALSE','HCCL_INTRA_ROCE_ENABLE':'0','HCCL_INTRA_PCIE_ENABLE':'1','HCCL_OP_EXPANSION_MODE':'AI_CPU'}.items():assert env[k]==v
server={}
try:
 with urllib.request.urlopen('http://172.16.12.32:29481/server_info?config_format=json',timeout=5) as f:info=json.load(f)
 def walk(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in ['cudagraph_capture_sizes','cudagraph_mode']:server.setdefault(k,[]).append(v)
    walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(info)
 if not server:
  text=json.dumps(info);server['capture_sizes_matches']=re.findall(r'cudagraph_capture_sizes[^\[]*(\[[^\]]+\])',text)
except Exception as e:server={'api_error':str(e)}
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 rows=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(rows)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in rows)
busy=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if any(z in s for z in ['msprof','msprof-export','export_profile','backup_artifacts.py','rsync ']) and 'ps -eo' not in s];assert not busy,busy
left=p['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>600
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'P_CID':cid,'actual_container_config':cfg,'server_info_graph_config':server,'runtime_source_sha256':patch['sha256'],'padding_observer_not_installed_in_this_build':True,'padding_env_flag_unused':env['GLM53_PREFILL_PADDING_PROBE'],'P_D_idle':True,'remaining_s':left,'scope':'Actual pinned container argv/source and native server configuration. Native dispatcher bucket512 inference is source-supported, not an all16 per-request metadata observation. No new data API or custom path selected; finite configuration control only.'})+'\nGLM53_RESULT_JSON_END')
