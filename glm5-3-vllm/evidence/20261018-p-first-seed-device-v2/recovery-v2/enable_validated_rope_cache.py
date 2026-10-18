from pathlib import Path
import datetime,hashlib,json,subprocess
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
v=json.loads((r/'rope-cache-layout-formal-verdict-v1.json').read_text());assert v['decision'].startswith('ACCEPTED') and min(z['candidate_over_native'] for z in v['C4_cycles'])>1.12 and v['quality']['full64_equal_cases']==26
for n in [32,33]:
 p=next((r/('rope-cache-layout-final-'+str(n)+'-snapshot')).glob('*/artifacts-backup.json'));x=json.loads(p.read_text());b=Path(x['local_archive']).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34'];pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default glm53-rope-cache-33-live1-20261019 -o json'],timeout=30));assert pod['metadata']['uid']=='5b2818f0-749c-4760-a178-37b498e76b9f' and pod['metadata']['labels']['owner']=='codex-glm53-rope-cache-live1-20261019' and pod['status']['phase']=='Running'
code="""from pathlib import Path
import datetime,json,subprocess,urllib.request
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-cache-layout-live1-20261019/node33');cid='246f4f1090349ae9b8018af114460537df89c7073547190f5d0949326edc2046';p=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['Id']==cid and d['Name']=='/'+p['container']['name'] and d['Image']==p['image']['id'] and d['State']['Running'];env=dict(z.split('=',1) for z in d['Config']['Env'] if '=' in z);assert env['GLM53_ROPE_CACHE_LAYOUT_PROBE']=='1' and env['GLM53_MTP_MOE_GRAPH_PROBE']=='0'
for k,v in {'HCCL_INTER_HCCS_DISABLE':'FALSE','HCCL_INTRA_ROCE_ENABLE':'0','HCCL_INTRA_PCIE_ENABLE':'1','HCCL_OP_EXPANSION_MODE':'AI_CPU'}.items():assert env[k]==v
idle={}
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 z=[x for x in s.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(z)>=3 and all(float(x.rsplit(' ',1)[1])==0 for x in z);idle[str(n)]=z
probe=root/'rope-cache-layout-probe';assert (probe/'disabled').exists() and not (probe/'enable').exists();left=p['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>600
(probe/'enable').touch(exist_ok=False)
with urllib.request.urlopen('http://172.16.12.33:29482/v1/models',timeout=5) as f:models=json.load(f)
assert any(z['id']=='glm5' for z in models['data'])
print(json.dumps({'enabled':True,'CID':cid,'helper':'rope_cache_layout','marker':str(probe/'enable'),'observer_disabled':True,'idle_before':idle,'remaining_s':left,'host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Enable the already validated candidate for ordinary multi-token requests in this exact owned finite research instance; no model/code/transport change, no extra benchmark or backend traffic. Source remains explicit opt-in.','rollback':'Remove this exact marker; native source restored by the wrapper after every lookup already.'}))
"""
q=subprocess.run(ssh+['kubectl exec -i -n default sglang-tuning-access-33-20261013 -- nsenter -t 1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=60);assert q.returncode==0,q.stderr;x=json.loads(q.stdout);x['lease_UID']=pod['metadata']['uid'];x['client_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(r/'rope-cache-layout-live-enabled-v1.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
