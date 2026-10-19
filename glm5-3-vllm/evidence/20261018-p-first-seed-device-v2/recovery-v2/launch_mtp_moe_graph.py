"""Refresh only backed-up owned finite P/D windows, preserving pinned config."""
from pathlib import Path
import datetime
import importlib.util
import json
import subprocess
import sys

repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
control='/tmp/glm53-active-20261018.sock'
ssh=['ssh','-S',control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
cids={32:'d9762493f4e60c450b06d964bf6c7b9d5fbcfd5261e44086a86f2031ff7ec33f',33:'7b65366a6d14d7840e2686118bb29e3621d470424e30f2648f5eb237a395906d'}
roots={32:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live2-20261018/node32',33:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live2-20261018/node33'}
assert json.loads((r/'mtp-moe-graph-cpu-check-v1.json').read_text())['passed']
manifest=json.loads((r/'mtp-moe-graph-build-manifest-v1.json').read_text())
spec=importlib.util.spec_from_file_location('lease',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plans={};receipts={};limits=[]
for role,node in [('P',32),('D',33)]:
    assert json.loads(next((r/('mtp-moe-graph-stage-'+str(node))).glob('*/metrics.json')).read_text())['staged']
    assert next((r/('before-mtp-moe-'+str(node)+'-snapshot')).glob('*/artifacts-backup.json')).exists()
    p=json.loads((r/('sfa-replay-order-live2-'+role+'-plan.json')).read_text());res=p['reservation']
    lease=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
    uid,allowed,expiry=m.fresh_lease(lease,res,datetime.datetime.now(datetime.timezone.utc));limits.append(allowed)
    items=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items']
    assert all(not sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in x['spec'].get('containers',[])) or x['metadata']['uid']==res['expected_uid'] or x['status']['phase'] in ['Failed','Succeeded'] for x in items)
    p['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live1-20261018/node'+str(node)
    p['container']['name']='glm53-mtp-moe-graph-'+str(node)+'-live1-20261018';p['runtime_patches']=manifest['patches']
    for key in list(p['env']):
        if key.startswith('GLM53_') and key!='GLM53_STATE_PROBE_OUTPUT':p['env'][key]='0'
    p['env'].update(GLM53_P_FIRST_DRAFT_EXPORT='1' if role=='P' else '0',GLM53_MTP_MOE_GRAPH_PROBE='0' if role=='P' else '1')
    i=p['serve_argv'].index('--kv-transfer-config')+1;kv=json.loads(p['serve_argv'][i]);kv['engine_id']='glm53-mtp-moe-graph-'+role.lower()+'-live1-20261018';p['serve_argv'][i]=json.dumps(kv,separators=(',',':'))
    p.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,objective='Ordinary PD native opaque MTP MoE core graph; target/attention/KV/outercombine unchanged',notes=['Refresh backedupownP/D due insufficientoldP remaining for anotherload+completeverification; sameowned7200s holders.','FixednativePIECEWISE P[256,1024]/sharedFalse and D FULL/K5/sharedTrue; weights/precision/operators/allSDMA unchanged.','Defaultoff; onlyloadedMTPopaqueMoEcore instance; actualall16 path before anyperformance; no graphbucket/rate/chunk sweep.'])
    plans[role]=p
life=min(3600,*limits);assert life>=1650,limits
for role,node in [('P',32),('D',33)]:
    p=plans[role];p['container']['lifetime_seconds']=life
    (r/('mtp-moe-graph-'+role+'-plan.json')).write_text(json.dumps(p,indent=2)+'\n')
    code="""from pathlib import Path
import json,subprocess,urllib.request
root=Path(OWN_ROOT_TOKEN);cid=OWN_CID_TOKEN;p=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==cid and d['Name']=='/'+p['container']['name'] and d['Image']==p['image']['id']
assert not (root/'sfa-replay-order-probe/enable').exists() and not (root/'mtp-ffn-graph-probe/enable').exists()
with urllib.request.urlopen('http://172.16.12.NODE:PORT/metrics',timeout=5) as f:s=f.read().decode()
idle=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(idle)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in idle)
stopped=[]
if NODE==32:
 proxy='8c8eac8d5d10bfdc993223fd0f2e4cad98091352f000656ffc8532e4e1362860';pl=json.loads((root/'official-proxy-v1/launch.json').read_text());pd=json.loads(subprocess.check_output(['docker','inspect',proxy]))[0];assert pl['CID']==proxy and pd['Image']==d['Image'] and not pd['HostConfig'].get('Devices') and pd['Name']=='/'+pl['argv'][pl['argv'].index('--name')+1]
 with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:assert json.load(f)['request_num']==0
 q=subprocess.run(['docker','stop','-t','30',proxy],capture_output=True,text=True,timeout=45);assert q.returncode==0;stopped.append(proxy)
q=subprocess.run(['docker','stop','-t','30',cid],capture_output=True,text=True,timeout=45);after=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert q.returncode==0 and not after['State']['Running'];stopped.append(cid)
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=10);assert smi.count('No running processes found in NPU')==8
print(json.dumps({'stopped_exact_only':stopped,'before':d['State'],'after':after['State'],'idle':idle,'NPU_idle':smi,'holder_unchanged':True}))
""".replace('OWN_ROOT_TOKEN',repr(roots[node])).replace('OWN_CID_TOKEN',repr(cids[node])).replace('NODE',str(node)).replace('PORT',str(29481 if node==32 else 29482))
    compile(code,'bounded_stop_'+role,'exec')
    q=subprocess.run(ssh+['kubectl exec -i -n default '+pods[node]+' -- nsenter -t 1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=120)
    assert q.returncode==0,q.stderr
    receipts[role]=json.loads(q.stdout)
    print(json.dumps({'phase':'old_owned_window_stopped','role':role,'only':receipts[role]['stopped_exact_only']}),flush=True)
(r/'mtp-moe-graph-old-windows-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
for role in ['P','D']:
    p=plans[role];res=p['reservation']
    lease=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
    uid,allowed,expiry=m.fresh_lease(lease,res,datetime.datetime.now(datetime.timezone.utc))
    p['container']['lifetime_seconds']=min(life,allowed)
    assert p['container']['lifetime_seconds']>=1650
    path=r/('mtp-moe-graph-'+role+'-plan.json')
    path.write_text(json.dumps(p,indent=2)+'\n')
    q=subprocess.run([sys.executable,'glm5-3-vllm/tools/start_capacity.py','--plan',str(path),'--control',control,'--output',str(r/('mtp-moe-graph-'+role+'-launch'))],cwd=repo,capture_output=True,text=True,timeout=240)
    (r/('mtp-moe-graph-'+role+'-launch-collector.txt')).write_text(q.stdout+q.stderr)
    print(json.dumps({'role':role,'life':life,'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}),flush=True)
    assert q.returncode==0
