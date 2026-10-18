"""One native P-only graph-bucket configuration control; keep loaded D native."""
from pathlib import Path
import datetime,importlib.util,json,subprocess,sys
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
def host(node,code):
 pod={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}[node]
 q=subprocess.run(ssh+['kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=120);assert q.returncode==0,q.stderr;return json.loads(q.stdout)
for n in [32,33]:
 p=next((r/('mtp-moe-live2-'+str(n)+'-snapshot')).glob('*/artifacts-backup.json'));x=json.loads(p.read_text());assert Path(x['local_archive']).stat().st_size==x['bytes']
v=json.loads((r/'mtp-moe-graph-formal-verdict-v1.json').read_text());assert v['decision'].startswith('REJECTED')
code="""from pathlib import Path
import datetime,json,subprocess,urllib.request
r=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live2-20261018/node33');cid='084035e84399bd88c3788c80f184590f309fd6730f93cc308f460487c75af213';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];p=json.loads((r/'plan.json').read_text());assert d['State']['Running'] and d['Id']==cid and d['Image']==p['image']['id'];assert not (r/'mtp-moe-graph-probe/enable').exists();left=p['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds()
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 rows=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(rows)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in rows)
print(json.dumps({'D_CID':cid,'D_remaining_s':left,'P_D_idle':True,'D_native':True}))
"""
d=host(33,code);assert d['D_remaining_s']>2100,d
p=json.loads((r/'mtp-moe-graph-live2-P-plan.json').read_text());res=p['reservation']
spec=importlib.util.spec_from_file_location('lease',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lease=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30));uid,allowed,expiry=m.fresh_lease(lease,res,datetime.datetime.now(datetime.timezone.utc));assert uid==res['expected_uid']
assigned=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items'];assert all(not sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in x['spec'].get('containers',[])) or x['metadata']['uid']==uid or x['status']['phase'] in ['Failed','Succeeded'] for x in assigned)
contract={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Necessary native configuration control before attributing P padding waste to custom execution mechanism. No graph257, chunk/rate sweep or D restart.','change':'P PIECEWISE capture [256,1024] -> [256,512,1024]; same model/W8A8C8/K1/TP16/maxbatch1024/native graph/runtime operators/SDMA. Native-only padding observer during functional then disabled before timing.','D':d,'measurement':'8 ordinary functional64/128; all16 P metadata/padded512 at N257; native-only C1 twelve formal/two warm and balanced C4 six blocks192formal/eight warm. Compare with existing same D native baseline, explicitly before/after P windows rather than within-window randomized config ABBA. Unchanged N256 control. No custom mechanism or maximum SLO claim.','restart_reason':'Native GPUModelRunner.update_config accepts model_config/load_config, not compilation_config. Capture buckets and dispatcher keys constructed at init. No live process injection. P-only restart after exact backup; D kept.'}
(r/'p-bucket-control-v1-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
code="""import json,subprocess,time
expected=[('e4d73792d53736a7608c4256a5d9168a2e1d4639d50b27a7a5702a83fb3c4a17','glm53-official-proxy-32-mtp-moe-graph-live2-20261018'),('12234db89fa8430727bba41ac176e8e21a81ecdd82a80eb07b26fc5c5b95c975','glm53-mtp-moe-graph-32-live2-20261018')];rows=[]
for cid,name in expected:
 d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['Name']=='/'+name and d['Id']==cid and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83';before=d['State'];q=subprocess.run(['docker','stop','-t','30',cid],capture_output=True,text=True,timeout=45);assert q.returncode==0;q=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert not q['State']['Running'];rows.append({'CID':cid,'before':before,'after':q['State']})
for i in range(20):
 s=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
 if s.count('No running processes found in NPU')==8:break
 time.sleep(1)
assert s.count('No running processes found in NPU')==8
print(json.dumps({'stopped_exact_owned':rows,'NPU_idle':s}))
"""
stopped=host(32,code);(r/'p-bucket-control-old-P-stop-receipt.json').write_text(json.dumps(stopped,indent=2)+'\n')
p['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/p-bucket-control-v1-20261019/node32';p['container']['name']='glm53-p-bucket-control-32-v1-20261019';p['container']['lifetime_seconds']=min(2400,allowed,int(d['D_remaining_s'])-180);assert p['container']['lifetime_seconds']>=1800
argv=p['serve_argv'];i=argv.index('--compilation-config')+1;cfg=json.loads(argv[i]);assert cfg=={'cudagraph_mode':'PIECEWISE','cudagraph_capture_sizes':[256,1024]};cfg['cudagraph_capture_sizes']=[256,512,1024];argv[i]=json.dumps(cfg,separators=(',',':'));i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-p-bucket-control-v1-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
p['env']['GLM53_PREFILL_PADDING_PROBE']='1';assert p['env']['GLM53_MTP_MOE_GRAPH_PROBE']=='0';p.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,objective='One native P graph bucket512 control; not a custom mechanism benefit',notes=[contract['change'],contract['measurement']]);plan=r/'p-bucket-control-P-plan.json';plan.write_text(json.dumps(p,indent=2)+'\n')
q=subprocess.run([sys.executable,'glm5-3-vllm/tools/start_capacity.py','--plan',str(plan),'--control','/tmp/glm53-active-20261018.sock','--output',str(r/'p-bucket-control-P-launch')],cwd=repo,capture_output=True,text=True,timeout=240);(r/'p-bucket-control-P-launch-collector.txt').write_text(q.stdout+q.stderr);print(q.stdout,flush=True);assert q.returncode==0,q.stderr
