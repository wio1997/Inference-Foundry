"""Same validated helper/config in fresh owned finite holders for formal timing."""
from pathlib import Path
import datetime
import importlib.util
import json
import subprocess
import sys

repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
proof=json.loads(next((r/'mtp-moe-copy-tasks-v2').glob('*/metrics.json')).read_text())
assert proof['completed'] and proof['all16_added_copy_SDMA']
assert json.loads((r/'mtp-moe-graph-functional-outcome.json').read_text())['quality_full64_equal_cases']==26
spec=importlib.util.spec_from_file_location('lease',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plans={};limits=[]
for role,node in [('P',32),('D',33)]:
    assert next((r/('mtp-moe-live1-'+str(node)+'-snapshot')).glob('*/artifacts-backup.json')).exists()
    p=json.loads((r/('mtp-moe-graph-'+role+'-plan.json')).read_text())
    saved=json.loads((r/('mtp-moe-graph-live2-lease'+str(node)+'.json')).read_text())
    res=p['reservation'];res.update(name=saved['metadata']['name'],expected_uid=saved['metadata']['uid'],owner_label=saved['metadata']['labels']['owner'])
    lease=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
    uid,allowed,expiry=m.fresh_lease(lease,res,datetime.datetime.now(datetime.timezone.utc));limits.append(allowed)
    assigned=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items']
    assert all(not sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in x['spec'].get('containers',[])) or x['metadata']['uid']==uid or x['status']['phase'] in ['Failed','Succeeded'] for x in assigned)
    smi=subprocess.check_output(ssh+['kubectl exec -i -n default '+pods[node]+' -- nsenter -t 1 -m -- npu-smi info'],timeout=30).decode();assert smi.count('No running processes found in NPU')==8
    (r/('mtp-moe-graph-live2-start-idle-'+str(node)+'.txt')).write_text(smi)
    p['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live2-20261018/node'+str(node)
    p['container']['name']='glm53-mtp-moe-graph-'+str(node)+'-live2-20261018'
    i=p['serve_argv'].index('--kv-transfer-config')+1;kv=json.loads(p['serve_argv'][i]);kv['engine_id']='glm53-mtp-moe-graph-'+role.lower()+'-live2-20261018';p['serve_argv'][i]=json.dumps(kv,separators=(',',':'))
    p.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,
             objective='Same all16-validated native MTP opaque MoE graph, fresh finite window for clean ordinary PD timing',
             notes=['No helper/model/precision/operator/config change from e4610b3ae3aa build; only freshholder/CID/engineID/lifetime.','Existing26x2 full64/cached/physicalC1 evidence reused; actualnewinstance path and C4physicalcase checked before formal.','AllSDMA; oldmodels/proxy backedup/stopped; nativeP PIECEWISE[256,1024]/sharedFalse, nativeD FULL/K5/sharedTrue.'])
    plans[role]=p
life=min(5400,*limits);assert life>=3600,limits
for role in ['P','D']:
    p=plans[role];p['container']['lifetime_seconds']=life;path=r/('mtp-moe-graph-live2-'+role+'-plan.json');path.write_text(json.dumps(p,indent=2)+'\n')
    q=subprocess.run([sys.executable,'glm5-3-vllm/tools/start_capacity.py','--plan',str(path),'--control','/tmp/glm53-active-20261018.sock','--output',str(r/('mtp-moe-graph-live2-'+role+'-launch'))],cwd=repo,capture_output=True,text=True,timeout=240)
    (r/('mtp-moe-graph-live2-'+role+'-launch-collector.txt')).write_text(q.stdout+q.stderr)
    print(json.dumps({'role':role,'life':life,'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}),flush=True);assert q.returncode==0
