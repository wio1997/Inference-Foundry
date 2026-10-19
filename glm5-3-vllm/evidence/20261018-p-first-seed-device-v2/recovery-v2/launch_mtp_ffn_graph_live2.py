from pathlib import Path
import datetime
import importlib.util
import json
import subprocess
import sys

repo = Path('/Users/wio/work/Inference-Foundry-execution-plan')
r = repo / 'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
ssh = ['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
def host(node, code):
    pod = {32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}[node]
    return json.loads(subprocess.check_output(ssh+['kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- python3 -'], input=code.encode(), timeout=70))

assert json.loads((r/'mtp-ffn-graph-cpu-check-v1.json').read_text())['passed']
assert json.loads(next((r/'mtp-ffn-graph-stage-v3').glob('*/metrics.json')).read_text())['staged']
assert next((r/'sfa-replay-live2-D-snapshot').glob('*/artifacts-backup.json')).exists()
manifest = json.loads((r/'mtp-ffn-graph-build-manifest-v3.json').read_text())
p = json.loads((r/'sfa-replay-order-live2-D-plan.json').read_text())
res = p['reservation']
lease = json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'], timeout=30))
spec = importlib.util.spec_from_file_location('lease', repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
uid, allowed, expiry = m.fresh_lease(lease,res,datetime.datetime.now(datetime.timezone.utc))
remaining = host(32, """import datetime,json,subprocess,urllib.request
cid='d9762493f4e60c450b06d964bf6c7b9d5fbcfd5261e44086a86f2031ff7ec33f'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/glm53-sfa-replay-order-32-live2-20261018'
start=datetime.datetime.fromisoformat(d['State']['StartedAt'].split('.')[0]+'+00:00');now=datetime.datetime.now(datetime.timezone.utc)
with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:assert json.load(f)['request_num']==0
print(json.dumps({'P_CID':cid,'remaining':5400-(now-start).total_seconds()}))
""")
life = int(min(3600, allowed, remaining['remaining']-180))
assert life >= 1800, (life,remaining,allowed)
p['remote']['run_dir'] = '/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live2-20261018/node33'
p['container'].update(name='glm53-mtp-ffn-graph-33-live2-20261018',lifetime_seconds=life)
p['runtime_patches'] = manifest['patches']
for key in list(p['env']):
    if key.startswith('GLM53_') and key != 'GLM53_STATE_PROBE_OUTPUT':p['env'][key]='0'
p['env']['GLM53_MTP_FFN_GRAPH_PROBE']='1'
i = p['serve_argv'].index('--kv-transfer-config')+1
kv = json.loads(p['serve_argv'][i]);kv['engine_id']='glm53-mtp-ffn-graph-d-live2-20261018'
p['serve_argv'][i] = json.dumps(kv,separators=(',',':'))
p.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,
         objective='Bounded native MTP pure FFN graph, attention/KV remain eager',
         notes=['OnlyDreload; retainedP and officialCPUproxy; watchdog capped by currentP remainingminus180s and own reservation.','Sameweights/precision/operators/nativeallSDMA; graphinputD2D copy requiresphysicalevidence.','Defaultmarkerabsent; CPU14guards; no unsupportedwholeMTPgraph flag.'])
plan = r/'mtp-ffn-graph-live2-D-plan.json';plan.write_text(json.dumps(p,indent=2)+'\n')
before = host(33, """import json,subprocess
cid='1bb3a19d7f6ed621b5fcf72189bbfa85c9a3ecb46753706ae5bf7cd1743a1ad2';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert not d['State']['Running'] and d['State']['ExitCode']==1 and d['Name']=='/glm53-mtp-ffn-graph-33-live1-20261018'
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=10);assert smi.count('No running processes found in NPU')==8
print(json.dumps({'failed_exact_CID':cid,'state':d['State'],'no_new_stop':True,'P_and_proxy_kept':True,'NPU_idle':smi}))
""")
(r/'mtp-ffn-graph-live2-D-reload-receipt.json').write_text(json.dumps({'lease':uid,'remaining':remaining,'lifetime':life,'old_D':before},indent=2)+'\n')
q = subprocess.run([sys.executable,'glm5-3-vllm/tools/start_capacity.py','--plan',str(plan),'--control','/tmp/glm53-active-20261018.sock','--output',str(r/'mtp-ffn-graph-live2-D-launch')],capture_output=True,text=True,timeout=240,cwd=repo)
(r/'mtp-ffn-graph-live2-D-launch-collector.txt').write_text(q.stdout+q.stderr)
print(json.dumps({'rc':q.returncode,'life':life,'stdout':q.stdout,'stderr':q.stderr}),flush=True)
assert q.returncode==0
