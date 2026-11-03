from pathlib import Path
from datetime import datetime,timezone
import socket,subprocess,json,re,hashlib,shutil,urllib.request
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
BASE='wio-vllm-v41-repro166-20261030';STAGE='wio-vllm-v41-repro166-stage-20261030'
BASE_ID='8757dcc7a2f8db672bbf6ddfc1ed29e6cff74d5dd89199770b7b6f3b27fd48e3'
STAGE_ID='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
SERVE_SHA='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
def cmd(args,timeout=20):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
containers=[];raw={}
for name,expected in [(BASE,BASE_ID),(STAGE,STAGE_ID)]:
 r=cmd(['docker','inspect',name]);raw[name+'_inspect_rc']=r['returncode']
 if r['returncode']:
  containers.append(dict(name=name,inspect_returncode=r['returncode'],stderr=r['stderr']));continue
 d=json.loads(r['stdout'])[0];top=cmd(['docker','top',name,'-eo','pid,comm']);raw[name+'_top']=top
 pids={int(l.split()[0]) for l in top['stdout'].splitlines()[1:] if l.split() and l.split()[0].isdigit()}
 containers.append(dict(name=name,id=d['Id'],image=d['Image'],running=d['State']['Running'],identity_matches=d['Id']==expected and d['Image']==IMAGE,
  restart_count=d['RestartCount'],pids=sorted(pids),top_rc=top['returncode'],diagnostic_instrumentation_env_present=any('instrumentation' in x or 'sitecustomize' in x for x in d['Config'].get('Env',[]))))
base=next((x for x in containers if x['name']==BASE),{});stage=next((x for x in containers if x['name']==STAGE),{})
if stage.get('running') and stage.get('identity_matches'):smi=cmd(['docker','exec',STAGE,'npu-smi','info'],30)
else:smi=cmd(['npu-smi','info'],30)
table_present=smi['returncode']==0 and 'Process id' in smi['stdout'];pids=set()
if table_present:
 for row in smi['stdout'].splitlines():
  col=[x.strip() for x in row.split('|')]
  if len(col)>4 and re.fullmatch(r'\d+\s+\d+',col[1]) and col[2].isdigit():pids.add(int(col[2]))
own=set(base.get('pids',[]));other=pids-own
serve=None
if base.get('running') and base.get('identity_matches'):
 r=cmd(['docker','exec',BASE,'sha256sum','/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/scripts/serve.sh']);serve=dict(returncode=r['returncode'],sha256=r['stdout'].split()[0] if r['returncode']==0 and r['stdout'].split() else None)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));http={}
for route in ['/health','/v1/models']:
 try:
  with opener.open('http://127.0.0.1:30075'+route,timeout=5) as resp:
   b=resp.read(65536);v=dict(status=resp.status,bytes=len(b))
   if route=='/v1/models':v['model_ids']=[x.get('id') for x in json.loads(b).get('data',[])]
   http[route]=v
 except Exception as e:http[route]=dict(error=type(e).__name__,message=str(e)[:250])
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030')
result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),containers=containers,serve_hash=serve,
 serve_hash_matches=bool(serve and serve['sha256']==SERVE_SHA),http=http,npu_process_table_present=table_present,npu_pids=sorted(pids),
 extra_npu_pids_outside_base=sorted(other),observed_NPU_processes_all_owned_by_base=table_present and len(pids)>=16 and not other,
 task_disk_free_bytes=shutil.disk_usage(P).free,raw_readonly_command_outputs={'npu_smi':smi,**raw},
 no_inference_device_or_service_mutation=True,
 limits=['Read-only npu-smi info queries driver state; no context/workload is launched.',
  'Ownership is a point-in-time observation and must be rechecked immediately before any isolated diagnostic.',
  'This check does not authorize killing other processes or replacing a changed container.'])
out=P/'analysis/whole-scope-continuation-20261030/profile-readonly-20261031/server-readonly-health-v11.json';assert not out.exists();payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result={k:v for k,v in result.items() if k!='raw_readonly_command_outputs'},full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
