"""Read-only serial failure capture while at most one old startup observer remains."""
import ast,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12B-P-startup-failure-capture-v1'
def plan():
 p=ROOT/'scripts/glm53_C32_L12B_P_transition_library_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='0f59ff934fa11980d17b590948250b22baffb088cbbe4b2daab2493e6545d146';s=importlib.util.spec_from_file_location('lib',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 stream=ROOT/'scripts/glm53_stream_C32_packed_native_logs_v1.py';assert hashlib.sha256(stream.read_bytes()).hexdigest()=='e03a299a23f50ee7d7fc7233c0f1cbb221043d8fde62c0cd3e5b04ea55cc5abc';plans={}
 cap=json.loads((ROOT/'results/sglang-C32-L12B-selected-P-pool-capacity-v1.json').read_text());assert cap['selected_variant']=='candidate'
 host=r'''import pathlib,subprocess,json,hashlib,urllib.request
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0];assert d['Id']==CID and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED
root=pathlib.Path(REMOTE);archive=pathlib.Path(ARCHIVE);assert not archive.exists();archive.mkdir(parents=True)
scope={};exec(compile(STREAM,'stream','exec'),scope);out={'project_id':'glm53','node':NODE,'container_id':CID,'container_state':d['State'],'logs':{},'model_requests':0,'SDK_init_calls':0}
for stream in ['stdout','stderr']:
 p=root/('model.'+stream+'.txt')
 if p.exists():
  with p.open('rb') as f:out['logs'][stream]=scope['summarize_log_stream'](f,archive/('model.'+stream+'.gz'))
for name in ['supervisor-status.json','provider-status.json']:
 p=root/name;out[name]=json.loads(p.read_text())
for name,args in [('npu',['npu-smi','info']),('top',['docker','top',CID,'-eo','pid,ppid,stat,comm,etime'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=30);out[name]={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));out['retained_D_health']={}
for n in [36,41]:
 url='http://172.16.12.'+str(n)+':30600'
 with opener.open(url+'/health',timeout=10) as f:assert f.status==200
 with opener.open(url+'/metrics',timeout=10) as f:raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2
 gauges=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges);out['retained_D_health'][str(n)]={'healthy_idle':True}
print(json.dumps(out))
'''
 for n in [34,35]:
  item=cap['providers'][str(n)];values={'CID':item['container_id'],'IMAGE':m.CURRENT['model_image'],'SHARED':item['run_id'],'REMOTE':item['remote_root'],'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+RID,'STREAM':stream.read_text(),'NODE':n};code=m.replace(host,values);ast.parse(code);plans[n]=code
 return m,plans

def main():
 m,plans=plan();review=json.loads((ROOT/'results/C32-L12B-P-startup-failure-capture-review-v1.json').read_text());assert review['host_codes_sha256']=={str(n):hashlib.sha256(c.encode()).hexdigest() for n,c in plans.items()}
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','serial_SSH':True,'max_additional_SSH':1,'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  for n,code in plans.items():
   r=m.stage().invoke(n,code,180);(run/('node-'+str(n)+'.json')).write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][str(n)]=r['rc'];print(n,r['rc'],flush=True);assert r['rc']==0
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
