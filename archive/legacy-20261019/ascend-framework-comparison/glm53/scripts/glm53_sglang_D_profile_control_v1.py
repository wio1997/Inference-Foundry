"""CPU43-only bounded native D profiler control; no inference, SDK or framework imports."""
import argparse,hashlib,json,socket,time,urllib.error,urllib.request
from pathlib import Path

def occupancy(body):
 import re
 result={}
 for line in body.splitlines():
  if not line.startswith('sglang:num_running_reqs{'):continue
  dp=int(re.search(r'dp_rank="([0-9]+)"',line).group(1));value=float(line.rsplit(' ',1)[1]);assert value.is_integer() and value>=0
  if dp in result:assert result[dp]==int(value)
  result[dp]=int(value)
 return result

def control(settings,request,pause=time.sleep,monotonic=time.monotonic,save=lambda x:None):
 assert settings['project_id']=='glm53' and settings['framework']=='sglang' and settings['diagnostic_only'] is True
 assert settings['profile_endpoint']=='http://172.16.12.36:30600'
 assert settings['metrics_endpoints']==['http://172.16.12.36:30600/metrics','http://172.16.12.41:30600/metrics']
 assert settings['profile_body']=={'output_dir':'/run/glm53/profile/'+settings['run_id'],'num_steps':8,'activities':['CPU','GPU'],'with_stack':False,'record_shapes':False,'profile_by_stage':False,'profile_id':settings['run_id'],'merge_profiles':False,'profile_prefix':'glm53-D','detailed_annotations':False}
 result={'project_id':'glm53','run_id':settings['run_id'],'diagnostic_only':True,'model_requests':0,'SDK_init_calls':0,'profile_start_attempts':0,'profile_stop_attempts':0,'events':[],'pass':False}
 def record(kind,value):result['events'].append({'kind':kind,'monotonic_seconds':monotonic(),'value':value});save(result)
 def get(url,body=None,timeout=10):
  value=request(url,body,timeout);record('HTTP',value);return value
 attempted=False
 try:
  info=get(settings['profile_endpoint']+'/get_server_info');assert info['status']==200
  args=json.loads(info['body']);assert args['disaggregation_mode']=='decode' and args['tp_size']==32 and args['dp_size']==16 and args['ep_size']==32 and args['device']=='npu' and args['disable_decode_cuda_graph'] and args['disable_overlap_schedule'] and not args['enable_hierarchical_cache']
  deadline=monotonic()+90;admitted=False
  while monotonic()<deadline:
   native=[get(url) for url in settings['metrics_endpoints']];assert all(x['status']==200 for x in native)
   a,b=[occupancy(x['body']) for x in native];assert set(a)==set(range(8)) and set(b)==set(range(8,16))
   counts=a|b
   if all(x>=1 for x in counts.values()):record('D16_running_admission',counts);admitted=True;break
   pause(1)
  assert admitted,'No all16-DP active decode admission within90s; do not start profiler'
  attempted=True;result['profile_start_attempts']=1
  start=get(settings['profile_endpoint']+'/start_profile',settings['profile_body'],90);result['start_response']=start;assert start['status']==200
  # Native8forward-step limit is primary. Wall guard always issues one stop,
  # including on partial/failed start. HTTP200 alone is not rank-completion proof.
  pause(12)
 except Exception as error:result['error']=repr(error)
 finally:
  if attempted:
   result['profile_stop_attempts']=1
   try:
    result['stop_response']=get(settings['profile_endpoint']+'/stop_profile',{},180)
    stop=result['stop_response'];assert stop['status']==200 or (stop['status']==500 and 'not in progress' in stop['body'].lower()),'Native stop did not acknowledge completion or already-inactive state'
   except Exception as error:result['stop_error']=repr(error)
  result['profile_completion']='PENDING_ALL32_NATIVE_TRACE_AND_STOP_AUDIT' if attempted else 'NOT_STARTED'
  result['pass']=not result.get('error') and not result.get('stop_error')
  save(result)
 return result

def request(url,body,timeout):
 assert url.startswith(('http://172.16.12.36:30600/','http://172.16.12.41:30600/'))
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));payload=None if body is None else json.dumps(body).encode();req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'},method='GET' if body is None else 'POST');started=time.monotonic_ns()
 try:
  with opener.open(req,timeout=timeout) as response:status=response.status;raw=response.read(8*1024**2+1)
 except urllib.error.HTTPError as error:status=error.code;raw=error.read(8*1024**2+1)
 assert len(raw)<=8*1024**2
 # Retain selected native settings and full HTTP status/body for controls.
 text=raw.decode()
 if url.endswith('/get_server_info'):
  info=json.loads(text);text=json.dumps({k:info[k] for k in ['disaggregation_mode','tp_size','dp_size','ep_size','device','disable_decode_cuda_graph','disable_overlap_schedule','enable_hierarchical_cache']})
 result={'url':url,'status':status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'dispatch_monotonic_ns':started,'terminal_monotonic_ns':time.monotonic_ns(),'body':text}
 if url.endswith('/metrics'):
  result['body']='\n'.join(x for x in text.splitlines() if x.startswith('sglang:num_running_reqs{'))
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--settings',required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--source-sha256',required=True);a=parser.parse_args()
 assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==a.source_sha256
 raw=Path(a.settings).read_bytes();assert hashlib.sha256(raw).hexdigest()==a.sha256;settings=json.loads(raw)
 root=Path('/data/wio/inference-foundry/glm53/runs')/settings['run_id'];assert root==Path(a.settings).parent
 with (root/'D-profile-control.once').open('x') as f:f.write(str(__import__('os').getpid()))
 def save(result):
  path=root/'D-profile-control-progress.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
 result=control(settings,request,save=save);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
