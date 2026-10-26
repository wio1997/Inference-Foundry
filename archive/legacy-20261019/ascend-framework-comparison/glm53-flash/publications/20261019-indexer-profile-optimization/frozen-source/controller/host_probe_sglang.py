import pathlib,json,subprocess,os,socket,base64,hashlib,traceback,time,shlex,urllib.request
__LAUNCH_COMMAND__
__PROFILE_EXPORT__
node=__NODE__
assert node in (46,47)
assert socket.gethostname()=={46:'B8-15U-Atlas900RCKA3-46',47:'B8-23U-Atlas900RCKA3-47'}[node]
R=pathlib.Path('/data/wio/inference-foundry/glm53-flash/runs/20261019-r247-b-indexer-optimized-profile/B')
R.mkdir(parents=True,exist_ok=False)
os.close(os.open(R/'once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
files=__FILES__
name=container_name(node)
image=IMAGE
created=False
out={'project_id':'glm53-flash','unit':'r247','framework':'sglang','node':node,
     'run_id':RUN_ID,'namespace':NAMESPACE,'container':name,'image':image,
     'generation_requests':0,'formal_requests':0,'performance_baseline':False,
     'cleanup_identity_verified':False,'candidate_delta':True,
     'original_46_overlay_unchanged':False}


def run(args,t=30):
 q=subprocess.run(args,capture_output=True,text=True,timeout=t)
 return {'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}


def save():
 q=R/'host-status.tmp';q.write_text(json.dumps(out,indent=2));q.replace(R/'host-status.json')


def identity(n):
 q=run(['docker','inspect',n]);assert q['rc']==0
 v=json.loads(q['stdout'])[0]
 assert v['Image']==image and v['Config']['Labels'].get('project_id')=='glm53-flash' and v['Config']['Labels'].get('run_id')==RUN_ID and v['Config']['Labels'].get('namespace')==NAMESPACE
 return v


def selected(name):
 if name in {'operator_details.csv','kernel_details.csv','step_trace_time.csv',
             'trace_view.json','profiler_metadata.json'}:
  return True
 if name.startswith('profiler_info') and name.endswith('.json'):
  return True
 return (name.startswith('forward-metadata-') or name.startswith('observer-status-')
         or name.startswith('observer-install') or name.startswith('observer-sitecustomize')
         or name.startswith('observer-patch-error'))


def profile_index():
 root=R/'profile-B'
 rows={};total=0;hashed=0
 if not root.exists():
  return {'present':False,'files':{},'bytes':0}
 for path in sorted(root.rglob('*')):
  if not path.is_file():continue
  if len(rows)>=20000:
   return {'present':True,'truncated':True,'files':rows,'bytes':total,'reason':'file_count_cap'}
  size=path.stat().st_size;total+=size
  rel=str(path.relative_to(R));digest=None
  if selected(path.name):
   if hashed+size<=8*1024**3:
    digest=hashlib.sha256(path.read_bytes()).hexdigest();hashed+=size
  rows[rel]={'bytes':size,'sha256':digest}
 return {'present':True,'files':rows,'bytes':total,'file_count':len(rows),'hashed_bytes':hashed}


def sdma_log(path):
 result={'events':0,'initialized':[],'push_submitted':0,'push_complete':0,
         'probe64_complete':0,'summaries':[],'parse_errors':[],
         'all_initialized_protocol_sdma':False,'expected_role':ROLES[node],
         'expected_role_initialized':False,'failed_calls':0}
 if not path.exists():return result
 try:
  stream=path.open('r',errors='replace')
 except BaseException as error:
  result['parse_errors'].append(repr(error));return result
 with stream:
  for line in stream:
   if 'GLM53_FLASH_B103_SDMA ' not in line:continue
   try:
    event=json.loads(line.split('GLM53_FLASH_B103_SDMA ',1)[1])
   except BaseException as error:
    result['parse_errors'].append(repr(error));continue
   result['events']+=1
   kind=event.get('kind')
   if kind=='initialized':
    result['initialized'].append({k:event.get(k) for k in ('device','role','protocol','session_id','store_url')})
   elif kind=='push_submitted':result['push_submitted']+=1
   elif kind=='push_complete':result['push_complete']+=1
   elif kind=='probe64_complete':result['probe64_complete']+=1
   elif kind=='performance_SDMA_summary':
    result['summaries'].append(event)
    result['failed_calls']=max(result['failed_calls'],int(event.get('failed_calls') or 0))
 result['all_initialized_protocol_sdma']=bool(result['initialized']) and all(
  'SDMA' in str(row.get('protocol','')).upper() for row in result['initialized'])
 result['expected_role_initialized']=any(row.get('role')==ROLES[node] for row in result['initialized'])
 return result


try:
 out['pre_npu']=run(['npu-smi','info'])
 assert out['pre_npu']['rc']==0 and out['pre_npu']['stdout'].count('No running processes found')==8,'NPU_OCCUPIED'
 ip='172.16.12.'+str(node)
 ports=[('0.0.0.0',19030),(ip,19031)]+([(ip,19032),(ip,19024)] if node==46 else [(ip,19100+i) for i in range(16)])
 for address,port in ports:
  s=socket.socket();s.bind((address,port));s.close()
 out['ports_checked']=ports
 available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines()
                    if l.startswith('MemAvailable:')))*1024
 out['host_mem_available_bytes']=available
 assert available>=128*1024**3
 assert run(['docker','image','inspect',image,'--format','{{.Id}}'])['stdout'].strip()==image
 for rel,e in files.items():
  key=pathlib.PurePosixPath(rel)
  assert not key.is_absolute() and '..' not in key.parts
  b=base64.b64decode(e['data'])
  assert hashlib.sha256(b).hexdigest()==e['sha256'],rel
  dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
 (R/'profile-B').mkdir(exist_ok=True)
 model=pathlib.Path(MODEL_PATH)
 for filename,sha in {'config.json':'33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943',
                      'quant_model_description.json':'9aae1f3c7657115b7e24294bf6d668f6ea062e2c8fe9eccad4a3862284fe665f',
                      'tokenizer_config.json':'98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc',
                      'tokenizer.json':'19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'}.items():
  assert hashlib.sha256((model/filename).read_bytes()).hexdigest()==sha,filename
 out['model_metadata_sha256']={filename:hashlib.sha256((model/filename).read_bytes()).hexdigest()
                               for filename in ('config.json','quant_model_description.json','tokenizer_config.json','tokenizer.json')}
 (R/'native-preflight-ready.json').write_text(json.dumps({'ready':True,'node':node,'run_id':RUN_ID})+'\n')
 manifest=json.loads((R/'combined-manifest.json').read_text())
 assert manifest['status']=='R247_CANDIDATE_DELTA_NOT_ORIGINAL_46'
 assert len(manifest['files'])==47
 args=build(node,str(R),manifest['files'],json.loads((R/f'argv{node}.json').read_text()))
 out['command']=args
 out['launch']=run(args,60);assert out['launch']['rc']==0
 created=True;save()
 deadline=time.monotonic()+1800
 out['service_ready']=False
 while time.monotonic()<deadline:
  if (R/'stop.json').exists():out['startup_stop_requested']=True;break
  state=run(['docker','inspect',name,'--format','{{.State.Running}} {{.State.ExitCode}}'])
  if not state['stdout'].startswith('true'):out['startup_exit']=state;break
  try:
   with urllib.request.urlopen('http://127.0.0.1:19030/ready',timeout=3) as response:
    if response.status==200:out['service_ready']=True;break
  except Exception:pass
  time.sleep(10)
 if not out['service_ready']:
  out['error']=('STARTUP_STOP_REQUESTED' if out.get('startup_stop_requested')
                else 'STARTUP_DEADLINE_1800')
 save()
 if out['service_ready']:
  for endpoint in ['model_info','server_info']:
   with urllib.request.urlopen('http://127.0.0.1:19030/'+endpoint,timeout=30) as response:
    info=json.load(response)
   out[endpoint]={k:info.get(k) for k in (['model_path','is_generation','tokenizer_path','context_length','served_model_name'] if endpoint=='model_info' else ['model_path','tp_size','dp_size','ep_size','chunked_prefill_size','max_prefill_tokens','disaggregation_mode','disaggregation_transfer_backend'])}
  out['preflight']=json.loads((R/'preflight.json').read_text());assert out['preflight']['passed']
  assert not (R/'request-context-enabled').exists()
  if node==47:(R/'destinations-exposed').write_text('Require actual writer-stop permit before owned Decode stop\n')
  (R/'service-ready.json').write_text(json.dumps({'project_id':'glm53-flash','node':node,'ready':True,'run_id':RUN_ID})+'\n')
  deadline=time.monotonic()+3000
  while time.monotonic()<deadline and not (R/'stop.json').exists():
   state=run(['docker','inspect',name,'--format','{{.State.Running}} {{.State.ExitCode}}'])
   if not state['stdout'].startswith('true'):raise RuntimeError('SERVICE_EXITED_DURING_PD_UNIT')
   time.sleep(3)
  assert (R/'stop.json').exists(),'CONTROLLER_STOP_DEADLINE'
  out['controller_stop']=json.loads((R/'stop.json').read_text())
  save()
except BaseException as error:
 out.update(error=repr(error),traceback=traceback.format_exc())
finally:
 identity_ok=False
 if created:
  try:
   identity(name);identity_ok=True;out['cleanup_identity_verified']=True
   if node==47 and (R/'destinations-exposed').exists():
    import runpy
    verify=runpy.run_path(str(R/'flash_pd_writer_first_cleanup.py'))['verify_decode_permit']
    permit_deadline=time.monotonic()+240
    out['decode_stop_permitted']=False
    while time.monotonic()<permit_deadline:
     try:
      permit=json.loads((R/'writer-stop-permit.json').read_text())
      verify(permit,run_id=RUN_ID,image=image,
             writer_container=container_name(46))
      out['decode_stop_permitted']=True;out['writer_stop_proof_sha256']=permit['sha256'];break
     except Exception as e:out['writer_stop_permit_observation']=repr(e)
     time.sleep(.5)
    if not out['decode_stop_permitted']:
     raise RuntimeError('Writer stop/idle unproven: retain owned Decode destinations, no docker stop')
   out['cleanup_stop']=run(['docker','stop','--time','30',name],45)
   out['post_inspect']=run(['docker','inspect',name,'--format','{{.State.Running}} {{.State.ExitCode}}'])
  except Exception as e:out['cleanup_error']=repr(e)
 for attempt in range(12):
  out['post_npu']=run(['npu-smi','info'])
  if out['post_npu']['stdout'].count('No running processes found')==8:break
  time.sleep(5)
 try:
  if created and identity_ok:
   with (R/'server-raw.log').open('wb') as f:
    p=subprocess.run(['docker','logs','--timestamps',name],stdout=f,
                     stderr=subprocess.STDOUT,timeout=60)
   out['logs_rc']=p.returncode
 except Exception as e:out['log_collection_error']=repr(e)
 out['server_logs']={}
 q=R/'server-raw.log'
 if q.exists():
  with q.open('rb') as f:f.seek(max(0,q.stat().st_size-256*1024));tail=f.read().decode(errors='replace')
  out['server_logs'][q.name]={'bytes':q.stat().st_size,
                              'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'tail':tail}
  out['log_tail']=tail
  try:out['sdma_log']=sdma_log(q)
  except Exception as e:out['sdma_log_error']=repr(e)
 try:
  out['profile_index']=profile_index()
  out['profile_export']=evaluate(out['profile_index'].get('files',{}),
                                 truncated=bool(out['profile_index'].get('truncated')),
                                 collection_error=None)
 except Exception as e:
  out['profile_index_error']=repr(e)
  out['profile_export']=evaluate({},collection_error=repr(e))
 save()
 print(json.dumps(out))
