import datetime,json,pathlib,re,shutil,socket,subprocess,time,urllib.request,uuid
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live1-20261018/node33');cid='1bb3a19d7f6ed621b5fcf72189bbfa85c9a3ecb46753706ae5bf7cd1743a1ad2';plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Image']==plan['image']['id'] and d['Name']=='/'+plan['container']['name']
left=plan['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>600,left
formal=json.loads((root/'mtp-ffn-graph-c4-functional-v2/result.json').read_text());assert formal['completed'] and formal['numeric_all_pass']
assert not (root/'mtp-ffn-graph-probe/disabled').exists();marker=root/'mtp-ffn-graph-probe/enable';assert not marker.exists();assert shutil.disk_usage(root).free>8*1024**3
cases=json.loads((root/'prefill-padding-cases-v1.json').read_text())['fixed'];assert [len(c['prompt_token_ids']) for c in cases]==[256,257]
out=root/'mtp-ffn-graph-profile-v1';out.mkdir(exist_ok=False);(out/'cases.json').write_text(json.dumps({'cases':cases})+'\n');before={p.name for p in (root/'profile').glob('*_ascend_pt')}
report={'completed':False,'rows':[],'container_id':cid,'profile_role':'D','scope':'One bounded native D profiler session: real P remote_decode to complete D8. Twoarm thenN257ABBA4. ExistingMTPFFN onlygraph; actualD2D transfer task coverage; no service timing.','native_wrapper_source':'5242ee153033afcf12dbfe022dbd807fa248d991ff5c53efbd1adb372d5575f4'}
def post(node,path,body,timeout=180,rid=None):
 req=urllib.request.Request('http://172.16.12.%d:%d%s'%(node,29481 if node==32 else 29482,path),json.dumps(body).encode(),{'Content-Type':'application/json',**({'X-Request-Id':rid} if rid else {})})
 with urllib.request.urlopen(req,timeout=timeout) as f:b=f.read()
 return json.loads(b) if b else None
def idle():
 for attempt in range(60):
  values=[]
  for node,port in [(32,29481),(33,29482)]:
   with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,port),timeout=5) as f:s=f.read().decode()
   rows=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(rows)>=3;values+=rows
  if all(float(z.rsplit(' ',1)[1])==0 for z in values):return
  time.sleep(.25)
 raise AssertionError(values)
def one(case,mode,phase):
 idle()
 if mode=='adaptive':marker.touch()
 else:marker.unlink(missing_ok=True)
 rid='mtp-ffn-profile-'+uuid.uuid4().hex;body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':8,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':False}
 pbody={**body,'max_tokens':1,'kv_transfer_params':{'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None}}
 row={'case_id':case['case_id'],'N':len(case['prompt_token_ids']),'mode':mode,'phase':phase,'request_id':rid,'sent_monotonic_ns':time.monotonic_ns(),'sent_wall_ns':time.time_ns()};report['rows'].append(row)
 row['P_response']=post(32,'/v1/completions',pbody,rid=rid);row['received_monotonic_ns']=time.monotonic_ns();row['received_wall_ns']=time.time_ns();params=row['P_response']['kv_transfer_params'];assert params['remote_num_prompt_tokens']==row['N'];body['kv_transfer_params']=params
 row['D_response']=post(33,'/v1/completions',body,rid=rid);row['D_finished_monotonic_ns']=time.monotonic_ns();row['D_finished_wall_ns']=time.time_ns();res=row['D_response'];text=res['choices'][0]['text'];m=re.search(r'[-+]?\d+(?:\.\d+)?',text)
 assert len(res['choices'][0]['token_ids'])==8 and res['usage']['completion_tokens']==8 and res['usage']['prompt_tokens']==row['N'];assert m and float(m.group())==case['expected']
 row['complete_answer_pass']=True;(out/'progress.json').write_text(json.dumps(report)+'\n')
attempted=False
try:
 idle();attempted=True;report['start_profile']=post(33,'/start_profile',{},timeout=360)
 for case in cases:one(case,'native','arm')
 for mode in ['native','adaptive','adaptive','native']:one(cases[1],mode,'active')
 idle();report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
finally:
 marker.unlink(missing_ok=True)
 if attempted:
  try:report['stop_profile']=post(33,'/stop_profile',{},timeout=360);report['profile_stopped']=True
  except Exception as exc:report['stop_failure']=str(exc)
dirs=sorted(p for p in (root/'profile').glob('*_ascend_pt') if p.name not in before);report['new_profile_dirs']=[str(p) for p in dirs];report['ranks']=sorted(int(re.search(r'_rank(\d+)_',p.name).group(1)) for p in dirs);report['source_files']=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size} for d in dirs for p in sorted(d.rglob('*')) if p.is_file()];report['source_bytes']=sum(x['bytes'] for x in report['source_files']);(out/'result.json').write_text(json.dumps(report)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'] and report.get('profile_stopped') and report['ranks']==list(range(16)) and report['source_bytes']<4*1024**3
