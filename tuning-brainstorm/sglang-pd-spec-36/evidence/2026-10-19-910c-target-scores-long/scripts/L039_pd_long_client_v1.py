"""Remote stdlib client: three original long P/D requests in fixed input order; never imports Torch or uses fake KV."""
import concurrent.futures,hashlib,json,math,os,pathlib,sys,threading,time,traceback,urllib.error,urllib.request
j=json.load(sys.stdin);plan=j['plan'];tag=plan['run'];root=pathlib.Path('/data/wio/sglang-mech36-20261018')/(tag+'-pd-client');assert not root.exists();root.mkdir(mode=0o700)
(root/'worker-handle.json').write_text(json.dumps({'run':tag,'pid':os.getpid(),'proc_start_ticks':pathlib.Path('/proc/self/stat').read_text().split()[21],'client_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'source_path':__file__},indent=2)+'\n')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));v={'run':tag,'execution_node':46,'status':'STARTING','intended_PD_pairs':3,'generation_attempted':False,'fake_bootstrap_used':False,'requests':{},'checks':[],'performance_verdict':'TARGET_TOP2_OBSERVATION_NOT_A_PERFORMANCE_SAMPLE','semantic_quality':'NOT_ASSESSED'}
def save():
 p=root/'result.tmp';p.write_text(json.dumps(v,indent=2)+'\n');p.replace(root/'result.json')
def http(base,path,body=None,timeout=8):
 data=None if body is None else json.dumps(body,separators=(',',':')).encode()
 req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'},method='GET' if body is None else 'POST');start=time.monotonic_ns()
 try:
  with opener.open(req,timeout=timeout) as response:raw=response.read(2*1024**2);status=response.status
  try:obj=json.loads(raw)
  except (ValueError,UnicodeDecodeError):obj=None
  return {'status':status,'start_ns':start,'end_ns':time.monotonic_ns(),'body_sha256':hashlib.sha256(raw).hexdigest(),'body':obj,'text':raw.decode(errors='replace') if obj is None else None}
 except urllib.error.HTTPError as error:return {'status':error.code,'start_ns':start,'end_ns':time.monotonic_ns(),'error_body':error.read(20000).decode(errors='replace')}
 except BaseException as error:return {'status':None,'start_ns':start,'end_ns':time.monotonic_ns(),'error':type(error).__name__+': '+str(error)}
try:
 assert plan['case']=='c3-long' and len(j['inputs'])==3 and plan['max_new_tokens']==256
 assert plan['bootstrap_host']=='172.16.12.44' and plan['bootstrap_port']==39098
 for item in j['inputs']:
  ids=item['input_ids'];assert len(ids)==32768 and all(type(t) is int and 0<=t<154880 for t in ids)
  assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==item['wire_ids_sha256']
 v['initial_flush']={role:http(url,'/flush_cache?timeout=10') for role,url in plan['endpoints'].items()}
 assert all(x['status']==200 for x in v['initial_flush'].values()),'Cannot establish idle/cold cache; no generation'
 barrier=threading.Barrier(2)
 def request_batch(role):
  body={'rid':[tag+'-'+x['label'] for x in j['inputs']],'input_ids':[x['input_ids'] for x in j['inputs']],'sampling_params':plan['sampling_params'],'stream':False,'return_logprob':True,'logprob_start_len':-1,'top_logprobs_num':2,'return_text_in_logprobs':False,'return_prompt_token_ids':True,'bootstrap_host':plan['bootstrap_host'],'bootstrap_port':plan['bootstrap_port'],'bootstrap_room':plan['rooms']}
  barrier.wait(timeout=15);result=http(plan['endpoints'][role],'/generate',body,plan['request_timeout_seconds']);result.update(role=role,request_sha256=hashlib.sha256(json.dumps(body,separators=(',',':')).encode()).hexdigest());return role,result
 started=time.monotonic_ns();v['generation_attempted']=True;v['batch_responses']={};save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(request_batch,role) for role in ['prefill','decode']]
  for f in concurrent.futures.as_completed(fs):
   role,result=f.result();v['batch_responses'][role]=result;save()
   assert result['status']==200 and isinstance(result['body'],list) and len(result['body'])==3,result
   for i,body in enumerate(result['body']):
    rid=tag+'-'+j['inputs'][i]['label'];assert body['meta_info']['id']==rid,'Returned batch order/identity mismatch'
    v['requests'][role+'-'+str(i)]={**result,'body':body,'index':i,'rid':rid,'room':plan['rooms'][i]}
   save()
 v['batch_client_window_ns']=time.monotonic_ns()-started
 for i in range(3):
  p=v['requests']['prefill-'+str(i)];d=v['requests']['decode-'+str(i)];assert p['status']==d['status']==200,(p,d)
  pb=p['body'];db=d['body'];assert isinstance(pb,dict) and isinstance(db,dict)
  pi=pb['output_ids'];di=db['output_ids'];pm=pb['meta_info'];dm=db['meta_info']
  assert len(pi)==1 and len(di)==256 and dm['completion_tokens']==256 and pm['prompt_tokens']==dm['prompt_tokens']==32768
  assert all(type(t) is int and 0<=t<154880 for t in di) and di[0]==pi[0],'Boundary token metadata differs between P/D'
  assert dm['finish_reason']['type']=='length' and dm['spec_verify_ct']>0
  assert dm['spec_num_proposed_drafts']==dm['spec_verify_ct']*3 and 0<=dm['spec_num_correct_drafts']<=dm['spec_num_proposed_drafts']
  assert db['prompt_token_ids']==j['inputs'][i]['input_ids'] and pb['prompt_token_ids']==j['inputs'][i]['input_ids']
  v['checks'].append({'request':i,'input_tokens':32768,'output_tokens':256,'P_D_first_token_equal':True,'returned_prompt_IDs_exact':True,'MTP_verify_ct':dm['spec_verify_ct'],'correct_drafts':dm['spec_num_correct_drafts'],'proposed_drafts':dm['spec_num_proposed_drafts'],'accept_histogram':dm.get('spec_correct_drafts_histogram'),'output_ids_sha256':hashlib.sha256(json.dumps(di,separators=(',',':')).encode()).hexdigest()})
 v['target_top2']=[]
 for i in range(3):
  body=v['requests']['decode-'+str(i)]['body'];ids=body['output_ids'];meta=body['meta_info'];selected=meta['output_token_logprobs'];top=meta['output_top_logprobs']
  offset=len(ids)-len(selected);assert offset in [0,1] and len(top)==len(selected),(len(ids),len(selected),len(top))
  assert [int(row[1]) for row in selected]==ids[offset:],'Chosen-logprob token IDs do not align'
  rows=[]
  for k,(chosen,alternatives) in enumerate(zip(selected,top)):
   position=offset+k
   if position==0 and (chosen[0] is None or not alternatives):continue
   assert chosen[0] is not None and math.isfinite(float(chosen[0]))
   assert len(alternatives)==2 and all(math.isfinite(float(a[0])) and 0<=int(a[1])<154880 for a in alternatives)
   alts=sorted(alternatives,key=lambda a:float(a[0]),reverse=True);rank=next((j+1 for j,a in enumerate(alts) if int(a[1])==ids[position]),None)
   rows.append({'position':position,'chosen_id':ids[position],'chosen_logprob':float(chosen[0]),'top2':[[float(a[0]),int(a[1])] for a in alts],'top2_logprob_margin':float(alts[0][0])-float(alts[1][0]),'chosen_rank_in_returned_top2':rank,'chosen_at_max_returned_logprob':float(chosen[0])==float(alts[0][0])})
  assert len(rows)>=255
  v['target_top2'].append({'request_index':i,'output_ids_sha256':v['checks'][i]['output_ids_sha256'],'logprob_alignment_offset':offset,'rows':rows,'all_chosen_at_returned_top2_max':all(z['chosen_at_max_returned_logprob'] for z in rows)})
 v['status']='PASS_FUNCTIONAL_PAIR_ONLY'
except BaseException as error:
 v['status']='FAIL';v['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
 # Abort only exact owned IDs, never abort_all or stop a service.
 v['own_abort_on_failure']={}
 for role,url in plan['endpoints'].items():
  for item in j['inputs']:
   rid=tag+'-'+item['label'];v['own_abort_on_failure'][role+'-'+rid]=http(url,'/abort_request',{'rid':rid,'abort_all':False})
finally:
 v['final_ready']={role:http(url,'/ready') for role,url in plan['endpoints'].items()}
 v['final_flush']={role:http(url,'/flush_cache?timeout=20',timeout=25) for role,url in plan['endpoints'].items()}
 if not all(x['status']==200 for x in [*v['final_ready'].values(),*v['final_flush'].values()]):v['status']='FAIL'
 v['scope']='Functional real P/D transfer+boundary/MTP token-count checks and idle cache flush only; no KV byte oracle, naturalEOS/cancel/retry coverage, quality conclusion or stable performance baseline.';save();print(json.dumps(v))
sys.exit(0 if v['status']=='PASS_FUNCTIONAL_PAIR_ONLY' else 1)
