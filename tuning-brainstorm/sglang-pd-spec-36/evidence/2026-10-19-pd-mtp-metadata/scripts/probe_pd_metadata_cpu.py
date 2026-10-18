#!/usr/bin/env python3
"""Actual CPU MetadataBuffers and actual EAGLE consumer; no native transfer."""
import argparse,datetime,hashlib,importlib,inspect,json,os,pathlib,platform,subprocess,sys,traceback,types
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);fixture_root=pathlib.Path(args.fixture_root);model=pathlib.Path('/model');protocol_path=pathlib.Path(args.protocol);protocol=json.loads(protocol_path.read_text());expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1';nodes=sorted({str(p) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for p in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==protocol['source_commit'];hashes={}
for f,h in expected.items():
 hashes[f]=sha(root/f);assert hashes[f]==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(model/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':hashes,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(protocol_path),'accelerator_device_nodes':nodes,'environment':{'python':sys.version,'machine':platform.machine(),'byteorder':sys.byteorder},'cases':[],'scope':'Full unmodified installed MetadataBuffers CPU branch and EAGLE consumer using actual runtime test override. CPU tensor copies simulate wire; receive-field fixture, not DecodeTransferQueue execution. Overlap relay recorder only, no FutureMap fence/native transport/model inference/performance.'};stage='import'
try:
 sys.path.insert(0,str(root/'python'));import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 M=importlib.import_module('sglang.srt.disaggregation.utils');C=importlib.import_module('sglang.srt.speculative.eagle_disaggregation');I=importlib.import_module('sglang.srt.speculative.eagle_info');U=importlib.import_module('sglang.srt.managers.overlap_utils');rt=importlib.import_module('sglang.srt.runtime_context');D=importlib.import_module('sglang.srt.layers.attention.dsa.utils');E=importlib.import_module('sglang.srt.environ')
 assert C.get_spec is rt.get_spec and not M.is_npu() and not D.is_cuda() and not D.is_hip() and not D.should_remap_pd_dsa_seed_to_local_slots();assert 'torch_npu' not in sys.modules
 loaded={}
 for mod in [M,C,I,U,rt,D,E]:
  f=pathlib.Path(mod.__file__);relative=str(f.relative_to(root));assert relative in expected and sha(f)==expected[relative];loaded[mod.__name__]={'path':str(f),'sha256':sha(f)}
 result['loaded_modules']=loaded;result['environment']['torch']=torch.__version__;config=json.loads((model/'config.json').read_text());assert config['hidden_size']==protocol['model_hidden_fixture'];model_dsa_dim=M.get_dsa_seed_metadata_dim(types.SimpleNamespace(**config));assert 0<=model_dsa_dim<=8192;result['model_shape_reference']={'hidden_size':config['hidden_size'],'index_share_for_mtp_iteration':config.get('index_share_for_mtp_iteration',False),'dsa_metadata_dim_from_actual_helper':model_dsa_dim,'scope':'Only fixture width from hashed config and actual config-only helper. CPU dtypes are declared fixtures, not inferred weight/KV dtype.'}
 stage='correctness';dtype_map={'float32':torch.float32,'bfloat16':torch.bfloat16};field_names=['output_ids','cached_tokens','output_token_logprobs_val','output_token_logprobs_idx','output_top_logprobs_val','output_top_logprobs_idx',None,None,None,'output_topk_p','output_topk_index','output_hidden_states','output_dsa_topk_indices','bootstrap_room']
 def equal(a,b):
  if a is None or b is None:assert a is None and b is None
  else:assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 def summary(t):
  if t is None:return None
  return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'sha256':hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
 def buffers(x):
  by={v.data_ptr():(n,v) for n,v in vars(x).items() if torch.is_tensor(v)};ptrs,lens,items=x.get_buf_infos();assert len(ptrs)==len(by)
  rows=[]
  for ptr,nbytes,itembytes in zip(ptrs,lens,items):
   name,t=by[ptr];assert t.device.type=='cpu' and nbytes==t.nbytes and itembytes==t[0].nbytes;rows.append((name,t))
  return rows
 def request(slot,serial,h,dtype,width,dsa_dim,old=False,seed_mode='valid',row=0):
  hidden=(torch.arange(h,dtype=torch.float32)%7+row*8+(32 if old else 0)).to(dtype);probs=-(torch.arange(width,dtype=torch.float32)+31) if old else (torch.arange(width,dtype=torch.float32)+1)/(width+1);indices=torch.arange(width,dtype=torch.int64)+1000+serial*32+row*64
  seed=torch.arange(dsa_dim,dtype=torch.int32)%64 if dsa_dim else None
  if seed_mode=='missing' or (seed_mode=='mixed_missing' and row==0):seed=None
  elif seed_mode=='all_negative' or (seed_mode=='mixed_negative' and row==0):seed=torch.full((dsa_dim,),-1,dtype=torch.int32) if dsa_dim else None
  return types.SimpleNamespace(metadata_buffer_index=slot,output_ids=[4000+serial*10+row],cached_tokens=31+row,cached_tokens_device=17+row,cached_tokens_host=7,cached_tokens_storage=2,multimodal_inputs=None,return_logprob=False,return_sampling_mask=False,output_topk_p=probs,output_topk_index=indices,hidden_states_tensor=hidden,output_dsa_topk_indices=seed,bootstrap_room=1000+serial*10+row)
 def allocate(size,h,dtype,dim,checksum):return M.MetadataBuffers(size,h,dtype,max_sampling_mask_tokens=8,output_dsa_topk_indices_dim=dim,kv_checksum_enabled=checksum)
 class RelayRecorder:
  def __init__(self):self.calls=[]
  def publish(self,indices,seq_lens):self.calls.append(('publish',indices,seq_lens))
  def stash(self,indices,payload):self.calls.append(('stash',indices,payload))
 def consume(rows,n,states,overlap,serial,expected_values):
  reqs=[]
  for row in rows:
   dsa=row[12].clone() if row[12] is not None else None
   if dsa is not None and torch.all(dsa<0).item():dsa=None
   reqs.append(types.SimpleNamespace(output_topk_p=row[9].clone(),output_topk_index=row[10].clone(),hidden_states_tensor=row[11].clone(),output_dsa_topk_indices=dsa))
  pool=torch.tensor([4,1,6][:n],dtype=torch.int64);lens=torch.tensor([128,32768,512][:n],dtype=torch.int64);last=torch.tensor([r[0][0].item() for r in rows],dtype=torch.int64);batch=types.SimpleNamespace(reqs=reqs,device='cpu',req_pool_indices=pool,seq_lens=lens,enable_overlap=overlap);relay=RelayRecorder()
  with rt.get_context().override_server_args(device='cpu',speculative_algorithm='EAGLE',speculative_eagle_topk=1,speculative_num_steps=3,speculative_num_draft_tokens=4,enable_multi_layer_eagle=(states==3),disaggregation_mode='decode'):
   assert C.get_spec is rt.get_spec and rt.get_spec().speculative_eagle_topk==1;info=C.build_eagle_disagg_draft_input(batch,last,relay)
  assert isinstance(info,I.EagleDraftInput) and info.capture_hidden_mode is C.CaptureHiddenMode.LAST
  wants={'topk_p':torch.stack([q.output_topk_p[:states] for q in expected_values]),'topk_index':torch.stack([q.output_topk_index[:states] for q in expected_values]),'hidden_states':torch.stack([q.hidden_states_tensor for q in expected_values])}
  dsa=[q.output_dsa_topk_indices for q in expected_values];wanted_seed=torch.stack(dsa) if all(t is not None for t in dsa) and not any(torch.all(t<0).item() for t in dsa) else None
  for name,t in wants.items():equal(getattr(info,name),t)
  equal(info.dsa_topk_indices,wanted_seed);equal(info.bonus_tokens,last);assert info.bonus_tokens.data_ptr()==last.data_ptr() and info.draft_probs is None
  for req in reqs:
   req.output_topk_p.fill_(99);req.output_topk_index.fill_(99);req.hidden_states_tensor.fill_(99)
   if req.output_dsa_topk_indices is not None:req.output_dsa_topk_indices.fill_(99)
  for name,t in wants.items():equal(getattr(info,name),t)
  equal(info.dsa_topk_indices,wanted_seed)
  if overlap:
   assert len(relay.calls)==2 and relay.calls[0][0]=='publish' and relay.calls[1][0]=='stash';assert relay.calls[0][1] is pool and relay.calls[0][2] is lens;payload=relay.calls[1][2];assert isinstance(payload,U.RelayPayload) and payload.topk_p is info.topk_p and payload.hidden_states is info.hidden_states and payload.bonus_tokens is last and payload.dsa_topk_indices is info.dsa_topk_indices;assert info.future_indices is pool and info.future_dsa_topk_indices_available==(wanted_seed is not None)
  else:assert not relay.calls and info.future_indices is None
  return {'overlap':overlap,'relay_scope':'Recorder checks call order/payload identity only; no FutureMap/stream fence' if overlap else 'not used','outputs':{name:summary(getattr(info,name)) for name in ['topk_p','topk_index','hidden_states','dsa_topk_indices','bonus_tokens']},'bonus_alias_expected':True,'stack_ownership_verified':True}
 with E.envs.SGLANG_ENABLE_DISAGG_SAMPLING_MASK.override(False),E.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(None):
  for case in protocol['cases']:
   result['in_progress']=case;n=case['batch'];h=case['hidden'];dtype=dtype_map[case['dtype']];dim=model_dsa_dim if case['dsa_dim']=='model_declared' else case['dsa_dim'];states=case['states'];serial=case['serial'];checksum=n==3;src=allocate(n+1,h,dtype,dim,checksum);dst=allocate(n+1,h,dtype,dim,checksum);layout=buffers(src);other=buffers(dst);assert [x[0] for x in layout]==[x[0] for x in other];slots=[2,0,1][:n]
   for row,slot in enumerate(slots):src.set_buf(request(slot,serial,h,dtype,16,dim,old=True,row=row))
   fresh=[request(slot,serial,h,dtype,states,dim,seed_mode=case.get('seed_mode','valid'),row=row) for row,slot in enumerate(slots)]
   for q in fresh:
    src.set_buf(q)
    if checksum:src.set_kv_checksum(q,q.bootstrap_room^123)
   for (_,a),(_,b) in zip(layout,other):b.copy_(a)
   snapshots=[dst.get_buf(slot) for slot in slots];saved=[[t.clone() if t is not None else None for t in row] for row in snapshots]
   for row,slot,q in zip(snapshots,slots,fresh):
    assert row[0][0].item()==q.output_ids[0] and row[13][0].item()==q.bootstrap_room;assert row[13].dtype==torch.uint64
    equal(row[9][:states],q.output_topk_p);equal(row[10][:states],q.output_topk_index);equal(row[11],q.hidden_states_tensor)
    if states<16:equal(row[9][states:],-(torch.arange(16,dtype=torch.float32)+31)[states:])
    for name,t in zip(field_names,row):
     if t is not None:assert t.data_ptr()!=getattr(dst,name)[slot].data_ptr()
    if checksum:assert dst.get_kv_checksum(slot)==(q.bootstrap_room^123)
   for _,tensor in other:tensor.zero_()
   for row,want in zip(snapshots,saved):
    for t,u in zip(row,want):equal(t,u)
   outputs=[consume(snapshots,n,states,overlap,serial,fresh) for overlap in [False,True]]
   result['cases'].append({'id':case['id'],'batch':n,'hidden':h,'dtype':case['dtype'],'dsa_dim':dim,'states':states,'seed_mode':case.get('seed_mode','valid'),'status':'PASS','wire_copy_scope':'CPU copy_ fixture, no connector/transport','wire_layout':[{'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'row_bytes':t[0].nbytes} for name,t in layout],'get_buf_clone_ownership':True,'stale_topk_tail_present_but_sliced':True,'checksums':checksum,'consumer':outputs});result.pop('in_progress')
  # Explicit unsupported-producer and capacity observations, not active PD failures.
  src=allocate(2,16,torch.float32,8,False);old=request(0,900,16,torch.float32,16,8);src.set_buf(old);bad=request(0,901,16,torch.float32,1,8,seed_mode='missing');bad.hidden_states_tensor=None;src.set_buf(bad);row=src.get_buf(0);equal(row[11],old.hidden_states_tensor);equal(row[12],old.output_dsa_topk_indices);assert row[13][0].item()==bad.bootstrap_room
  result['hidden_none_observation']={'old_spec_fields_retained':True,'new_bootstrap_room_written':True,'scope':'HiddenNone producer fixture; reachability in realEAGLEprefill unproven, not an activePDbug.'}
  rejects=[]
  for kind in ['topk17','hidden-width']:
   buf=allocate(2,16,torch.float32,8,False);q=request(0,902,16,torch.float32,1,8);buf.set_buf(q);old_room=int(buf.bootstrap_room[0,0].item());bad=request(0,903,16,torch.float32,17 if kind=='topk17' else 1,8)
   if kind=='hidden-width':bad.hidden_states_tensor=torch.ones(15)
   try:buf.set_buf(bad);raise AssertionError('Expected tensor shape rejection')
   except RuntimeError as exc:rejects.append({'kind':kind,'error_type':type(exc).__name__,'error':str(exc),'bootstrap_room_remained_old':int(buf.bootstrap_room[0,0].item())==old_room,'scope':'CPU shape-boundary fixture; writes may be partial, no native transport/publication tested'})
   buf.set_buf(q);equal(buf.get_buf(0)[11],q.hidden_states_tensor)
  result['shape_rejections']=rejects
 assert len(result['cases'])==len(protocol['cases']) and C.get_spec is rt.get_spec and 'torch_npu' not in sys.modules
 result.update({'probe_status':'PASS','correctness':'PASS','case_count':len(result['cases']),'consumer_calls':2*len(result['cases']),'metadata_backend':'CPU uint64bootstrap, no NPUint64/D2D result','remap_selector_actual_result':False,'native_executed':False,'performance':'NOT_MEASURED'})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in models.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
