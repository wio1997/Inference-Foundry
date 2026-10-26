#!/usr/bin/env python3
"""Actual share-state CPU lifecycle plus labeled source-branch spy; no NPU work."""
import argparse,ast,copy,datetime,hashlib,importlib,json,os,pathlib,platform,subprocess,sys,traceback,types
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);model=pathlib.Path('/model');protocol_path=pathlib.Path(args.protocol);p=json.loads(protocol_path.read_text());expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1';nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit'];hashes={}
for f,h in expected.items():
 hashes[f]=sha(root/f);assert hashes[f]==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(model/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':hashes,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(protocol_path),'accelerator_device_nodes':nodes,'environment':{'python':sys.version,'machine':platform.machine()},'pipeline_cases':[],'capture_cases':[],'truth_cases':[],'cleanup_cases':[],'route_cases':[],'scope':'Source-extracted unmodified class/CP helper/Ascendroute bodies with actual TorchCPU, ForwardMode/SpecInput and libraryruntime testoverride; CPstrategy None fixture. Full module remainsUNAVAILABLE. NPUgate is extracted actual AST branch with spy, not entireNPUattention/indexer or nativework. No performance or PDmodel result.'};stage='import';result['cycle_cases']=[]
try:
 sys.path.insert(0,str(root/'python'));import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 M=importlib.import_module('sglang.srt.disaggregation.utils');F=importlib.import_module('sglang.srt.model_executor.forward_batch_info');I=importlib.import_module('sglang.srt.speculative.eagle_info');rt=importlib.import_module('sglang.srt.runtime_context');MC=importlib.import_module('sglang.srt.configs.model_config');D=importlib.import_module('sglang.srt.layers.attention.dsa.utils');FM=F.ForwardMode
 assert 'torch_npu' not in sys.modules
 import contextlib,enum
 def bodyhash(node):return hashlib.sha256(ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
 share_tree=ast.parse((root/'python/sglang/srt/layers/attention/index_topk_share.py').read_text());share=copy.deepcopy(next(n for n in share_tree.body if isinstance(n,ast.ClassDef) and n.name=='IndexTopKShareState'));cp_tree=ast.parse((root/'python/sglang/srt/layers/cp/utils.py').read_text());cp_functions=[copy.deepcopy(n) for n in cp_tree.body if isinstance(n,ast.FunctionDef) and n.name in ['is_cp_active','cp_gather_after_forward']];assert len(cp_functions)==2
 state_env={'contextmanager':contextlib.contextmanager,'get_cp_strategy':lambda:None};future=ast.parse('from __future__ import annotations').body[0];exec(compile(ast.Module(body=[future,*cp_functions,share],type_ignores=[]),'frozen-state-source','exec'),state_env);State=state_env['IndexTopKShareState'];CP=types.SimpleNamespace(is_cp_active=state_env['is_cp_active']);result['state_execution']={'mode':'SOURCE_EXTRACTED_CLASS_NOT_FULL_MODULE','class_body_ast_sha256':bodyhash(share),'body_unchanged':True,'CP_helper_bodies_unchanged':True,'CP_strategy_fixture':None,'full_module_status':'L017-R005 UNAVAILABLE: missing sgl_kernel CPU extension; not resolved here'}
 enum_tree=ast.parse((root/'python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py').read_text());enum_class=copy.deepcopy(next(n for n in enum_tree.body if isinstance(n,ast.ClassDef) and n.name=='AttnForwardMethod'));route_tree=ast.parse((root/'python/sglang/srt/models/deepseek_common/attention_backend_handler.py').read_text());route=copy.deepcopy(next(n for n in route_tree.body if isinstance(n,ast.FunctionDef) and n.name=='handle_attention_ascend'));route_env={'IntEnum':enum.IntEnum,'auto':enum.auto};exec(compile(ast.Module(body=[enum_class,route],type_ignores=[]),'frozen-route-source','exec'),route_env);H=types.SimpleNamespace(handle_attention_ascend=route_env['handle_attention_ascend']);registrations=[n for n in ast.walk(route_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='register' and len(n.args)==2 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='ascend'];assert len(registrations)==1 and isinstance(registrations[0].args[1],ast.Name) and registrations[0].args[1].id=='handle_attention_ascend';result['route_execution']={'mode':'SOURCE_EXTRACTED_FUNCTION_AND_ENUM','function_body_ast_sha256':bodyhash(route),'body_unchanged':True,'registry':'Static registration verified; runtime registry module not imported'}
 loaded={}
 for mod in [M,F,I,rt,MC,D]:
  f=pathlib.Path(mod.__file__);relative=str(f.relative_to(root));assert relative in expected and sha(f)==expected[relative];loaded[mod.__name__]={'path':str(f),'sha256':sha(f)}
 result['loaded_modules']=loaded;result['environment']['torch']=torch.__version__;config=types.SimpleNamespace(**json.loads((model/'config.json').read_text()));assert MC.is_deepseek_dsa(config) and config.index_share_for_mtp_iteration;model_width=MC.get_dsa_mtp_topk_width(config);assert model_width==2048;result['model_reference']={'index_share_for_mtp_iteration':True,'index_topk':config.index_topk,'index_kpool':getattr(config,'index_kpool',1),'seed_width':model_width,'scope':'Hashedconfig-only shape source; no modelweights or activeNPUconfig'}
 stage='source-branch';native_path=root/p['npu_module'];tree=ast.parse(native_path.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forward_dsa_prepare_npu');gate=next(n for n in func.body if isinstance(n,ast.If) and 'm.skip_topk' in ast.unparse(n.test) and 'prev_topk_indices' in ast.unparse(n.test));helper=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_use_dsa_dcp_partial_attention'));branch_sha=hashlib.sha256(ast.dump(gate,include_attributes=False).encode()).hexdigest();fn=ast.parse('def source_gate(m,prev_topk_indices,forward_batch,hidden_states):\n q_lora=positions=layer_scatter_modes=dynamic_scale=None').body[0];fn.body += [copy.deepcopy(gate),ast.Return(value=ast.Name(id='topk_indices',ctx=ast.Load()))];ast.fix_missing_locations(fn);env={'get_parallel':rt.get_parallel,'dsa_use_prefill_cp':D.dsa_use_prefill_cp};exec(compile(ast.Module(body=[helper,fn],type_ignores=[]),str(native_path),'exec'),env);source_gate=env['source_gate'];result['extracted_npu_branch']={'function':'forward_dsa_prepare_npu','branch_ast_sha256':branch_sha,'body_unchanged':True,'scope':'Branch plus actual DCPpredicate only, CPinactive; m.indexer is spy, no nativeoperator'}
 def equal(a,b):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 def summary(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
 def values(n,w,offset):return (torch.arange(n*w,dtype=torch.int32).reshape(n,w)+offset)%32768
 def fb(spec,mode,reuse=False):return types.SimpleNamespace(spec_info=spec,forward_mode=mode,reuse_dsa_topk_indices=reuse,input_ids=None,attn_cp_metadata=None)
 class Spy:
  def __init__(self,value,skip=True,nextn=True):self.value=value;self.calls=0;self.skip_topk=skip;self.is_nextn=nextn;self.layer_id=0
  def indexer(self,*unused):self.calls+=1;return self.value
 class ExpectedUnwind(Exception):pass
 stage='actual-CPU-state'
 with rt.get_context().override_server_args(device='cpu',tp_size=1,dp_size=1,pp_size=1,dcp_size=1,speculative_algorithm='EAGLE',speculative_eagle_topk=1,speculative_num_steps=3,speculative_num_draft_tokens=4):
  assert not rt.get_parallel().dcp_enabled
  for case in p['pipeline_cases']:
   result['in_progress']=case;n=case['batch'];w=case['width'];seed=values(n,w,7) if case['initial_seed'] else None;spec=I.EagleDraftInput(dsa_topk_indices=seed);batch=fb(spec,FM.DECODE);spy=Spy(values(n,w,101));observations=[];assert not CP.is_cp_active(batch)
   with State.mtp_iteration(batch,enabled=case['share'],keep_carry_seed=case['keep']) as context:
    assert (context is None)==(not case['share'])
    for step in range(3):
     state=State.from_mtp_carry(batch);before=state.topk_indices;out=source_gate(spy,before,batch,None);assert out is (spy.value if before is None else before);state.update(out);state.publish();observations.append({'step':step,'indexer_spy_calls_so_far':spy.calls,'had_previous':before is not None,'published_alias':spec.dsa_topk_indices is out if case['share'] else False})
   want_calls=3 if not case['share'] else (0 if case['keep'] and seed is not None else 1);assert spy.calls==want_calls
   if case['share']:assert spec.dsa_topk_indices is None and not batch.reuse_dsa_topk_indices
   else:assert spec.dsa_topk_indices is seed and not batch.reuse_dsa_topk_indices
   result['pipeline_cases'].append({**case,'status':'PASS','expected_spy_calls':want_calls,'actual_spy_calls':spy.calls,'steps':observations,'exit_cleanup_or_disabled_preservation':True,'scope':'Extractedstate source + sourcegate selection count, not fullmodule/nativeindexer count/time'});result.pop('in_progress')
  for case in p['capture_cases']:
   result['in_progress']=case;n=case['batch'];w=case['width'];selected=case['selected'];source=values(n*4,w,201);sel=torch.tensor([i*4+3 for i in range(n)],dtype=torch.int64) if selected else None;rows=n if selected else n*4;buf=torch.full((rows,w),-1,dtype=torch.int32);ptr=buf.data_ptr();old=values(n,w,9);spec=I.EagleDraftExtendInput(hidden_states=None,dsa_seed_topk_capture=buf,dsa_seed_topk_select=sel);spec.dsa_topk_indices=old;mode=getattr(FM,case['mode']);batch=fb(spec,mode,case['reuse']);assert not CP.is_cp_active(batch);state=State.from_mtp_carry(batch);eligible=case['mode'] in ['EXTEND','DRAFT_EXTEND_V2','TARGET_VERIFY'];want_publish=case['reuse'] or eligible;assert state.should_publish==want_publish;state.update(source);state.publish();assert state.topk_indices is source and buf.data_ptr()==ptr
   if case['reuse']:assert spec.dsa_topk_indices is source
   else:assert spec.dsa_topk_indices is old
   if eligible:equal(buf,source[sel] if selected else source)
   else:assert torch.all(buf==-1).item()
   before=buf.clone();source.fill_(77);equal(buf,before)
   result['capture_cases'].append({**case,'status':'PASS','capture_eligible':eligible,'should_publish':want_publish,'stable_capture_pointer':True,'capture_copy_independent':True,'carry_alias_intentional':case['reuse'],'capture':summary(buf)});result.pop('in_progress')
  for case in p['truth_cases']:
   seed=values(2,8,3) if case['has_prev'] else None;spy=Spy(values(2,8,13),case['skip'],case['nextn']);batch=fb(I.EagleDraftInput(),FM.DECODE);out=source_gate(spy,seed,batch,None);expect=(not case['skip']) or (case['nextn'] and seed is None);assert spy.calls==int(expect);assert out is (spy.value if expect else seed);result['truth_cases'].append({**case,'status':'PASS','spy_calls':spy.calls,'returns_none':out is None,'scope':'Sourcepredicate fixture, not nativeor reachableconfiguration proof'})
  for n in p.get('cleanup_batches',[2,3]):
   spec=I.EagleDraftInput(dsa_topk_indices=values(n,model_width,4));batch=fb(spec,FM.DECODE)
   try:
    with State.mtp_iteration(batch,keep_carry_seed=True):
     state=State.from_mtp_carry(batch);state.update(values(n,model_width,5));state.publish();raise ExpectedUnwind()
   except ExpectedUnwind:pass
   assert not batch.reuse_dsa_topk_indices and spec.dsa_topk_indices is None;result['cleanup_cases'].append({'batch':n,'width':model_width,'status':'PASS','exception_propagated_and_finally_cleared':True})
  worker_tree=ast.parse((root/'python/sglang/srt/speculative/eagle_worker_v2.py').read_text());methods=[m for c in worker_tree.body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_draft_extend_for_decode'];assignments=[n for m in methods for n in m.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='select_index' for t in n.targets)];assert len(assignments)==1;assign=copy.deepcopy(assignments[0]);select_fn=ast.parse('def actual_select(self,batch,batch_result):\n pass').body[0];select_fn.body=[assign,ast.Return(value=ast.Name(id='select_index',ctx=ast.Load()))];ast.fix_missing_locations(select_fn);sel_env={'torch':torch};exec(compile(ast.Module(body=[select_fn],type_ignores=[]),'actual-select-expression','exec'),sel_env);result['select_expression_ast_sha256']=hashlib.sha256(ast.dump(assign,include_attributes=False).encode()).hexdigest()
  for case in p['cycle_cases']:
   result['in_progress']=case;n=case['batch'];w=2048;carry=values(n,w,31) if case['initial_seed'] else None;rounds=[]
   for turn in range(2):
    spec=I.EagleDraftInput(dsa_topk_indices=carry);batch=fb(spec,FM.DECODE);draft_spy=Spy(values(n,w,101+turn));had_seed=carry is not None
    with State.mtp_iteration(batch,enabled=case['share'],keep_carry_seed=True):
     for step in range(3):
      state=State.from_mtp_carry(batch);out=source_gate(draft_spy,state.topk_indices,batch,None);state.update(out);state.publish()
    expected_draft=3 if not case['share'] else (0 if had_seed else 1);assert draft_spy.calls==expected_draft and not batch.reuse_dsa_topk_indices
    if case['share']:assert spec.dsa_topk_indices is None
    accept=torch.tensor([1,4,2][:n],dtype=torch.int32);ctx_batch=types.SimpleNamespace(seq_lens=torch.full((n,),32768,dtype=torch.int64));selector=sel_env['actual_select'](types.SimpleNamespace(speculative_num_draft_tokens=4,device='cpu'),ctx_batch,types.SimpleNamespace(accept_lens=accept));assert selector.tolist()==[i*4+a-1 for i,a in enumerate(accept.tolist())]
    capture=torch.full((n*4,w),-1,dtype=torch.int32) if case['share'] else None;extend_info=I.EagleDraftExtendInput(hidden_states=None,num_correct_drafts=accept-1,num_accept_tokens=accept,dsa_seed_topk_capture=capture,dsa_seed_topk_select=None);extend=fb(extend_info,FM.DRAFT_EXTEND_V2);ext_values=values(n*4,w,401+turn);ext_spy=Spy(ext_values);state=State.from_mtp_carry(extend);assert state.topk_indices is None;state.update(source_gate(ext_spy,None,extend,None));state.publish();assert ext_spy.calls==1
    if case['share']:equal(capture,ext_values);carry=capture[selector];equal(carry,ext_values[selector]);saved=carry.clone();capture.zero_();equal(carry,saved)
    else:carry=None
    rounds.append({'turn':turn,'had_carry_seed':had_seed,'draft_gate_spy_calls':draft_spy.calls,'extend_gate_spy_calls':ext_spy.calls,'total_spy_calls':draft_spy.calls+ext_spy.calls,'accept_lens_fixture':accept.tolist(),'select_index':selector.tolist(),'next_seed':summary(carry) if carry is not None else None,'fixture_rows':{'draft_per_step':n,'draft_extend':n*4},'scope':'Selection counts and row fixtures only; actualnative/padded/raggedwork andtime unknown'})
   result['cycle_cases'].append({**case,'status':'PASS','rounds':rounds,'next_round_seed_owned_after_capture_reuse':case['share']});result.pop('in_progress')
  for case in p['route_cases']:
   batch=types.SimpleNamespace(forward_mode=getattr(FM,case['mode']),seq_lens=torch.tensor([case['context']],dtype=torch.int64));attn=types.SimpleNamespace(use_dsa=case['dsa']);route=H.handle_attention_ascend(attn,batch);want='DSA_NPU' if case['dsa'] else ('MHA_NPU' if case['mode']=='EXTEND' else 'MLA_NPU');assert route.name==want;result['route_cases'].append({**case,'status':'PASS','actual_route':route.name,'scope':'Source-extracted handler/enum with actualForwardMode framefixture, not runtime registry/model execution'})
 assert 'torch_npu' not in sys.modules
 result.update({'probe_status':'PASS','correctness':'PASS','counts':{k:len(result[k]) for k in ['pipeline_cases','capture_cases','truth_cases','cleanup_cases','route_cases','cycle_cases']},'CP':'Inactive verified through actualhelper; no gather tested','native_executed':False,'performance':'NOT_MEASURED'})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in models.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
