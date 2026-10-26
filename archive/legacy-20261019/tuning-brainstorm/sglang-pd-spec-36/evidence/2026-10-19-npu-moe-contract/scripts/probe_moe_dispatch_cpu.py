#!/usr/bin/env python3
"""Source class/wrapper lifecycle with actual CPU casts and explicit native-op spies."""
import argparse,ast,copy,datetime,enum,hashlib,importlib.util,json,os,pathlib,platform,subprocess,sys,traceback,types,typing
from abc import ABC,abstractmethod
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);model=pathlib.Path('/model');protocol_path=pathlib.Path(args.protocol);p=json.loads(protocol_path.read_text());expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1'
nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit'];hashes={}
for f,h in expected.items():
 hashes[f]=sha(root/f);assert hashes[f]==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(model/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':hashes,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(protocol_path),'accelerator_device_nodes':nodes,'environment':{'python':sys.version,'machine':platform.machine()},'basic_cases':[],'lifecycle_cases':[],'mutation_cases':[],'hook_cases':[],'source_bodies':{},'scope':'Complete BaseDispatcher file loaded by file-spec, without parent package startup. Ascend dispatcher/classes/native wrapper bodies extracted unchanged. Real Torch CPU casts; torch.ops.npu replaced only in source environment by explicit spies. No native permutation, quantization, finalize numeric values, runtime module registry, transport, model or performance.'};stage='torch-import'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);result['environment']['torch']=torch.__version__
 config=json.loads((model/'config.json').read_text());H=config['hidden_size'];E=config['n_routed_experts'];assert H==6144 and E==256 and config['num_experts_per_tok']==8;result['model_reference']={'hidden':H,'routed_experts':E,'routed_topk':8,'scope':'Config-only dimensions; T=2/3 token rows and k=9 fused-shared layout are fixtures, not actual runtime batch/placement.'}
 spec=importlib.util.spec_from_file_location('if_actual_dispatch_base',root/p['base_source']);base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base);assert sha(pathlib.Path(base.__file__))==expected[p['base_source']];result['base_module']={'path':base.__file__,'sha256':sha(pathlib.Path(base.__file__)),'complete_file_loaded':True,'parent_package_startup':False}
 future=ast.parse('from __future__ import annotations').body[0]
 def bodyhash(node):return hashlib.sha256(ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
 def selected(path,names):
  tree=ast.parse((root/path).read_text());return [next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name) for name in names]
 def execute(nodes,env,label):
  for n in nodes:result['source_bodies'][label+':'+n.name]={'body_ast_sha256':bodyhash(n),'body_unchanged':True}
  exec(compile(ast.Module(body=[future,*copy.deepcopy(nodes)],type_ignores=[]),label,'exec'),env)
 events=[]
 class ExpectedNativeBoundaryError(Exception):pass
 class NativeSpy:
  def __init__(self):self.init_calls=[];self.final_calls=[];self.fail_init=False;self.fail_final=False;self.current=None;self.result=None
  def npu_moe_init_routing_v2(self,hidden,ids,**kw):
   events.append('native-init-spy')
   if self.fail_init:self.fail_init=False;raise ExpectedNativeBoundaryError('injected init boundary')
   T,W=hidden.shape;K=ids.shape[1];rows=T*K;assert ids.dtype==torch.int32 and kw=={'active_num':rows,'expert_num':self.expert_num,'expert_tokens_num_type':1,'expert_tokens_num_flag':True,'active_expert_range':[0,self.expert_num],'quant_mode':self.quant_mode}
   dtype={-1:torch.bfloat16,1:torch.int8,3:torch.float8_e4m3fn}[self.quant_mode];out=torch.empty((rows,W),dtype=dtype);idx=torch.arange(rows,dtype=torch.int32);counts=torch.bincount(ids.reshape(-1).to(torch.int64),minlength=self.expert_num).to(torch.int32);scale=torch.ones((rows,W//32),dtype=torch.uint8) if self.quant_mode==3 else torch.ones(rows,dtype=torch.float32);rec={'hidden':hidden,'ids':ids,'out':out,'idx':idx,'counts':counts,'scale':scale,'kwargs':kw};self.init_calls.append(rec);self.current=rec;return out,idx,counts,scale
  def npu_moe_finalize_routing(self,hidden,**kw):
   events.append('native-finalize-spy');self.final_calls.append({'hidden':hidden,**kw})
   if self.fail_final:self.fail_final=False;raise ExpectedNativeBoundaryError('injected finalize boundary')
   assert all(kw[k] is None for k in ['skip1','skip2','bias']) and kw['drop_pad_mode']==2
   T,W=self.current['hidden'].shape;self.result=torch.full((T,W),7,dtype=self.current['hidden'].dtype);return self.result
 class TorchProxy:
  def __init__(self,spy):self.ops=types.SimpleNamespace(npu=spy)
  def __getattr__(self,name):return getattr(torch,name)
 def make():
  spy=NativeSpy();module=types.ModuleType('if_extracted_ascend_dispatch_'+str(len(result['basic_cases'])+len(result['lifecycle_cases'])+len(result['mutation_cases'])+len(result['hook_cases'])));sys.modules[module.__name__]=module;env=module.__dict__;env.update({'torch':TorchProxy(spy),'ABC':ABC,'abstractmethod':abstractmethod,'NamedTuple':typing.NamedTuple,'Optional':typing.Optional,'BaseDispatcher':base.BaseDispatcher,'DispatchOutputFormat':base.DispatchOutputFormat,'CombineInputFormat':base.CombineInputFormat,'Enum':enum.Enum,'MXFP8_QUANT_MODE':3,'get_parallel':lambda:types.SimpleNamespace(tp_size=1)})
  execute(selected(p['utils_source'],['DispatcherOutputDtype','get_ascend_dispatcher_output_dtype']),env,'source-dtype')
  execute(selected(p['init_source'],['_normalize_mxfp_scale','BaseInitRouting','NPUMoEInitRouting_v2']),env,'source-init')
  execute(selected(p['final_source'],['BaseFinalizeRouting','NPUFinalizeRouting']),env,'source-final')
  execute(selected(p['dispatcher_source'],['AscendTPDispatchOutput','AscendTPCombineInput','AscendTPDispatcher']),env,'source-dispatch')
  return env,spy
 def setup(T,K,dtype_name='bf16',wd='float32',idd='int64'):
  env,spy=make();N=E+(1 if K==9 else 0);d=env['AscendTPDispatcher'](types.SimpleNamespace(num_experts=N,top_k=K));assert d.init.quant_mode==-1 and d.group_list_type==1;d.set_quant_config({'dispatcher_output_dtype':dtype_name});spy.quant_mode=d.init.quant_mode;spy.expert_num=N
  hidden=torch.zeros((T,H),dtype=torch.bfloat16);weights=torch.linspace(0.1,0.9,T*K,dtype=getattr(torch,wd)).reshape(T,K);ids=torch.arange(T*K,dtype=getattr(torch,idd)).reshape(T,K)%E
  if K==9:ids[:,-1]=E
  return env,spy,d,hidden,weights,ids
 def dispatch(d,h,w,ids):return d.dispatch(h,(w,ids,None))
 def combine(env,d,T):return d.combine(env['AscendTPCombineInput'](torch.zeros((T,H),dtype=torch.bfloat16)))
 def expect_error(fn,kind):
  try:fn()
  except kind:return
  raise AssertionError('Expected boundary error absent')
 stage='basic'
 for case in p['basic_cases']:
  result['in_progress']=case;T=case['token_rows'];K=case['topk'];env,spy,d,h,w,ids=setup(T,K,case['output_dtype'],case['weight_dtype'],case['id_dtype']);expect_error(lambda:combine(env,d,T),RuntimeError);out=dispatch(d,h,w,ids);raw=spy.init_calls[-1];assert out is d._dispatch_output and out.hidden_states is raw['out'] and out.expanded_row_idx is raw['idx'];assert out.expert_tokens.dtype==torch.int64 and torch.equal(out.expert_tokens,raw['counts'].to(torch.int64));assert torch.equal(out.topk_weights,w.to(h.dtype)) and torch.equal(out.topk_ids,ids.to(torch.int32));wa=out.topk_weights is w;ia=out.topk_ids is ids;assert wa==(w.dtype==h.dtype) and ia==(ids.dtype==torch.int32)
  if spy.quant_mode==-1:assert out.hidden_states_scale is None
  elif spy.quant_mode==3:assert list(out.hidden_states_scale.shape)==[T*K,H//64,2] and out.hidden_states_scale.data_ptr()==raw['scale'].data_ptr()
  else:assert out.hidden_states_scale is raw['scale']
  assert out.format is base.DispatchOutputFormat.ASCEND_TP;final=combine(env,d,T);got=spy.final_calls[-1];assert final is spy.result and d._dispatch_output is None and got['scales'] is out.topk_weights and got['export_for_source_row'] is out.topk_ids and got['expanded_src_to_dst_row'] is out.expanded_row_idx;expect_error(lambda:combine(env,d,T),RuntimeError)
  result['basic_cases'].append({**case,'status':'PASS','quant_mode':spy.quant_mode,'active_num':T*K,'num_experts_fixture':spy.expert_num,'weight_alias':wa,'id_alias':ia,'scale_shape':list(out.hidden_states_scale.shape) if out.hidden_states_scale is not None else None,'scale_fixture_dtype':str(out.hidden_states_scale.dtype) if out.hidden_states_scale is not None else None,'combine_clears_pending':True,'init_and_finalize_calls':[len(spy.init_calls),len(spy.final_calls)]});result.pop('in_progress')
 stage='lifecycle'
 for case in p['lifecycle_cases']:
  T=case['token_rows'];env,spy,d,h,w,ids=setup(T,8);first=dispatch(d,h,w,ids);scenario=case['scenario']
  if scenario=='double_dispatch':
   second=dispatch(d,h,w+0.25,(ids+1)%E);assert second is d._dispatch_output and second is not first;combine(env,d,T);assert spy.final_calls[-1]['scales'] is second.topk_weights;observation='Second dispatch overwrites single pending slot; first output object still exists. Interleaving fixture only.'
  elif scenario=='init_failure':
   spy.fail_init=True;expect_error(lambda:dispatch(d,h,w,ids),ExpectedNativeBoundaryError);assert d._dispatch_output is first;combine(env,d,T);assert spy.final_calls[-1]['scales'] is first.topk_weights;observation='Injected init failure retains previous pending object. Subsequent fixture combine reads it; real native recovery not tested.'
  elif scenario=='finalize_failure':
   spy.fail_final=True;expect_error(lambda:combine(env,d,T),ExpectedNativeBoundaryError);assert d._dispatch_output is first;combine(env,d,T);assert d._dispatch_output is None;observation='Injected finalize failure retains pending state; success clears. No production retry recommendation.'
  else:raise AssertionError('Unknown lifecycle case')
  result['lifecycle_cases'].append({**case,'status':'PASS','observation':observation,'scope':'Conditional boundary failure/interleaving construction, no observed active bug'})
 stage='ownership-mutation'
 for case in p['mutation_cases']:
  T=case['token_rows'];env,spy,d,h,w,ids=setup(T,8,'bf16',case['weight_dtype'],case['id_dtype']);out=dispatch(d,h,w,ids);oldw=out.topk_weights.clone();oldi=out.topk_ids.clone();w.add_(0.25);ids.add_(1);wa=w.dtype==h.dtype;ia=ids.dtype==torch.int32;assert torch.equal(out.topk_weights,w if wa else oldw) and torch.equal(out.topk_ids,ids if ia else oldi);combine(env,d,T)
  result['mutation_cases'].append({**case,'status':'PASS','original_weight_mutation_visible':wa,'original_id_mutation_visible':ia,'scope':'Intentional producer mutation after dispatch; no native correctness claim or upstream violation observed'})
 stage='hooks'
 for T in p['hook_rows']:
  env,spy,d,h,w,ids=setup(T,8);events.clear()
  def pre_d(*unused):events.append('pre-dispatch')
  def post_d(*unused):events.append('post-dispatch')
  def pre_c(*unused):events.append('pre-combine')
  def post_c(*unused):events.append('post-combine')
  handles=[d.register_pre_dispatch_hook(pre_d),d.register_post_dispatch_hook(post_d),d.register_pre_combine_hook(pre_c),d.register_post_combine_hook(post_c)];dispatch(d,h,w,ids);combine(env,d,T);assert events==['pre-dispatch','native-init-spy','post-dispatch','pre-combine','native-finalize-spy','post-combine'];with_hooks=events.copy()
  for handle in handles:handle.remove()
  events.clear();dispatch(d,h,w,ids);combine(env,d,T);assert events==['native-init-spy','native-finalize-spy'];result['hook_cases'].append({'token_rows':T,'status':'PASS','order':with_hooks,'removed_hooks_not_called':True,'scope':'Actual complete BaseDispatcher hook implementation with source-extracted subclass and spies'})
 assert 'torch_npu' not in sys.modules and 'sgl_kernel_npu' not in sys.modules
 result.update({'probe_status':'PASS','correctness':'PASS','counts':{k:len(result[k]) for k in ['basic_cases','lifecycle_cases','mutation_cases','hook_cases']},'native_executed':False,'full_ascend_module_imported':False,'performance':'NOT_MEASURED','native_spy_payloads':'Constructed fixtures, not NPU permutation/scale/finalize values; MXFP8 scale uint8 fixture tests view shape only.'})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in models.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
