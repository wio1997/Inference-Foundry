#!/usr/bin/env python3
"""Unchanged W8A8 methods on CPU fixtures; format/quant/GMM boundaries are spies."""
import argparse,ast,copy,datetime,hashlib,json,os,pathlib,platform,subprocess,sys,traceback,types
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);model=pathlib.Path('/model');protocol_path=pathlib.Path(args.protocol);p=json.loads(protocol_path.read_text());expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1'
nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit'];hashes={}
for f,h in expected.items():
 hashes[f]=sha(root/f);assert hashes[f]==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(model/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':hashes,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(protocol_path),'accelerator_device_nodes':nodes,'environment':{'python':sys.version,'machine':platform.machine()},'load_cases':[],'apply_cases':[],'expected_rejections':[],'source_bodies':{},'scope':'Extracted unmodified W8A8 load/apply and base helper bodies with actual TorchCPU Parameters/casts/transposes. Constructor not run. FuseEP=False, format_cast identity, activation quantizer and GMM are explicit boundary fixtures. No native format/quant/GMM values, fullmodule/registry, weights or performance.'};stage='torch-import'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);result['environment']['torch']=torch.__version__;config=json.loads((model/'config.json').read_text());assert config['hidden_size']==6144 and config['moe_intermediate_size']==2048
 tree=ast.parse((root/p['method_source']).read_text());base=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_NPUMoEMethodBase');w8=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NPUW8A8Int8MoEMethod');helpers=[next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['_set_dispatcher_output_dtype','_validate_weight_prefix','_get_bias_args']];methods=[next(n for n in w8.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['process_weights_after_loading','apply']]
 for node in helpers+methods:result['source_bodies'][node.name]={'body_ast_sha256':hashlib.sha256(ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False).encode()).hexdigest(),'body_unchanged':True,'decorators_preserved':True}
 class FormatSpy:
  def __init__(self):self.calls=[]
  def __call__(self,t):self.calls.append(t);return t
 format_spy=FormatSpy();env={'torch':torch,'npu_format_cast':format_spy};cls=ast.ClassDef(name='ExtractedW8Methods',bases=[],keywords=[],body=copy.deepcopy(helpers+methods),decorator_list=[]);ast.fix_missing_locations(cls);exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0],cls],type_ignores=[]),'unchanged-w8-method-bodies','exec'),env);Methods=env['ExtractedW8Methods']
 class DispatchFixture:
  def __init__(self):self.calls=[]
  def set_quant_config(self,c):self.calls.append(c.copy())
 def method():
  m=Methods();m.maybe_process_fuseep_weights=lambda layer:False;return m
 def dims(prefix):return (16,16) if prefix=='w13' else (16,8)
 stage='static-load'
 for case in p['load_cases']:
  result['in_progress']=case;prefix=case['prefix'];N=case['local_experts'];a,b=dims(prefix);layer=torch.nn.Module();layer.dispatcher=DispatchFixture();w=(torch.arange(N*a*b,dtype=torch.int32)%127-63).to(torch.int8).reshape(N,a,b);s=(torch.arange(N*a,dtype=torch.float32)/17+0.001).reshape(N,a,1).to(getattr(torch,case['scale_dtype']));layer.register_parameter(prefix+'_weight',torch.nn.Parameter(w,requires_grad=False));layer.register_parameter(prefix+'_weight_scale',torch.nn.Parameter(s,requires_grad=False));oldw=layer.get_parameter(prefix+'_weight');olds=layer.get_parameter(prefix+'_weight_scale');original_weight=w.clone();original_scale=s.clone()
  if case['offset']:
   off=torch.arange(N*a,dtype=torch.float32).reshape(N,a,1)/101;layer.register_parameter(prefix+'_weight_offset',torch.nn.Parameter(off,requires_grad=False));oldoff=layer.get_parameter(prefix+'_weight_offset')
  format_spy.calls.clear();m=method();m.process_weights_after_loading(layer,prefix);neww=layer.get_parameter(prefix+'_weight');news=layer.get_parameter(prefix+'_weight_scale');assert neww is oldw and torch.equal(neww,original_weight.transpose(1,2).contiguous()) and neww.is_contiguous();assert news.dtype==torch.bfloat16 and news.shape==(N,a) and torch.equal(news,original_scale.squeeze(-1).to(torch.bfloat16)) and not news.requires_grad;assert len(format_spy.calls)==1 and format_spy.calls[0].data_ptr()==neww.data_ptr();scale_alias=news.data_ptr()==olds.data_ptr();assert scale_alias==(case['scale_dtype']=='bfloat16')
  if case['offset']:assert layer.get_parameter(prefix+'_weight_offset').data_ptr()==oldoff.data_ptr() and layer.get_parameter(prefix+'_weight_offset').shape==(N,a)
  assert layer.dispatcher.calls==([{'dispatcher_output_dtype':'int8'}] if prefix=='w13' else [])
  result['load_cases'].append({**case,'status':'PASS','fixture_hidden':16,'fixture_intermediate':8,'weight_shape_before':list(original_weight.shape),'weight_shape_after':list(neww.shape),'scale_shape_after':list(news.shape),'scale_dtype_after':str(news.dtype),'scale_storage_alias':scale_alias,'offset_squeeze_alias':case['offset'],'format_spy_calls':1,'dispatcher_quant_config_calls':layer.dispatcher.calls,'scope':'One designated load-method call, real CPU static transforms; nativeformat identity fixture, model loading cadence notexecuted'});result.pop('in_progress')
 class QuantSpy:
  def __init__(self):self.calls=[]
  def __call__(self,x):
   q=torch.zeros_like(x,dtype=torch.int8);s=torch.ones(x.shape[0],dtype=torch.float32);self.calls.append((x,q,s));return q,s
 class MatmulSpy:
  def __init__(self):self.calls=[]
  def forward(self,info,prefix,x,counts,out_dtype,**kw):
   self.calls.append({'info':info,'prefix':prefix,'x':x,'counts':counts,'out_dtype':out_dtype,'kwargs':kw});return torch.empty((x.shape[0],16),dtype=out_dtype)
 stage='apply-boundaries'
 for case in p['apply_cases']:
  result['in_progress']=case;prefix=case['prefix'];T=case['token_rows'];rows=T*8;N=2;cols=16 if prefix=='w13' else 8;m=method();m.hidden_states_quantizer=QuantSpy();m.matmul=MatmulSpy();static_scale=torch.nn.Parameter(torch.ones((N,16),dtype=torch.bfloat16),requires_grad=False);info=types.SimpleNamespace(**{prefix+'_weight_scale':static_scale});sb=torch.ones((N,16),dtype=torch.float32);wb=sb*2
  if case['bias'] in ['scale','both']:setattr(info,prefix+'_scale_bias',sb)
  if case['bias'] in ['weight','both']:setattr(info,prefix+'_weight_bias',wb)
  supplied_scale=torch.ones(rows,dtype=torch.float32) if case['scale_present'] else None;original_scale_ptr=static_scale.data_ptr();counts=torch.tensor([rows,0],dtype=torch.int64)
  for repeat in range(3):
   x=torch.full((rows,cols),repeat,dtype=torch.int8 if case['scale_present'] else torch.bfloat16);out=m.apply(info,x,counts,supplied_scale,torch.bfloat16,prefix,group_list_type=1);call=m.matmul.calls[-1];kw=call['kwargs'];assert call['info'] is info and call['counts'] is counts and call['prefix']==prefix and call['out_dtype']==torch.bfloat16 and kw['group_list_type']==1 and kw['transposed'] is True and kw['scale'][0] is static_scale and static_scale.data_ptr()==original_scale_ptr
   if case['scale_present']:assert call['x'] is x and kw['per_token_scale'][0] is supplied_scale and not m.hidden_states_quantizer.calls
   else:assert call['x'] is m.hidden_states_quantizer.calls[-1][1] and kw['per_token_scale'][0] is m.hidden_states_quantizer.calls[-1][2]
   if case['bias']=='none':assert 'bias' not in kw
   else:assert kw['bias'][0] is (sb if case['bias'] in ['scale','both'] else wb)
   assert out.dtype==torch.bfloat16
  expected_quant=0 if case['scale_present'] else 3;assert len(m.hidden_states_quantizer.calls)==expected_quant and len(m.matmul.calls)==3
  result['apply_cases'].append({**case,'status':'PASS','iterations':3,'dynamic_quant_boundary_calls':expected_quant,'gmm_boundary_calls':3,'static_scale_same_object_and_storage':True,'bias_source':'scale_bias' if case['bias'] in ['scale','both'] else ('weight_bias' if case['bias']=='weight' else None),'scope':'Nativequant/GMM replacedbyspies, only branch/argument/lifetime evidence; no numeric matmul or quantization'});result.pop('in_progress')
 for prefix in ['w13','w2']:
  layer=torch.nn.Module();scale=torch.nn.Parameter(torch.ones((2,16,1)),requires_grad=False);layer.register_parameter(prefix+'_weight_scale',scale);format_spy.calls.clear()
  try:method().process_weights_after_loading(layer,prefix)
  except AttributeError as exc:assert layer.get_parameter(prefix+'_weight_scale') is scale and not format_spy.calls;result['expected_rejections'].append({'prefix':prefix,'status':'PASS','error':str(exc),'scope':'Missing-weightfixture rejected beforestaticmutation, not actualmodelloadfailure'})
  else:raise AssertionError('Missing weight not rejected')
 assert 'torch_npu' not in sys.modules and 'sgl_kernel_npu' not in sys.modules
 result.update({'probe_status':'PASS','correctness':'PASS','counts':{k:len(result[k]) for k in ['load_cases','apply_cases','expected_rejections']},'native_executed':False,'fullmodule_imported':False,'performance':'NOT_MEASURED','metadata_reference':p['quant_metadata_reference'],'model_or_real_weights_loaded':False})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in models.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
