#!/usr/bin/env python3
"""Execute unchanged source bodies against CPU/meta tensor fixtures, never native ops."""
import argparse,ast,copy,datetime,hashlib,itertools,json,os,pathlib,platform,subprocess,sys,traceback,types
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);fixture=pathlib.Path(args.fixture_root);model=pathlib.Path('/model');protocol_path=pathlib.Path(args.protocol);p=json.loads(protocol_path.read_text());expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1'
nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit'];hashes={}
for f,h in expected.items():
 hashes[f]=sha(root/f);assert hashes[f]==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(model/f)==h
vendor=pathlib.Path(p['meta_source_path']);assert sha(vendor)==p['meta_source_sha256']==sha(fixture/'fixtures/L018-meta.py')
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':hashes,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(protocol_path),'accelerator_device_nodes':nodes,'environment':{'python':sys.version,'machine':platform.machine()},'meta_cases':[],'compact_gate_cases':[],'pd_view_cases':[],'expected_rejections':[],'extracted_bodies':{},'scope':'Unmodified source bodies only, with real TorchCPU/meta tensors and explicit object/getter fixtures. No full module or runtime registry import, NPU constructor/allocation/writes, transport, index values or performance.'}
stage='torch-import'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);result['environment']['torch']=torch.__version__
 future=ast.parse('from __future__ import annotations').body[0]
 def bodyhash(node):return hashlib.sha256(ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
 def compile_nodes(nodes,env,label):
  out=[]
  for n in nodes:
   c=copy.deepcopy(n);c.decorator_list=[];assert bodyhash(c)==bodyhash(n);result['extracted_bodies'][label+':'+n.name]={'body_ast_sha256':bodyhash(n),'body_unchanged':True,'decorators_removed_only':bool(n.decorator_list)};out.append(c)
  exec(compile(ast.Module(body=[future,*out],type_ignores=[]),label,'exec'),env)
 meta_tree=ast.parse(vendor.read_text());metas=[next(n for n in meta_tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['npu_lightning_indexer_forward','npu_quant_lightning_indexer_forward']];meta_env={'torch':torch};compile_nodes(metas,meta_env,'installed-meta')
 result['vendor_meta']={'path_candidate':str(vendor),'sha256':sha(vendor),'git':'UNKNOWN','package_imported':False,'runtime_registry_executed':False}
 indexer_tree=ast.parse((root/p['indexer_source']).read_text());cls=next(n for n in indexer_tree.body if isinstance(n,ast.ClassDef) and n.name=='DSANPUIndexerMixin');fwd=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_npu');adapters={}
 for quant,expr in [(True,'topk_indices.squeeze(1)'),(False,'topk_indices[0].squeeze(1)')]:
  choices=[n for n in ast.walk(fwd) if isinstance(n,ast.Return) and ast.unparse(n.value)==expr];assert len(choices)==1;value=copy.deepcopy(choices[0].value);adapters[quant]=compile(ast.Expression(value),p['indexer_source'],'eval');result['extracted_bodies']['sglang-return:'+str(quant)]={'expression_ast_sha256':hashlib.sha256(ast.dump(value,include_attributes=False).encode()).hexdigest(),'expression_unchanged':True}
 stage='meta-contract'
 for case in p['meta_cases']:
  result['in_progress']=case;n=case['batch'];qpr=case['queries_per_request'];context=case['context'];quant=case['quant'];rows=n*qpr;pages=(context+127)//128+1;dtype=torch.float8_e4m3fn if quant else torch.bfloat16
  query=torch.empty((rows,2,128),device='meta',dtype=dtype);key=torch.empty((pages,128,1,128),device='meta',dtype=dtype);weights=torch.empty((rows,2),device='meta',dtype=torch.bfloat16);common={'actual_seq_lengths_query':torch.arange(qpr,rows+1,qpr,dtype=torch.int32),'actual_seq_lengths_key':torch.full((n,),context,dtype=torch.int32),'block_table':torch.empty((n,pages-1),device='meta',dtype=torch.int32),'layout_query':'TND','layout_key':'PA_BSND','sparse_count':2048,'sparse_mode':3}
  if quant:
   out=meta_env['npu_quant_lightning_indexer_forward'](query,key,weights,torch.empty((rows,2),device='meta'),torch.empty((pages,128,1),device='meta'),0,0,**common);assert isinstance(out,torch.Tensor);contract='Tensor'
  else:
   out=meta_env['npu_lightning_indexer_forward'](query,key,weights,**common);assert isinstance(out,tuple) and len(out)==2 and list(out[1].shape)==[0];contract='(indices,empty_values)'
  adapted=eval(adapters[quant],{'topk_indices':out});assert adapted.device.type=='meta' and adapted.dtype==torch.int32 and list(adapted.shape)==[rows,2048]
  result['meta_cases'].append({**case,'status':'PASS','return_contract':contract,'adapted_shape':list(adapted.shape),'adapted_dtype':str(adapted.dtype),'fixture_query_heads':2,'scope':'Meta inference and source return-expression only; values/validation/nativework unknown'});result.pop('in_progress')
 stage='compact-gate'
 gate_tree=ast.parse((root/p['configurator_source']).read_text());gate=next(n for n in gate_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_should_elide_dsa_index_k');state={};env={'get_memory':lambda:state['memory'],'get_disagg':lambda:state['disagg']};compile_nodes([gate],env,'compact-gate')
 for case in p['gate_cases']:
  state['memory']=types.SimpleNamespace(enable_hisparse=case['hisparse'],enable_hierarchical_cache=case['hicache'],enable_unified_cache_external_linker=case['external_linker']);state['disagg']=types.SimpleNamespace(disaggregation_mode=case['mode']);got=env['_should_elide_dsa_index_k'](is_draft_worker=case['draft']);want=case['mode']=='null' and not any(case[k] for k in ['draft','hisparse','hicache','external_linker']);assert got==want
  if case['mode'] in ['prefill','decode']:assert not got
  result['compact_gate_cases'].append({**case,'status':'PASS','elide_gate':got,'scope':'Source getter fixtures; outer isDSA/arch35 requirements still apply'})
 stage='pd-views'
 pool_tree=ast.parse((root/p['pool_source']).read_text());poolcls=next(n for n in pool_tree.body if isinstance(n,ast.ClassDef) and n.name=='NPUMLATokenToKVPool');wanted=['_get_disagg_buffer_view','get_contiguous_buf_infos','get_dcp_remote_decode_layout'];methods=[next(n for n in poolcls.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in wanted];pool_env={'torch':torch};compile_nodes(methods,pool_env,'pool-methods');Pool=type('PoolFixture',(),{name:pool_env[name] for name in wanted});Pool._raise_if_native_kv_cache_disabled=lambda self:None
 config=json.loads((model/'config.json').read_text());rank=config['kv_lora_rank'];rope=config['qk_rope_head_dim'];index_dim=config['index_head_dim'];assert rank%128==0 and index_dim==128;result['shape_reference']={'kv_lora_rank':rank,'qk_rope_head_dim':rope,'index_head_dim':index_dim,'source':'Hashed config only; actual KV dtype and allocated capacity unknown'}
 for case in p['pd_view_cases']:
  result['in_progress']=case;pool=Pool();pool.layer_num=1 if case['draft'] else 2;pool.is_draft_worker=case['draft'];pool.dcp_size=case['dcp'];pool.page_size=pool.index_page_size=128;pool.dsa_kv_cache_store_fp8=case['layout']=='DSA_FP8';pool.index_head_dim=None if case['layout']=='MLA_BF16' else index_dim;pool.num_indexer_layers=pool.layer_num if pool.index_head_dim is not None else 0
  size=512;kvpages=size//128+(pool.dcp_size if case['draft'] else 1);index_size=size if case['draft'] else size*pool.dcp_size;indexpages=index_size//128+pool.dcp_size;kdim=rank+rank//128*4+rope*2 if pool.dsa_kv_cache_store_fp8 else rank;vdim=0 if pool.dsa_kv_cache_store_fp8 else rope;dtype=torch.float8_e4m3fn if pool.dsa_kv_cache_store_fp8 else torch.bfloat16
  pool.k_buffer=torch.zeros((pool.layer_num,kvpages,128,1,kdim),dtype=dtype);pool.v_buffer=torch.zeros((pool.layer_num,kvpages,128,1,vdim),dtype=torch.bfloat16);pool.index_k_buffer=None if pool.index_head_dim is None else torch.zeros((pool.num_indexer_layers,indexpages,128,1,index_dim),dtype=dtype);pool.index_k_scale_buffer=torch.zeros((pool.num_indexer_layers,indexpages,128,1),dtype=torch.float32) if pool.dsa_kv_cache_store_fp8 else None
  groups=[('K',list(pool.k_buffer),case['draft'])]
  if not pool.dsa_kv_cache_store_fp8:groups.append(('V',list(pool.v_buffer),case['draft']))
  if pool.index_head_dim is not None:groups.append(('indexK',list(pool.index_k_buffer),True))
  if pool.index_k_scale_buffer is not None:groups.append(('indexScale',list(pool.index_k_scale_buffer),True))
  entries=[(kind,t,glob) for kind,ts,glob in groups for t in ts];ptrs,lens,items=pool.get_contiguous_buf_infos();layout=pool.get_dcp_remote_decode_layout();assert len(ptrs)==len(lens)==len(items)==len(layout)==len(entries);observed=[]
  for i,(kind,t,glob) in enumerate(entries):
   factor=pool.dcp_size if glob else 1;want_item=128*factor*t[0,0].numel()*t.element_size();assert ptrs[i]==t.data_ptr() and lens[i]==t.nbytes and items[i]==want_item and layout[i]==glob;view=pool._get_disagg_buffer_view(t,page_size=128,uses_global_slots=glob);assert view.data_ptr()==t.data_ptr() and view.untyped_storage().data_ptr()==t.untyped_storage().data_ptr();observed.append({'kind':kind,'kernel_shape':list(t.shape),'transfer_shape':list(view.shape),'bytes':lens[i],'transfer_item_bytes':items[i],'global_slots':glob,'aliases_storage':True})
  result['pd_view_cases'].append({**case,'status':'PASS','fixture_capacity_tokens':size,'layers':pool.layer_num,'entries':observed,'registered_bytes':sum(lens),'scope':'Extracted pool methods with explicit CPU allocation fixtures following source geometry. Constructor/nativewrites/realallocation/transport notrun.'});result.pop('in_progress')
 for label,t,pg,dcp in [('page-shape',torch.zeros((4,127,1,8)),128,2),('nonintegral-global-pages',torch.zeros((5,128,1,8)),128,2)]:
  pool=Pool();pool.dcp_size=dcp
  try:pool._get_disagg_buffer_view(t,page_size=pg,uses_global_slots=True)
  except RuntimeError as exc:result['expected_rejections'].append({'id':label,'status':'PASS','error':str(exc),'scope':'Explicit malformed fixture, not observed runtime failure'})
  else:raise AssertionError('Malformed page fixture not rejected')
 assert 'torch_npu' not in sys.modules and 'sgl_kernel_npu' not in sys.modules
 result.update({'probe_status':'PASS','correctness':'PASS','counts':{k:len(result[k]) for k in ['meta_cases','compact_gate_cases','pd_view_cases','expected_rejections']},'native_executed':False,'performance':'NOT_MEASURED','full_module_imported':False,'runtime_registry_executed':False})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in models.items():assert sha(model/f)==h
assert sha(vendor)==p['meta_source_sha256'];print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
