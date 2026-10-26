#!/usr/bin/env python3
"""Source-extracted branch predicates, no torch import and no device work."""
import ast,copy,datetime,hashlib,itertools,json,pathlib,types
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/installed/python/sglang/srt';O=R/'runs/L009-R002-layout-cpu';O.mkdir(parents=True,exist_ok=False)
env={};hashes={};locations={}
def source(p):
 b=(S/p).read_bytes();hashes[p]=hashlib.sha256(b).hexdigest();return ast.parse(b)
def get(p,name,cls=None):
 t=source(p);items=t.body if cls is None else next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls).body;n=next(n for n in items if isinstance(n,ast.FunctionDef) and n.name==name);locations[(cls+'.' if cls else '')+name]={'file':p,'line':n.lineno,'end_line':n.end_lineno,'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()};return n
def install(nodes):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*copy.deepcopy(nodes)],type_ignores=[])),'<fixed-qk-predicates>','exec'),env)
model='models/deepseek_v2.py';constructor=get(model,'__init__','DeepseekV2AttentionMLA')
assignments=[n for n in ast.walk(constructor) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='is_neox_style' for t in n.targets) and "'rope_interleave'" in ast.unparse(n.value)];assert len(assignments)==1
f=ast.parse('def style_from_config(config):\n pass').body[0];f.body=[assignments[0],ast.Return(value=ast.Name(id='is_neox_style',ctx=ast.Load()))];install([f])
prepare=get('hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py','forward_dsa_prepare_npu');prep=next(n for n in prepare.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mla_preprocess_used' for t in n.targets));top=next(n for n in prepare.body if isinstance(n,ast.If) and ast.unparse(n.test)=='mla_preprocess_used');neox=next(n for n in top.orelse if isinstance(n,ast.If) and ast.unparse(n.test)=='m.rotary_emb.is_neox_style');alt=next(n for n in neox.body if isinstance(n,ast.If) and ast.unparse(n.test)=='m.alt_stream is not None')
ret=lambda s:ast.Return(value=ast.Constant(value=s))
f=ast.parse('def qk_branch(m, forward_batch):\n pass').body[0];f.body=[copy.deepcopy(prep),ast.If(test=copy.deepcopy(top.test),body=[ret('preprocess')],orelse=[ast.If(test=copy.deepcopy(neox.test),body=[ast.If(test=copy.deepcopy(alt.test),body=[ret('neox-alt')],orelse=[ret('neox-serial')])],orelse=[ret('interleaved')])])];install([f])
NS=types.SimpleNamespace;rows=[]
for enabled,extend,style,has_alt in itertools.product([False,True],repeat=4):
 env['is_mla_preprocess_enabled']=lambda:enabled;m=NS(rotary_emb=NS(is_neox_style=style),alt_stream=object() if has_alt else None);fb=NS(forward_mode=NS(is_extend_or_draft_extend_or_mixed=lambda:extend));got=env['qk_branch'](m,fb);want='preprocess' if enabled and not extend else 'interleaved' if not style else 'neox-alt' if has_alt else 'neox-serial';assert got==want;rows.append({'mlapo_fixture':enabled,'extend_or_draft_extend_or_mixed_fixture':extend,'neox_fixture':style,'alt_present_fixture':has_alt,'selected_source_branch':got})
style_rows=[]
for label,config in [('absent',NS()),('interleaved',NS(rope_interleave=True)),('neox',NS(rope_interleave=False))]:
 got=env['style_from_config'](config);assert got==(label=='neox');style_rows.append({'input':label,'computed_is_neox_style':got})
raw=R/'runs/L001-R001-environment/stdout.json';d=json.loads(raw.read_text());candidate=d['model_candidates'][0];meta=candidate['files'][0];cfg=meta['config'];assert meta['name']=='config.json' and cfg['rope_interleave'] is True
observed={'source_raw_path':str(raw),'source_raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'observed_model_path':candidate['path'],'model_config_file_sha256':meta['sha256'],'architectures':cfg['architectures'],'rope_interleave':cfg['rope_interleave'],'indexer_rope_interleave':cfg['indexer_rope_interleave'],'scope':'Observed raw model config, not introspected live HF config or rotary object and not task runtime contract.'}
style=env['style_from_config'](NS(rope_interleave=observed['rope_interleave']));observed_rows=[r for r in rows if r['neox_fixture']==style];assert all(r['selected_source_branch']!='neox-alt' for r in observed_rows)
mc=get(model,'__init__','DeepseekV2Model');allocation=next(n for n in mc.body if isinstance(n,ast.Assign) and any(ast.unparse(t)=='self.alt_stream' for t in n.targets));assert isinstance(allocation.value,ast.IfExp);f=ast.parse('def wants_alt_stream():\n pass').body[0];f.body=[ast.Return(value=copy.deepcopy(allocation.value.test))];install([f]);allocation_rows=[]
for cuda,musa,npu,rocm in itertools.product([False,True],repeat=4):
 env['_is_cuda']=cuda;env['_is_musa']=musa;env['envs']=NS(SGLANG_NPU_USE_MULTI_STREAM=NS(get=lambda:npu),SGLANG_ROCM_USE_MULTI_STREAM=NS(get=lambda:rocm));got=env['wants_alt_stream']();assert got==(cuda or musa or npu or rocm);allocation_rows.append({'cuda_fixture':cuda,'musa_fixture':musa,'npu_multistream_fixture':npu,'rocm_multistream_fixture':rocm,'allocation_condition':got})
defaults={};t=source('environ.py')
for n in ast.walk(t):
 if isinstance(n,ast.Assign):
  for target in n.targets:
   if isinstance(target,ast.Name) and target.id in ['SGLANG_NPU_USE_MULTI_STREAM','SGLANG_NPU_USE_MLAPO']:
    assert isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='EnvBool';defaults[target.id]={'declared_default':ast.literal_eval(n.value.args[0]),'line':n.lineno}
assert len(defaults)==2
factory='layers/rotary_embedding/factory.py';install([get(factory,'get_rope_wrapper')]);env['_use_aiter']=False;env['get_rope']=lambda *args:NS(is_neox_style=args[4]);env['get_rope_cpu']=lambda *args:NS(is_neox_style=args[4]);factory_rows=[]
for style in [False,True]:
 got=env['get_rope_wrapper'](64,64,1024,10000,is_neox_style=style,device='npu');assert got.is_neox_style==style;factory_rows.append({'input_style':style,'output_flag_via_factory_spy':got.is_neox_style,'real_rotary_constructor_executed':False})
result={'correctness':'PASS','source_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes':hashes,'extracted_nodes':locations,'branch_rows':rows,'layout_rows':style_rows,'allocation_rows':allocation_rows,'declared_defaults':defaults,'factory_spy_rows':factory_rows,'observed_config':observed,'observed_config_conditional_rows':observed_rows,'decision':'The named neox Q-projection/KV-norm alt-stream subcandidate is conditionally NOT_APPLICABLE when the observed rope_interleave=True config is used unchanged and the constructed flag is preserved. Actual loaded rotary object was not introspected. This does not establish that all model multistream paths are inactive.','scope':'Exact source predicates and configuration-to-flag argument flow only; branch bodies replaced by labels and rotary factory replaced by a spy. No torch import, stream/event construction, NPU execution, numerical equivalence or performance test. Other MLAPO/indexer/MoE streams remain independent. Overall serving E0.'}
p=O/'result.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');receipt={'correctness':'PASS','exit_code':0,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'branch_cases':len(rows),'layout_cases':len(style_rows),'allocation_cases':len(allocation_rows),'factory_spy_cases':len(factory_rows),'scope':result['scope']};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k.endswith('_cases')}))
