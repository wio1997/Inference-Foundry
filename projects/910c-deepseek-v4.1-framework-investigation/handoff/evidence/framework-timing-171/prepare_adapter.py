"""Freeze the actual adapter, fail on missing functions, and compile transformed code."""
import ast,hashlib,importlib.util,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
TOOL=HERE/'tool' if (HERE/'tool').exists() else HERE.parents[1]/'tools/framework-timing'
sys.path.insert(0,str(TOOL));import ft_hook
old=HERE.parent/'idle-attribution-171'
rows=json.loads((old/'source-map.json').read_text())+json.loads((old/'additional-source-map.json').read_text())
rows += [dict(r,runtime_path=r['path'],bundle_path='source/'+r['name'],base=str(HERE)) for r in json.loads((HERE/'source-map.json').read_text()) if not r.get('missing')]
manifest={'schema':1,'framework':'frozen vLLM Ascend / DeepSeek-V4.1-Flash','modules':{}}
modules={
'vllm_ascend.worker.model_runner_v1': ('NPUModelRunner', ['execute_model','sample_tokens','_update_states','_update_states_after_model_execute','_prepare_inputs','_sync_metadata_across_dp','_determine_batch_execution_and_padding','_build_attention_metadata','_pad_query_start_loc_for_fia','_prepare_device_metadata_for_forward','_sanitize_placeholder_input_ids_for_forward','_calc_spec_decode_metadata','_copy_spec_decode_metadata_to_device','_sample','_bookkeeping_sync','propose_draft_token_ids','_model_forward','_update_full_graph_params_if_needed'],['execute_model','sample_tokens','_update_states','_prepare_inputs','_sync_metadata_across_dp','_determine_batch_execution_and_padding','_build_attention_metadata','_model_forward']),
'vllm.v1.worker.gpu_model_runner':('GPUModelRunner',['_preprocess','_update_states'],['_preprocess']),
'vllm.v1.engine.core':('EngineCore',['step','step_with_batch_queue','post_step','add_request'],['step','step_with_batch_queue','post_step','add_request']),
'vllm.v1.core.sched.scheduler':('Scheduler',['schedule','update_from_output','update_draft_token_ids','update_draft_token_ids_in_output','add_request'],['schedule','update_from_output','update_draft_token_ids','update_draft_token_ids_in_output','add_request']),
'vllm_ascend.attention.context_parallel.dsa_v41_cp':(None,[],[]),
'vllm_ascend.attention.context_parallel.dsa_cp':(None,[],[]),
'vllm_ascend.attention.dsa_v41':(None,[],[]),
'vllm_ascend.worker.device_metadata':(None,[],[]),
'vllm_ascend.worker.block_table':('MultiGroupBlockTable',['compute_slot_mapping','commit_block_table'],[]),
'vllm_ascend.spec_decode.llm_base_proposer':('SpecDecodeBaseProposer',['_propose','prepare_inputs_padded','build_draft_attn_metadata'],[]),
'vllm_ascend.spec_decode.dspark_proposer':('DSparkProposer',['set_inputs_first_pass','set_per_group_attn_metadata'],[]),
}
found=[]
for module,(cls,names,coarse) in modules.items():
 suffix='/'+module.replace('.','/')+'.py'
 row=next(r for r in rows if r['runtime_path'].endswith(suffix))
 base=pathlib.Path(row.get('base',old));path=base/row['bundle_path'];source=path.read_bytes()
 assert hashlib.sha256(source).hexdigest()==row['sha256'],module
 tree=ast.parse(source);classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
 # Resolve real class names from source rather than assume them.
 if module.endswith('llm_base_proposer'):cls=next(k for k,v in classes.items() if any(isinstance(n,ast.FunctionDef) and n.name=='_propose' for n in v.body))
 if module.endswith('dspark_proposer'):cls=next(k for k,v in classes.items() if any(isinstance(n,ast.FunctionDef) and n.name=='set_inputs_first_pass' for n in v.body))
 if cls:
  available={n.name for n in classes[cls].body if isinstance(n,ast.FunctionDef)}
  if module.endswith('block_table'):names=[n for n in names if n in available]
  assert set(names)<=available,(module,set(names)-available)
 functions=[cls+'.'+name for name in names]
 coarses=[cls+'.'+name for name in coarse]
 if module=='vllm.v1.engine.core':
  functions+=['EngineCoreProc._process_input_queue','EngineCoreProc._process_engine_step'];coarses+=functions[-2:]
 if module.endswith('device_metadata'):
  functions=[c.name+'.'+n.name for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ['submit','wait','release','acquire','run','execute','finish','enqueue']]
 spec={'sha256':row['sha256'],'functions':functions,'coarse':coarses,'metadata_builders':'.attention.' in module,'source_path':row['runtime_path']}
 if module=='vllm.v1.engine.core':spec['timed_calls']=['future.result','exec_model_fut.result','self.model_executor.execute_model','self.model_executor.sample_tokens']
 tr=ft_hook.Transformer(module,row['runtime_path'],spec);new=tr.visit(tree);ast.fix_missing_locations(new);compile(new,row['runtime_path'],'exec')
 assert tr.changed,module
 manifest['modules'][module]=spec;found+=tr.changed
(HERE/'adapter.json').write_text(json.dumps(manifest,indent=2))
(HERE/'instrumented-functions.json').write_text(json.dumps(found,indent=2))
print(json.dumps({'modules':len(modules),'functions':len(found),'adapter_sha256':hashlib.sha256((HERE/'adapter.json').read_bytes()).hexdigest()}))
