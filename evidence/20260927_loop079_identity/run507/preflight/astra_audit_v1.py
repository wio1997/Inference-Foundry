import sys
sys.dont_write_bytecode=True
import ast, contextlib, copy, hashlib, importlib, json, os, shutil, subprocess, tempfile, types
from pathlib import Path
R=Path('/data/wio/Inference_Foundry');O=R/'evidence/20260927_loop079_identity/run507/preflight'
S=O/'reviewed_v1';sys.path.insert(0,str(S))
import loop079_target_frontier_run507 as h
import loop079_target_frontier_patch_run507 as p
import loop079_target_frontier_validate_run507 as base
import loop079_target_frontier_update_validate_run507 as u
N=types.SimpleNamespace;checks=[];sha=lambda b:hashlib.sha256(b).hexdigest()
def ck(n,v,d=None):checks.append(dict(test=n,passed=bool(v),detail=d))
def reject(f):
 try:f();return False
 except (RuntimeError,ValueError,KeyError,TypeError):return True
def write(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
man=[]
for k,path in p.SOURCES.items():
 old=path.read_bytes();ck('source_pin_'+k,sha(old)==p.PINS[k]);new=p.patch(k,old.decode()).encode();compile(new,str(path),'exec')
 man.append(dict(key=k,source=str(path),original=sha(old),patched=sha(new)))
ck('six_actual_patched_hashes',man==base.SOURCE_MANIFEST,man)
gsha=next(x['patched'] for x in man if x['key']=='graph')
ck('helper_graph_hash',h.EXPECTED_BINDINGS['graph_call']==gsha)
ck('validator_graph_hash',base.PINS['graph_call']==gsha)
controller=(S/'run_loop079_target_frontier_run507_b.sh').read_text()
ck('shell_syntax',subprocess.run(['bash','-n',str(S/'run_loop079_target_frontier_run507_b.sh')]).returncode==0)
# Execute only extracted source functions with CPU doubles, never module imports.
graphsrc=p.SOURCES['graph'].read_text();tree=ast.parse(graphsrc)
select=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_select_graph_params')
ctx=N(batch_descriptor=N(has_lora=False),cudagraph_runtime_mode='FULL')
ns=dict(get_forward_context=lambda:ctx,CUDAGraphMode=N(FULL='FULL'))
exec('from __future__ import annotations\n'+ast.unparse(select),ns)
main=N(attn_params={96:[object(),object()]},handles={96:[object(),object()]},events={96:[object(),object()]})
lora=N(attn_params={96:[]},handles={96:[]},events={96:[]})
selector=lambda:ns['_select_graph_params']({False:main,True:lora})
fakegraph=types.ModuleType('vllm_ascend.compilation.acl_graph');fakegraph.get_graph_params=selector
sys.modules['vllm_ascend.compilation.acl_graph']=fakegraph
snap=h._mla_params_snapshot(96)
ck('main_nonLoRA_context_snapshot',snap['params_id']==id(main) and snap['lists']['events']['count']==2)
ctx.batch_descriptor.has_lora=True
ck('LoRA_context_selects_other_params',h._mla_params_snapshot(96)['params_id']==id(lora));ctx.batch_descriptor.has_lora=False
attn=types.ModuleType('vllm_ascend.attention');mla=types.ModuleType('vllm_ascend.attention.mla_v1');mla._EXTRA_CTX=N(is_draft_model=False);attn.mla_v1=mla
sys.modules['vllm_ascend.attention']=attn;sys.modules['vllm_ascend.attention.mla_v1']=mla
backend=object();stream=N(stream_id=102,device_index=0,device_type=20)
class Handoff:
 def forward(self):pass
hand=Handoff();h._backend_from_update=lambda x:backend
def update_callable(*args):pass
impl=N(update_graph_params=update_callable)
replay=dict(entry_id=123,capture_generation=1,graph_id=456,selected_observation_ordinal=1)
h.CAPTURES={123:dict(generation=1,mla_params=snap)};h.TLS.selected=True
def reset():
 h.D=dict(selected=True,graph_update=[dict(end_ns=None,phase='after')],actual_update=None,
  serving=N(runtime=N(target=N(binding=N(forward=hand.forward)))),
  setup=dict(graph_update_bound=dict(backend=dict(object_id=id(backend)),configured_update_stream=vars(stream))),replay=copy.deepcopy(replay))
def begin():h.actual_update_begin(backend,impl,update_callable,stream,N(attn_metadata={'a':1,'b':2}),96,None)
reset();begin();h.actual_update_end();ck('selected_after_replay_identity_and_inferred_two',h.D['actual_update']['source_inferred_successful_event_records']==2)
reset();h.D['replay']=None;ck('before_or_missing_replay_rejected',reject(begin))
reset();h.CAPTURES[123]['mla_params']=copy.deepcopy(snap);h.CAPTURES[123]['mla_params']['params_id']+=1;ck('different_capture_GraphParams_rejected',reject(begin));h.CAPTURES[123]['mla_params']=snap
reset();mla._EXTRA_CTX.is_draft_model=True;ck('draft_branch_rejected',reject(begin));mla._EXTRA_CTX.is_draft_model=False
# Pin and execute only the installed MLA update function with fake stream/FIA/event APIs.
mlapath=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/mla_v1.py');raw=mlapath.read_bytes();ck('actual_MLA_source_pin',sha(raw)==u.MLA_SHA)
cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='AscendMLAImpl')
func=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_graph_params'));func.decorator_list=[]
calls=[]
class Event:
 def record(self,s):calls.append('record')
params=N(attn_params={},handles={},events={},workspaces={})
npu=N(stream=lambda s:contextlib.nullcontext(),graph_task_update_begin=lambda *x:calls.append('begin'),graph_task_update_end=lambda *x:calls.append('end'))
space=dict(_EXTRA_CTX=N(is_draft_model=False),get_graph_params=lambda:params,torch=N(npu=npu),torch_npu=N(npu_fused_infer_attention_score_v2=N(out=lambda *a,**kw:calls.append('FIA'))))
exec('from __future__ import annotations\n'+ast.unparse(func),space)
param=tuple([None]*18)
for keys,lengths,expected in [(0,(2,2,2),0),(3,(0,2,2),0),(3,(2,1,2),1),(2,(2,2,2),2)]:
 calls.clear();params.attn_params={96:[param]*lengths[0]};params.handles={96:[object() for _ in range(lengths[1])]};params.events={96:[Event() for _ in range(lengths[2])]}
 md={str(i):N(decode=N(seq_lens_list=[1])) for i in range(keys)}
 space['update_graph_params'](stream,N(attn_metadata=md),96)
 ck('actual_source_zip_'+str((keys,lengths)),calls==['begin','FIA','end','record']*expected,calls.copy())
with tempfile.TemporaryDirectory(prefix='astra_run507_') as tmp:
 root=Path(tmp)/'candidate';shutil.copytree(R/'evidence/20260927_loop079_identity/run502/b_candidate',root)
 for path in sorted((root/'capture').glob('rank*_cohort*.json')):
  row=json.loads(path.read_text());rank=row['rank'];c=row['cohort'];rep=row['replay'];row['setup']['source_pins']['bindings']['graph_call']['sha256']=gsha
  snapshot=dict(params_id=90000+rank,num_tokens=96,lists={name:dict(count=1,object_ids=[1000+rank]) for name in ['attn_params','handles','events']})
  a=row['actual_update'];a.update(replay_entry_id=rep['entry_id'],replay_capture_generation=rep['capture_generation'],replay_graph_id=rep['graph_id'],replay_selected_ordinal=rep['selected_observation_ordinal'],mla_params_capture=snapshot,mla_params_selected=copy.deepcopy(snapshot),attention_key_count=1,list_counts={n:1 for n in snapshot['lists']},zip_iteration_count=1,draft_metadata_argument_is_none=True,source_inferred_successful_event_records=1,device_event_completion='unobserved',native_event_id='unobserved')
  if c==5:
   gp=root/'graph_dump'/f'rank{rank}_cohort5_acl_graph.json';mp=gp.with_suffix('.meta.json');meta=json.loads(mp.read_text());meta['path']=str(gp);meta['graph_update_backend']['selected_actual_update']=a;write(mp,meta);row['debug_dump']['path']=str(gp);row['debug_dump']['meta_path']=str(mp)
  write(path,row)
 for name,action in [('source_check.json','check'),('install.json','install'),('restore.json','restore')]:write(root/name,dict(action=action,files=man))
 write(root/'patch_state/manifest.json',man)
 ck('synthetic_update_positive',u.validate(root)['valid'])
 ck('synthetic_base_final_positive',base.final_admit(root)['valid'])
 cp=root/'capture/rank0_cohort1.json';before=cp.read_bytes();row=json.loads(before);row['actual_update']['source_inferred_successful_event_records']=999;write(cp,row)
 ck('update_validator_wrong_count_rejects',reject(lambda:u.validate(root)))
 ck('final_admission_wrong_new_count_rejects',reject(lambda:base.final_admit(root)))
 cp.write_bytes(before)
 for label,mut in [('wrong_generation',lambda a:a.__setitem__('replay_capture_generation',99)),('wrong_params_owner',lambda a:a['mla_params_selected'].__setitem__('params_id',99)),('native_completion_claim',lambda a:a.__setitem__('device_event_completion','complete')),('bad_object_id',lambda a:a['mla_params_selected']['lists']['events']['object_ids'].__setitem__(0,0))]:
  row=json.loads(before);mut(row['actual_update']);write(cp,row);ck('new_negative_'+label,reject(lambda:u.validate(root)));cp.write_bytes(before)
 # CPU-only cleanup function extraction, with every external operation mocked.
 cleanup_source=(R/'scripts/loop079_target_frontier_cleanup_cpu_test_run494.py').read_text();cns={'__file__':str(R/'scripts/loop079_target_frontier_cleanup_cpu_test_run494.py')};exec(compile(cleanup_source,'cleanup_cpu','exec'),cns);cns['CONTROLLER']=S/'run_loop079_target_frontier_run507_b.sh';cns['main']();ck('Run507_controller_seven_cleanup_mock_paths',True)
 # Explicitly preserve a failing Run507 update validator process exit through cleanup.
 functions=controller[controller.index('verify_stopped() {'):controller.index('\ntrap cleanup EXIT')]
 out=Path(tmp)/'cleanup';out.mkdir();(out/'patch_state').mkdir();(out/'patch_state/manifest.json').write_text('{}')
 script=f'''set -euo pipefail
OUT={out}
ROOT=/fake
CONTAINER=fake
SOURCES=(/fake/source)
RUN_STARTED=1
bash() {{ return 0; }}
curl() {{ return 1; }}
docker() {{ return 0; }}
npu-smi() {{ return 0; }}
python3() {{ return 0; }}
sha256sum() {{ return 0; }}
cmp() {{ return 0; }}
{functions}
trap cleanup EXIT
exit 7
'''
 pr=subprocess.run(['bash','-c',script],capture_output=True,text=True);ck('update_failure_exit_preserved_by_cleanup',pr.returncode==7,pr.stderr)
result=dict(checks=checks,all_expected=all(c['passed'] for c in checks),reviewed_scripts_sha256={x.name:sha(x.read_bytes()) for x in S.iterdir()},limitations='Synthetic records test schema/control paths only; actual MLA source is AST-executed with CPU API mocks, never imported.')
write(O/'cpu_review_results_v1.json',result);print(json.dumps(result,indent=2))
