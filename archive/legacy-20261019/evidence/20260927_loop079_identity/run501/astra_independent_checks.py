import sys,json,hashlib,tempfile,runpy,contextlib,io,copy,os
from pathlib import Path
sys.path.insert(0,'scripts')
import loop079_target_frontier_validate_cpu_test_run494 as t
import loop079_target_frontier_validate_run494 as v
import loop079_target_frontier_patch_run494 as p
out={};actual=json.load(open('/tmp/astra_run501_source_check.json'))['files']
out['source_manifest_all_six']=actual==v.SOURCE_MANIFEST
out['source_independent_all_six']=all(hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()==row['original'] and hashlib.sha256(p.patch(row['key'],Path(row['source']).read_text()).encode()).hexdigest()==row['patched'] for row in v.SOURCE_MANIFEST)
out['serving_patched']=actual[1]['patched'];out['graph_patched']=actual[4]['patched']
# Reproduce old attacks and test a broader malformed-identity matrix while keeping byte/metadata joins consistent.
cases={'missing_all_fields':lambda ns:ns.__setitem__(slice(None),[{}]),'duplicate_task_identity':lambda ns:ns.append(copy.deepcopy(ns[0])), 'mixed_native_models':lambda ns:ns[1]['args'].update({'Model Id':50}), 'kernel_arguments_missing':lambda ns:ns[0]['args'].pop('Kernel Args'), 'unsupported_task_type':lambda ns:ns[1]['args'].update({'Task Type':'UNKNOWN'})}
for field in ('Model Id','Stream Id','Task Id'):
 for tag,val in [('null',None),('bool',True),('string','1'),('negative',-1)]:cases[field+'_'+tag]=lambda ns,field=field,val=val:ns[0]['args'].update({field:val})
for name,mutate in cases.items():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);t.fixture(root);graph=root/'graph_dump/rank0_cohort5_acl_graph.json';mp=root/'graph_dump/rank0_cohort5_acl_graph.meta.json';rp=root/'capture/rank0_cohort5.json'
  ns=json.loads(graph.read_text());mutate(ns);t.write(graph,ns);raw=graph.read_bytes();digest=hashlib.sha256(raw).hexdigest()
  meta=json.loads(mp.read_text());meta.update(bytes=len(raw),sha256=digest,node_count=len(ns),task_count=len(ns))
  args=[n.get('args',{}) for n in ns];streams=[str(a['Stream Id']) for a in args if a.get('Stream Id') is not None]
  meta.update(stream_ids=sorted(set(streams)),stream_task_counts={s:streams.count(s) for s in sorted(set(streams))})
  models=[a.get('Model Id') for a in args];meta['native_model_ids']=sorted(set(models),key=str)
  t.write(mp,meta);row=json.loads(rp.read_text());row['debug_dump'].update(bytes=len(raw),sha256=digest,node_count=len(ns));t.write(rp,row)
  try:v.validate_root(root);out[name]='ACCEPTED'
  except Exception as exc:out[name]='rejected: '+str(exc)
for name,mutate in {
 'wrong_debug_owner':lambda m:m.update(debug_dump_owner_id=99),
 'missing_debug_owner':lambda m:m.pop('debug_dump_owner_id'),
 'nonhex_backend_hash':lambda m:m['graph_update_backend']['impl']['source'].update(sha256='z'*64),
 'wrong_stream_count':lambda m:m['stream_task_counts'].update({'0':99}),
 'wrong_model_metadata':lambda m:m.update(native_model_ids=[50]),
 'wrong_selected_impl':lambda m:m['graph_update_backend']['selected_actual_update'].update(impl_id=99),
}.items():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);t.fixture(root);mp=root/'graph_dump/rank0_cohort5_acl_graph.meta.json';m=json.loads(mp.read_text());mutate(m);t.write(mp,m)
  try:v.validate_root(root);out[name]='ACCEPTED'
  except Exception as exc:out[name]='rejected: '+str(exc)
with contextlib.redirect_stdout(io.StringIO()):n=runpy.run_path('scripts/loop079_target_frontier_cpu_test_run494.py')
f=n['f'];d=n['frontier_d'];serving=n['serving'];handoff=n['handoff'];e=n['e']
first=f._post_drain_backend(serving,d)
out['actual_getter_update_call_counts']=n['update_calls'].copy()
class OtherImpl:
 @staticmethod
 def update_graph_params(*args,**kwargs):pass
class OtherBackend:
 @staticmethod
 def get_impl_cls():return OtherImpl
fn=handoff.graph_update;cells=dict(zip(fn.__code__.co_freevars,fn.__closure__));saved=cells['attn_backend'].cell_contents
cells['attn_backend'].cell_contents=OtherBackend
try:f._post_drain_backend(serving,d);out['same_callable_backend_cell_swap']='ACCEPTED'
except Exception as exc:out['same_callable_backend_cell_swap']='rejected: '+str(exc)
cells['attn_backend'].cell_contents=saved
other=n['DumpGraph']();e.aclgraph.debug_dump=other.debug_dump;old_calls=e.aclgraph.calls
with tempfile.TemporaryDirectory() as tmp:
 os.environ[f.DUMP_ENV]=tmp
 try:
  f.post_drain_dump(d,serving)
  out['same_method_different_graph_owner']=dict(result='ACCEPTED',selected_calls_delta=e.aclgraph.calls-old_calls,wrong_graph_calls=other.calls)
 except Exception as exc:out['same_method_different_graph_owner']='rejected: '+str(exc)
 os.environ.pop(f.DUMP_ENV)
del e.aclgraph.debug_dump
# Execute generated original-update branch with a throwing implementation. The return hook must not run.
f.D=d;d['selected']=True;f.TLS.selected=True;d['actual_update']=None;d['_actual_update_objects']=None;d['graph_update'][-1]['end_ns']=None
backend=f._backend_from_update(handoff);impl=backend.get_impl_cls()
def throwing(*args,**kwargs):raise RuntimeError('intentional CPU-only update failure')
impl.update_graph_params=staticmethod(throwing)
try:n['ns_update']['update_full_graph_params'](backend,n['s'],None,96,None)
except RuntimeError as exc:out['throwing_update_escaped']=str(exc)
out['throwing_update_return_count']=d['actual_update']['return_count']
try:f._post_drain_backend(serving,d);out['throwing_update_dump']='ACCEPTED'
except Exception as exc:out['throwing_update_dump']='rejected: '+str(exc)
for run,name in [('run492','simple_graph.json'),('run493','multistream_graph.json')]:
 nodes=json.loads((Path('/data/wio/Inference_Foundry/evidence/20260927_loop079_identity')/run/name).read_text())
 out[run+'_installed_format']=v.native_tasks(nodes)
print(json.dumps(out,indent=2))
