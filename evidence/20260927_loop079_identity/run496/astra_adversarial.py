import sys,json,hashlib,tempfile,runpy,contextlib,io
from pathlib import Path
sys.path.insert(0,'scripts')
import loop079_target_frontier_validate_cpu_test_run494 as t
import loop079_target_frontier_validate_run494 as v
results={}
actual=json.load(open('/tmp/astra_run494_source_check.json'))['files']
results['actual_source_manifest_matches_validator']=actual==v.SOURCE_MANIFEST
results['manifest_mismatches']=[dict(key=a['key'],actual=a['patched'],expected=b['patched']) for a,b in zip(actual,v.SOURCE_MANIFEST) if a!=b]
for label,nodes in [('missing_all_task_stream_fields',[{}]),('null_task_stream_fields',[{'args':{'Stream Id':None,'Task Id':None,'Task Type':None}}]),('duplicate_task_identity',[{'name':'task','args':{'Model Id':1,'Stream Id':0,'Task Id':0,'Task Type':'KERNEL_AIVEC'}}]*2)]:
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);t.fixture(root)
  graph=root/'graph_dump/rank0_cohort5_acl_graph.json';meta_path=root/'graph_dump/rank0_cohort5_acl_graph.meta.json';rowpath=root/'capture/rank0_cohort5.json'
  t.write(graph,nodes);raw=graph.read_bytes();digest=hashlib.sha256(raw).hexdigest()
  meta=json.loads(meta_path.read_text());meta.update(bytes=len(raw),sha256=digest,node_count=len(nodes),stream_ids=sorted({str(n.get('args',{}).get('Stream Id')) for n in nodes if isinstance(n.get('args'),dict) and n['args'].get('Stream Id') is not None}));t.write(meta_path,meta)
  row=json.loads(rowpath.read_text());row['debug_dump'].update(bytes=len(raw),sha256=digest,node_count=len(nodes));t.write(rowpath,row)
  try: v.validate_root(root);results[label]='ACCEPTED'
  except Exception as e:results[label]=f'rejected {e}'
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);t.fixture(root)
 meta_path=root/'graph_dump/rank0_cohort5_acl_graph.meta.json';meta=json.loads(meta_path.read_text())
 meta['graph_update_backend']['update_graph_params']['source']['sha256']='z'*64;t.write(meta_path,meta)
 try:v.validate_root(root);results['nonhex_backend_hash']='ACCEPTED'
 except Exception as e:results['nonhex_backend_hash']=f'rejected {e}'
# Execute stock CPU mocks, then replace the backend in the same update closure.
# This changes neither the update callable ID nor immutable stream identity.
with contextlib.redirect_stdout(io.StringIO()): ns=runpy.run_path('scripts/loop079_target_frontier_cpu_test_run494.py')
f=ns['f'];serving=ns['serving'];handoff=ns['handoff'];d=ns['frontier_d']
first=f._post_drain_backend(serving,d)
class OtherImpl:
 @staticmethod
 def update_graph_params(*args,**kwargs):pass
class OtherBackend:
 @staticmethod
 def get_impl_cls():return OtherImpl
fn=handoff.graph_update
cells=dict(zip(fn.__code__.co_freevars,fn.__closure__))
cells['attn_backend'].cell_contents=OtherBackend
try:
 after=f._post_drain_backend(serving,d)
 results['backend_closure_swap_after_selected']='ACCEPTED' if after['attn_backend']['object_id']!=first['attn_backend']['object_id'] else 'unchanged'
except Exception as e:results['backend_closure_swap_after_selected']=f'rejected {e}'
print(json.dumps(results,indent=2))
