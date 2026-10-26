"""Supplemental native optional-field audit; never rewrites runtime reports or fills nulls."""
import ast,copy,hashlib,json,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE_RUN='runs/20261017T1896-sglang-cache-annotation-source-v1'
SHAS={'srt/managers/scheduler_components/output_streamer.py':'e0eb1e235c6afcb47734824fada07bbd1a1f0c66af947359715e8d019baeffc9','srt/entrypoints/openai/protocol.py':'264114cff2cbdfa1a967c58d2c1a3317220fa579df3ea4e34a44a807dcbbc0bd','srt/entrypoints/openai/utils.py':'11014c3b21ca44692bf264ed7eb3e9b0cdbed25de64a6573e026315b14028fec'}
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def source_formatter():
 wire=json.loads((ROOT/SOURCE_RUN/'sglang.json').read_text());assert wire['rc']==0
 obj=json.loads(wire['stdout']);assert obj['model_requests']==obj['SDK_init_calls']==0 and not obj['profiling_started']
 files={x['relative']:x for x in obj['packages']['sglang']['files']}
 for relative,digest in SHAS.items():
  item=files[relative];assert hashlib.sha256(item['source'].encode()).hexdigest()==item['sha256']==digest and len(item['source'].encode())==item['bytes']
 # Execute only the captured pure formatter AST, with no framework imports.
 node=copy.deepcopy(next(n for n in ast.walk(ast.parse(files['srt/managers/scheduler_components/output_streamer.py']['source'])) if isinstance(n,ast.FunctionDef) and n.name=='get_cached_tokens_details'))
 node.returns=None;node.decorator_list=[]
 for a in node.args.args:a.annotation=None
 module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));scope={};exec(compile(module,'captured-native-cache-formatter','exec'),scope)
 return scope['get_cached_tokens_details']
def check_tiers(P,D):
 assert isinstance(P,dict) and isinstance(D,dict)
 assert all(type(P.get(k)) is int and P[k]>=0 for k in ['device','host','storage'])
 assert P.get('storage_backend')=='NpuMemcacheStore' and P['host']==0 and 74368<=sum(P[k] for k in ['device','host','storage'])<=74400
 assert all(type(D.get(k)) is int and D[k]>=0 for k in ['device','host']) and D['host']==0
 assert D['device']==P['device'] and D['host']==P['host']
 if 'storage' in D:assert type(D['storage']) is int and D['storage']>=0 and D['storage']==P['storage']
 else:assert P['storage']==0 and 74368<=D['device']<=74400 and set(D)=={'device','host'}
 return {'P':{k:P[k] for k in ['device','host','storage']},'D':{k:D.get(k) for k in ['device','host','storage']},'D_storage_field_present':'storage' in D,'logical_prefix_verified':True,'scope':'Native source accounting inherited from P; D storage null stays null. No direct D L3-read claim.'}
def review(report):
 assert report['framework']=='sglang';formatter=source_formatter();rows=[]
 for row in report['formal']['rows']:
  r=row['result'];P=r['terminal']['gateway']['prefill_sglext']['cached_tokens_details'];D=r['native']['extra_details_data']['glm53_native_sglext']['cached_tokens_details'];checked=check_tiers(P,D)
  req=types.SimpleNamespace(cached_tokens_device=P['device'],cached_tokens_host=P['host'],cached_tokens_storage=P['storage'],cached_tokens=sum(P[k] for k in ['device','host','storage']))
  for enabled,actual in [(True,P),(False,D)]:
   owner=types.SimpleNamespace(enable_hicache_storage=lambda:enabled,_get_storage_backend_type=lambda:'NpuMemcacheStore');assert formatter(owner,req)==actual
  rows.append({'request_id':row['request_id'],'original_cache_pass':r['cache_observation']['pass']}|checked)
 assert len(rows)==4*report['concurrency']
 return {'project_id':'glm53','run_id':report['run_id'],'pass':True,'review_revision':'native-optional-D-storage-v1','original_baseline_accepted':report['baseline_accepted'],'original_cache_passes':sum(x['original_cache_pass'] for x in rows),'independently_verified_prefix_requests':len(rows),'D_storage_null_requests':sum(not x['D_storage_field_present'] for x in rows),'raw_fields_changed':False,'runtime_or_model_changes':False,'raw_report_sha256':sha(f"runs/{report['run_id']}/formal-result.remote.json"),'source_capture_sha256':sha(SOURCE_RUN+'/sglang.json'),'source_file_sha256':SHAS,'source_coverage_scope':'Actual installed SG34 formatter/protocol from the same qualified image and source used by all four SG nodes; D HiCache disabled confirmed by native D36 server info. Metadata transfer semantics separately pinned in1850.','rows':rows}
