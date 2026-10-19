"""Read-only L13 D two-step native evidence over the frozen mixed P/D runtime.

Four node programs are derived from the byte-frozen L12C mixed-root capture,
with the reviewed D-only native profile extension appended for nodes 36/41 and
one CPU43 diagnostic metric re-derivation. Importing this module performs no
remote action; execute is the only remote entry point and is fixed to one ID.
"""
import ast,concurrent.futures,hashlib,importlib.util,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CAPTURE_ID='20261019T-L13-sglang-C32-D-two-step-native-v1'
DIAG_RID='20261019T-L13-sglang-C32-D-native-decode-two-step-profile-v1'
BASE_CAPTURE_SHA='e08773c054c1e6a636e9ff6de3e5a0f22257c14ddd6b16e53b0bf36a31d86610'
TARGET_GRAPH_SHA='8cad4c626dd24b028d35bdea17863f047d150ac9e15e7e1486669fe5ad668e03'
MIXED_IDENTITY_SHA='ff5a92342a3a385256b619e9c004294b881634678326d706eca026413c5e6a98'
PROFILE_SOURCE_SHA='22a9868de8d52da60f53dd573b232a113b71198aff4e7a96a6295a9e24d7ae6c'
CORE_SHA='c8e661f21c6a7413cd5e8ac34e490d7e457151c40811d566be4652ce2278a3a3'
CODEC_SHA='ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5'
CAPACITY='results/sglang-C32-L12C-selected-P-pool-capacity-v1.json'
SETTINGS='config/C32-target-graph-runtime-v3/formal-C32-target-graph-v1.json'
DIAG_CONFIG='config/C32-L13-D-native-decode-v1/D-two-step-diagnostic-v1.json'

def load(name,file,sha):
 p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def profile_extension(old):
 p=ROOT/'scripts/glm53_C32_P_native_profile_capture_v2.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==PROFILE_SOURCE_SHA
 spec=importlib.util.spec_from_file_location('reviewed_profile_template',p);template=importlib.util.module_from_spec(spec);spec.loader.exec_module(template)
 text=template.PROFILE
 node_from='if NODE in [34,35]:';node_to='if NODE in [36,41]:'
 assert text.count(node_from)==1;text=text.replace(node_from,node_to)
 rank_from='assert rank_ids==set(range(16) if NODE==34 else range(16,32))';rank_to='assert rank_ids==set(range(16) if NODE==36 else range(16,32))'
 assert text.count(rank_from)==1;text=text.replace(rank_from,rank_to)
 core=(ROOT/'scripts/glm53_profile_csv_analysis_v1.py').read_text();assert hashlib.sha256(core.encode()).hexdigest()==CORE_SHA
 text=old.render(text,{'CORE':core})
 assert 'if NODE in [36,41]:' in text and 'range(16) if NODE==36' in text
 return text

def diagnostic_cpu(old,raw_report_sha):
 assert type(raw_report_sha) is str and re.fullmatch('[0-9a-f]{64}',raw_report_sha)
 cpu=old.CPU_HOST
 replacements=[
  ("formal=pathlib.Path('/data/wio/inference-foundry/glm53/runs/'+FORMAL_RID+'');output=formal/'output';report_raw=(output/'formal-result.json').open('rb').read(64*1024**2+1);assert len(report_raw)<=64*1024**2;report=json.loads(report_raw);assert report['project_id']=='glm53' and report['framework']=='sglang' and report['concurrency']==CONCURRENCY",
   "diag=pathlib.Path('/data/wio/inference-foundry/glm53/runs/'+DIAG_RID+'');output=diag/'output';report_raw=(output/'diagnostic-result.json').open('rb').read(64*1024**2+1);assert len(report_raw)<=64*1024**2;assert hashlib.sha256(report_raw).hexdigest()==RAW_REPORT_SHA;report=json.loads(report_raw);assert report['project_id']=='glm53' and report['framework']=='sglang' and report['concurrency']==CONCURRENCY and report['diagnostic_only'] is True and report['performance_certificate_allowed'] is False and report['phase']=='COMPLETE'"),
  ("out['formal_report_identity']={'sha256':hashlib.sha256(report_raw).hexdigest(),'bytes':len(report_raw),'namespace':report['namespace'],'phase':report['phase'],'baseline_accepted':report['baseline_accepted']}",
   "out['diagnostic_report_identity']={'sha256':hashlib.sha256(report_raw).hexdigest(),'bytes':len(report_raw),'namespace':report['cache_namespace'],'phase':report['phase'],'diagnostic_only':report['diagnostic_only'],'performance_certificate_allowed':report['performance_certificate_allowed']}"),
  ("metric=output/'formal-D-metrics.jsonl.gz'","metric=output/'diagnostic-D-metrics.jsonl.gz'"),
  ("out['independent_formal_metric_archive_audit']","out['independent_diagnostic_metric_archive_audit']"),
  ("out['formal_native_target_graph_counters']","out['diagnostic_native_target_graph_counters']"),
  ("'delta_by_global_EP_rank':{k:graph_last[k]-graph_first[k] for k in graph_first},'native_endpoint_rows':graph_sample_rows,",
   "'delta_by_global_EP_rank':{k:graph_last[k]-graph_first[k] for k in graph_first},'all32productive_Graph_deltas':all(graph_last[k]>graph_first[k] for k in graph_first),'native_endpoint_rows':graph_sample_rows,"),
  ("'scope':'All32 ranks cumulative Graph counter progression within original formal D metrics archive","'scope':'All32 ranks cumulative Graph counter progression within original diagnostic D metrics archive"),
 ]
 for before,after in replacements:
  assert cpu.count(before)==1,before;cpu=cpu.replace(before,after)
 assert 'formal' not in cpu.lower() and 'baseline_accepted' not in cpu and "report['namespace']" not in cpu
 assert 'RAW_REPORT_SHA' in cpu and 'DIAG_RID' in cpu
 return cpu

def plans(capture_id,raw_report_sha):
 assert capture_id==CAPTURE_ID and Path(capture_id).name==capture_id
 assert type(raw_report_sha) is str and re.fullmatch('[0-9a-f]{64}',raw_report_sha)
 old=load('old_capture','glm53_sglang_C32_target_graph_native_capture_v3.py',TARGET_GRAPH_SHA)
 identity=load('mixed_identity','glm53_C32_L12C_mixed_model_identity_v1.py',MIXED_IDENTITY_SHA)
 capacity_raw=(ROOT/CAPACITY).read_bytes();cap=json.loads(capacity_raw);assert cap['pass'] and cap['selected_variant']=='candidate'
 assert set(cap['providers'])=={'34','35','36','41'}
 settings=json.loads((ROOT/SETTINGS).read_text());diag_raw=(ROOT/DIAG_CONFIG).read_bytes();diag=json.loads(diag_raw)
 assert diag['run_id']==DIAG_RID and diag['diagnostic_only'] is True and diag['performance_certificate_allowed'] is False
 assert diag['model_image']==settings['model_image'] and diag['meta_container_id']==cap['meta_container_id']
 for n,item in cap['providers'].items():
  d=diag['runtime_containers'][n];assert (d['container_id'],d['name'],d['remote_root'])==(item['container_id'],item['name'],item['remote_root'])
 text=old.HOST.replace("root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3');",'root=pathlib.Path(NODE_REMOTE_ROOT);')
 assert text!=old.HOST
 stream=old.STREAM.replace("if 'GLM53_C32_PACKED_DISPATCH_' in line", "if 'GLM53_C32_NATIVE_MC_MAPPING_REUSED=' in line or 'GLM53_C32_NATIVE_HICACHE_' in line or 'GLM53_NATIVE_HICACHE_FATAL' in line or 'GLM53_C32_PACKED_DISPATCH_' in line")
 assert stream!=old.STREAM
 text+='''
out['native_optional_extension_worker_maps']=[]
if NATIVE_P:
 sdk_pids={event['pid'] for item in out['persistent_audit'].values() for event in item['rows'] if event['event']=='initialize_complete'}
 for row in rows:
  if row['namespace_pid'] not in sdk_pids:continue
  raw=(pathlib.Path('/proc')/str(row['host_pid'])/'maps').read_bytes();assert len(raw)<=8*1024**2
  paths=sorted({line.split(maxsplit=5)[5].strip() for line in raw.decode().splitlines() if 'libmf_hybm_accoffload.so' in line})
  assert paths==['/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so']
  out['native_optional_extension_worker_maps'].append({'namespace_pid':row['namespace_pid'],'host_pid':row['host_pid'],'loaded_extension_paths':paths,'process_maps_sha256':hashlib.sha256(raw).hexdigest()})
 assert len(sdk_pids)==16 and len(out['native_optional_extension_worker_maps'])==16
'''
 profile=profile_extension(old);text+=profile
 codes={}
 for n,item in cap['providers'].items():
  node=int(n);values={'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NODE':node,'IDENTITY':identity.render(node,cap),'IDENTITY_ENV':{},'STREAM':stream,'SDK_SOURCE':old.SDK_SOURCE,'PORT':30500 if node in [34,35] else 30600,'META':cap['meta_container_id'],'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+capture_id,'FORMAL_RID':DIAG_RID,'CONCURRENCY':32,'NEED_FORMAL':False,'NODE_REMOTE_ROOT':item['remote_root'],'NATIVE_P':node in [34,35] and cap['selected_variant']=='candidate'}
  codes[node]=old.render(text,values);tree=ast.parse(codes[node]);assert not ({x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}&set(values))
 cpu=diagnostic_cpu(old,raw_report_sha)
 codes[43]=old.render(cpu,{'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+capture_id,'DIAG_RID':DIAG_RID,'CONCURRENCY':32,'RAW_REPORT_SHA':raw_report_sha})
 tree=ast.parse(codes[43]);assert not ({x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}&{'ARCHIVE','DIAG_RID','CONCURRENCY','RAW_REPORT_SHA'})
 meta={'capture_id':capture_id,'diagnostic_run_id':DIAG_RID,'base_capture_source_sha256':BASE_CAPTURE_SHA,'target_graph_source_sha256':TARGET_GRAPH_SHA,'mixed_identity_source_sha256':MIXED_IDENTITY_SHA,'profile_template_source_sha256':PROFILE_SOURCE_SHA,'profile_CSV_analysis_source_sha256':CORE_SHA,'raw_failure_codec_sha256':CODEC_SHA,'capacity_source_sha256':hashlib.sha256(capacity_raw).hexdigest(),'diagnostic_config_sha256':hashlib.sha256(diag_raw).hexdigest(),'profile_extension_sha256':hashlib.sha256(profile.encode()).hexdigest(),'profile_extension_bytes':len(profile.encode())}
 return old,codes,meta

def invoke_with_raw_failure(old,node,code,archive):
 import shlex
 pod,host=old.base.NODES[node];mapped=old.base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 local=ROOT/'runs'/Path(archive).name
 (local/('node-'+str(node)+'-mapping-wire.json')).write_text(json.dumps(mapped,indent=2)+'\n');assert mapped['rc']==0 and mapped['stdout']=='172.16.12.'+str(node)
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code+old.codec.ENCODER;ast.parse(code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 wire=old.base.invoke([command],code,timeout=900);(local/('node-'+str(node)+'-raw-wire.json')).write_text(json.dumps(wire,indent=2)+'\n')
 if wire['rc']!=0:return wire
 return old.codec.decode(wire)

def execute(capture_id,review_path):
 assert capture_id==CAPTURE_ID
 review_raw=(ROOT/review_path).read_bytes();review=json.loads(review_raw)
 assert review['project_id']=='glm53' and review.get('capture_id')==capture_id and review.get('diagnostic_run_id')==DIAG_RID
 frozen_raw_report_sha=review['raw_diagnostic_report_sha256'];assert type(frozen_raw_report_sha) is str and re.fullmatch('[0-9a-f]{64}',frozen_raw_report_sha)
 diag_run=ROOT/'runs'/DIAG_RID;diagnostic=json.loads((diag_run/'receipt.json').read_text())
 assert diagnostic['project_id']=='glm53' and diagnostic['run_id']==DIAG_RID and diagnostic['diagnostic_only'] is True
 assert diagnostic.get('node43_rc')==0 and diagnostic.get('client_rc') is not None and diagnostic.get('host_wait_timeout') is not True
 local_report=(diag_run/'diagnostic-result.remote.json').read_bytes();assert hashlib.sha256(local_report).hexdigest()==frozen_raw_report_sha
 report=json.loads(local_report);assert report['phase']=='COMPLETE' and report['diagnostic_only'] is True and report['performance_certificate_allowed'] is False and report['backend_HTTP_attempts']==64
 old,codes,meta=plans(capture_id,frozen_raw_report_sha)
 expected={str(n):hashlib.sha256(c.encode()).hexdigest() for n,c in codes.items()};assert review['host_codes_sha256']==expected
 run=ROOT/'runs'/capture_id;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()))
 archive='/data/wio/inference-foundry/glm53/runs/'+capture_id
 receipt={'project_id':'glm53','capture_id':capture_id,'diagnostic_run_id':DIAG_RID,'diagnostic_report_sha256':frozen_raw_report_sha,'review_path':str(review_path),'review_sha256':hashlib.sha256(review_raw).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profile_completion_basis':'Stable 16 worker folders and six native CSVs per D node plus core CSV analysis; model-log marker lines are retained verbatim but are not completion counts.','raw_profile_traces_remain_remote':True,'model_requests':0,'SDK_init_calls':0,'max_SSH':2,'nodes':{}}
 receipt.update(meta)
 results={}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs={pool.submit(invoke_with_raw_failure,old,n,code,archive):n for n,code in codes.items()}
   for future in concurrent.futures.as_completed(jobs):
    n=jobs[future]
    try:r=future.result()
    except Exception as error:r={'error':repr(error)}
    results[n]=r;(run/('node-'+str(n)+'.json')).write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][str(n)]=r.get('rc',r.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
  assert len(receipt['nodes'])==5 and all(x==0 for x in receipt['nodes'].values())
  data=json.loads(results[43]['stdout']);identity=data['diagnostic_report_identity'];audit=data['independent_diagnostic_metric_archive_audit'];graphs=data['diagnostic_native_target_graph_counters']
  assert identity['sha256']==frozen_raw_report_sha
  assert audit['all_native_bodies_SHA_and_DP_gauges_recomputed'] and not audit['errors'] and audit['sample_pairs']==audit['complete_pairs']
  deltas=graphs['delta_by_global_EP_rank'];assert len(deltas)==32 and all(x>0 for x in deltas.values()) and graphs['all32productive_Graph_deltas'] is True
  receipt.update(diagnostic_report_bytes=identity['bytes'],diagnostic_metric_archive_sha256=audit['archive_sha256'],sample_pairs=audit['sample_pairs'],native_HTTP_rows=audit['native_HTTP_rows'],all32productive_Graph_deltas=True,profile_nodes=[36,41]);receipt['pass']=True
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt
