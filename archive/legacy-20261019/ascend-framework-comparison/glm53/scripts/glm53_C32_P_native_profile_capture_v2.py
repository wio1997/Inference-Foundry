"""Read-only native capture plus all32 current C32 P profile CSV analyses.

Uses the reviewed 2029 native capture transport. The template extension and its
source digest are recorded separately. Raw traces remain on their owned nodes.
"""
import ast,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='8cad4c626dd24b028d35bdea17863f047d150ac9e15e7e1486669fe5ad668e03'
CORE_SHA='c8e661f21c6a7413cd5e8ac34e490d7e457151c40811d566be4652ce2278a3a3'
PROFILE=r'''
if NODE in [34,35]:
 import traceback
 prof=root/'profile'/FORMAL_RID
 info={'root':str(prof),'exists':prof.is_dir(),'files':[],'workers':{},'raw_traces_transferred_to_Mac':False,'analysis_pass':False,'model_log_markers':[]}
 out['native_profile']=info
 try:
  import time
  ready_start=time.monotonic();deadline=ready_start+240;previous_signature=None;ready=False
  while time.monotonic()<deadline:
   folders_now=sorted(x for x in prof.iterdir() if x.is_dir()) if prof.is_dir() else []
   wanted_now=[folder/'ASCEND_PROFILER_OUTPUT'/name for folder in folders_now for name in ['api_statistic.csv','kernel_details.csv','op_statistic.csv','operator_details.csv','step_trace_time.csv','task_time.csv']]
   if len(folders_now)==16 and all(path.is_file() and path.stat().st_size>0 for path in wanted_now):
    signature=[(str(path),path.stat().st_size,path.stat().st_mtime_ns) for path in wanted_now]
    if signature==previous_signature:ready=True;break
    previous_signature=signature
   time.sleep(3)
  info['CSV_ready_wait_seconds']=time.monotonic()-ready_start;assert ready,'All16 local Pprofiler CSV exports not complete/stable within240s; preserve originals'
  assert prof.is_dir()
  initialized={x['pid']:x for item in out['persistent_audit'].values() for x in item['rows'] if x['event']=='initialize_complete'}
  assert len(initialized)==16
  folders=sorted(x for x in prof.iterdir() if x.is_dir());assert len(folders)==16
  files=sorted(x for x in prof.rglob('*') if x.is_file());assert len(files)<=8192
  total=0;identities={};rank_ids=set();indexed={}
  for path in files:
   assert not path.is_symlink() and path.resolve().is_relative_to(prof.resolve())
   stat=path.stat();assert stat.st_size<=4*1024**3
   digest=hashlib.sha256()
   with path.open('rb') as file:
    for chunk in iter(lambda:file.read(1024**2),b''):digest.update(chunk)
   after=path.stat();assert (stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Profiler output still changing'
   total+=stat.st_size;assert total<=32*1024**3
   relative=str(path.relative_to(prof));row={'relative':relative,'bytes':stat.st_size,'sha256':digest.hexdigest(),'mtime_ns':stat.st_mtime_ns};info['files'].append(row);indexed[relative]=row
  info['total_bytes']=total
  exec(compile(CORE,'current-C32-native-profile-CSV','exec'),globals())
  seen_pids=set()
  for folder in folders:
   match=re.search(r'_(\d+)_\d{17}_ascend_pt$',folder.name);assert match
   pid=int(match.group(1));assert pid in initialized and pid not in seen_pids;seen_pids.add(pid)
   metadata=list(folder.glob('profiler_info_*.json'));assert len(metadata)==1
   native_info=json.loads(metadata[0].read_text());rank=native_info['rank_id'];assert type(rank) is int and rank not in rank_ids;rank_ids.add(rank)
   wanted=['ASCEND_PROFILER_OUTPUT/'+name for name in ['api_statistic.csv','kernel_details.csv','op_statistic.csv','operator_details.csv','step_trace_time.csv','task_time.csv']]+[metadata[0].name]
   selected={}
   for relative in wanted:
    entry=indexed[folder.name+'/'+relative];selected[relative]={'bytes':entry['bytes'],'sha256':entry['sha256']}
   settings={'namespace_pid':pid,'device_id':initialized[pid]['device_id'],'TP_rank':rank,'files':selected,'profiler_info_name':metadata[0].name}
   worker=analyze_worker(folder,settings)
   spans={}
   for row in csv_rows(folder/'ASCEND_PROFILER_OUTPUT/operator_details.csv',['Name','Host Self Duration(us)','Host Total Duration(us)','Device Self Duration(us)']):
    name=row['Name']
    if not re.search(r'prefill|decode|draft|verify|forward|batch_size|batch size',name,re.I):continue
    entry=spans.setdefault(name,{'count':0,'host_self_sum_us':0.0,'host_total_sum_us':0.0,'device_self_sum_us':0.0});entry['count']+=1
    for field,column in [('host_self_sum_us','Host Self Duration(us)'),('host_total_sum_us','Host Total Duration(us)'),('device_self_sum_us','Device Self Duration(us)')]:
     value=number(row[column])
     if value is not None:entry[field]+=value
   worker['span_like_host_operator_aggregates']=[{'name':name}|value for name,value in sorted(spans.items(),key=lambda item:item[1]['host_total_sum_us'],reverse=True)[:40]]
   classes={}
   for row in csv_rows(folder/'ASCEND_PROFILER_OUTPUT/api_statistic.csv',['Level','API Name','Time(us)','Count']):
    name=row['API Name'];kind='sync' if re.search('Synchronize',name,re.I) else 'launch' if re.search('Launch',name,re.I) else 'memcpy' if re.search('Memcpy',name,re.I) else None
    if kind is None:continue
    value=number(row['Time(us)']);count=number(row['Count'])
    if value is None or count is None:continue
    entry=classes.setdefault(kind,{'count':0,'time_sum_us':0.0,'native_names':[]});entry['count']+=count;entry['time_sum_us']+=value;entry['native_names'].append(name)
   worker['native_API_class_aggregates']=classes
   worker['additional_aggregate_scope']='Name-based grouping of native CSV rows. Nested spans and API waits may overlap; no addition to critical-path duration or inferred kernel batch size.'
   info['workers'][folder.name]=worker
  assert seen_pids==set(initialized)
  assert rank_ids==set(range(16) if NODE==34 else range(16,32))
  info['all16_local_model_workers_profiled']=True
  info['analysis_pass']=len(info['workers'])==16
 except Exception as error:
  info['error']=repr(error);info['traceback']=traceback.format_exc()
 for stream in ['stdout','stderr']:
  with (root/('model.'+stream+'.txt')).open(errors='replace') as file:
   for line in file:
    if any(marker in line for marker in ['Profiling starts','Profiling done','Stop profiling','Profiling is not in progress']):
     assert len(info['model_log_markers'])<512;info['model_log_markers'].append({'stream':stream,'line':line.rstrip()[:4000]})
'''

def load():
    p=ROOT/'scripts/glm53_sglang_C32_target_graph_native_capture_v3.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==BASE_SHA
    spec=importlib.util.spec_from_file_location('native_base',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    core=(ROOT/'scripts/glm53_profile_csv_analysis_v1.py').read_text();assert hashlib.sha256(core.encode()).hexdigest()==CORE_SHA
    extension=m.render(PROFILE,{'CORE':core})
    # NODE and FORMAL_RID are rendered by the original frozen capture function.
    m.HOST=m.HOST+extension
    return m,hashlib.sha256(extension.encode()).hexdigest()

def execute(relative,digest,capture_id):
    assert capture_id=='20261018T-L11-sglang-C32-P-two-step-native-v1'
    settings=json.loads((ROOT/relative).read_text());assert settings['run_id']=='20261018T-L11-sglang-C32-P-two-step-profile-v1'
    prior=json.loads((ROOT/'runs'/settings['run_id']/'receipt.json').read_text())
    assert prior.get('node43_rc')==0 and prior.get('client_rc') is not None and not prior.get('host_wait_timeout')
    module,extension_sha=load()
    receipt=module.execute(relative,digest,capture_id,mode='baseline')
    receipt.update(extension_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profile_template_extension_sha256=extension_sha,profile_CSV_analysis_source_sha256=CORE_SHA,raw_traces_transferred_to_Mac=False)
    (ROOT/'runs'/capture_id/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
