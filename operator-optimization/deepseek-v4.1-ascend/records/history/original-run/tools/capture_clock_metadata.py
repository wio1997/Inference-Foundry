"""Read frozen profiler conversion source and trace clock anchors; no NPU use."""
import hashlib,json,shlex,subprocess,datetime,base64
from pathlib import Path
T=Path(__file__).resolve().parents[1]
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
code=r'''
import pathlib,json,hashlib,base64
B=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
root=pathlib.Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
torch=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler')
paths={'cann_info_conf_reader.py':root/'common_func/info_conf_reader.py','cann_host_start.py':root/'profiling_bean/basic_info/host_start.py','torch_profiler_interface.py':torch/'profiler_interface.py','torch_profiler_config.py':torch/'analysis/_profiler_config.py','torch_cann_file_parser.py':torch/'analysis/prof_parse/_cann_file_parser.py','torch_path_manager.py':torch/'analysis/prof_common_func/_path_manager.py'}
sources={}
for k,p in paths.items():
 b=p.read_bytes();sources[k]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'content_base64':base64.b64encode(b).decode()}
rows=[]
for p in B.glob('*_ascend_pt'):
 r=json.loads(next(p.glob('profiler_info_*.json')).read_text())['rank_id']
 h=json.loads(next(p.glob('PROF_*/host/start_info')).read_text())
 d=json.loads(next(q for q in p.glob('PROF_*/device_*/start_info.*') if not q.name.endswith('.done')).read_text())
 info=json.loads(next(q for q in p.glob('PROF_*/device_*/info.json.*') if not q.name.endswith('.done')).read_text())
 a=dict(s.split(': ',1) for s in next(p.glob('PROF_*/host/host_start.log')).read_text().splitlines() if ': ' in s)
 off=lambda x:int(x['collectionTimeBegin'])*1000-int(x['clockMonotonicRaw'])
 rows.append({'profile_directory':p.name,'rank':r,'device_offset_ns':off(d),'host_offset_ns':off(h),'device_host_offset_difference_ns':off(d)-off(h),'host_intercept_ns':int(a['clock_monotonic_raw'])-10*int(a['cntvct']),'hwts_frequency_mhz':info['DeviceInfo'][0]['hwts_frequency']})
rows.sort(key=lambda x:x['rank'])
print(json.dumps({'sources':sources,'rows':rows,'spans':{k:max(x[k] for x in rows)-min(x[k] for x in rows) for k in ['device_offset_ns','host_offset_ns','host_intercept_ns']}}))
'''
r=subprocess.run([ssh,shlex.join(['docker','exec',stage,'python3','-c',code])],capture_output=True,text=True,timeout=120,check=True)
d=json.loads(r.stdout);assert len(d['rows'])==16
expected={'cann_info_conf_reader.py':'044899552770c109ea9e18ad0282c097cccce22b22b564af2ce52ac9566b23e4','cann_host_start.py':'75c65c4e7dfc17c13b76b18b255aaf6f4849408f300cc008aefc9d9b4501e012','torch_profiler_interface.py':'8183b6470d552c33907784252f4009050aa7066955371c460232bc60b8e0af65','torch_profiler_config.py':'1d96b590e3904a2201a51d09778438d6c83cc4862e2e59cd52a03adec54bd3eb','torch_cann_file_parser.py':'fb47549af5aa1547461884d43150107eeca9131869959495bee8e5403fc9d4a3'}
# Correction: compare complete known digest as recorded by the independent source audit.
expected['torch_profiler_config.py']='1d96b590e3904a2201a51d09778438d6c83cc4862e2e59cd52a03adec54bd3eb'
for name,v in d['sources'].items():
 b=base64.b64decode(v.pop('content_base64'));assert hashlib.sha256(b).hexdigest()==v['sha256']
 if name in expected:assert v['sha256']==expected[name],(name,v['sha256'])
 p=T/'sources/primary/frozen-image-clock'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
 v['snapshot_path']=str(p.relative_to(T))
d.update(status='CLOCK_SOURCE_AND_METADATA_AUDIT',timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_device_work=True,
 limitations=['Metadata anchor spans are observations at nonidentical times, not certified bounds on cross-device or cross-process event timing error.','Device clock conversion uses separate anchors; exporter/parser path and anchor capture delay and drift are not proven.','Cross-DP operation pairing is a separate unresolved hypothesis.'],
 interpretation='Time conversion is source documented; do not subtract an observed anchor span as if it were a physical synchronization error bound.')
p=T/'evidence/clock-source-and-anchor-audit.json';assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'ranks':len(d['rows']),'spans_ns':d['spans'],'source_files':len(d['sources'])}))
