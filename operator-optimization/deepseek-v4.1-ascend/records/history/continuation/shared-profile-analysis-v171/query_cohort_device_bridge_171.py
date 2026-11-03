import sys,json
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
H=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v171')
code=r'''import json,hashlib,ctypes,subprocess
from pathlib import Path
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis');rows=[]
finals={}
for p in (A/'joint168/workers').glob('*/FINAL.json'):
 d=json.loads(p.read_text());finals[d['pid']]={'path':str(p),'keys':list(d),'flat_metadata':{k:v for k,v in d.items() if not isinstance(v,(list,dict))}}
for p in sorted((A/'joint169').glob('*-SUMMARY.json')):
 s=json.loads(p.read_text());rp=A/'joint169'/(s['process_uuid']+'-REPLAYS.json');rr=json.loads(rp.read_text());ds=sorted({r['device_id'] for r in rr});hs=A/'joint168/derived'/(s['process_uuid']+'-rows.jsonl');first=json.loads(hs.open().readline());rows.append({'pid':s['pid'],'process_uuid':s['process_uuid'],'source_ep_rank':first['ep_rank'],'source_history_first_keys':list(first),'native_replay_device_ids':ds,'native_replay_pids':sorted({r['native_pid'] for r in rr}),'native_replay_count':len(rr),'final_source_metadata':finals[s['pid']]})
files={}
for name in ['parallel_group_info.json','RANK_DEVICE_MAP.json','NPU_INFO.json','profiler_metadata.json']:
 ps=sorted((A/'joint168').rglob(name));files[name]=[str(p) for p in ps]
groups=[]
for name,ps in files.items():
 for f in ps[:2]:
  p=Path(f);b=p.read_bytes()
  try:d=json.loads(b)
  except Exception as e:d={'parse_error':str(e)}
  groups.append({'path':f,'sha256':hashlib.sha256(b).hexdigest(),'data':d})
markers=[str(p) for p in (A/'joint168/runtime').glob('*_ascend_pt/PROF_*/device_*/info.json.*') if not p.suffix[1:].isdigit()]
ctypes.CDLL('libdcmi.so');paths=sorted({x.split()[-1] for x in Path('/proc/self/maps').read_text().splitlines() if 'libdcmi.so' in x and '/' in x});libs=[{'path':p,'resolved':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths if Path(p).is_file()]
x=json.loads(subprocess.run(['docker','inspect','wio-vllm-v41-repro171-20261102'],capture_output=True,text=True,check=True).stdout)[0]
r={'process_device_rows':rows,'capture_group_files':files,'group_examples':groups,'auxiliary_info_markers':markers,'dcmi_library_pins':libs,'current_container':{'id':x['Id'],'created':x['Created'],'image':x['Image'],'hostname':x['Config']['Hostname'],'started_at':x['State']['StartedAt']}}
(A/'joint171/COHORT-DEVICE-BRIDGE-INPUTS.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))'''
r=invoke('target',code,timeout=30);(H/'cohort-device-bridge-inputs.json').write_text(json.dumps(r,indent=2)+'\n');print('processrows',len(r['process_device_rows']),'groupfilecounts',{k:len(v) for k,v in r['capture_group_files'].items()},'markers',len(r['auxiliary_info_markers']));print('firstprocess',json.dumps(r['process_device_rows'][:1],ensure_ascii=False));print('group_examples',json.dumps(r['group_examples'],ensure_ascii=False)[:7000]);print('libraries',r['dcmi_library_pins'])
