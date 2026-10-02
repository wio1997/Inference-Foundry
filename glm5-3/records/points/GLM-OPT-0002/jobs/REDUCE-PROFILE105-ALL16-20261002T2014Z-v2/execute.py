from pathlib import Path
import json,sys,time,subprocess,shlex,hashlib,ast
sys.path.insert(0,'/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime')
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];exp=p/'jobs/EXPORT-PROFILE105-ALL16-20261002T1954Z';deadline=time.monotonic()+2400
while not(exp/'result.json').exists():
 assert time.monotonic()<deadline;time.sleep(10)
a=json.loads((exp/'result.json').read_text());assert a['status']=='completed'and a['execution']['inner_exit_code']==0
a=json.loads((exp/'reduction.json').read_text());assert a['profile_count']==16
first=json.loads((p/'jobs/REDUCE-PROFILE105-RANK0-20261002T1951Z/reduction.json').read_text());lo,hi=first['common_device_window_us'];code=(j/'device_reduce.py').read_text();ast.parse(code)
with(j/'reduce.stdout').open('wb')as out,(j/'reduce.stderr').open('wb')as err:
 z=subprocess.run(['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(['python3','-c',code,str(lo),str(hi)])],stdout=out,stderr=err,timeout=1200)
z.check_returncode();d=json.loads((j/'reduce.stdout').read_text());assert d['profile_count']==16
gate=json.loads((p/'runs/GLM-RUN-0105/profile_gate.json').read_text());by={int(x['container_pid']):x for x in gate['workers']};assert {x['pid']for x in d['profiles']}==set(by)
for x in d['profiles']:x['actual_native_worker']=by[x['pid']]
d.update(at=utc(),source_export_reference=dict(path=str(exp/'reduction.json'),sha256=hashlib.sha256((exp/'reduction.json').read_bytes()).hexdigest()))
atomic_json(j/'reduction.json',d);b=(j/'reduction.json').read_bytes();atomic_json(j/'result.json',dict(schema_version=1,job_id=j.name,status='completed',summary='Run105 all16worker native DB task/graph/nongraph/host API/clock/connection interval reduction completed; noGPUrerun/hardwarebound',execution=dict(inner_exit_code=0,acceptance='passed',processes=[]),findings=[],evidence=[dict(id='reduction',path=str(j/'reduction.json'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator='all16ownership/clock/nativeconnections/stepintervalunions')],unknowns=d['limits'],decision_request=None,next_check_at=None));print(json.dumps(dict(profiles=16,window=d['window_s'],perrank=[dict(pid=x['pid'],device=x['device'],graph=x['graph_compute_union_ms'],nongraph=x['non_graph_compute_union_ms'],steps=x['step_marker_count'],statistics=x['step_statistics'])for x in d['profiles']])))
