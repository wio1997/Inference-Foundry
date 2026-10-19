from pathlib import Path
import hashlib
import importlib.util
import json
import time
r=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
start=time.monotonic()
while not(r/'early-sample-shadow-reviewed-v1.json').exists():
    assert time.monotonic()-start<1600
    time.sleep(15)
gate=json.loads((r/'early-sample-shadow-reviewed-v1.json').read_text())
assert gate['completed'] and gate['bitwise_all'] and gate['observed_ranks']==list(range(16))
assert gate['remaining_model_s']>1000
contract=json.loads((r/'early-sample-shadow-contract-v2.json').read_text())
script=r/'profile_early_sample_shadow_v1.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()==contract['profile_script_sha256']
x=m.collect(script,'early_sample_shadow_c4_profile_v1','early-sample-shadow-c4-profile-v1',timeout=360)
assert x['completed'] and x['shadow_restored_off'] and x['observer_restored_off']
assert len(x['profiles'])==1 and x['profiles'][0]['profile_stopped'] and x['profiles'][0]['ranks']==list(range(16))
print(json.dumps(dict(completed=True,phase='profile',raw_bytes=x['profiles'][0]['source_bytes'])),flush=True)
script=r/'export_early_sample_shadow_v1.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()==contract['export_script_sha256']
x=m.collect(script,'early_sample_shadow_c4_export_v1','early-sample-shadow-c4-export-v1',timeout=650)
assert x['completed'] and len(x['modes'])==1 and x['modes'][0]['compute_devices_used']==[]
print(json.dumps(dict(completed=True,phase='CPU-only-export',modes=x['modes'])),flush=True)
