python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, re, socket, subprocess, time, urllib.request

assert socket.gethostname() == '<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
gate_file=Path(__GATE_FILE__).resolve(strict=True)
proof_file=Path(__P_RUNTIME_PROOF_FILE__).resolve(strict=True)
assert gate_file.is_relative_to(client) and proof_file.is_relative_to(client)
kind=__DIAGNOSTIC_KIND__
assert kind in ('long','usage')
gate=json.loads(gate_file.read_bytes())
proof=json.loads(proof_file.read_bytes())
expected=proof['result']
now=datetime.datetime.now(datetime.timezone.utc)
age=(now-datetime.datetime.fromisoformat(gate['validated_at'])).total_seconds()
assert 0<=age<480
assert gate['task_id']==root.name and gate['stage']=='diagnostic' and gate['profile_epoch']==1178
assert gate['engine']=='sglang' and gate['model_id']=='ds41flash'
assert all(gate[key] for key in ('ready','pd_transfer_verified','configuration_verified','workload_guard_actual_verified'))
assert proof['terminal_verified'] and proof['raw_files_verified'] and expected['math1']=='91'
assert expected['completed']==3
for label in ('transfer_count_delta_by_P_rank','D_full_graph_counter_delta_by_rank'):
    rows=expected[label]
    assert set(map(int,rows))==set(range(8)) and min(rows.values())>0
assert gate['roles']['prefill']['container_id']==proof['handle']['router']['P_container_ID']
assert gate['roles']['decode']['container_id']==proof['handle']['router']['D_container_ID']
assert hashlib.sha256(proof_file.read_bytes()).hexdigest()==gate['source_receipts_sha256']['PD_runtime']

frozen_file=client/'SG_PD_g1178_full_GSM8K_control_g1217/frozen_actual_manifest_g1249.json'
frozen=json.loads(frozen_file.read_bytes())
assert frozen['complete_1319'] and frozen['completed']==1319
assert frozen['request_worker_sha256']=='aefe2729cd0803b037c828e900ebe24d40e76dcafb7ba4f96905bd05894ad245'
model=root/'sg_pd_g1178'
decode=json.loads((model/'state.json').read_bytes())
assert decode['pid']==2826773 and decode['generation']==1178
assert decode['container_id']==gate['roles']['decode']['container_id']
assert str(model/'node_controller.py').encode() in Path('/proc/2826773/cmdline').read_bytes().split(b'\0')

cid='d8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
procs=subprocess.check_output(['docker','top',cid,'-eo','pid,args'],text=True,timeout=10).splitlines()
assert len(procs)==2 and 'sleep 72000' in procs[1]
role_queue={'prefill':('num_prefill_bootstrap_queue_reqs','num_prefill_inflight_queue_reqs'),
            'decode':('num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs')}
for role,meta in gate['roles'].items():
    with urllib.request.urlopen(meta['base_url']+'/ready',timeout=6) as f:
        assert f.status==200
    with urllib.request.urlopen(meta['base_url']+'/metrics',timeout=8) as f:
        raw=f.read().decode()
    for key in ('num_running_reqs','num_queue_reqs',*role_queue[role]):
        values=[float(v) for v in re.findall(r'^sglang:'+key+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',raw,re.M)]
        assert values and sum(values)==0, (role,key,values)

control=client/f'SG_PD_post_quality_{kind}_control_g1259'
assert not control.exists()
if kind=='long':
    source=client/'SG_diagnostic_tools_g1254/verify_SG_long_context_g1202.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='7fa1f15aa13b5e25d49754bebf516482cfa893ee9d9419360be5dfe487a0d1ea'
    assets=root/'client_ds41_g503/long_quality_g703'
    output=client/'SG_PD_g1178_long_retrieval_g1259'
    args=['--assets',str(assets),'--output',str(output),'--runtime-proof',str(control/'runtime_result.json')]
    limit=3690
else:
    source=client/'SG_PD_usage_source_g1227/verify_SG_AISBench_usage_g1227.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='44ed698ba925154b484bafa6a90c826a64050ad1c7032dbc4b2521872ab389d2'
    request=source.parent/'native_protocol_probe_request.json'
    assert hashlib.sha256(request.read_bytes()).hexdigest()=='c3e55a36beb6639f13f1ab8d6489ccd140061dcdde1577dff4f3705a07cdfc2d'
    output=client/'SG_PD_usage_g1227'
    args=['--root',str(root),'--request',str(request)]
    limit=660
assert not output.exists()
assert gate['guard_remaining_seconds']-age>limit+300
control.mkdir(exist_ok=False)
with (control/'runtime_result.json').open('x') as f:
    json.dump(expected,f,indent=2)
argv=['timeout','--signal=TERM','--kill-after=20',str(limit),
      'python3','-u',str(source),'--gate',str(gate_file),*args]
with (control/'diagnostic.log').open('x') as f:
    proc=subprocess.Popen(argv,cwd=control,stdin=subprocess.DEVNULL,
                          stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
report={'at':now.isoformat(),'kind':kind,'outer_PID':proc.pid,'argv':argv,'output':str(output),
        'quality_manifest_sha256':hashlib.sha256(frozen_file.read_bytes()).hexdigest(),
        'gate_sha256':hashlib.sha256(gate_file.read_bytes()).hexdigest(),
        'scope':'One independent post-quality diagnostic. Preserve the original handle and raw evidence.'}
with (control/'actual_launch_handle.json').open('x') as f:
    json.dump(report,f,indent=2)
time.sleep(2)
try:
    command=Path('/proc',str(proc.pid),'cmdline').read_bytes().split(b'\0')
except FileNotFoundError:
    command=[]
print('G1259_SG_POST_QUALITY_DIAGNOSTIC_JSON '+json.dumps({
    **report,'actual_outer_live':proc.poll() is None and str(source).encode() in command,
    'immediate_exit_code':proc.poll()}))
PY
