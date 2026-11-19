python3 - <<'PY'
from pathlib import Path
import datetime,hashlib,json,socket

assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
control=client/'SG_PD_post_quality_usage_retry_control_g1259'
handle=json.loads((control/'actual_launch_handle.json').read_bytes())
assert handle['kind']=='usage_retry' and handle['outer_PID']==2955349
source=client/'SG_PD_usage_source_g1227/verify_SG_AISBench_usage_g1273.py'
assert str(source) in handle['argv']
assert hashlib.sha256(source.read_bytes()).hexdigest()=='0a460ac86fb4e7f62fc6f410aeab8de62fd5b88d3a75c0ed758469b2fe1d0746'
try:
    args=Path('/proc/2955349/cmdline').read_bytes().split(b'\0')
except FileNotFoundError:
    args=[]
live=str(source).encode() in args
output=Path(handle['output'])
assert output==client/'SG_PD_usage_retry_g1273'
stream=output/'response.sse'
raw=stream.read_bytes() if stream.is_file() else b''
frames=[line[5:].strip() for line in raw.split(b'\n')[:-1] if line.startswith(b'data:')]
decoded=[json.loads(line) for line in frames if line!=b'[DONE]']
last_usage=[row['usage'] for row in decoded if row.get('usage') is not None]
report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_outer_live':live,'outer_PID':handle['outer_PID'],
        'partial_SSE_bytes':len(raw),'complete_frames':len(frames),'saw_DONE':b'[DONE]' in frames,
        'actual_final_usage_fields_seen':last_usage,'contract_verified':False}
contract=output/'usage_contract.json'
if contract.is_file() and not live:
    value=json.loads(contract.read_bytes())
    assert value['reviewed'] and len(value['evidence'])==5
    assert last_usage==[value['actual_SGLang_final_usage']]
    for row in value['evidence']:
        p=(output/row['path']).resolve(strict=True)
        assert p.parent==output and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    report.update(contract_verified=True,contract=str(contract),contract_SHA256=hashlib.sha256(contract.read_bytes()).hexdigest(),
                  summary=json.loads((output/'summary.json').read_bytes()))
elif not live:
    failure=output/'terminal_failure.json'
    report['failure']=json.loads(failure.read_bytes()) if failure.is_file() else None
    report['log_tail']=(control/'diagnostic.log').read_text(errors='replace')[-4000:]
print('G1276_SG_USAGE_RETRY_OBSERVATION_JSON '+json.dumps(report))
PY
