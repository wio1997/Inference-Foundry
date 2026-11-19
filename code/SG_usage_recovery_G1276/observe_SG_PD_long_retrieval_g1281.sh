python3 - <<'PY'
from pathlib import Path
import datetime,hashlib,json,socket

assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT/client_ds41_g1122')
control=root/'SG_PD_post_quality_long_control_g1259'
handle=json.loads((control/'actual_launch_handle.json').read_bytes())
assert handle['kind']=='long' and handle['outer_PID']==2956446
source=root/'SG_diagnostic_tools_g1254/verify_SG_long_context_g1202.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='7fa1f15aa13b5e25d49754bebf516482cfa893ee9d9419360be5dfe487a0d1ea'
assert str(source) in handle['argv']
try:
    argv=Path('/proc/2956446/cmdline').read_bytes().split(b'\0')
except FileNotFoundError:
    argv=[]
alive=str(source).encode() in argv
output=Path(handle['output'])
assert output==root/'SG_PD_g1178_long_retrieval_g1259'
summary={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_outer_live':alive,'outer_PID':handle['outer_PID']}
for label in ('state','result','manifest'):
    path=output/(label+'.json')
    summary[label]=json.loads(path.read_bytes()) if path.is_file() else None
if summary['manifest']:
    state=summary['manifest']
    assert state['request_manifest_sha256']=='2b2a82681901b3e19e0ed8b799b1e12085a9f7bd4f2caf90d30d62750e484b5d'
    assert state['selected_lengths']==[32768,65536,131072,262144]
    summary['manifest']={'request_manifest_sha256':state['request_manifest_sha256'],'selected_lengths':state['selected_lengths']}
summary['audits']=[json.loads(path.read_bytes()) for path in sorted(output.glob('*_audit.json'))]
if not alive:
    summary['log_tail']=(control/'diagnostic.log').read_text(errors='replace')[-3500:]
print('G1281_SG_LONG_RETRIEVAL_OBSERVATION_JSON '+json.dumps(summary))
PY
