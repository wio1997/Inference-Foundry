python3 - <<'PY'
from pathlib import Path
import datetime
import json
import os
import re
import socket
import subprocess
import time
import urllib.request

D = __DECODE_PROOF__
host = socket.gethostname()
assert host == '<PREFILL_HOSTNAME>' and os.geteuid() == 0
root = Path('/PRIVATE_TASK_ROOT')
run = root / 'sg_pd_g1178'
now = datetime.datetime.now(datetime.timezone.utc)
assert -60 <= (now - datetime.datetime.fromisoformat(D['at'])).total_seconds() < 600
assert D['ready'] == 200 and D['graph_capture_all8_verified'] and D['all8_ownership_and_FDs_verified']
assert D['handle']['epoch'] == 1178 and D['handle']['role'] == 'd'
assert D['handle']['guard_PID'] == 2227468
assert all(vals and sum(vals) == 0 for vals in D['idle'].values())

def cmdline(pid):
    return Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')

def fds(pid):
    links=set()
    for fd in Path('/proc',str(pid),'fd').iterdir():
        try: links.add(os.readlink(fd))
        except FileNotFoundError: pass
    return links

def get(url):
    with urllib.request.urlopen(url, timeout=7) as response:
        assert response.status == 200
        return response.read()

handle = json.loads((run/'actual_launch_handle.json').read_text())
state = json.loads((run/'state.json').read_text())
pid = handle['pid']
cid = state['container_id']
assert pid == 2240397 and handle['role']=='p' and handle['epoch']==1178
assert str(run/'node_controller.py').encode() in cmdline(pid)
assert str(root/'workload_guard_g1120.py').encode() in cmdline(2227468)
coord = Path('/PRIVATE_COORD_ROOT')
assert str(coord/'deepseek-v41-p3d4-workload.lock') in fds(2227468)
container = json.loads(subprocess.check_output(['docker','inspect',cid],timeout=12))[0]
labels = container['Config']['Labels']
assert container['State']['Running'] and container['Image']==handle['image']
assert labels['codex.run']==run.name and labels['codex.controller_pid']==str(pid)
owned = fds(pid)
for rank in range(8):
    stem=coord/f'{host}-npu{rank}'
    row=json.loads(stem.with_suffix('.owner.json').read_text())
    assert row['pid']==pid and row['state']=='claimed' and row['run_directory']==str(run)
    assert str(stem.with_suffix('.lock')) in owned

prefill='http://<ROLE_HOST>:19093'
decode='http://<ROLE_HOST>:19094'
for expected_mode,base in [('prefill',prefill),('decode',decode)]:
    get(base+'/ready')
    report=json.loads(get(base+'/server_info'))
    args=report.get('server_args',report)
    assert args['disaggregation_mode']==expected_mode and args['disaggregation_transfer_backend']=='ascend'
    assert args['speculative_algorithm']=='DSPARK' and args['speculative_num_draft_tokens']==6
    assert (args['tp_size'], args['ep_size'], args['pp_size']) == (8,8,1)
    assert not args['disable_radix_cache'] and not args['disable_overlap_schedule']
    if expected_mode=='decode':
        assert args['disaggregation_decode_enable_radix_cache']
        assert args['cuda_graph_config']['decode']['backend']=='full'
    metrics=get(base+'/metrics').decode()
    for key in ('num_running_reqs','num_queue_reqs'):
        vals=[float(v) for v in re.findall(r'^sglang:'+key+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',metrics,re.M)]
        assert vals and sum(vals)==0

layout=json.loads(get('http://127.0.0.1:18998/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1'))
assert layout['dsv41_spec_layout']['num_draft_tokens']==6
assert layout['kv_layout']=='v4,c1:v4,c2:v4' and layout['attn_tp_size']==8
router_handle=run/'router_handle.json'
assert not router_handle.exists(), 'Existing router belongs to this epoch. Observe its handle instead of starting again.'
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('0.0.0.0',18090))
argv=['flock','--nonblock',str(coord/f'{host}-port18090.lock'),
      'docker','exec','-w','/task/'+run.name,cid,
      'python3','-m','sglang_router.launch_router','--host','0.0.0.0','--port','18090',
      '--pd-disaggregation','--prefill',prefill,'18998','--decode',decode,
      '--max-concurrent-requests','16','--queue-size','100','--request-timeout-secs','3600',
      '--disable-retries','--health-check-endpoint','/ready']
with (run/'router.log').open('x') as log:
    child=subprocess.Popen(argv,cwd=run,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'at':now.isoformat(),'router_PID':child.pid,'argv':argv,'P_container_ID':cid,'D_container_ID':D['container_ID'],
         'P_controller_PID':pid,'D_controller_PID':D['handle']['pid'],'guard_PID':2227468,'bootstrap':layout}
with router_handle.open('x') as f:
    json.dump(receipt,f,indent=2)
print('G1189_SG_ROUTER_START_JSON '+json.dumps(receipt),flush=True)
deadline=time.monotonic()+55
while time.monotonic()<deadline:
    assert child.poll() is None,'Router exited. Review its log and handle.'
    try:
        table=json.loads(get('http://127.0.0.1:18090/workers'))
    except urllib.error.URLError:
        time.sleep(2)
        continue
    ready={(w['worker_type'],w['url']) for w in table['workers']
           if w['is_healthy'] and w['load']==0 and w['model_id']=='ds41flash'}
    if ready=={('prefill',prefill),('decode',decode)} and len(table['workers'])==2:
        report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'handle':receipt,'workers':table,
                'router_live':True,'scope':'Router health and exact P/D endpoint registration. No model requests submitted.'}
        print('G1189_SG_ROUTER_READY_JSON '+json.dumps(report))
        break
    time.sleep(2)
else:
    print('G1189_SG_ROUTER_PENDING_JSON '+json.dumps({'router_live':child.poll() is None,'handle':receipt,
                                                    'scope':'Readiness wait expired. Observe this same process.'}))
PY
