"""Check real P/D generation, per-rank KV transfer, and Decode graph replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path('/PRIVATE_TASK_ROOT')
RUN = ROOT/'sg_pd_g1178'
OUT = RUN/'PD_runtime_gate_g1190'
BASE = {'prefill':'http://<ROLE_HOST>:19093', 'decode':'http://<ROLE_HOST>:19094'}
ROUTER = 'http://127.0.0.1:18090'
MODEL = 'ds41flash'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(name, value):
    result = OUT/name
    temp = result.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(result)

def get(url, body=None, timeout=7):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        assert response.status == 200, response.status
        return response.read()

def proc_has(pid, arg):
    return arg.encode() in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')

def verify_epoch():
    handle = json.loads((RUN/'router_handle.json').read_text())
    state = json.loads((RUN/'state.json').read_text())
    assert state['generation'] == 1178 and state['role'] == 'p'
    assert state['pid'] == handle['P_controller_PID'] == 2240397
    assert state['container_id'] == handle['P_container_ID']
    assert proc_has(state['pid'],str(RUN/'node_controller.py'))
    container = json.loads(subprocess.check_output(['docker','inspect',state['container_id']],timeout=10))[0]
    assert container['State']['Running'] and container['Config']['Labels']['codex.controller_pid']==str(state['pid'])
    folder = Path('/proc',str(handle['guard_PID']))
    assert handle['guard_PID'] == 2227468 and proc_has(handle['guard_PID'],str(ROOT/'workload_guard_g1120.py'))
    assert any(f.resolve()==Path('/PRIVATE_COORD_ROOT/deepseek-v41-p3d4-workload.lock') for f in (folder/'fd').iterdir())
    assert proc_has(handle['router_PID'],'sglang_router.launch_router')
    workers = json.loads(get(ROUTER+'/workers'))['workers']
    assert len(workers)==2
    assert {(w['worker_type'],w['url']) for w in workers if w['is_healthy'] and w['load']==0 and w['model_id']==MODEL}==set(BASE.items())
    return handle

def series(text, name, role, mode=None):
    prefix = 'sglang:'+name
    rows = {}
    for line in text.splitlines():
        if not line.startswith(prefix+'{'):
            continue
        head, value = line.rsplit(' ',1)
        labels = dict(re.findall(r'([a-z_]+)="([^"]*)"',head))
        if labels.get('engine_type') != role or labels.get('model_name') != MODEL:
            continue
        if mode is not None and labels.get('mode') != mode:
            continue
        assert labels['pp_rank']=='0' and labels['tp_rank']==labels['moe_ep_rank']
        rank=int(labels['tp_rank'])
        assert rank not in rows
        rows[rank]=float(value)
    return rows

def snapshot(tag):
    result = {}
    for role, url in BASE.items():
        metrics = get(url+'/metrics').decode()
        (OUT/f'{tag}_{role}.prom').write_text(metrics)
        result[role] = metrics
    return result

def delta(before, after, field, role, mode=None):
    start = series(before[role],field,role,mode)
    end = series(after[role],field,role,mode)
    assert set(start) <= set(end)
    result={rank:value-start.get(rank,0.0) for rank,value in end.items()}
    assert all(v>=0 for v in result.values()),(field,result)
    return result

def main():
    assert socket.gethostname()=='<PREFILL_HOSTNAME>'
    handle = verify_epoch()
    OUT.mkdir(exist_ok=False)
    start=time.monotonic()
    def alarm(*_):
        raise TimeoutError('P/D functional gate exceeded 600 seconds')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(600)
    save('launch_state.json',{'phase':'actual_running','at':now(),'pid':os.getpid(),'router':handle})
    try:
        before=snapshot('before')
        question={'model':MODEL,'messages':[{'role':'user','content':'Case C16-g999-01. Compute 14 times 6, then add 7. Reply with only the integer.'}],
                  'temperature':0,'seed':0,'max_tokens':128,'stream':False,
                  'chat_template_kwargs':{'thinking':False}}
        save('math_request.json',question)
        raw=get(ROUTER+'/v1/chat/completions',question,timeout=120)
        (OUT/'math_response.json').write_bytes(raw)
        response=json.loads(raw)
        assert response['model']==MODEL and response['choices'][0]['finish_reason']=='stop'
        assert response['choices'][0]['message']['content'].strip()=='91'
        assert response['usage']['prompt_tokens']>0 and response['usage']['completion_tokens']>0
        prompt='SG P/D verification 1178. Read the terminology. '+('row column matrix product. '*48)+'\nExplain matrix multiplication in a paragraph.'
        request={'model':MODEL,'prompt':prompt,'temperature':0,'seed':0,'max_tokens':64,
                 'min_tokens':64,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
        save('completion_request.json',request)
        uses=[]
        for tag in ('fresh','replay'):
            raw=get(ROUTER+'/v1/completions',request,timeout=160)
            (OUT/f'{tag}_response.sse').write_bytes(raw)
            events=[line[5:].strip() for line in raw.splitlines() if line.startswith(b'data:')]
            assert events[-1]==b'[DONE]' and events.count(b'[DONE]')==1
            parsed=[json.loads(item) for item in events[:-1]]
            usage=[item['usage'] for item in parsed if item.get('usage')]
            assert len(usage)==1 and usage[0]['prompt_tokens']>=128 and usage[0]['completion_tokens']==64
            ends=[c['finish_reason'] for e in parsed for c in e.get('choices',[]) if c.get('finish_reason')]
            assert ends==['length']
            uses.append({'case':tag,'usage':usage[0],'SHA256':hashlib.sha256(raw).hexdigest()})
        deadline=time.monotonic()+25
        while True:
            after=snapshot('after')
            transfers=delta(before,after,'kv_transfer_latency_ms_count','prefill')
            graphs=delta(before,after,'cuda_graph_passes_total','decode','decode_cuda_graph')
            if set(transfers)==set(graphs)==set(range(8)) and min(transfers.values())>0 and min(graphs.values())>0:
                break
            assert time.monotonic()<deadline,(transfers,graphs)
            time.sleep(2)
        mb=delta(before,after,'kv_transfer_total_mb_sum','prefill')
        failed=delta(before,after,'num_bootstrap_failed_reqs_total','prefill')
        assert set(mb)==set(range(8)) and min(mb.values())>0
        assert all(value==0 for value in failed.values())
        verify_epoch()
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
        save('result.json',{'phase':'complete_requires_parent_review','math1':'91','completed':3,'usage':uses,
            'transfer_count_delta_by_P_rank':transfers,'transfer_mb_delta_by_P_rank':mb,
            'D_full_graph_counter_delta_by_rank':graphs,'bootstrap_failure_delta_by_P_rank':failed,
            'file_SHA256':hashes,'elapsed_s':time.monotonic()-start,
            'scope':'Short real P/D requests and positive transfer/replay counters. Full quality and benchmark are separate.'})
    except BaseException as exc:
        save('failure.json',{'at':now(),'error':repr(exc),'pid':os.getpid()})
        raise

if __name__ == '__main__':
    main()
