"""Verify live SGLang SSE token usage with the native reference request bytes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

CLIENT_IMAGE = 'sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
CLIENT_ID = 'd8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
REQUEST_SHA = 'c3e55a36beb6639f13f1ab8d6489ccd140061dcdde1577dff4f3705a07cdfc2d'
SOURCES = {
    'base_api.py': '203a416444b972193274a13e1593328a288b6879a3d62e4f201bbb9adfa81755',
    'vllm_custom_api.py': '8170e1ca344340a4919e05fb61df7bfdcd8d7f7b7cca9d427924d793c89da086',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, data=None, timeout=15):
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        assert response.status == 200
        return response.read()


def sse_usage(raw):
    events = [line[5:].strip() for line in raw.splitlines() if line.startswith(b'data:')]
    assert events and events[-1] == b'[DONE]' and events.count(b'[DONE]') == 1
    objects = [json.loads(event) for event in events[:-1]]
    assert objects and all(event['model']=='ds41flash' for event in objects)
    usage = [event['usage'] for event in objects if event.get('usage') is not None]
    assert len(usage) == 1 and objects[-1]['usage'] == usage[0]
    assert usage[0]['prompt_tokens'] == 32768 and usage[0]['completion_tokens'] == 1024
    assert usage[0]['total_tokens'] == 33792
    finishes = [choice['finish_reason'] for event in objects for choice in event.get('choices', [])
                if choice.get('finish_reason') is not None]
    assert finishes == ['length']
    assert any(choice.get('text') for event in objects for choice in event.get('choices', []))
    return {'usage': usage[0], 'events': len(objects), 'done': True, 'finish': finishes[0]}


def main(args):
    assert socket.gethostname() == '<ROLE_HOSTNAME>'
    task = args.root.resolve(strict=True)
    client_root = task / 'client_ds41_g1122'
    output = client_root / 'SG_PD_usage_retry_g1273'
    assert not output.exists(), 'Preserve any existing attempt'
    gate = json.loads(args.gate.read_bytes())
    at = datetime.datetime.now(datetime.timezone.utc)
    assert 0 <= (at-datetime.datetime.fromisoformat(gate['validated_at'])).total_seconds() < 600
    assert gate['task_id'] == task.name and gate['profile_epoch'] == 1178 and gate['engine'] == 'sglang'
    for key in ('ready', 'pd_transfer_verified', 'configuration_verified', 'resources_verified',
                'workload_guard_actual_verified'):
        assert gate[key] is True, key
    assert gate['model_id'] == 'ds41flash'
    assert set(gate['roles']) == {'prefill','decode'}
    state = json.loads((task/'sg_pd_g1178/state.json').read_bytes())
    assert state['generation'] == 1178 and state['role'] == 'd'
    assert state['container_id'] == gate['roles']['decode']['container_id']
    assert str(task/'sg_pd_g1178/node_controller.py').encode() in Path(
        '/proc', str(state['pid']), 'cmdline').read_bytes().split(b'\0')
    client = json.loads(subprocess.check_output(['docker', 'inspect', CLIENT_ID], timeout=12))[0]
    assert client['Id']==CLIENT_ID and client['Image']==CLIENT_IMAGE and client['State']['Running']
    assert not client['HostConfig']['Devices'] and client['HostConfig']['CpusetCpus']=='184-187'
    processes = subprocess.check_output(['docker', 'top', CLIENT_ID, '-eo', 'pid,args'],
                                        text=True, timeout=10).splitlines()
    assert len(processes) == 2 and 'sleep 72000' in processes[1]
    installed = Path('/proc', str(client['State']['Pid']), 'root',
                     'benchmark/ais_bench/benchmark/models/api_models')
    data = {name: (installed/name).read_bytes() for name in SOURCES}
    assert {name:sha(raw) for name,raw in data.items()} == SOURCES
    body = args.request.read_bytes()
    assert sha(body)==REQUEST_SHA
    payload = json.loads(body)
    assert isinstance(payload.pop('prompt'), str)
    assert payload == {'model':'ds41flash','max_tokens':1024,'min_tokens':1024,'ignore_eos':True,
                       'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
    router = gate['proxy_url'].rstrip('/')
    assert json.loads(fetch(router+'/v1/models'))['data'][0]['id']=='ds41flash'
    metrics = {}
    for role, item in gate['roles'].items():
        base = item['base_url'].rstrip('/')
        fetch(base+'/ready')
        raw = fetch(base+'/metrics')
        for key in (b'num_running_reqs', b'num_queue_reqs'):
            values = re.findall(rb'^sglang:'+key+rb'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)$',raw,re.M)
            assert values and sum(map(float,values)) == 0, (role,key,values)
        metrics[role] = raw

    output.mkdir(exist_ok=False)
    for name,raw in data.items():
        (output/name).write_bytes(raw)
    (output/'request.json').write_bytes(body)
    for role,raw in metrics.items():
        (output/f'before_{role}.prom').write_bytes(raw)
    started=time.monotonic()
    first_frame_s=None
    first_content_s=None
    frames=0
    response_path=output/'response.sse'
    def alarm(*_):
        raise TimeoutError('SSE usage retry exceeded its 1800-second bound')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1800)
    try:
        request=urllib.request.Request(router+'/v1/completions',data=body,
                                       headers={'Content-Type':'application/json'})
        with response_path.open('xb') as log, urllib.request.urlopen(request,timeout=1700) as response:
            assert response.status==200
            for line in response:
                log.write(line)
                log.flush()
                if not line.startswith(b'data:'):
                    continue
                frames+=1
                elapsed=time.monotonic()-started
                if first_frame_s is None:
                    first_frame_s=elapsed
                event=line[5:].strip()
                if first_content_s is None and event!=b'[DONE]':
                    item=json.loads(event)
                    if any(choice.get('text') for choice in item.get('choices',[])):
                        first_content_s=elapsed
    except BaseException as exc:
        if isinstance(exc,urllib.error.HTTPError):
            (output/'HTTP_error.txt').write_bytes(exc.read())
        failure={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(exc),
                 'frames_seen':frames,'first_SSE_frame_s':first_frame_s,'first_nonempty_content_s':first_content_s,
                 'partial_response_bytes':response_path.stat().st_size if response_path.is_file() else 0}
        (output/'terminal_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
    finally:
        signal.alarm(0)
    raw=response_path.read_bytes()
    summary = sse_usage(raw)
    for role,item in gate['roles'].items():
        (output/f'after_{role}.prom').write_bytes(fetch(item['base_url'].rstrip('/')+'/metrics'))
    summary.update(at=at.isoformat(), cpu_container=CLIENT_ID, gate_sha256=sha(args.gate.read_bytes()),
                   same_native_diagnostic_request_sha256=REQUEST_SHA, sg_epoch=1178,
                   frames_seen=frames, first_SSE_frame_s=first_frame_s, first_nonempty_content_s=first_content_s,
                   complete_SSE_s=time.monotonic()-started, maximum_request_bound_s=1800,
                   prior_independent_probe_retained='SG_PD_usage_g1227')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    names = [*SOURCES,'request.json','response.sse','summary.json']
    contract = {'input_tokens_source':'server_usage.prompt_tokens', 'output_tokens_source':'server_usage.completion_tokens',
                'reviewed':True, 'evidence':[{'path':name,'sha256':sha((output/name).read_bytes())} for name in names],
                'actual_SGLang_final_usage':summary['usage'],
                'basis':'Live pinned AISBench files request include_usage and parse server fields. The same native unscored prompt returned 32768/1024 and SSE DONE on SGLang G1178.',
                'scope':'Independent standard text/SSE protocol check, outside all reserved AISBench phases.'}
    with (output/'usage_contract.json').open('x') as file:
        json.dump(contract,file,indent=2)
        file.write('\n')
    print('G1273_SG_AISBENCH_USAGE_PROTOCOL_JSON '+json.dumps(summary),flush=True)


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,required=True)
    cli.add_argument('--gate',type=Path,required=True)
    cli.add_argument('--request',type=Path,required=True)
    main(cli.parse_args())
