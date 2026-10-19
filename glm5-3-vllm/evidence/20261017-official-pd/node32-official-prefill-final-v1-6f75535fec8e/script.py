#!/usr/bin/env python3
"""Read the two owned official-PD engine states and preserve their native logs."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--label', required=True); args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.label)
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    role = 'prefill' if node == 32 else 'decode'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    root = base/f'{role}-node{node}-v1'
    cid = json.loads((root/'launch.json').read_text())['container_id']
    doc = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert doc['Name'] == f'/glm53-official-pd-{role}-{node}-v1-20261017'
    p = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=30)
    raw = p.stdout+b'\nSTDERR\n'+p.stderr; lines = raw.decode(errors='replace').splitlines()
    out = base/'consumer-audits'/f'node{node}-{args.label}'; out.mkdir(parents=True, exist_ok=False)
    (out/'console.txt.gz').write_bytes(gzip.compress(raw, compresslevel=1))
    report = {'node': node, 'role': role, 'container_id': cid, 'state': doc['State'],
        'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'api_ready': False,
        'raw_console_sha256': hashlib.sha256(raw).hexdigest(), 'raw_console_bytes': len(raw),
        'errors': [s for s in lines if re.search(r'ERROR|Traceback|RuntimeError|ValidationError', s)][-45:],
        'connector_graph_transfer_lines': [s for s in lines if re.search(r'Mooncake|TransferEngine|transfer|remote|replay|Graph capturing finished|HCCS|SDMA|RDMA|protocol|Loading.*checkpoint|Loading model weights', s, re.I)][-75:],
        'completed_transfer_lines': [s for s in lines if 'KV cache transfer for requests' in s],
        'actual_graph_replay': [s for s in lines if 'ACL graph replay is active' in s],
        'graph_capture': [s for s in lines if 'Graph capturing finished' in s],
        'transport_lines': [s for s in lines if re.search(r'HCCS|SDMA|RDMA|[Ll]ink[Tt]ype', s)][-30:],
        'tail': lines[-6:], 'raw_logs_directory': str(out)}
    report['inference_errors']=[s for s in report['errors'] if 'profiler.py: The profiling data cannot be parsed during the daemon process' not in s]
    report['automatic_profiler_parse_errors']=[s for s in report['errors'] if 'profiler.py: The profiling data cannot be parsed during the daemon process' in s]
    url = f'http://172.16.12.{node}:{29481 if node == 32 else 29482}'
    try:
        with urllib.request.urlopen(url+'/v1/models', timeout=3) as response:
            report['models'] = json.load(response)
        report['api_ready'] = any(m['id'] == 'glm5' for m in report['models']['data'])
        with urllib.request.urlopen(url+'/metrics', timeout=5) as response:
            metrics = response.read(); (out/'metrics.txt').write_bytes(metrics)
        report['transfer_metrics'] = [s for s in metrics.decode().splitlines()
            if not s.startswith('#') and re.search(r'mooncake|external|kv_transfer|requests_running|requests_waiting', s, re.I)]
        if node==32:
            with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=3) as response:
                report['proxy_health']=json.load(response)
    except Exception as exc:
        report['api_error'] = str(exc)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
