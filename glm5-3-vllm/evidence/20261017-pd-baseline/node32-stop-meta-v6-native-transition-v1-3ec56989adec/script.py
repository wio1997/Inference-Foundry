#!/usr/bin/env python3
"""Close only the finite Meta-v5 after both owned model instances have exited."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',choices=('source-ready','post-pool'),default='source-ready');args=parser.parse_args()
    p_revision,meta_revision=(14,5) if args.deployment=='source-ready' else (15,6)
    # Local source must be closed first; caller separately retains D13 close evidence.
    p = json.loads((base/f'pd-node32-v{p_revision}/launch.json').read_text())['container_id']
    assert not json.loads(subprocess.check_output(['docker','inspect',p,'--format','{{json .State}}']))['Running']
    c = json.loads((base/f'meta32-v{meta_revision}/launch.json').read_text())['container_id']
    doc = json.loads(subprocess.check_output(['docker','inspect',c]))[0]
    assert doc['Name'] == f'/glm53-memcache-meta32-baseline-v{meta_revision}-20261017'
    assert doc['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    if doc['State']['Running']:
        subprocess.run(['docker','stop','-t','30',c],check=True,capture_output=True,timeout=40)
    state = json.loads(subprocess.check_output(['docker','inspect',c,'--format','{{json .State}}']))
    logs = subprocess.run(['docker','logs','--timestamps',c],capture_output=True,timeout=30)
    raw = logs.stdout+b'\nSTDERR\n'+logs.stderr
    out = base/f'consumer-audits/meta32-v{meta_revision}-lifecycle'
    out.mkdir(parents=True,exist_ok=False)
    (out/'console.txt.gz').write_bytes(gzip.compress(raw,compresslevel=1))
    report = {'container_id':c,'state':state,'raw_console_bytes':len(raw),'raw_console_sha256':hashlib.sha256(raw).hexdigest()}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not state['Running'] and not state['OOMKilled'] and state['ExitCode'] == 0


if __name__ == '__main__':
    main()
