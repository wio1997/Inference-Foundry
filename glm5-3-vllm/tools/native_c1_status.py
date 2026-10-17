#!/usr/bin/env python3
"""Read native model startup, effective config and graph evidence without profiling."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.request


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,choices=(1,2),default=1);args=parser.parse_args()
    node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
    mode='eager' if node==32 else 'graph'
    root=Path(f'/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/native-{mode}-node{node}-v{args.revision}')
    launch=json.loads((root/'launch.json').read_text());cid=launch['container_id']
    state=json.loads(subprocess.check_output(['docker','inspect',cid,'--format','{{json .State}}']))
    p=subprocess.run(['docker','logs','--timestamps',cid],capture_output=True,timeout=30)
    raw=p.stdout+b'\nDOCKER_STDERR\n'+p.stderr
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=root/f'console-{stamp}.txt';out.write_bytes(raw)
    text=raw.decode(errors='replace');lines=text.splitlines()
    errors=[s for s in lines if re.search(r'ERROR|Traceback|RuntimeError|ValidationError',s)]
    graphs=[s for s in lines if re.search(r'Native benchmark startup|graph|async.schedul',s,re.I)]
    result={'node':node,'mode':mode,'container_id':cid,'state':state,'api_ready':False,'console':{'path':str(out),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'native_errors':errors[-40:],'graph_and_scheduler_evidence':graphs[-60:],'console_tail':lines[-10:]}
    try:
        with urllib.request.urlopen(f'http://172.16.12.{node}:{29481 if node==32 else 29482}/v1/models',timeout=3) as response:o=json.load(response)
        result['api_ready']=any(x['id']=='glm53-w8a8c8-native' for x in o['data'])
    except Exception as exc:result['api_error']=str(exc)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
