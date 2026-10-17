#!/usr/bin/env python3
"""Bounded process and native-error inspection for this task's model container."""
import json
from pathlib import Path
import re
import socket
import subprocess


def main():
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    root = Path(f'/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node{node}-fixed-v5')
    container = json.loads((root/'launch.json').read_text())['container_id']
    code = '''import pathlib,json
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if not any(s in cmd for s in ('vllm','spawn_main','resource_tracker')): continue
  stat=(p/'status').read_text()
  fields={k:v.strip() for k,v in (s.split(':',1) for s in stat.splitlines() if ':' in s) if k in ('Name','State','Threads','VmRSS')}
  rows.append({'pid':p.name,'cmdline':cmd[:300],'status':fields,'wchan':(p/'wchan').read_text()})
 except (OSError,ValueError): pass
print(json.dumps(rows))
'''
    p = subprocess.run(['docker', 'exec', container, 'python3', '-c', code], capture_output=True, text=True, timeout=30)
    smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=25)
    errors = []
    for f in sorted((root/'ascend-log').rglob('*')):
        if not f.is_file() or f.stat().st_size > 32*1024*1024:
            continue
        for s in f.read_text(errors='replace').splitlines():
            if re.search(r'ERROR|ERR\]|timeout|failed|invalid', s, re.I):
                errors.append(str(f.relative_to(root))+': '+s[-2000:])
    print(json.dumps({'node': node, 'processes': json.loads(p.stdout) if p.returncode == 0 else p.stderr,
                      'npu_smi': smi.stdout, 'native_errors': errors[-60:]}, indent=2))


if __name__ == '__main__':
    main()
