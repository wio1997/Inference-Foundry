#!/usr/bin/env python3
"""Run the bounded CPU input-mode gate in a second private owned directory."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from launch_common_frame_torch_cpu import REMOTE
ROOT=Path('audit/execution-search')
def main():
    old=json.loads((ROOT/'common-frame-torch-cpu.json').read_text());guard=json.loads((ROOT/'common-frame-cpu-import-guard.json').read_text());deps=json.loads((ROOT/'common-frame-key-dependencies.json').read_text());helper=next(r for r in deps['sources'] if r['runtime_path'].endswith('/dsa_cp.py'))
    probe=Path('tools/probe_common_frame_cpu_modes.py');payload=dict(container_root='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/cpu-frame-run-20261110T110329Z',probe_source=probe.read_text(),probe_sha256=sha256(probe.read_bytes()).hexdigest(),spec=dict(helper_source_path=helper['runtime_path'],helper_source_sha256=helper['runtime_sha256'],torch_init_sha256=guard['torch_init_sha256'],prototype_source_path=old['container_root']+'/probe.py',prototype_source_sha256=old['probe_sha256']))
    (ROOT/'common-frame-cpu-modes-launch.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='probe_source'},ensure_ascii=False,indent=2)+'\n')
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(payload)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);d.update(launcher_sha256=sha256(Path(__file__).read_bytes()).hexdigest());(ROOT/'common-frame-cpu-modes.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(returncode=d['returncode'],host_root=d['host_root'],elapsed_s=d['elapsed_s'],stdout=d['stdout'],stderr=d['stderr']),ensure_ascii=False))
if __name__=='__main__':main()
