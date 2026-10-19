#!/usr/bin/env python3
"""Preserve stopped raw profiles and export using CPU/driver queries only."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
    role='prefill' if node==32 else 'decode'
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd');root=base/f'{role}-node{node}-v1'
    out=base/'consumer-audits'/f'node{node}-profile-export-v1';out.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(root).free>8*1024**3
    manifest=json.loads((base/'consumer-audits'/f'node{node}-profile-summary-v1/profile-manifest.json').read_text())
    preserved=root/'profile-raw-preserved';preserved.mkdir(exist_ok=False)
    for item in manifest:
        source=root/item['path'];target=preserved/source.relative_to(root/'profile')
        assert source.is_file() and not source.is_symlink() and digest(source)==item['sha256']
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        assert digest(target)==item['sha256']
    image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    argv=['docker','run','--rm','--pull','never','--network','none','--name',f'glm53-official-profile-export-{node}-v1-20261017',
        '--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','64g','--pids-limit','4096',
        '--tmpfs','/tmp:rw,size=2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1',
        '--device','/dev/davinci_manager','-v',str(root/'profile')+':/profile','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
        '--entrypoint','/usr/bin/timeout',image,'--signal=TERM','--kill-after=15','420','python3','-c',
        "from torch_npu.profiler.profiler import analyse; analyse('/profile', max_process_number=4, export_type='text')"]
    start=time.monotonic()
    with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
        p=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=450)
    report={'node':node,'returncode':p.returncode,'elapsed_s':time.monotonic()-start,'argv':argv,
        'compute_devices':[],'query_devices':['/dev/davinci_manager'],'preserved_raw_files':len(manifest),
        'preserved_raw_bytes':sum(x['bytes'] for x in manifest),'preserved_raw_sha256_verified':True,
        'trace_paths':[str(p.relative_to(root)) for p in sorted((root/'profile').rglob('trace_view.json'))]}
    for name in ('stdout.txt','stderr.txt'):
        report[name]={'sha256':digest(out/name),'tail':(out/name).read_text(errors='replace').splitlines()[-8:]}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    assert p.returncode==0


if __name__=='__main__':main()
