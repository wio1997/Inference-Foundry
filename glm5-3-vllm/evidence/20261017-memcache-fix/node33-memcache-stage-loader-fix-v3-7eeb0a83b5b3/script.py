#!/usr/bin/env python3
"""Stage a task-local wheel-library de-duplication; never modify the base image."""
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3')
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
STAGE = r'''
import hashlib,json,pathlib,shutil
src=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
dst=pathlib.Path('/fix/site-packages');dst.mkdir()
for name in ('memcache_hybrid','memcache_hybrid.libs','memfabric_hybrid','memfabric_hybrid.libs'):
    assert (src/name).is_dir(), name
    shutil.copytree(src/name,dst/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
aliases=[]
for package in ('memcache_hybrid','memfabric_hybrid'):
    for lib in sorted((dst/package/'lib').glob('libmf_*.so')):
        candidates=list((dst/(package+'.libs')).glob(lib.stem+'-*.so'))
        if not candidates: continue
        assert len(candidates)==1, (lib,candidates)
        canonical=candidates[0]
        old_sha=hashlib.sha256(lib.read_bytes()).hexdigest()
        new_sha=hashlib.sha256(canonical.read_bytes()).hexdigest()
        import os
        target=os.path.relpath(canonical,lib.parent)
        lib.unlink();lib.symlink_to(target)
        aliases.append({'alias':str(lib.relative_to(dst)),'target':target,'old_sha256':old_sha,'canonical_sha256':new_sha})
assert any('libmf_memcache.so' in a['alias'] for a in aliases)
assert any('libmf_smem.so' in a['alias'] for a in aliases)
init=dst/'memcache_hybrid/__init__.py'
before=init.read_text()
after=before.replace('ctypes.CDLL(so_path)', 'ctypes.CDLL(so_path, mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)')
assert after != before, 'Expected explicit MemCache preloader mode'
init.write_text(after)
(dst.parent/'python-patch.json').write_text(json.dumps({'path':'memcache_hybrid/__init__.py','before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'change':'Load MemCache/UBSIO with LOCAL|DEEPBIND, matching upstream isolation mechanism; no change to exit or cleanup.'},indent=2)+'\n')
(dst.parent/'aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
print(json.dumps(aliases,indent=2))
'''


def main():
    assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32', 'B5-5U-Atlas900RCKA3-33')
    ROOT.mkdir(parents=True, exist_ok=False)
    argv = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
            '--read-only', '--cpus', '2', '--memory', '2g',
            '-v', str(ROOT)+':/fix', '--entrypoint', 'python3', IMAGE, '-c', STAGE]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    (ROOT/'stage.stdout').write_text(p.stdout)
    (ROOT/'stage.stderr').write_text(p.stderr)
    assert p.returncode == 0, p.stderr
    print(json.dumps({'root': str(ROOT), 'aliases': json.loads(p.stdout)}, indent=2))


if __name__ == '__main__':
    main()
