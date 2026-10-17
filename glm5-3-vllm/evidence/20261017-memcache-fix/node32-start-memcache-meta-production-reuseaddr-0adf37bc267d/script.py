#!/usr/bin/env python3
"""Start task-owned metadata with verified loader isolation and a finite lifetime."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/meta32-production-v3')
FIX = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3')
NAME = 'glm53-memcache-meta32-production-v3-20261017'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert not ROOT.exists()
    assert (FIX/'aliases.json').is_file() and (FIX/'python-patch.json').is_file()
    for port in (27400, 27401, 27402):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('172.16.12.32', port))
    ROOT.mkdir(parents=True)
    config = ('ock.mmc.meta_service_url = tcp://172.16.12.32:27400\n'
              'ock.mmc.meta_service.config_store_url = tcp://172.16.12.32:27401\n'
              'ock.mmc.meta_service.metrics_url = http://172.16.12.32:27402\n'
              'ock.mmc.log_level = info\n'
              'ock.mmc.log_output_target = screen\n')
    (ROOT/'mmc-meta.conf').write_text(config)
    env = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE, '--format', '{{json .Config.Env}}']))
    ld = next((v.split('=', 1)[1] for v in env if v.startswith('LD_LIBRARY_PATH=')), '')
    py = next((v.split('=', 1)[1] for v in env if v.startswith('PYTHONPATH=')), '')
    argv = ['docker', 'run', '-d', '--pull', 'never', '--name', NAME, '--network', 'host',
            '--cpus', '4', '--memory', '4g', '--pids-limit', '512', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '-v', str(ROOT)+':/out',
            '-v', str(FIX/'site-packages')+':/opt/memcache-runtime:ro',
            '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '-e', 'PYTHONPATH=/opt/memcache-runtime'+(':'+py if py else ''),
            '-e', 'OMP_NUM_THREADS=1',
            '-e', 'LD_LIBRARY_PATH=/opt/memcache-runtime/memcache_hybrid/lib:/opt/memcache-runtime/memfabric_hybrid/lib:'+ld,
            '-e', 'MMC_META_CONFIG_PATH=/out/mmc-meta.conf',
            '-e', 'ASCEND_PROCESS_LOG_PATH=/out/ascend-log',
            '--entrypoint', '/usr/bin/timeout', IMAGE, '--signal=TERM', '--kill-after=10', '7200',
            'python3', '-c', 'from memcache_hybrid import MetaService; raise SystemExit(MetaService.main())']
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    result = {'argv': argv, 'rc': proc.returncode, 'container_id': proc.stdout.strip(),
              'stderr': proc.stderr, 'config_sha256': hashlib.sha256(config.encode()).hexdigest(),
              'aliases_sha256': hashlib.sha256((FIX/'aliases.json').read_bytes()).hexdigest(),
              'python_patch': json.loads((FIX/'python-patch.json').read_text()), 'metadata_ready': False}
    (ROOT/'launch.json').write_text(json.dumps(result, indent=2)+'\n')
    assert proc.returncode == 0, result
    deadline = time.monotonic()+90
    while time.monotonic() < deadline:
        try:
            for port in (27400, 27401, 27402):
                with socket.create_connection(('172.16.12.32', port), timeout=1):
                    pass
            result['metadata_ready'] = True
            break
        except OSError:
            time.sleep(1)
    logs = subprocess.run(['docker', 'logs', result['container_id']], capture_output=True, text=True, timeout=10)
    result['logs'] = logs.stdout+logs.stderr
    (ROOT/'launch.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['metadata_ready']


if __name__ == '__main__':
    main()
