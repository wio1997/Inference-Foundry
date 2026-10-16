#!/usr/bin/env python3
"""Start a finite task-owned MemCache metadata service on node32; no data pool/NPU/SSD."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/meta32-v1')
NAME = 'glm53-memcache-meta32-20261017'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert not ROOT.exists(), 'Do not overwrite a prior metadata launch'
    prior = subprocess.run(['docker', 'inspect', NAME], capture_output=True)
    assert prior.returncode != 0, 'Container already exists'
    for port in (27400, 27401, 27402):
        with socket.socket() as sock:
            sock.bind(('172.16.12.32', port))
    ROOT.mkdir(parents=True)
    config = ('ock.mmc.meta_service_url = tcp://172.16.12.32:27400\n'
              'ock.mmc.meta_service.config_store_url = tcp://172.16.12.32:27401\n'
              'ock.mmc.meta_service.metrics_url = http://172.16.12.32:27402\n'
              'ock.mmc.log_level = info\n'
              'ock.mmc.log_output_target = screen\n')
    (ROOT/'mmc-meta.conf').write_text(config)
    argv = ['docker', 'run', '-d', '--pull', 'never', '--name', NAME, '--network', 'host',
            '--cpus', '2', '--memory', '4g', '--pids-limit', '256', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '-v', str(ROOT)+':/out',
            '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '-e', 'MMC_META_CONFIG_PATH=/out/mmc-meta.conf',
            '-e', 'ASCEND_PROCESS_LOG_PATH=/out/ascend-log',
            '--entrypoint', '/usr/bin/timeout', IMAGE, '--signal=TERM', '--kill-after=10', '2700',
            'python3', '-c', 'from memcache_hybrid import MetaService; raise SystemExit(MetaService.main())']
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    result = {'argv': argv, 'rc': proc.returncode, 'container_id': proc.stdout.strip(),
              'stderr': proc.stderr, 'config_sha256': hashlib.sha256(config.encode()).hexdigest(),
              'metadata_ready': False, 'scope': 'isolated metadata/config service; no NPU data or SSD'}
    (ROOT/'launch.json').write_text(json.dumps(result, indent=2)+'\n')
    assert proc.returncode == 0, result
    deadline = time.monotonic()+12
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(('172.16.12.32', 27400), timeout=1):
                pass
            result['metadata_ready'] = True
            break
        except OSError:
            time.sleep(1)
    logs = subprocess.run(['docker', 'logs', result['container_id']], capture_output=True, text=True, timeout=10)
    result['logs'] = logs.stdout+logs.stderr
    (ROOT/'launch.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['metadata_ready'], 'Metadata socket not ready; preserve failure and inspect logs'


if __name__ == '__main__':
    main()
