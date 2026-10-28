#!/usr/bin/env python3
"""Recreate missing project containers from measured metadata; default is plan."""
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    configs = json.loads((package / 'config/container-manifest.json').read_text())
    extra = json.loads((package / 'config/container-extra.json').read_text())
    if args.execute:
        assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-172', 'wrong server'
    for name, cfg in configs.items():
        cmd = ['docker', 'run', '-d', '--pull=never', '--name', name, '--restart=no',
               '--runtime', extra[name]['runtime'], '--read-only', '--network', cfg['network_mode'],
               '--ipc', cfg['ipc_mode'], '--shm-size', str(cfg['shm_size']), '--log-driver=none']
        if cfg['privileged']:
            cmd.append('--privileged')
        for cap in cfg['cap_add']:
            cmd += ['--cap-add', cap]
        for value in cfg['ulimits']:
            cmd += ['--ulimit', f"{value['Name']}={value['Soft']}:{value['Hard']}"]
        for key, value in cfg['environment_overrides'].items():
            cmd += ['-e', key + '=' + value]
        for bind in cfg['binds']:
            cmd += ['-v', bind]
        for device in cfg['devices']:
            cmd += ['--device', f"{device['PathOnHost']}:{device['PathInContainer']}:{device['CgroupPermissions']}"]
        assert cfg['entrypoint'] == ['bash']
        cmd += ['-w', cfg['working_dir'], '--entrypoint', 'bash', cfg['image_id'], *cfg['command']]
        print(shlex.join(cmd))
        if not args.execute:
            continue
        found = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
        if found.returncode == 0:
            actual = json.loads(found.stdout)[0]
            assert actual['Image'] == cfg['image_id'], 'existing name/image differs'
            assert actual['State']['Running'], 'existing stopped container: inspect before starting'
            print(name, 'already running; retained')
            continue
        image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', cfg['image_id']], text=True))[0]
        assert image['Id'] == cfg['image_id']
        for bind in cfg['binds']:
            assert Path(bind.split(':', 1)[0]).exists(), bind
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
