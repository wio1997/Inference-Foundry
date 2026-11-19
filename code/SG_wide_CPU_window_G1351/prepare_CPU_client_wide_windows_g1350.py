"""Create unactivated CPU lease tickets from the inspected predecessor and wide-phase plan."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('contract', 'windows', 'output', 'canonical-lock'):
        cli.add_argument('--'+key, required=True, type=Path)
    cli.add_argument('--activation-reserve-seconds', required=True, type=int)
    args = cli.parse_args()
    contract, contract_hash = read(args.contract)
    windows, windows_hash = read(args.windows)
    lease = contract['lease']
    host = contract['configuration']
    assert not host['Devices'] and not host['DeviceRequests'] and not host['Privileged']
    assert host['CpusetCpus']=='184-187' and host['NanoCpus']==4_000_000_000 and host['Memory']==32*1024**3
    assert contract['entrypoint']==['sleep'] and contract['sleep_seconds']==72000
    assert args.activation_reserve_seconds > 0
    mounts={row['Destination']:row for row in contract['mounts']}
    root=mounts['/task']
    client=mounts[contract['working_dir']]
    assert not root['RW'] and client['RW'] and Path(client['Source']).parent==Path(root['Source'])
    suffix='-cpu'+host['CpusetCpus']+'.lock'
    assert args.canonical_lock.is_absolute() and args.canonical_lock.name.endswith(suffix)
    execution_host = args.canonical_lock.name.removesuffix(suffix)
    controller = Path(__file__).with_name('cpu_client_long_window_g1350.py')
    source_sha=sha256(controller.read_bytes()).hexdigest()
    args.output.mkdir(exist_ok=False)
    for width in ('131072','262144'):
        minimum=windows['candidate_windows'][width]['remaining_CPU_and_pair_lease_required_at_formal_submit_seconds']
        assert minimum>windows['current_CPU_lease_limit_seconds']==70000
        lifetime=minimum+args.activation_reserve_seconds
        width_k=int(width)//1024
        ticket={
            'schema':1,'host':execution_host,'task_root':root['Source'],'client_directory':client['Source'],
            'canonical_cpu_lock':str(args.canonical_lock),'cpuset':host['CpusetCpus'],
            'predecessor_container_ID':lease['container_ID'],'predecessor_immutable_config_sha256':lease['immutable_config_SHA256'],
            'cpu_client_image_ID':contract['image'],'input_tokens':int(width),
            'successor_container_name':f'yxy-ds41-aisbench-g1350-{width_k}k',
            'minimum_required_formal_window_seconds':minimum,'lease_seconds':lifetime,
            'container_sleep_seconds':lifetime+(contract['sleep_seconds']-windows['current_CPU_lease_limit_seconds']),
            'predecessor_contract_sha256':contract_hash,'workload_windows_sha256':windows_hash,'controller_sha256':source_sha,
        }
        location=args.output/f'{width_k}k'
        location.mkdir()
        (location/'assignment.json').write_text(json.dumps(ticket,indent=2)+'\n')
        print(json.dumps({'width':int(width),'minimum_formal_window_seconds':minimum,'CPU_lease_candidate_seconds':lifetime}))


if __name__=='__main__':
    main()
