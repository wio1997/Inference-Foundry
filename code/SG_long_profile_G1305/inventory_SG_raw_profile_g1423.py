"""List retained native CPU and device trace files for one SGLang profile role."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

PROFILER_INFO = re.compile(r'profiler_info_(\d+)\.json')
DEVICE_STREAM = re.compile(r'(?:ffts_profile|stars_soc)\.data\..+')
EXPECTED_RANKS = frozenset(range(8))


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def raw_rank(info, rank):
    folder = info.parent
    sessions = sorted(path for path in folder.glob('PROF_*') if path.is_dir())
    cpu = folder / 'FRAMEWORK/torch.op_range'
    cpu_bytes = cpu.stat().st_size if cpu.is_file() else 0
    database = folder / 'ASCEND_PROFILER_OUTPUT' / f'ascend_pytorch_profiler_{rank}.db'
    details = {
        'rank': rank, 'profiler_info_sha256': sha256(info.read_bytes()).hexdigest(),
        'raw_metadata_ok': True, 'raw_sessions': len(sessions),
        'framework_op_range_bytes': cpu_bytes, 'parsed_DB_file_present': nonempty(database),
    }
    if len(sessions) != 1:
        return {
            **details,
            'raw_cpu_and_device_files_present': False,
            'reason': 'expected_one_raw_profiler_session',
        }
    session = sessions[0]
    dev = session / f'device_{rank}'
    device_folders = sorted(path.name for path in session.glob('device_*') if path.is_dir())
    device_data = [
        path for path in (dev / 'data').glob('*')
        if DEVICE_STREAM.fullmatch(path.name) and not path.name.endswith('.done') and nonempty(path)
    ]
    end_files = (session / 'host/end_info', dev / f'end_info.{rank}')
    end_complete = all(nonempty(path) and nonempty(path.with_name(path.name + '.done'))
                       for path in end_files)
    complete = bool(cpu_bytes and device_data and end_complete and device_folders == [dev.name])
    return {
        **details,
        'device_trace_data_files': len(device_data),
        'device_trace_data_bytes': sum(path.stat().st_size for path in device_data),
        'device_trace_families': sorted({path.name.split('.data.', 1)[0] for path in device_data}),
        'expected_device_directory_only': device_folders == [dev.name],
        'host_and_device_end_records_present': end_complete,
        'raw_cpu_and_device_files_present': complete,
    }


def inventory(root):
    ranks = {}
    for info in sorted(root.glob('*_ascend_pt/profiler_info*.json')):
        match = PROFILER_INFO.fullmatch(info.name)
        if match is None:
            continue
        rank = int(match[1])
        payload = json.loads(info.read_bytes())
        if type(payload.get('rank_id')) is not int or payload['rank_id'] != rank:
            raise ValueError(f'Profiler metadata rank disagrees with filename for rank {rank}')
        if rank not in EXPECTED_RANKS or rank in ranks:
            raise ValueError(f'Unexpected or duplicate metadata for rank {rank}')
        ranks[rank] = raw_rank(info, rank)
    complete = {rank for rank, item in ranks.items() if item['raw_cpu_and_device_files_present']}
    parsed = {rank for rank, item in ranks.items() if item.get('parsed_DB_file_present')}
    return {
        'raw_cpu_and_device_ranks': sorted(complete),
        'missing_rank_metadata': sorted(EXPECTED_RANKS - ranks.keys()),
        'incomplete_raw_ranks': sorted(ranks.keys() - complete),
        'parsed_db_file_ranks': sorted(parsed),
        'raw_complete_ranks_missing_parsed_db': sorted(complete - parsed),
        'all8_raw_file_presence_verified': complete == EXPECTED_RANKS,
        'ranks': [ranks[rank] for rank in sorted(ranks)],
        'scope': ('File inventory for a closed, unmerged capture. Parsed database presence '
                  'still requires actual event, logical and rank coverage validation.'),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--role', choices=('prefill', 'decode'), required=True)
    args = cli.parse_args()
    root = args.root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Expected the profiler role output directory')
    report = inventory(root)
    print(json.dumps({'role': args.role, **report}, indent=2))


if __name__ == '__main__':
    main()
