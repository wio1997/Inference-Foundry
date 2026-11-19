"""Generate an Ascend DB from one closed rank directory staged for offline analysis."""

import argparse
from contextlib import closing
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sqlite3


A2_PROFILER_SHA = '1a04902572383b74a401449e73929b2c4ce728d6c81bb59341876b03c7a82168'


def export_rank(root, rank):
    root = root.resolve(strict=True)
    info, = root.glob('profiler_info_*.json')
    assert info.name == f'profiler_info_{rank}.json' and json.loads(info.read_bytes())['rank_id'] == rank
    session, = root.glob('PROF_*')
    for entry in ('host/end_info', f'device_{rank}/end_info.{rank}'):
        assert (session / entry).stat().st_size > 0 and (session / (entry + '.done')).stat().st_size > 0
    assert (root / 'FRAMEWORK/torch.op_range').stat().st_size > 0
    output = root / 'ASCEND_PROFILER_OUTPUT'
    assert not output.exists()
    api = import_module('torch_npu.profiler.profiler')
    profiler_sha = sha256(Path(api.__file__).read_bytes()).hexdigest()
    assert profiler_sha == A2_PROFILER_SHA
    api.analyse(profiler_path=str(root), max_process_number=1, export_type=api.Constant.Db)
    database = output / f'ascend_pytorch_profiler_{rank}.db'
    assert database.stat().st_size > 0
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        devices = [int(row[0]) for row in db.execute('SELECT DISTINCT deviceId FROM TASK')]
        cpu = db.execute('SELECT COUNT(*) FROM PYTORCH_API').fetchone()[0]
        compute = db.execute('SELECT COUNT(*) FROM TASK WHERE globalTaskId IN (SELECT globalTaskId FROM COMPUTE_TASK_INFO)').fetchone()[0]
    assert devices == [rank] and cpu > 0 and compute > 0
    return {'profiler_source_SHA256': profiler_sha, 'rank': rank, 'selector': api.Constant.Db,
            'parsed_database_bytes': database.stat().st_size, 'CPU_API_rows': cpu, 'compute_TASK_rows': compute,
            'database_device_ids': devices}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--rank-dir', type=Path, required=True)
    cli.add_argument('--rank', type=int, choices=range(8), required=True)
    args = cli.parse_args()
    print('G1445_RANK_DB_JSON ' + json.dumps(export_rank(args.rank_dir, args.rank), sort_keys=True))
