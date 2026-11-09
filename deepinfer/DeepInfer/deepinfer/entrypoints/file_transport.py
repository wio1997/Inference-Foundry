"""File transport identity and receipts; independent of model computation."""
import hashlib
import json
from pathlib import Path
import time

PROTOCOL = 'deepinfer-original-engine-v1'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    temporary.replace(path)


def process_birth(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[19]


def source_identity():
    root = Path(__file__).resolve().parents[2]
    return hashlib.sha256((root/'ascend910c/source-manifest.json').read_bytes()).hexdigest()


def read_ready(directory, *, world_size, source=None):
    rows = [json.loads((Path(directory)/f'ready-rank{rank}.json').read_text()) for rank in range(world_size)]
    identity = rows[0]['source'] if source is None else source
    for rank, row in enumerate(rows):
        if row.get('protocol') != PROTOCOL or row.get('rank') != rank or row.get('source') != identity:
            raise ValueError('ready receipts differ from the original Engine deployment')
        if row.get('state') != 'READY' or not row.get('pid') or not row.get('starttime'):
            raise ValueError('original Engine ready receipt is incomplete')
    return rows


def heartbeat(directory, dp_rank, peers, *, job_id=None):
    try:
        live = all(process_birth(row['pid']) == row['starttime'] for row in peers)
    except OSError:
        live = False
    value = dict(protocol=PROTOCOL, source=peers[0]['source'], dp_rank=dp_rank,
                 observed=time.time(), healthy=live, job_id=job_id)
    atomic_json(Path(directory)/f'heartbeat-dp{dp_rank}.json', value)
    if not live:
        raise RuntimeError('an original Worker process incarnation changed')


def ready_health(directory, peers, *, dp_size, max_age=90):
    directory = Path(directory)
    if any(directory.glob('fatal-*.json')):
        return False
    now = time.time()
    for rank in range(dp_size):
        try:
            row = json.loads((directory/f'heartbeat-dp{rank}.json').read_text())
        except (OSError, ValueError):
            return False
        if (row.get('protocol') != PROTOCOL or row.get('source') != peers[0]['source']
                or row.get('dp_rank') != rank or row.get('healthy') is not True
                or not 0 <= now-row.get('observed', 0) <= max_age):
            return False
    return True
