"""Join Prefill and Decode marks to the same frozen AISBench requests."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

from audit_decode_prefix_trace_g1389 import MARKER as DECODE_MARKER, PREFIX as DECODE_PREFIX
from bind_decode_prefix_profile_g1391 import DIGEST, PROMPT_FIELD, verified_requests
from parse_SG_step_graph_g1213 import sqlite_integer
from prefill_room_annotation_g1485 import POLL_PREFIX, SEND_PREFIX


ROOM = re.compile(r'room=([a-f0-9]{16}) prompt=(\d+)')


def read_pinned(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Unexpected artifact digest for {path.name}')
    return raw


def sqlite_scopes(path, expected, prefixes):
    digest = sha256()
    with path.open('rb') as raw:
        for piece in iter(lambda: raw.read(1024*1024), b''):
            digest.update(piece)
    if digest.hexdigest() != expected:
        raise ValueError(f'Unexpected database digest for {path.name}')
    if any(Path(f'{path}{suffix}').exists() for suffix in ('-wal', '-shm', '-journal')):
        raise ValueError('An immutable export is required')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        for prefix in prefixes:
            rows = db.execute(
                'SELECT names.value, api.startNs, api.endNs FROM PYTORCH_API AS api '
                'JOIN STRING_IDS AS names ON api.name = names.id '
                'WHERE substr(names.value, 1, ?) = ? ORDER BY CAST(api.startNs AS INTEGER)',
                (len(prefix), prefix),
            )
            for label, start, end in rows:
                yield label, sqlite_integer(start), sqlite_integer(end)


def collect_p(rows, prompt_tokens):
    joined = {}
    for label, start, end in rows:
        stage, sep, metadata = label.partition(' ')
        if not sep or stage + sep not in (SEND_PREFIX, POLL_PREFIX):
            raise ValueError('Unexpected Prefill marker')
        match = ROOM.fullmatch(metadata)
        if match is None or int(match[2]) != prompt_tokens or not 0 <= start <= end:
            raise ValueError('Malformed or mismatched Prefill room event')
        by_stage = joined.setdefault(match[1], {})
        key = 'last_chunk_call' if stage + sep == SEND_PREFIX else 'scheduler_transfer_observed'
        if key in by_stage:
            raise ValueError('Duplicate Prefill stage for one room')
        by_stage[key] = [start, end]
    for parts in joined.values():
        if len(parts) != 2 or parts['last_chunk_call'][1] > parts['scheduler_transfer_observed'][0]:
            raise ValueError('Missing or reversed Prefill stages')
    return joined


def collect_d(rows, expected, prompt_tokens):
    joined = {}
    seen = set()
    for label, start, end in rows:
        base, sep, digest = label.rpartition(PROMPT_FIELD)
        match = DECODE_MARKER.fullmatch(base)
        if not sep or match is None or not DIGEST.fullmatch(digest) or digest not in expected:
            raise ValueError('Decode mark does not bind to an actual frozen request')
        if match[1] == 'unavailable' or int(match[2]) != prompt_tokens or match[3] != '0' or not 0 <= start <= end:
            raise ValueError('Decode room identity, shape, fresh transfer or timing mismatch')
        if digest in seen or match[1] in joined:
            raise ValueError('Ambiguous Decode prompt or room')
        joined[match[1]] = {**expected[digest], 'D_kv_commit_ns': [start, end]}
        seen.add(digest)
    if seen != set(expected):
        raise ValueError('Decode trace has incomplete frozen request coverage')
    return joined


def join(p_rows, d_rows, expected, prompt_tokens):
    prefill = collect_p(p_rows, prompt_tokens)
    decode = collect_d(d_rows, expected, prompt_tokens)
    if not prefill or prefill.keys() != decode.keys():
        raise ValueError('Prefill and Decode do not have the same complete room set')
    combined = [{**decode[room], 'P_local_scopes_ns': prefill[room]} for room in decode]
    return {'joined_requests':len(combined),'requests':sorted(combined,key=lambda row:row['data_id']),
            'clock_scope':'P markers share the Prefill export clock. D markers use the Decode export clock. '
                          'No cross-host timestamp subtraction is computed.',
            'boundary_scope':'P final-chunk invocation and the transfer scheduler return after its stream call, '
                             'joined with D KV commit, using one fresh room per frozen request. HTTP and Router '
                             'require separate instrumentation.'}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for label in ('prefill','decode','manifest','details'):
        cli.add_argument('--'+label,type=Path,required=True)
        cli.add_argument('--'+label+'-sha256',required=True)
    cli.add_argument('--prompt-tokens',type=int,default=32768)
    cli.add_argument('--output-tokens',type=int,default=1024)
    args=cli.parse_args()
    manifest=json.loads(read_pinned(args.manifest,args.manifest_sha256))
    raw=read_pinned(args.details,args.details_sha256)
    expected=verified_requests(manifest,[json.loads(row) for row in raw.splitlines() if row.strip()],
                               args.prompt_tokens,args.output_tokens)
    p=sqlite_scopes(args.prefill,args.prefill_sha256,(SEND_PREFIX,POLL_PREFIX))
    d=sqlite_scopes(args.decode,args.decode_sha256,(DECODE_PREFIX,))
    report=join(p,d,expected,args.prompt_tokens)
    print(json.dumps({'dataset_sha256':manifest['dataset_sha256'],**report},indent=2,sort_keys=True))


if __name__=='__main__':
    main()
