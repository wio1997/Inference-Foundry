"""Recompute the first 32K warmup's request timing from frozen AISBench SQLite."""

import argparse
from contextlib import ExitStack, closing
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean, median

from raw_sqlite_timing_audit_g751 import npy_float64


RECORD_SHA256 = '3ea6203e9094ef212a6220abf914bc3984f57a14c40a7bade3e522ba66868d7a'
REQUESTS = 16
INPUT = 32768
OUTPUT = 1024


def load_times(detail, rows):
    root = (detail.parent/'db_data').resolve(strict=True)
    pins, clocks, seen = {}, [], set()
    with ExitStack() as cleanup:
        connections = {}
        for row in rows:
            name = row['db_name']
            assert Path(name).name == name
            ref = row['time_points']
            assert set(ref) == {'__db_ref__'} and type(ref['__db_ref__']) is int
            key = name, ref['__db_ref__']
            assert key not in seen, 'Timing reference repeated across requests'
            seen.add(key)
            if name not in connections:
                path = (root/name).resolve(strict=True)
                assert path.parent == root
                wal = path.with_name(path.name+'-wal')
                assert not wal.exists() or wal.stat().st_size == 0
                pins[name] = sha256(path.read_bytes()).hexdigest()
                connections[name] = cleanup.enter_context(closing(sqlite3.connect(
                    path.as_uri()+'?mode=ro&immutable=1', uri=True)))
            record = connections[name].execute('SELECT arr_blob FROM numpy_store WHERE id=?',(key[1],)).fetchone()
            assert record is not None
            clocks.append(npy_float64(record[0]))
    assert all(sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in pins.items())
    return clocks, pins


def summarize(rows, clocks, reference):
    origin = min(times[0] for times in clocks)
    duration = max(times[-1] for times in clocks)-origin
    per_request = []
    for row, times in zip(rows, clocks):
        start, first, last = times[0], times[1], times[-1]
        per_request.append({'request_id':row['id'], 'data_id':row['data_id'], 'start_offset_s':start-origin,
            'first_token_offset_s':first-origin,'finish_offset_s':last-origin, 'TTFT_s':first-start,
            'post_first_token_s':last-first,'E2EL_s':last-start,'TPOT_ms':(last-first)*1000/(OUTPUT-1),
            'time_point_count':len(times)})
    per_request.sort(key=lambda row:row['first_token_offset_s'])
    active = peak = 0
    entries = [(times[0], 1) for times in clocks] + [(times[-1],-1) for times in clocks]
    for _, delta in sorted(entries):
        active += delta
        peak = max(active,peak)
    assert active == 0 and peak == reference['observed_max_concurrency'] == 16
    assert math.isclose(duration,reference['duration_s'],rel_tol=0,abs_tol=1e-6)
    for field, measured in [('ttft_mean_ms',1000*mean(row['TTFT_s'] for row in per_request)),
                            ('tpot_mean_ms',mean(row['TPOT_ms'] for row in per_request)),
                            ('e2el_mean_ms',1000*mean(row['E2EL_s'] for row in per_request))]:
        assert math.isclose(measured, reference[field], rel_tol=0, abs_tol=0.050001), (field,measured,reference[field])
    onsets = [row['first_token_offset_s'] for row in per_request]
    gaps = [late-early for early,late in zip(onsets,onsets[1:])]
    longest = max(per_request,key=lambda row:row['E2EL_s'])
    ratio = sum(row['TTFT_s'] for row in per_request)/sum(row['E2EL_s'] for row in per_request)
    return {'recomputed_duration_s':duration,'peak_concurrency':peak,'per_request_first_token_order':per_request,
            'median_inter_request_first_token_gap_s':median(gaps),'first_token_gap_range_s':[min(gaps),max(gaps)],
            'request_start_offset_range_s':[0,max(row['start_offset_s'] for row in per_request)],
            'prefirst_share_of_summed_request_E2EL':ratio,'longest_E2EL_request_id':longest['request_id'],
            'longest_E2EL_prefirst_share':longest['TTFT_s']/longest['E2EL_s'],
            'all_reference_aggregate_timings_match':True}


def audit(run_root):
    phase = run_root/'warmup1_32768'
    detail, = phase.rglob('*_details.jsonl')
    raw = detail.read_bytes()
    assert sha256(raw).hexdigest() == RECORD_SHA256
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == REQUESTS and {row['id'] for row in rows} == set(range(REQUESTS))
    assert len({row['data_id'] for row in rows}) == REQUESTS
    assert all(row['success'] is True and not row['error_info'] and row['input_tokens'] == INPUT
               and row['output_tokens'] == OUTPUT for row in rows)
    reference_path = run_root/'warmup1_32768_audit.json'
    reference_raw = reference_path.read_bytes()
    reference = json.loads(reference_raw)
    assert reference['fully_validated'] and reference['requests'] == REQUESTS
    assert reference['evidence'][str(detail.relative_to(phase))] == RECORD_SHA256
    clocks, db_pins = load_times(detail,rows)
    results = summarize(rows,clocks,reference)
    assert sha256(detail.read_bytes()).hexdigest()==RECORD_SHA256 and reference_path.read_bytes()==reference_raw
    return {'scope':'Original unscored SGLang 32K W1. One client clock. Prefirst timing includes the full client-to-first-token path.',
            'detail_SHA256':RECORD_SHA256,'source_phase_audit_SHA256':sha256(reference_raw).hexdigest(),
            'timing_database_SHA256':db_pins,'shape':{'requests':REQUESTS,'input':INPUT,'output':OUTPUT},**results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    results = audit(args.run_root.resolve(strict=True))
    with args.output.open('x') as stream:
        json.dump(results,stream,indent=2)
        stream.write('\n')
