"""Task-scoped verification of completed frozen AISBench timestamp evidence."""
import argparse
import ast
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct


def read_array(blob):
    if blob[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError('expected pinned NumPy v1 timestamp encoding')
    length = struct.unpack('<H', blob[8:10])[0]
    header = ast.literal_eval(blob[10:10 + length].decode())
    if header['descr'] != '<f8' or header['fortran_order'] or len(header['shape']) != 1:
        raise ValueError('timestamp array layout differs from verified AISBench encoding')
    values = struct.unpack('<' + 'd' * math.prod(header['shape']), blob[10 + length:])
    if len(values) < 3 or not all(map(math.isfinite, values)) or any(a > b for a, b in zip(values, values[1:])):
        raise ValueError('invalid or non-monotonic request timestamps')
    return values


def summarize(root):
    details = list(root.rglob('*_details.jsonl'))
    if len(details) != 1:
        raise ValueError('select exactly one completed concurrency result directory')
    rows = [json.loads(line) for line in details[0].read_text().splitlines() if line.strip()]
    if (len(rows) != 32 or len({row['data_id'] for row in rows}) != 32
            or any(not row['success'] or row['input_tokens'] != 1024 or row['output_tokens'] != 256 for row in rows)):
        raise ValueError('frozen workload is incomplete, failed, duplicated or has wrong token lengths')
    databases = {}
    for path in root.rglob('*.db'):
        if path.name in databases:
            raise ValueError('ambiguous database identity')
        databases[path.name] = path
    timestamps = []
    for row in rows:
        path = databases[Path(row['db_name']).name]
        ref = row['time_points']['__db_ref__']
        with sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True) as connection:
            record = connection.execute('SELECT arr_blob FROM numpy_store WHERE id=?', (ref,)).fetchone()
        if record is None:
            raise ValueError('missing raw timestamp reference')
        timestamps.append(read_array(record[0]))
    seconds = max(t[-1] for t in timestamps) - min(t[0] for t in timestamps)
    result = dict(requests=32, successful=32, failed=0, input_tokens=32768,
        output_tokens=8192, benchmark_seconds=seconds,
        output_token_throughput=8192 / seconds,
        ttft_median_ms=statistics.median((t[1] - t[0]) * 1000 for t in timestamps),
        tpot_median_ms=statistics.median((t[-1] - t[1]) * 1000 / 255 for t in timestamps),
        timestamps_finite_nondecreasing=True, raw_details=str(details[0]))
    summary_path = details[0].with_name(details[0].name.removesuffix('_details.jsonl') + '.json')
    official = json.loads(summary_path.read_text())
    reported = float(official['Output Token Throughput']['total'].split()[0])
    if abs(reported - result['output_token_throughput']) > 0.000051:
        raise ValueError('raw throughput disagrees with official rounded global output throughput')
    if official['Success Requests']['total'] != 32 or official['Failed Requests']['total'] != 0:
        raise ValueError('official success/failure counts disagree')
    result['official_global_throughput_agrees'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
