"""读取标准AISBench原SQLite并复算逐请求时序；仅使用Python标准库。"""
import ast
import hashlib
import json
import lzma
import math
from pathlib import Path
import sqlite3
import statistics
import struct


def array_from_npy(blob):
    assert blob[:6] == b'\x93NUMPY', 'Expected original numpy .npy blob'
    major, minor = blob[6:8]
    assert major in [1, 2, 3]
    offset = 10 if major == 1 else 12
    length = struct.unpack('<H' if major == 1 else '<I', blob[8:offset])[0]
    header = ast.literal_eval(blob[offset:offset+length].decode('utf-8' if major == 3 else 'latin1'))
    assert len(header['shape']) == 1 and header['descr'] in ['<f8', '<f4']
    count = header['shape'][0]
    values = struct.unpack('<'+str(count)+('d' if header['descr'] == '<f8' else 'f'), blob[offset+length:])
    assert len(values) >= 2 and all(math.isfinite(x) for x in values)
    assert all(a <= b for a,b in zip(values, values[1:])), 'Non-monotonic request timestamps'
    return values


def load_db(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as conn:
        tables = {x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {'numpy_store', 'sqlite_sequence'}, tables
        return {i:array_from_npy(blob) for i,blob in conn.execute('SELECT id, arr_blob FROM numpy_store')}


def quantile(values, q):
    values = sorted(values)
    i = (len(values)-1)*q
    lo,hi = math.floor(i),math.ceil(i)
    return values[lo]+(values[hi]-values[lo])*(i-lo)


def phase_witness(root, manifest):
    evidence = manifest['raw_evidence']
    databases = {Path(s).name:load_db(root/s) for s in evidence['timing_databases']}
    with lzma.open(root/evidence['details'],'rt') as f:
        raw = [json.loads(line) for line in f if line.strip()]
    result = []
    for row in raw:
        points = databases[row['db_name']][row['time_points']['__db_ref__']]
        start,first,end = points[0],points[1],points[-1]
        output = row['output_tokens']
        result.append(dict(data_id=row['data_id'],input_tokens=row['input_tokens'],output_tokens=output,
                           time_points_count=len(points),start_time=start,end_time=end,
                           TTFT=(first-start)*1000,E2EL=(end-start)*1000,
                           TPOT=((end-first)/(output-1) if output>1 else 0)*1000))
    stats = {key:dict(Average=statistics.mean(r[key] for r in result),Min=min(r[key] for r in result),
                     Max=max(r[key] for r in result),Median=quantile([r[key] for r in result],.5),
                     P75=quantile([r[key] for r in result],.75),P90=quantile([r[key] for r in result],.9),
                     P99=quantile([r[key] for r in result],.99)) for key in ['TTFT','TPOT','E2EL']}
    return result,stats


def main():
    root = Path(__file__).resolve().parents[1]
    rows = []
    for entry in json.loads((root/'records/runs.json').read_text()):
        manifest = json.loads((root/entry['path']).read_text())
        if 'raw_evidence' not in manifest:continue
        requests,stats = phase_witness(root,manifest)
        rows.append(dict(run_id=entry['id'],requests=len(requests),statistics_ms=stats))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
