#!/usr/bin/env python3
"""Query the three frozen EP1 gaps in the already captured profiler data, CPU only."""
import argparse, csv, hashlib, json, sqlite3
from decimal import Decimal
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--trace-root', required=True)
args = ap.parse_args()
spec = json.loads(FROZEN_INPUT)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def ns(value):
    x = Decimal(value.strip()) * 1000
    if x != x.to_integral_value():
        raise ValueError('sub-nanosecond value')
    return int(x)

def union(rows, start, end):
    clipped = sorted((max(start,a), min(end,b)) for a,b in rows if a < end and b > start and a < b)
    out = []
    for a,b in clipped:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a,b])
    return out

def length(rows):
    return sum(b-a for a,b in rows)

rank = spec['rank']
base = Path(args.trace_root) / rank / 'ASCEND_PROFILER_OUTPUT'
cp = base / 'kernel_details.csv'
db = base / 'ascend_pytorch_profiler_1.db'
if sha(cp) != spec['csv_sha256'] or sha(db) != spec['db_sha256']:
    raise RuntimeError('frozen file identity mismatch')
windows = spec['windows']
if sorted(w['op_id'] for w in windows) != [402,405,408]:
    raise RuntimeError('window identity mismatch')
cr = {w['op_id']: [] for w in windows}
with cp.open(newline='') as f:
    reader = csv.DictReader(f)
    fields = ['Name','Type','Model ID','Stream ID','Task ID','Start Time(us)','Duration(us)']
    if not set(fields).issubset(reader.fieldnames or ()):
        raise RuntimeError('required CSV fields absent')
    for row in reader:
        start = ns(row['Start Time(us)']); end = start + ns(row['Duration(us)'])
        for w in windows:
            if start < w['end_ns'] and end > w['start_ns']:
                v = {k: row[k] for k in fields}; v.update(start_ns=start, end_ns=end)
                cr[w['op_id']].append(v)
                if len(cr[w['op_id']]) > 5000: raise RuntimeError('CSV bound exceeded')
c = sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)
c.row_factory = sqlite3.Row
c.execute('PRAGMA query_only=ON')
c.execute('PRAGMA temp_store=MEMORY')
devices = c.execute('SELECT DISTINCT deviceId FROM RANK_DEVICE_MAP').fetchall()
if len(devices) != 1 or devices[0][0] != 1: raise RuntimeError('unexpected device')
out = []
for w in windows:
    a,b,main = w['start_ns'],w['end_ns'],w['main_stream']
    if not a < b: raise RuntimeError('invalid interval')
    rows = [dict(x) for x in c.execute('''SELECT t.startNs,t.endNs,t.streamId,t.modelId,t.taskId,t.globalTaskId,t.connectionId,s.value AS task_type_name
       FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType
       WHERE t.deviceId=? AND ((t.startNs<? AND t.endNs>?) OR (t.startNs>=? AND t.startNs<? AND t.endNs=t.startNs))
       ORDER BY t.startNs,t.streamId LIMIT 5001''', (1,b,a,a,b))]
    if len(rows)>5000: raise RuntimeError('DB bound exceeded')
    # No model filter; stream identity is useful only as a local exported label.
    main_rows = [x for x in rows if x['streamId'] == main]
    other_rows = [x for x in rows if x['streamId'] != main and x['streamId'] is not None]
    unlabelled = [x for x in rows if x['streamId'] is None]
    seg = lambda rr: [(x['startNs'],x['endNs']) for x in rr]
    main_u = union(seg(main_rows),a,b); other_u = union(seg(other_rows),a,b)
    total_u = union(seg(rows),a,b)
    csv_u = union([(x['start_ns'],x['end_ns']) for x in cr[w['op_id']]],a,b)
    gaps = []; pos=a
    for s,e in total_u:
        if s>pos: gaps.append([pos,s])
        pos=max(pos,e)
    if pos<b: gaps.append([pos,b])
    out.append({'op_id':w['op_id'],'start_ns':a,'end_ns':b,'duration_ns':b-a,'main_stream':main,
      'db_main_union_ns':length(main_u),'db_other_numeric_stream_union_ns':length(other_u),
      'db_unlabelled_stream_union_ns':length(union(seg(unlabelled),a,b)),
      'db_all_stream_union_ns':length(total_u),'db_all_stream_uncovered_ns':(b-a)-length(total_u),
      'db_uncovered_intervals_ns':gaps,'csv_all_stream_union_ns':length(csv_u),
      'db_rows':rows,'csv_rows':cr[w['op_id']]})
c.close()
print(json.dumps({'scope':'EP1 only; three frozen [gating-end, MC2 CSV start) gaps; all DB models/streams and positive CSV task rows; same device clock',
 'inputs':{'csv_sha256':sha(cp),'db_sha256':sha(db)},'samples':out,
 'limits':['Only exported rows; no claim about capture completeness or device idle.',
  'Positive-duration EVENT WAIT records indicate recorded wait intervals, not proven blocking cause or idle cores.',
  'Zero-duration events are included in DB rows but contribute no duration.',
  'Union of overlapping, nested, and duplicated rows is not additive; CSV and DB are separate views.',
  'Timing overlap does not establish data or event dependency; do not compare different device clocks.']},indent=2))
