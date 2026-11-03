#!/usr/bin/env python3
"""Rank-local, same-CSV overlap of 3 layer-0 MC2 samples; read saved files only."""
import argparse, csv, hashlib, json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--task-root', required=True)
ap.add_argument('--trace-root', required=True)
a = ap.parse_args()
root = Path(a.task_root)
wp = root / 'captured-layer-attribution/module-call-metrics.json'
windows = [w for w in json.loads(wp.read_text())
           if '_ep8_' in w['rank'] and w['layer'] == 0 and w['run_in_model'] in (1, 2, 3)]
assert len(windows) == 3 and len({w['rank'] for w in windows}) == 1
assert {w['run_in_model'] for w in windows} == {1, 2, 3}
rank = windows[0]['rank']
p = Path(a.trace_root) / rank / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
records = []
with p.open() as f:
    for row in csv.DictReader(f):
        start = int(Decimal(row['Start Time(us)'].strip()) * 1000)
        end = start + int(Decimal(row['Duration(us)'].strip()) * 1000)
        if end <= start: continue
        records.append({'start': start, 'end': end, 'type': row['Type'], 'stream': row['Stream ID'], 'model': row['Model ID']})

def union(spans):
    spans = sorted(spans); result = []
    for lo, hi in spans:
        if lo >= hi: continue
        if not result or lo > result[-1][1]: result.append([lo, hi])
        else: result[-1][1] = max(result[-1][1], hi)
    return result

def measure(spans): return sum(hi - lo for lo, hi in union(spans))

out = []
for w in sorted(windows, key=lambda w: w['run_in_model']):
    own = [r for r in records if r['type'] == 'DispatchFFNCombine' and r['stream'] == str(w['main_stream_id'])
           and r['model'] == str(w['model_id']) and w['window_start_ns'] <= r['start'] and r['end'] <= w['window_end_ns']]
    assert len(own) == 1
    m = own[0]; dur = m['end'] - m['start']
    other, unknown = [], []
    for r in records:
        if r['start'] >= m['end'] or r['end'] <= m['start']: continue
        s = str(r['stream']).strip()
        if s == str(w['main_stream_id']): continue
        lo, hi = max(r['start'], m['start']), min(r['end'], m['end'])
        entry = {**r, 'lo': lo, 'hi': hi}
        if s.isdigit(): other.append(entry)
        else: unknown.append(entry)
    def summarize(rows):
        spans = [(r['lo'], r['hi']) for r in rows]
        types, streams, models = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in rows:
            types[r['type']].append((r['lo'], r['hi']))
            streams[r['stream']].append((r['lo'], r['hi']))
            models[r['model']].append((r['lo'], r['hi']))
        return {'row_count':len(rows),'union_us':measure(spans)/1000,
                'by_type':{k:{'rows':len(v),'union_us':measure(v)/1000} for k,v in sorted(types.items())},
                'by_stream':{k:{'rows':len(v),'union_us':measure(v)/1000} for k,v in sorted(streams.items())},
                'by_model':{k:{'rows':len(v),'union_us':measure(v)/1000} for k,v in sorted(models.items())}}
    other_summary=summarize(other); unknown_summary=summarize(unknown)
    # Keep physical-stream hcom names separate, not mixed with N/A summary rows.
    comm = [r for r in other if r['type'].lower().startswith('hcom_')]
    out.append({'run':w['run_in_model'],'main_stream':w['main_stream_id'],'model':w['model_id'],
                'mc2_start_ns':m['start'],'mc2_end_ns':m['end'],'mc2_duration_us':dur/1000,
                'other_numeric_streams':other_summary,'other_hcom_rows':summarize(comm),
                'unknown_stream_rows':unknown_summary,
                'no_numeric_other_stream_row_us':(dur-measure([(r['lo'],r['hi']) for r in other]))/1000})
result={'rank':rank,'scope':'EP8 only; layer 0, model/stream-bracketed interior runs 1/2/3; overlap by clipped intervals from one rank kernel CSV',
        'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'kernel_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'samples':out,'limitations':['Temporal overlap in exported CSV is not proof of task identity, causality, resource sharing, or useful hidden communication.',
          'Other streams may include unrelated request/model work; none is attributed to MC2 by interval overlap.',
          'N/A stream rows are summarized separately because exporter may duplicate or summarize communication; do not sum their durations with numeric-stream rows.',
          'No matching exported row is not device idle; MC2-internal communication or arithmetic is not decomposed.',
          'Timestamp association is rank local, contingent on comparable timestamps in this CSV; cross-rank synchronization is not assumed.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
