"""Compare existing128/129 AIS and host intervals; no model, tensor math or remote calls."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
import statistics

W = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-dir', type=Path, default=W/'evidence/migration103-20261112')
parser.add_argument('--array-reader', type=Path, default=W.parent.parent/'glm53flash-910c/plans/aisbench-tp8-20261110/summarize_raw_aisbench.py')
args = parser.parse_args()
E = args.evidence_dir.resolve()
spec = importlib.util.spec_from_file_location('raw_ais', args.array_reader)
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)
inputs = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    inputs[str(p.relative_to(E))] = sha(p)
    return json.loads(p.read_text())


def overlap(a, b, intervals):
    return sum(max(0, min(b, y)-max(a, x)) for x, y in intervals)


def stream_rows(root, c, reproduction):
    folder = root/f'aisbench-c{c}-results'
    details, = folder.rglob('*_details.jsonl')
    inputs[str(details.relative_to(E))] = sha(details)
    rows = [json.loads(s) for s in details.read_text().splitlines() if s.strip()]
    dbs = {p.name:p for p in folder.rglob('*.db')}
    for p in dbs.values():
        inputs[str(p.relative_to(E))] = sha(p)
    saved = {r['data_id']:r for r in reproduction['raw_timing_endpoints'][f'C{c}']['rows']}
    out = {}
    for r in rows:
        p = dbs[Path(r['db_name']).name]
        with sqlite3.connect('file:'+str(p)+'?mode=ro', uri=True) as con:
            blob, = con.execute('SELECT arr_blob FROM numpy_store WHERE id=?', (r['time_points']['__db_ref__'],)).fetchone()
        ts = raw.read_array(blob)
        q = saved[r['data_id']]
        assert (ts[0],ts[1],ts[-1]) == (q['request_start'],q['first_token'],q['request_end'])
        assert r['success'] and r['input_tokens']==1024 and r['output_tokens']==256
        assert hashlib.sha256(r['prediction'].encode()).hexdigest()==q['prediction_sha256']
        out[r['data_id']] = dict(q,
            input_text_sha256=hashlib.sha256(r['input'].encode()).hexdigest(),
            recorded_timestamps=len(ts), unique_post_start_timestamps=len(set(ts[1:])),
            maximum_post_first_recorded_gap_s=max(b-a for a,b in zip(ts[1:],ts[2:])))
    assert len(out)==32 and len(rows)==32
    return list(sorted(out.values(), key=lambda r:r['data_id']))


def summarize(rows, c, events):
    lo=min(r['request_start'] for r in rows); hi=max(r['request_end'] for r in rows)
    es=[e for e in events if lo<=e['started_monotonic_ns']/1e9<=hi]
    intervals=[(e['started_monotonic_ns']/1e9,e['host_return_monotonic_ns']/1e9) for e in es]
    assert all(b<=x for (a,b),(x,y) in zip(intervals,intervals[1:])), 'host intervals must not overlap'
    active=0
    for _, change in sorted([(r['request_start'],1) for r in rows]+[(r['request_end'],-1) for r in rows]):
        active+=change
        assert 0<=active<=c
    assert active==0
    ttft=sum(r['first_token']-r['request_start'] for r in rows)/c
    post=sum(r['request_end']-r['first_token'] for r in rows)/c
    idle=hi-lo-ttft-post
    assert idle>=-1e-8 and math.isclose(ttft+post+idle,hi-lo,abs_tol=1e-8)
    post_in=sum(overlap(r['first_token'],r['request_end'],intervals) for r in rows)/c
    ttft_in=sum(overlap(r['request_start'],r['first_token'],intervals) for r in rows)/c
    event_rows=[]
    for e,(a,b) in zip(es,intervals):
        event_rows.append(dict(start_offset_s=a-lo,host_interval_s=b-a,spans=e['spans'],hit=e['hit'],
            input_tokens=sum(y-x for x,y in e['spans']),
            client_requests_waiting_first_token_at_start=sum(r['request_start']<=a<r['first_token'] for r in rows),
            client_requests_after_first_token_at_start=sum(r['first_token']<=a<r['request_end'] for r in rows)))
    count=sum(sum(y-x for x,y in e['spans']) for e in es)
    return dict(makespan_s=hi-lo,output_tps=8192/(hi-lo),
        ttft_area_per_slot_s=ttft,post_first_area_per_slot_s=post,unoccupied_area_per_slot_s=idle,
        ttft_area_overlapping_host_prefill_per_slot_s=ttft_in,
        post_first_area_overlapping_host_prefill_per_slot_s=post_in,
        post_first_area_outside_host_prefill_per_slot_s=post-post_in,
        pure_prefill_calls=len(es),hit_calls=sum(e['hit'] for e in es),
        scheduled_pure_prefill_tokens=count,
        computed_pure_prefill_tokens=sum(y-x for e in es if not e['hit'] for x,y in e['spans']),
        average_scheduled_tokens_per_pure_prefill_call=count/len(es),
        scheduled_token_histogram=dict(Counter(sum(y-x for x,y in e['spans']) for e in es)),
        segment_count_histogram=dict(Counter(len(e['spans']) for e in es)),
        single_segment_calls=sum(len(e['spans'])==1 for e in es),
        total_host_prefill_interval_s=sum(b-a for a,b in intervals),
        recorded_timestamp_count=sum(r['recorded_timestamps'] for r in rows),
        unique_post_start_timestamp_count=sum(r['unique_post_start_timestamps'] for r in rows),
        per_request_median_maximum_post_first_recorded_gap_s=statistics.median(r['maximum_post_first_recorded_gap_s'] for r in rows),
        events=event_rows,requests=rows)


out={'created_at_utc':datetime.now(timezone.utc).isoformat(),
     'analysis_script_sha256':sha(Path(__file__)), 'array_reader_sha256':sha(args.array_reader),
     'new_model_loads':0,'remote_calls':0,'arms':{},'candidate_minus_previous_seconds':{},
     'comparison':'single historical batch per arm;observational decomposition,not randomized causal evidence',
     'general_quality_accepted':False,'exclusive30pct_accepted':False,'goal_complete':False}
for g in (128,129):
    root=E/f'native-g{g}-complete'; B=root/'native-run'
    repro=read(E/f'native{g}-reproduction.json')
    state=read(B/'communication/native-tp8/rank0/native-prefill-state.json')
    assert state['closed'] and state['counts'].get('events_dropped',0)==0
    result=read(B/'communication/native-tp8/rank0/result.json')
    a={'source_commit':repro['source_identity']['commit'],
       'whole_run_mtp_including_engineering_and_warmups':result['mtp'],
       'whole_run_async_schedule':result['async_schedule']}
    for c in (4,8):
        a[f'C{c}']=summarize(stream_rows(root,c,repro),c,state['events'])
    out['arms'][str(g)]=a
for c in (4,8):
    phase=f'C{c}';a=out['arms']['128'][phase];b=out['arms']['129'][phase]
    assert all(x['data_id']==y['data_id'] and x['input_text_sha256']==y['input_text_sha256'] for x,y in zip(a['requests'],b['requests']))
    fields=('makespan_s','ttft_area_per_slot_s','post_first_area_per_slot_s','unoccupied_area_per_slot_s',
            'post_first_area_overlapping_host_prefill_per_slot_s','post_first_area_outside_host_prefill_per_slot_s')
    delta={k:b[k]-a[k] for k in fields}
    assert math.isclose(delta['makespan_s'],sum(delta[k] for k in fields[1:4]),abs_tol=1e-8)
    assert math.isclose(delta['post_first_area_per_slot_s'],sum(delta[k] for k in fields[4:]),abs_tol=1e-8)
    delta['full_text_identical_requests']=sum(x['prediction_sha256']==y['prediction_sha256'] for x,y in zip(a['requests'],b['requests']))
    delta['input_texts_identical']=32
    out['candidate_minus_previous_seconds'][phase]=delta
out['interpretation_limits']=[
    'Request time areas divided by configured concurrency form an exact makespan identity,not a hardware critical-path decomposition.',
    'Post-first time contains other-request prefill,decode,communication,scheduling and transport;not isolated decode kernel time.',
    'Host prefill intervals contain waits/enqueue/bookkeeping and cannot be equated with NPU execution.Overlaps are time coincidences,not causal attribution.',
    'Spans are recorded only for pure eager prefill;no request IDs or complete mixed/decode event trace were recorded.',
    'Identical input text and token counts verified;complete token ID arrays not present in this evidence.',
    'MTP counters cover all72 requests;no phase boundaries,therefore no formal-C4 acceptance rate can be inferred.',
    'Recorded timestamp counts reflect transport/client recording,not verified model steps or accepted draft token counts.',
    'One batch each does not estimate repeatability or attribute changed batching to the preparation edit.',
    'Existing130/131 controls remain untriggered;this analysis does not reopen model-load budget.']
out['inputs_sha256']=inputs
p=E/'native129-existing-timeline-analysis.json'
p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':p.name,'deltas':out['candidate_minus_previous_seconds'],
    'phases':{g:{c:{k:v for k,v in a[c].items() if k not in ('events','requests')} for c in ('C4','C8')} for g,a in out['arms'].items()}},ensure_ascii=False))
