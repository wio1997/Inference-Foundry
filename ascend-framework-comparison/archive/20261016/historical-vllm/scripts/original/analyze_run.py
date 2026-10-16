"""Aggregate AISBench timing without copying prompt or output text."""
import argparse,ast,io,json,pathlib,sqlite3,statistics,struct
parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=pathlib.Path)
parser.add_argument('--concurrency',type=int,required=True)
parser.add_argument('--expected-requests',type=int)
parser.add_argument('--expected-output',type=int,default=600);args=parser.parse_args()
files=list((args.run_dir/'outputs/full').rglob('gsm8k_details.jsonl'))
assert len(files)==1, f'Expected one completed result file, found {len(files)}'
path=files[0];conns={};rows=[];failures=0;missing=0
def load_times(r):
    db=r['db_name']
    if db not in conns:conns[db]=sqlite3.connect('file:'+str(path.parent/'db_data'/db)+'?mode=ro&immutable=1',uri=True)
    b=conns[db].execute('select arr_blob from numpy_store where id=?',(r['time_points']['__db_ref__'],)).fetchone()[0]
    assert b[:6]==b'\x93NUMPY'
    hlen=struct.unpack('<H' if b[6]==1 else '<I',b[8:10] if b[6]==1 else b[8:12])[0]
    start=10 if b[6]==1 else 12;meta=ast.literal_eval(b[start:start+hlen].decode())
    assert meta['descr'] in ('<f8','=f8') and len(meta['shape'])==1
    return struct.unpack('<'+str(meta['shape'][0])+'d',b[start+hlen:])
for line in path.open():
    r=json.loads(line)
    if not r['success']:failures+=1
    try:ts=load_times(r)
    except (KeyError,TypeError):missing+=1;continue
    if len(ts)<2:missing+=1;continue
    rows.append({'start':ts[0],'first':ts[1],'end':ts[-1],
                 'output_tokens':r['output_tokens'],'input_tokens':r['input_tokens'],
                 'success':r['success']})
assert rows
t0=min(r['start'] for r in rows);t1=max(r['end'] for r in rows);span=t1-t0
ok=[r for r in rows if r['success']]
def pct(values,p):
    a=sorted(values);x=(len(a)-1)*p/100;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def distribution(values):return {f'p{p}':pct(values,p) for p in (50,75,90,95,99)}
def phase(start_key):
    events=sorted([(r[start_key],1) for r in rows]+[(r['end'],-1) for r in rows])
    active=peak=0;area=0;last=t0;full=[]
    for t,d in events:
        area+=active*(t-last)
        if active>=args.concurrency and t>last:full.append((last,t))
        active+=d;peak=max(peak,active);last=t
    phase_span=t1-min(r[start_key] for r in rows)
    return {'peak':peak,'mean_over_full_window':area/span,
            'phase_window_s':phase_span,'mean_over_phase_window':area/phase_span,
            'fraction_of_window_at_configured_cap':sum(b-a for a,b in full)/span,
            'first_full_time_relative_s':full[0][0]-t0 if full else None,
            'last_full_time_relative_s':full[-1][1]-t0 if full else None}
ttft=distribution([1000*(r['first']-r['start']) for r in ok])
tpot=distribution([1000*(r['end']-r['first'])/(r['output_tokens']-1) for r in ok if r['output_tokens']>1])
integrity=(not failures and not missing and
           all(r['output_tokens']==args.expected_output for r in ok) and
           (args.expected_requests is None or len(rows)==args.expected_requests))
sla={'ttft_p50_ms':(ttft['p50'],4000),'ttft_p75_ms':(ttft['p75'],8000),
     'ttft_p90_ms':(ttft['p90'],12000),'ttft_p99_ms':(ttft['p99'],30000),
     'tpot_p50_ms':(tpot['p50'],18),'tpot_p90_ms':(tpot['p90'],40)}
summary={'timed_requests':len(rows),'successes':len(ok),'failures':failures,'missing_timing':missing,
         'configured_concurrency':args.concurrency,'duration_s':span,
         'arrival_span_s':max(r['start'] for r in rows)-t0,
         'output_tokens':sum(r['output_tokens'] for r in ok),
         'output_tps':sum(r['output_tokens'] for r in ok)/span,'successful_request_per_s':len(ok)/span,
         'output_token_counts':sorted(set(r['output_tokens'] for r in ok)),
         'input_token_min_max':[min(r['input_tokens'] for r in ok),max(r['input_tokens'] for r in ok)],
         'ttft_ms':ttft,'tpot_ms':tpot,
         'ttft_mean_ms':statistics.mean(1000*(r['first']-r['start']) for r in ok),
         'tpot_mean_ms':statistics.mean(1000*(r['end']-r['first'])/(r['output_tokens']-1) for r in ok if r['output_tokens']>1),
         'e2e_ms':distribution([1000*(r['end']-r['start']) for r in ok]),
         'e2e_mean_ms':statistics.mean(1000*(r['end']-r['start']) for r in ok),
         'http_concurrency':phase('start'),'decode_phase_concurrency':phase('first'),
         'sample_sla':{k:{'observed':v,'limit':limit,'strict_pass':v<limit,'difference':v-limit,
                          'relative_difference_pct':100*(v/limit-1)} for k,(v,limit) in sla.items()},
         'integrity_pass':integrity,'expected_requests':args.expected_requests,
         'all_sample_gates_pass':integrity and all(v<limit for v,limit in sla.values()),
         'quantile_method':'linear interpolation, p*(n-1); successful requests, full run',
         'caveat':'Observed sample gates only, not a proof of long-term production tail stability; full-run denominator may be incomplete if failed requests lack timing.'}
(args.run_dir/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
