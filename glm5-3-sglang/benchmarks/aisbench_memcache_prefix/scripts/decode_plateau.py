"""Explicitly separate sampled server occupancy from AIS client steady metrics."""
import json
from pathlib import Path

def analyze(rows, load_path):
    result=dict(valid=False,definition='Longest contiguous sequence of sampled D running-count sum16, with poll gaps<=10s; discard first and last samples; require>=30s remaining. Occupancy samples are aggregated, non-atomic server snapshots. Throughput counts cumulative SSE usage increments RECEIVED inside this client-monotonic window, not NPU generation timestamps. This is not native AISBench steady.',
        target_observed_decode_running=16)
    points=[]
    for line in Path(load_path).read_text().splitlines():
        x=json.loads(line);roles={}
        for role in ['prefill','decode']:
            loads=(x.get(role)or{}).get('loads',[])
            expected=4 if role=='prefill'else 16
            if len(loads)==expected and {v.get('dp_rank')for v in loads}==set(range(expected)):
                roles[role]=dict(running=sum(v['num_running_reqs']for v in loads),
                    waiting=sum(v['num_waiting_reqs']for v in loads),
                    server_rank_timestamps=[v.get('timestamp')for v in loads])
        points.append(dict(t=x['monotonic'],roles=roles))
    result['sample_count']=len(points)
    result['sampled_peaks']={role:{k:max((x['roles'].get(role,{}).get(k,0)for x in points),default=0)
        for k in ['running','waiting']}for role in ['prefill','decode']}
    groups=[];current=[]
    for x in points:
        if x['roles'].get('decode',{}).get('running')==16:
            if current and x['t']-current[-1]['t']>10:groups.append(current);current=[]
            current.append(x)
        elif current:groups.append(current);current=[]
    if current:groups.append(current)
    groups=[g for g in groups if len(g)>=3 and g[-2]['t']-g[1]['t']>=30]
    if not groups:
        result['reason']='No sampled decode16 platform satisfies predeclared duration/gap criteria'
        return result
    group=max(groups,key=lambda g:g[-2]['t']-g[1]['t']);lo,hi=group[1]['t'],group[-2]['t']
    result.update(window_client_monotonic=[lo,hi],duration_s=hi-lo,occupancy_samples=group[1:-1])
    total=0;receiving=[];errors=[];event_counts=[]
    for row in rows:
        events=(row.get('extra_details_data')or{}).get('usage_events',[])
        last_t=float('-inf');last_n=0;inside=0;positive=[]
        for event in events:
            t,n=event['t'],event['completion_tokens']
            if t<last_t or not isinstance(n,int)or n<last_n or n>600:
                errors.append(dict(id=row.get('id'),error='Non-monotonic streaming usage'));break
            delta=n-last_n
            if delta>0:
                positive.append(t)
                if lo<t<=hi:inside+=delta
            last_t,last_n=t,n
        if len(positive)<2 or last_n!=600:
            errors.append(dict(id=row.get('id'),error='No auditable progressive600-token usage',positive_events=len(positive),final=last_n))
        event_counts.append(dict(id=row.get('id'),positive_events=len(positive)))
        total+=inside
        if inside:receiving.append(dict(id=row.get('id'),tokens_received=inside))
    result.update(valid=not errors,errors=errors,progressive_event_counts=event_counts,
        receiving_requests=receiving,output_tokens_received=total,
        received_output_tokens_per_second=total/(hi-lo) if not errors else None)
    return result
