#!/usr/bin/env python3
"""Bounded offline recheck of saved 16-rank MC2 layer records."""
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
inp=root/'diagnostic-20261029/target-layer-mc2-16rank.json'
hier=root/'diagnostic-20261029/sidecar-hierarchy-20261029.json'
early='e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023'
raw=inp.read_bytes();hraw=hier.read_bytes()
if hashlib.sha256(hraw).hexdigest()!=early:raise ValueError('hierarchy hash mismatch')
d=json.loads(raw);h=json.loads(hraw)
if d.get('hierarchy_input_sha256')!=early or len(d['ranks'])!=16 or d['rank_errors'] or d['missing_expected_ep'] or d['unexpected_ep']:
    raise ValueError('invalid source coverage')
scope_map={}
for s in h['layer_samples']:
    if s['role']!='conditional_target':continue
    key=(s['ep'],s['call_id'])
    if key in scope_map:raise ValueError('duplicate sidecar scope')
    scope_map[key]=(s['layer_index'],s['parent_digest'],s['pid'])
groups=defaultdict(list);seen=set();bad_start=[];zero_cycle=[];seen_ep=set()
for rank in d['ranks']:
    ep=rank['ep']
    if ep in seen_ep or rank['status']!='completed':raise ValueError('bad rank')
    seen_ep.add(ep)
    counts=rank['counts']
    if counts.get('target_scope_count')!=80 or counts.get('target_tasks_uniquely_matched')!=80 or counts.get('target_scopes_no_verified_task')!=0:
        raise ValueError('incomplete scope coverage')
    if counts.get('flow_finish_invalid_ts',0):raise ValueError('invalid finish timestamp')
    if counts.get('flow_start_nonunique',0):bad_start.append({'ep':ep,'discarded_flow_ids':counts['flow_start_nonunique']})
    for layer in rank['layers']:
        cid=layer['layer_scope_id']; idx=layer['layer']; digest=layer['scheduler_digest']
        if (ep,cid) in seen:raise ValueError('duplicate scope')
        seen.add((ep,cid))
        if scope_map.get((ep,cid))!=(idx,digest,rank['pid']):raise ValueError('scope join mismatch')
        if layer['trace_scope_count']!=1 or layer['matched_task_count']!=1 or not layer['sidecar_thread_matches_trace'] or len(layer['tasks'])!=1:
            raise ValueError('nonunique verified match')
        t=layer['tasks'][0];csv=t['csv']
        if csv['Name']!='DispatchFFNCombine' or csv['Type']!='DispatchFFNCombine':raise ValueError('type mismatch')
        n=Decimal(csv['aic_total_cycles']);ratio=Decimal(csv['aic_mac_ratio']);dur=Decimal(csv['Duration(us)'])
        if not (n.is_finite() and ratio.is_finite() and dur.is_finite() and n>0 and 0<=ratio<=1 and dur>0):
            zero_cycle.append((ep,cid));continue
        groups[(digest,idx)].append({'ep':ep,'call_id':cid,'aic_cycles':str(n),'aic_mac_ratio':str(ratio),'duration_us':str(dur)})
if len(seen)!=1280 or set(scope_map)!=seen or seen_ep!=set(range(16)) or zero_cycle:
    raise ValueError('incomplete matching or unknown metric')
digests=sorted({k[0] for k in groups})
if len(digests)!=4 or len(groups)!=160:raise ValueError('unexpected digest/layer cardinality')
entries=[]
for (digest,idx),rows in sorted(groups.items()):
    if len(rows)!=8 or len({z['ep'] for z in rows})!=8:raise ValueError('rank uniqueness failed')
    ns=[Decimal(z['aic_cycles']) for z in rows];ratios=[Decimal(z['aic_mac_ratio']) for z in rows];ds=[Decimal(z['duration_us']) for z in rows]
    proxy=sum((n*r for n,r in zip(ns,ratios)),Decimal(0));total=sum(ns,Decimal(0));mean=proxy/total
    sds=sorted(ds)
    entries.append({'digest':digest,'layer':idx,'ranks':sorted(rows,key=lambda z:z['ep']),
                    'aic_cycles_sum':str(total),'aic_cycle_times_displayed_ratio_sum':str(proxy),
                    'cycle_weighted_displayed_ratio':str(mean),
                    'ratio_displayed_min':str(min(ratios)),'ratio_displayed_max':str(max(ratios)),
                    'duration_us_min':str(sds[0]),'duration_us_median':str((sds[3]+sds[4])/2),
                    'duration_us_max':str(sds[-1]),'duration_us_range':str(sds[-1]-sds[0])})
minima=[]
for digest in digests:
    vals=[e for e in entries if e['digest']==digest]
    if {e['layer'] for e in vals}!=set(range(40)):raise ValueError('layers incomplete')
    eps={tuple(x['ep'] for x in e['ranks']) for e in vals}
    if len(eps)!=1:raise ValueError('rank composition varies by layer')
    low=min(Decimal(e['cycle_weighted_displayed_ratio']) for e in vals)
    minima.append({'digest':digest,'minimum_weighted_displayed_ratio':str(low),'layers_at_minimum':[e['layer'] for e in vals if Decimal(e['cycle_weighted_displayed_ratio'])==low]})
out={'input_sha256':hashlib.sha256(raw).hexdigest(),'hierarchy_sha256':early,'matched_scopes':len(seen),
     'digest_layer_rank_groups':len(entries),'flow_start_nonunique_discarded':bad_start,
     'digest_minima':minima,'groups':entries,
     'limitations':['Weighted displayed ratios are reported-field arithmetic, not whole-device utilization.',
                    'Exported ratio precision and rounding are unknown; differences may not be physically significant.',
                    'Device task durations are compared individually; no cross-rank or cross-stream time sum.',
                    'Flow joins show uniquely matched retained candidates, not an exhaustive proof of all tasks.',
                    'Scope digest identity is conditional and not unique across all steps; effective routing rows remain unknown.']}
print(json.dumps(out,indent=2))
