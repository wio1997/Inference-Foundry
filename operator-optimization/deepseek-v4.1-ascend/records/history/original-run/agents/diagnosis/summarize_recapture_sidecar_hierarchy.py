#!/usr/bin/env python3
"""Offline, bounded attribution of sidecar scope nesting; no device operations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'))
root = p.parse_args().root
summary_raw = (root/'metadata-summary-early.json').read_bytes()
sha = hashlib.sha256(summary_raw).hexdigest()
if sha != '2e322b98e2f972e35d74cb2d5fc98dd7071a0ff48f38f8b47a62d91a2b48538a':
    raise ValueError('summary changed')
early = json.loads(summary_raw)
rankmap = {}
for d in (root/'runtime/diagnostic-traces').glob('*_ascend_pt'):
    m = re.fullmatch(r'dp(\d+)_pp\d+_tp(\d+)_dcp\d+_ep(\d+)_rank(\d+)_(\d+)_\d+_ascend_pt', d.name)
    if not m or d.is_symlink() or not d.is_dir(): raise ValueError('invalid trace rank directory')
    dp,tp,ep,rank,pid = map(int,m.groups())
    if pid in rankmap: raise ValueError('duplicate rank pid')
    rankmap[pid] = {'dp':dp, 'tp':tp, 'ep':ep, 'rank':rank}
if len(rankmap) != 16 or len({x['ep'] for x in rankmap.values()}) != 16:
    raise ValueError('unexpected rank map')
begins = {}; ends = {}; grouped=defaultdict(list); event_counts=Counter()
for pidstr, meta in early['per_pid'].items():
    if not pidstr.isdigit(): raise ValueError('bad pid')
    path = root/'runtime/metadata'/('events-'+pidstr+'.jsonl')
    if path.is_symlink() or not path.is_file() or path.stat().st_size>100_000_000: raise ValueError('bad file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']: raise ValueError('file hash mismatch')
    for line_no, line in enumerate(raw.splitlines(), 1):
        if len(line)>1_000_000: raise ValueError('oversized line')
        e=json.loads(line)
        if e.get('pid') != int(pidstr): raise ValueError('pid mismatch')
        kind=e.get('kind')
        if kind not in ('call_begin','call_end'): continue
        event_counts[kind]+=1
        pid,tid,ts,cid,label=e.get('pid'),e.get('tid'),e.get('perf_counter_ns'),e.get('call_id'),e.get('label')
        if type(tid) is not int or type(ts) is not int or not isinstance(cid,str) or not re.fullmatch(str(pid)+r'-\d+',cid):
            raise ValueError('invalid call identity')
        x=(pid,tid,cid)
        table=begins if kind=='call_begin' else ends
        if x in table: raise ValueError('duplicate begin/end')
        table[x]=e
        grouped[(pid,tid)].append((line_no,kind,cid,ts,label))
if event_counts != {'call_begin':2757,'call_end':2757}: raise ValueError('unexpected event count')
if set(begins)!=set(ends): raise ValueError('unpaired or wrong-thread call')
for key,b in begins.items():
    e=ends[key]
    if b['label']!=e['label'] or e.get('outcome')!='ok' or e['perf_counter_ns']<b['perf_counter_ns']:
        raise ValueError('mismatched label/outcome/time')
strict=Counter(); parent={}; invalid_threads=set()
for thread, events in grouped.items():
    stack=[]; last=-1
    for _,kind,cid,ts,label in events:
        if ts<last: invalid_threads.add(thread)
        last=ts
        if kind=='call_begin':
            parent[(thread[0],thread[1],cid)]=tuple(stack)
            stack.append(cid)
        else:
            if not stack or stack[-1]!=cid:
                invalid_threads.add(thread); strict['crossing_or_missing_end']+=1
                if cid in stack: stack.remove(cid)
            else: stack.pop()
    if stack: invalid_threads.add(thread); strict['unclosed_thread']+=1
strict['invalid_threads']=len(invalid_threads)

ROLE_LAYER='DeepseekV41DecoderLayer.forward'
ROLE_DRAFT='DeepseekV41DSparkModel.forward'
ROLE_RUNNER='NPUModelRunner.execute_model'
data=[]; stats=Counter()
for key,b in begins.items():
    pid,tid,cid=key
    if pid not in rankmap or b['label']!=ROLE_LAYER: continue
    anc=[begins.get((pid,tid,x),{}) for x in parent.get(key,())]
    labels=[x.get('label') for x in anc]
    reasons=[]
    if (pid,tid) in invalid_threads: reasons.append('non_strict_thread')
    if labels.count(ROLE_LAYER): reasons.append('nested_layer')
    if labels.count(ROLE_DRAFT)>1 or labels.count(ROLE_RUNNER)>1: reasons.append('multiple_model_or_runner')
    if reasons: role='unknown'
    elif ROLE_DRAFT in labels: role='draft'
    elif ROLE_RUNNER in labels: role='conditional_target'
    else: role='unknown'; reasons.append('no_draft_or_runner_parent')
    runner=next((a for a in reversed(anc) if a.get('label')==ROLE_RUNNER),None)
    schedule=runner.get('schedule') if runner else None
    digest=schedule.get('schedule_digest_sha256') if isinstance(schedule,dict) else None
    if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest): digest=None
    if digest is not None:
        canonical=json.dumps({'scheduled_total':schedule.get('scheduled_total'), 'requests':schedule.get('requests')},sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(canonical).hexdigest()!=digest: raise ValueError('runner schedule digest mismatch')
    layer=b.get('layer_index')
    if type(layer) is not int or layer<0: layer=None
    elapsed=ends[key]['perf_counter_ns']-b['perf_counter_ns']
    data.append({'pid':pid,'ep':rankmap[pid]['ep'],'call_id':cid,'layer_index':layer,
                 'role':role,'parent_digest':digest,'cpu_sidecar_wall_ns':elapsed,'unknown_reasons':reasons})
    stats[role]+=1
def hist(nums):
    if not nums: return None
    return {'count':len(nums),'min_ns':min(nums),'median_ns':statistics.median(nums),'max_ns':max(nums)}
aggregates=defaultdict(list)
rank_stats=defaultdict(Counter)
for x in data:
    rank_stats[x['ep']][x['role']]+=1
    aggregates[(x['role'], x['layer_index'], x['parent_digest'])].append(x['cpu_sidecar_wall_ns'])
groups=[{'role':k[0],'layer_index':k[1],'parent_digest':k[2], 'cpu_sidecar_wall':hist(v)}
        for k,v in sorted(aggregates.items(),key=lambda z:(z[0][0],str(z[0][1]),str(z[0][2])))]
result={'sidecar_event_counts':dict(event_counts),'ranks':16,'strict_nesting':dict(strict),
        'roles':dict(stats),'rank_distribution':[{'ep':ep, **dict(rank_stats[ep])} for ep in sorted(rank_stats)],
        'by_role_layer_and_parent_digest':groups,
        'layer_samples':data,
        'limitations':['CPU sidecar wall is not NPU time/utilization; hooks add measurement overhead.',
                       'Sidecar same-thread nesting is conditional attribution, not proof of model graph replay or device execution.',
                       'Digest is absent if no runner ancestor, not zero work; distinct calls can share a digest.',
                       'Trace flow and scope-to-device mapping require independent evidence; not provided here.']}
print(json.dumps(result,sort_keys=True,separators=(',',':')))
