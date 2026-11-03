#!/usr/bin/env python3
"""Read the saved Engram diagnostic log/statuses; emit anonymous aggregates only."""
import collections, hashlib, json, pathlib
RUN='001d14496b2146bba25dae0f27c780b5'
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029') / ('engram-diagnostic-'+RUN) / 'runtime'
progress=json.loads((base/'engram-client-results/progress.json').read_text()); window=progress['window_id']
assert progress['run_id']==RUN and progress['state']=='COMPLETE'
p=base/'metadata/engram-metadata.jsonl'; before=p.stat(); raw=p.read_bytes(); after=p.stat()
assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
sha=hashlib.sha256(raw).hexdigest(); assert sha=='b32f03b3aca61939fbf76e56a6c7faaa0d1d8a9c4d225515f03e72082c44cfbf'
es=[json.loads(x) for x in raw.splitlines()]
statuses=collections.defaultdict(list)
for s in (base/'metadata').glob('observer-*.json'):
    v=json.loads(s.read_text());statuses[v.get('process_instance')].append(v)
B=collections.defaultdict(list); E=collections.defaultdict(list)
for e in es:
    if e.get('event') in ('embed_begin','embed_end'):
        (B if e['event']=='embed_begin' else E)[(e['process_instance'],e['call'])].append(e)
pairs=[]
for k,v in B.items():
    if len(v)!=1 or len(E.get(k,[]))!=1:continue
    b,e=v[0],E[k][0]; ss=statuses[k[0]]
    if not (len(ss)==1 and ss[0].get('run_id')==RUN and ss[0].get('window_id')==window and ss[0].get('pid')==b.get('pid')==e.get('pid') and b.get('window_id')==e.get('window_id')==window and e.get('outcome')=='returned' and isinstance(b.get('tensor'),dict)):continue
    pairs.append((b,e))
assert len(pairs)==826
out={'log_sha256':sha,'status_matched_unique_returned_pairs':len(pairs)}
for field in ('tensor_instance','storage_wrapper_instance'):
    groups=collections.defaultdict(list)
    for b,e in pairs:
        val=b['tensor'].get(field)
        if val is not None:groups[(b['process_instance'],val)].append((b,e))
    values=list(groups.values())
    out[field]={'nonnull_records':sum(map(len,values)),'null_records':len(pairs)-sum(map(len,values)),'distinct_per_process_tokens':len(values),'repeat_groups':sum(len(v)>1 for v in values),'repeat_records':sum(len(v) for v in values if len(v)>1),'max_records_per_token':max(map(len,values),default=0)}
    if field=='storage_wrapper_instance':
        twos=[sorted(v,key=lambda x:x[0]['monotonic_ns']) for v in values if len(v)==2]
        out['storage_wrapper_same_token_pairs']={'groups':len(twos),'same_shape':sum(a[0]['tensor']['shape']==b[0]['tensor']['shape'] for a,b in twos),'same_num_tokens':sum(a[0]['num_tokens']==b[0]['num_tokens'] for a,b in twos),'same_thread':sum(a[0]['tid']==b[0]['tid'] for a,b in twos),'consecutive_call_ids':sum(b[0]['call']==a[0]['call']+1 for a,b in twos),'first_returned_before_second_begin_host':sum(a[1]['monotonic_ns']<=b[0]['monotonic_ns'] for a,b in twos),'different_tensor_tokens':sum(a[0]['tensor']['tensor_instance']!=b[0]['tensor']['tensor_instance'] for a,b in twos),'first_offset_0_second_offset_24':sum((a[0]['tensor']['storage_offset'],b[0]['tensor']['storage_offset'])==(0,24) for a,b in twos)}
print(json.dumps(out,indent=2))
