#!/usr/bin/env python3
"""Aggregate task-owned Engram metadata without exposing tensor identities or contents."""
import json,hashlib,collections,sys
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized Python unsupported')
run='001d14496b2146bba25dae0f27c780b5'
base=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-'+run+'/runtime')
progress=json.loads((base/'engram-client-results/progress.json').read_text());window=progress['window_id'];log=base/'metadata/engram-metadata.jsonl';before=log.stat();raw=log.read_bytes();after=log.stat()
if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('log changed while reading')
events=[json.loads(x) for x in raw.splitlines()];beg=collections.defaultdict(list);end=collections.defaultdict(list)
for e in events:
 if e.get('event') not in ('embed_begin','embed_end') or e.get('window_id')!=window:continue
 key=(e.get('process_instance'),e.get('call'));(beg if e['event']=='embed_begin' else end)[key].append(e)
groups=collections.Counter();eligible=0;paired=0;processes=set();bad_meta=0
for key,bs in beg.items():
 if len(bs)!=1:continue
 b=bs[0];tensor=b.get('tensor');
 if not isinstance(tensor,dict):bad_meta+=1;continue
 if len(end[key])==1 and end[key][0].get('outcome')=='returned':paired+=1
 eligible+=1;processes.add(key[0]);sig=(tuple(tensor.get('shape',[])),tuple(tensor.get('stride',[])),tensor.get('storage_offset'),tensor.get('is_contiguous'),tensor.get('dtype'),tensor.get('device_type'));groups[sig]+=1
states=[]
for f in (base/'metadata').glob('observer-*.json'):
 try:states.append(json.loads(f.read_text()))
 except (ValueError,OSError):pass
print(json.dumps({'run_id':run,'window_id':window,'client_state':progress['state'],'measurement':progress.get('measure1'),'log_sha256':hashlib.sha256(raw).hexdigest(),'log_bytes':len(raw),'all_events':dict(collections.Counter(x.get('event') for x in events)),'window_begin_keys':len(beg),'unique_begin_valid_metadata':eligible,'unique_returned_pairs':paired,'invalid_metadata':bad_meta,'contributing_process_instances':len(processes),'signatures':[{'shape':k[0],'stride':k[1],'storage_offset':k[2],'contiguous':k[3],'dtype':k[4],'device':k[5],'count':v} for k,v in groups.most_common()],'observer_status_files':len(states),'status_incomplete':sum(bool(x.get('state_incomplete')) for x in states),'status_states':dict(collections.Counter(x.get('status') for x in states)),'limits':['This is only retained Python wrapper data in one formal window; graph replay and missing/capped events unknown.','Status count does not prove rank coverage or exact missing call count.','Metadata input stride does not establish tensor lifetime or value correctness of a changed implementation.']},indent=2))
