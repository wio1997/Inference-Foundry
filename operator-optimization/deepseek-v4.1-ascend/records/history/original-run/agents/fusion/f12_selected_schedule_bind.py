#!/usr/bin/env python3
"""Read hash-bound historical sidecar; report anonymized four F12 schedule descriptors.
Run inside the original stage container; never output request hashes or prompt content.
"""
import collections, hashlib, json, pathlib
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
def load(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
early=R/'diagnostic-20261029/metadata-summary-early.json'
assert sha(early)=='2e322b98e2f972e35d74cb2d5fc98dd7071a0ff48f38f8b47a62d91a2b48538a'
pairs=R/'agents/diagnosis/indexer-attention-event-identity.json'; hierarchy=R/'diagnostic-20261029/sidecar-hierarchy-20261029.json'
assert sha(pairs)=='8377661b150c2024a62622b5855c36c057c119d32493a8996a8f49c409d2badc'
assert sha(hierarchy)=='e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023'
p=load(pairs)['pairs']; h=load(hierarchy)['layer_samples']; hm={x['call_id']:x for x in h}; assert len(hm)==len(h)
scope=collections.defaultdict(list); shapes=collections.defaultdict(set)
for row in p:
 x=hm[row['scope_id']]; assert (x['ep'],x['layer_index'],x['role'])==(row['ep'],row['layer'],'conditional_target')
 key=(x['ep'],x['parent_digest']);scope[key].append(row['layer'])
 middle=[t for t in row['middle_tasks'] if t['info'] and t['info']['name']=='_prepare_indexer_indices_kernel'];assert len(middle)==1
 shape=middle[0]['info']['inputShapes'].strip('\"').split(';')[0];dims=[int(n) for n in shape.split(',')];assert len(dims)==2 and dims[1]==512
 shapes[key].add(dims[0])
assert len(scope)==4 and all(sorted(v)==[2,8,14,20,24,28,32,36] for v in scope.values())
wanted={d for ep,d in scope}; schedules=collections.defaultdict(list); runners=collections.defaultdict(list)
for pid,meta in load(early)['per_pid'].items():
 path=R/'diagnostic-20261029/runtime/metadata'/('events-'+pid+'.jsonl')
 assert path.is_file() and not path.is_symlink() and sha(path)==meta['sha256']
 for line in path.read_bytes().splitlines():
  x=json.loads(line); sched=x.get('schedule');
  if not isinstance(sched,dict) or sched.get('schedule_digest_sha256') not in wanted:continue
  digest=sched['schedule_digest_sha256']; canonical=json.dumps({'scheduled_total':sched['scheduled_total'],'requests':sched['requests']},sort_keys=True,separators=(',',':')).encode(); assert hashlib.sha256(canonical).hexdigest()==digest
  if x.get('label')=='Scheduler.schedule' and x.get('kind')=='call_end' and x.get('outcome')=='ok': schedules[digest].append(sched)
  if x.get('label')=='NPUModelRunner.execute_model' and x.get('kind')=='call_begin': runners[digest].append(sched)
result=[]
for (ep,d),layers in sorted(scope.items()):
 ss=schedules[d]; rr=runners[d]; assert len(ss)==1 and len(rr)==8
 s=ss[0]; rows=[];assert len(shapes[(ep,d)])==1
 for z in s['requests']:
  state=[v for v in s.get('scheduler_state_after_schedule',[]) if v.get('request_hmac_sha256')==z['request_hmac_sha256']]
  assert len(state)<=1
  rows.append({k:z.get(k) for k in ('scheduled_tokens','new_computed_tokens','cached_computed_tokens','spec_token_count')} | {'post_computed_tokens':state[0].get('num_computed_tokens_post_schedule') if state else None,'prompt_tokens':state[0].get('num_prompt_tokens') if state else None})
 assert sum(row['scheduled_tokens'] for row in rows)==s['scheduled_total']
 # Explicitly numeric sort: output no HMAC and no claim about input order.
 rows.sort(key=lambda x:tuple(str(x[k]) for k in sorted(x)))
 result.append({'ep':ep,'digest':d,'scope_layers':sorted(layers),'matched_scheduler_events':len(ss),'matched_runner_events':len(rr),'scheduled_total':s['scheduled_total'],'prepare_physical_rows':next(iter(shapes[(ep,d)])),'descriptors_sorted_without_identity':rows})
print(json.dumps({'inputs_sha256':{'pairs':sha(pairs),'hierarchy':sha(hierarchy),'metadata_summary':sha(early)},'groups':result,'limitations':['Digest matches content only and may repeat outside recorded window; scopes identify submission ancestry not device tensor contents.','Rows sorted numerically as strings without request IDs; no row order or per-kernel request mapping.','New/cached computed are scheduler descriptor values, not independently observed device C; device p/L/Q, CP slicing, graph padding and runtime data mutation remain unverified.','A null spec count is unavailable, not zero.']},indent=2))
