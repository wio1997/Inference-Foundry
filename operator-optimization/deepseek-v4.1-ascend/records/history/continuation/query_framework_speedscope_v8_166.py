from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
ROOT=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/runtime/logs/framework-20261030T091949Z-3d9bcf87')
pins={'pyspy-DP0.speedscope.json':'bc6d1bdfe401f604847942d0334008aea14306644d0bea7399a40924b849365c',
      'pyspy-DP1.speedscope.json':'e001b7faf9643cd536971f55113a54017539e2098801d0508a875e743aa50d65',
      'result.json':'f6b3f4ad79503ffa143765884d36a1712546b15b104ed71edc267b1fee4206fb',
      'measure-results.json':'908ef136357f6621e9d1cab2ca16bdde9b8a98d6fc27bbf28b36b93f93fe3ebe'}
data={}
for name,pin in pins.items():
    p=ROOT/name;before=p.stat();b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==pin,(name,'sha')
    data[name]=json.loads(b);after=p.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
rows=[]
for dp in ['DP0','DP1']:
    d=data['pyspy-'+dp+'.speedscope.json'];frames=d['shared']['frames'];profiles=[]
    for p in d['profiles']:
        selected=[];leaves=Counter();ancestry=Counter()
        for stack,weight in zip(p['samples'],p['weights']):
            if not stack:continue
            leaves[stack[-1]]+=weight
            if any(frames[i]['name']=='_wait_for_response' and frames[i]['file'].endswith('multiproc_executor.py') for i in stack):
                ancestry[tuple(stack)]+=weight
        wanted={i for stack in p['samples'] for i in stack if frames[i]['name'] in ['_wait_for_response','wait','acquire_read','check','should_warn','timeout_ms','memory_fence','sched_yield','step_with_batch_queue','execute_dummy_batch']}
        profiles.append(dict(main='MainThread' in p['name'],samples=len(p['samples']),sample_weight_s=sum(p['weights']),
                             selected_locations=[dict(frame_id=i,frame=frames[i],leaf_weight_s=leaves[i]) for i in sorted(wanted)],
                             all_leaf_locations=[dict(frame=frames[i],weight_s=w) for i,w in leaves.most_common()],
                             response_wait_full_stack_groups=[dict(frames=[frames[i] for i in stack],sample_weight_s=w) for stack,w in ancestry.most_common()]))
    assert sum(p['main'] for p in profiles)==1
    rows.append(dict(DP=dp,profiles=profiles))
measure=data['measure-results.json'];result=data['result.json']
public_cases=dict(result_keys=list(result),request_keys=list(measure[0]) if measure else [],
                  request_hash_fields=[{k:v for k,v in r.items() if 'hash' in k.lower() or k in ['temperature','top_p','seed','max_tokens','stream']} for r in measure])
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/framework-speedscope-callsites-v8.json')
assert not out.exists()
full=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_root=str(ROOT),source_sha256=pins,source_stats_unchanged=True,rows=rows,case_metadata=public_cases,
          no_device_or_inference_work=True,limits=['Samples include idle tail and have no per-sample absolute times. Weight is notCPUtime.',
                                                  'Source file/line identifies sampled Python location, not current loaded binary or per-request causal bottleneck.'])
payload=json.dumps(full,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(created_utc=full['created_utc'],source_sha256=pins,source_stats_unchanged=True,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                      main_profiles=[dict(DP=r['DP'],selected_locations=p['selected_locations'],response_stack_groups=len(p['response_wait_full_stack_groups']),
                                          top3_response_stacks=p['response_wait_full_stack_groups'][:3]) for r in rows for p in r['profiles'] if p['main']],case_metadata=public_cases,
                      no_device_or_inference_work=True,limits=full['limits']),ensure_ascii=False))
