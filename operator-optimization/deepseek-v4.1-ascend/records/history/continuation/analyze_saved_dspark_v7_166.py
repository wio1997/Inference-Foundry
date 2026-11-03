from pathlib import Path
from collections import Counter
from bisect import bisect_right
from datetime import datetime, timezone
import socket, json, hashlib, statistics

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
p=A/'ep3-dspark-and-moe-linked-tasks-v7-v2.json'
before=p.stat()
raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='e806fc67206b77a7d90d1d70f289eab84656e78fb813c1636509c901d51c4a98'
d=json.loads(raw)
assert d['database_stat_unchanged'] and len(d['tasks'])==3307
chains=d['callchains']


def frames(t):
    return [f['frame'] for a in t['api_candidates'] for f in chains[str(a['callchainId'])]]


def has(t,part):return any(part in f for f in frames(t))


def sh(s):
    return [tuple(int(n) for n in a.split(',')) for a in s.strip('"').split(';') if a and all(v.strip().lstrip('-').isdigit() for v in a.split(','))]


def short(t):
    out={k:t[k] for k in ['taskId','startNs','endNs','streamId','op_type','input_shapes','output_shapes','input_dtypes','output_dtypes']}
    out['API']=[a['name'] for a in t['api_candidates']]
    relevant=[f for f in frames(t) if any(s in f for s in ['deepseek_v41/dspark.py(', 'deepseek_v41/model.py(', 'moe_comm_method.py(', 'rope_dsv4.py(', 'vocab_parallel_embedding.py(', 'layernorm.py('])]
    out['source_frames']=list(dict.fromkeys(relevant))[:6]
    return out


def types(ts):return dict(Counter(t['op_type'] for t in ts))


tasks=sorted(d['tasks'],key=lambda t:(t['startNs'],t['streamId']))
repeat=[t for t in tasks if t['op_type']=='Tile' and any('/models/deepseek_v41/dspark.py(' in f and '): forward' in f for f in frames(t))]
pre=[t for t in tasks if t['op_type']=='HcPre']
post=[t for t in tasks if t['op_type']=='HcPost']
assert len(repeat)==13 and len(pre)==len(post)==78
starts=[r['startNs'] for r in repeat]
forward_groups=[]
for i,r in enumerate(repeat):
    lo=r['startNs'];hi=starts[i+1] if i+1<len(starts) else 2**63-1
    ps=[t for t in pre if lo<=t['startNs']<hi]
    qs=[t for t in post if lo<=t['startNs']<hi]
    init=[t for t in tasks if lo<=t['startNs']<(ps[0]['startNs'] if ps else hi) and any(f'dspark.py({n}): forward' in f for n in [249,254,258,259,260,261] for f in frames(t))]
    forward_groups.append(dict(index=i,repeat=short(r),hcpre_count=len(ps),hcpost_count=len(qs),
                               first_hcpre=short(ps[0]) if ps else None,initialization_types=types(init),initialization=[short(t) for t in init]))
embed=[t for t in tasks if has(t,'dspark.py(249): forward')]
projection=[t for t in tasks if t['op_type'] in ['MatMulV2','MatMulV3'] and has(t,'dspark.py(290): _project_shared_kv')]
context=[t for t in tasks if has(t,'): precompute_and_store_context_kv')]
assert len(projection)==39
projection_starts=[t['startNs'] for t in projection]
pipelines=[]
for i,pr in enumerate(projection):
    lo=pr['startNs'];hi=projection_starts[i+1] if i+1<len(projection_starts) else 2**63-1
    ts=[t for t in context if lo<=t['startNs']<hi]
    keep=[t for t in ts if (t['op_type']=='RmsNorm' and has(t,'dspark.py(290):')) or
          (t['op_type']=='RotaryPositionEmbedding' and has(t,'dspark.py(68):')) or
          (t['op_type']=='ConcatD' and has(t,'dspark.py(293):')) or
          (t['op_type']=='ScatterNdUpdateSk' and has(t,'dspark.py(246):'))]
    pipelines.append(dict(index=i,projection=short(pr),physical_context_rows=sh(pr['input_shapes'])[0][0],
                           context_op_counts=types(ts),selected_chain=[short(pr)]+[short(t) for t in keep]))
marked=[t for t in tasks if any(has(t,f'dspark.py({n}): {fn}') for n,fn in [(332,'compute_logits'),(321,'markov_embed'),(324,'markov_bias')])]
markov_groups={}
for method,line in [('compute_logits',332),('markov_embed',321),('markov_bias',324)]:
    ts=[t for t in marked if has(t,f'dspark.py({line}): {method}')]
    counts=Counter((t['op_type'],t['input_shapes'],t['output_shapes'],t['input_dtypes'],t['output_dtypes']) for t in ts)
    markov_groups[method]=dict(records=len(ts),op_counts=types(ts),signatures=[dict(op_type=k[0],input_shapes=k[1],output_shapes=k[2],input_dtypes=k[3],output_dtypes=k[4],count=v) for k,v in counts.items()])
source_correlations={name:[short(t) for t in tasks if has(t,name)][:2] for name in ['dspark.py(249): forward','dspark.py(258): forward','dspark.py(265): forward']}
after=p.stat()
assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),source_file=str(p),source_sha256=hashlib.sha256(raw).hexdigest(),source_stat_unchanged=True,
             task_count=3307,unique_api_count_hist=dict(Counter(len(t['api_candidates']) for t in tasks)),
             F28=dict(repeat_records=len(repeat),embedding_callsite_op_counts=types(embed),
                       observed_repeat_source_frames=list(dict.fromkeys(f for t in repeat for f in frames(t) if '/models/deepseek_v41/dspark.py(' in f)),
                       repeat_input_shape_counts=dict(Counter(t['input_shapes'] for t in repeat)),
                       first_HCpre_input_shape_counts=dict(Counter(g['first_hcpre']['input_shapes'] if g['first_hcpre'] else 'MISSING' for g in forward_groups)),
                       hcpre_count_per_ordered_repeat_group=dict(Counter(g['hcpre_count'] for g in forward_groups)),
                       hcpost_count_per_ordered_repeat_group=dict(Counter(g['hcpost_count'] for g in forward_groups))),
             F29=dict(context_projection_records=len(projection),context_projection_input_shape_counts=dict(Counter(t['input_shapes'] for t in projection)),
                       context_op_counts=types(context),pipeline_norm_count_hist=dict(Counter(sum(t['op_type']=='RmsNorm' for t in g['selected_chain']) for g in pipelines)),
                       pipeline_rotation_count_hist=dict(Counter(sum(t['op_type']=='RotaryPositionEmbedding' for t in g['selected_chain']) for g in pipelines)),
                       pipeline_scatter_count_hist=dict(Counter(sum(t['op_type']=='ScatterNdUpdateSk' for t in g['selected_chain']) for g in pipelines))),
             F21_draft=dict(records=sum(t['op_type']=='DispatchFFNCombine' for t in tasks),
                            input_shape_counts=dict(Counter(t['input_shapes'] for t in tasks if t['op_type']=='DispatchFFNCombine')),
                            output_shape_counts=dict(Counter(t['output_shapes'] for t in tasks if t['op_type']=='DispatchFFNCombine'))),
             F30=markov_groups,no_database_requery_or_device_work=True,overall_complete=False,
             limits=['Ordered-repeat and ordered-projection groups are device-order associations within one rank, not tensor, layer, request or iteration identity proofs.',
                     'Missing unlinked APIs or unexported work remain outside these task associations.',
                     'Source callsite plus frozen semantics can refine candidates; historical binary/numerical, actual positions/slots/aliases and event generations are separate evidence.',
                     'Context and query KV share cache semantics but are not interchangeable; Markov/logits belong F30, not F29.'])
result=dict(summary=summary,forward_groups=forward_groups,context_pipelines=pipelines,source_correlations=source_correlations)
out=A/'ep3-dspark-pipeline-audit-v7-v2.json'
assert not out.exists(),'refuse_overwrite'
payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
out.write_text(payload)
indexes=sorted({0,min(range(len(forward_groups)),key=lambda i:sh(forward_groups[i]['repeat']['input_shapes'])[0][0]),max(range(len(forward_groups)),key=lambda i:sh(forward_groups[i]['repeat']['input_shapes'])[0][0])})
pipe_indexes=sorted({0,max(range(len(pipelines)),key=lambda i:pipelines[i]['physical_context_rows'])})
public=dict(summary=json.loads(json.dumps(summary)),full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
             selected_forward_groups=[forward_groups[i] for i in indexes],selected_context_pipelines=[pipelines[i] for i in pipe_indexes],source_correlations=source_correlations)
for v in public['summary']['F30'].values():
    v['signature_group_count']=len(v['signatures'])
    v['signature_display_truncated']=len(v['signatures'])>8
    v['signatures']=v['signatures'][:8]
while len(json.dumps(public,ensure_ascii=False).encode())>32000:
    public['display_reduced_full_data_retained_server']=True
    if public['source_correlations']:
        public['source_correlations']={}
    elif len(public['selected_forward_groups'])>1:
        public['selected_forward_groups']=public['selected_forward_groups'][-1:]
    elif len(public['selected_context_pipelines'])>1:
        public['selected_context_pipelines']=public['selected_context_pipelines'][-1:]
    else:
        break
assert len(json.dumps(public,ensure_ascii=False).encode())<48000,'public_budget'
print(json.dumps(public,ensure_ascii=False))
