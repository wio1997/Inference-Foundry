#!/usr/bin/env python3
"""Same-capture host scope accounting for selected rank 0/8 later windows."""
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path

BASE=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d/analysis/operator-export/copies')

def ns(x): return int(Decimal(str(x))*1000)

def union(vv):
    out=[]
    for a,b in sorted(vv):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return sum(b-a for a,b in out)

results=[]
for spec in SPECS:
    p=BASE/spec['source_dir']/'ASCEND_PROFILER_OUTPUT/trace_view.json'
    data=json.loads(p.read_text()); data=data['traceEvents'] if isinstance(data,dict) else data
    events=[]
    for x in data:
        if x.get('ph')=='X' and x.get('cat')=='cpu_op' and x.get('dur',0)>0:
            events.append({'s':ns(x['ts']),'e':ns(x['ts'])+ns(x['dur']),'name':x['name'],'pid':x.get('pid'),'tid':x.get('tid')})
    roots=[x for x in events if x['name']=='prepare input']
    roots.sort(key=lambda x:x['s'])
    summaries=[]
    for ordinal,root in enumerate(roots):
        descendants=[dict(x) for x in events if x['pid']==root['pid'] and x['tid']==root['tid'] and root['s']<=x['s'] and x['e']<=root['e'] and x!=root]
        descendants.sort(key=lambda x:(x['s'],-x['e']))
        # Compute exclusive recorded-scope times using interval unions. No CPU
        # activity or scheduling state is inferred from a missing child scope.
        nodes=[dict(root,children=[])];stack=[0]
        containment_violations=0
        for x in descendants:
            while len(stack)>1 and x['s']>=nodes[stack[-1]]['e']:stack.pop()
            while len(stack)>1 and x['e']>nodes[stack[-1]]['e']:
                containment_violations+=1;stack.pop()
            parent=stack[-1]
            nodes[parent]['children'].append((x['s'],x['e']))
            nodes.append(dict(x,children=[]));stack.append(len(nodes)-1)
        byname=collections.defaultdict(lambda:{'count':0,'inclusive_sum_ns':0,'exclusive_sum_ns':0})
        for x in nodes[1:]:
            v=byname[x['name']];v['count']+=1;v['inclusive_sum_ns']+=x['e']-x['s'];v['exclusive_sum_ns']+=x['e']-x['s']-union(x['children'])
        gap_matches=[]
        for gap in spec['gaps']:
            a,b=gap['start_epoch_ns'],gap['end_epoch_ns']
            clipped=max(0,min(root['e'],b)-max(root['s'],a))
            if clipped:gap_matches.append({'gap_ordinal':gap['ordinal'],'overlap_ns':clipped,'gap_ns':b-a})
        summaries.append({'ordinal':ordinal,'s_ns':root['s'],'e_ns':root['e'],'duration_ns':root['e']-root['s'],
                          'descendant_scope_count':len(descendants),'descendant_union_ns':union([(x['s'],x['e']) for x in descendants]),
                          'root_exclusive_recorded_scope_ns':root['e']-root['s']-union(nodes[0]['children']),
                          'containment_violations':containment_violations,'gaps':gap_matches,
                          'by_name':dict(sorted(byname.items(),key=lambda kv:-kv[1]['exclusive_sum_ns']))})
    results.append({'rank':spec['rank'],'source_dir':spec['source_dir'],'trace_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'prepare_input_scopes':summaries})

sources={}
for relative in ['vllm_ascend/worker/model_runner_v1.py','vllm_ascend/attention/dsa_v41.py','vllm_ascend/attention/context_parallel/dsa_v41_cp.py','vllm_ascend/attention/dsa_v1.py','vllm_ascend/compilation/acl_graph.py']:
    p=Path('/vllm-workspace/vllm-ascend')/relative
    if p.is_file():sources[relative]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps({'scope':'recorded worker prepare-input scopes; same official later capture',
                  'limits':['Exclusive time means uncovered by nested recorded cpu_op scopes; it includes Python/native work, waits and profiler overhead.',
                             'Source snapshots are current frozen checkout evidence, not proof of a particular historical branch.'],
                  'rows':results,'sources':sources},indent=2))
