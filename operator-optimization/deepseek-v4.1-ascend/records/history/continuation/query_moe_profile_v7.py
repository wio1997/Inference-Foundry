from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal
from datetime import datetime, timezone
import json, hashlib, statistics

N=Path(__file__).parent
O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
P=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'


def stats(v):
    return dict(n=len(v),min=min(v),median=statistics.median(v),max=max(v)) if v else None


def main():
    sha=hashlib.sha256(P.read_bytes()).hexdigest()
    assert sha=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
    rows=[]
    for line in P.open():
        d=json.loads(line)
        if d.get('kind')!='rank':continue
        for s in d['data']['layers']:
            found=[]
            for g in s['operator_groups']:
                for a in g['tasks']:
                    c=a['csv']
                    if c['Type']!='DispatchFFNCombine':continue
                    start=int(Decimal(a['device_start_us'])*1000)
                    dur=int(Decimal(a['trace_duration_us'])*1000)
                    found.append(dict(task_id=c['Task ID'],stream=c['Stream ID'],start_ns=start,end_ns=start+dur,duration_ns=dur,
                                      host_start_us=a['host_start_us'],type=c['Type'],name=c['Name'],
                                      input_shapes=c['Input Shapes'],output_shapes=c['Output Shapes'],
                                      input_dtypes=c['Input Data Types'],output_dtypes=c['Output Data Types'],
                                      accelerator_core=c.get('Accelerator Core'),block_num=c.get('Block Num'),mix_block_num=c.get('Mix Block Num')))
            rows.append(dict(ep=s['ep'],digest=s['scheduler_digest'],layer=s['layer'],scope_id=s['layer_scope_id'],count=len(found),tasks=found))
    assert len(rows)==1280
    targets=[]
    for digest in sorted({r['digest'] for r in rows if r['ep']==3}):
        group=[r for r in rows if r['ep']==3 and r['digest']==digest and r['count']==1]
        for label,r in [('MIN',min(group,key=lambda r:r['tasks'][0]['duration_ns'])),('MAX',max(group,key=lambda r:r['tasks'][0]['duration_ns']))]+[(f'L{r["layer"]}',r) for r in group if r['layer'] in (0,36)]:
            t=r['tasks'][0]
            targets.append(dict(label=f'F21_{digest[:8]}_{label}',ep=3,layer=r['layer'],scope_id=r['scope_id'],digest=digest,**t))
    groups=defaultdict(list)
    for r in rows:
        for t in r['tasks']:groups[(r['ep'],r['digest'])].append(t['duration_ns'])
    cfg=json.loads((R/'evidence/model-config.json').read_text())['text_config']
    summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),retained_scopes=len(rows),count_hist=dict(Counter(r['count'] for r in rows)),
                 physical_input_shape_counts=dict(Counter(t['input_shapes'] for r in rows for t in r['tasks'])),
                 output_shape_counts=dict(Counter(t['output_shapes'] for r in rows for t in r['tasks'])),
                 per_rank_digest_duration_ns=[dict(ep=ep,digest=d,stats=stats(vals)) for (ep,d),vals in sorted(groups.items())],
                 configured_moe={k:cfg[k] for k in cfg if any(s in k for s in ['expert','moe_intermediate'])},
                 selected_db_anchors=len(targets),source_sha256=sha,new_source_read=False,new_device_work=False,
                 overall_complete=False,limits=['One outer composite interval does not expose dispatch/GMM1/activation/GMM2/combine timing.',
                                               'Physical shape is not actual expert count or effective valid rows.',
                                               'Per-rank/digest durations are exported measurements, not request critical path or internal wait decomposition.',
                                               'New166 routing snapshots are a different run and cannot be same-call joined to these169 timings.'])
    (O/'moe-profile-query-v7.json').write_text(json.dumps(dict(summary=summary,scope_results=rows),ensure_ascii=False,indent=2)+'\n')
    (O/'moe-profile-summary-v7.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (O/'moe-db-targets-v7.json').write_text(json.dumps(dict(targets=targets),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_rank_digest_duration_ns'},ensure_ascii=False))


if __name__=='__main__':main()
