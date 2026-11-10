"""Refine two representative ranks using observed submission boundaries only."""
from pathlib import Path
import json,sys
from analyze_scopes import span,merge,intersection,duration

root=Path(sys.argv[1]);out=[]
for prefix in ['dp0_pp0_tp0','dp1_pp0_tp0']:
    path=next((root/'traces').glob(prefix+'*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
    raw=json.load(open(path));es=raw['traceEvents'] if isinstance(raw,dict) else raw
    xs=[e for e in es if e.get('ph')=='X' and e.get('dur')]
    begins=sorted(span(e)[0] for e in xs if e.get('name')=='MODEL_EXECUTE');lo,hi=begins[0],begins[-1]
    free=merge([span(e) for e in xs if e.get('name')=='Free'])
    scopes=sorted([e for e in xs if e.get('name')=='prepare input'],key=lambda e:float(e['ts']))
    records=[]
    for scope in scopes:
        a,b=span(scope)
        if a<lo or b>hi:continue
        cpu=[e for e in xs if e.get('cat')=='cpu_op' and span(e)[0]>=a and span(e)[1]<=b]
        gloo=[e for e in cpu if e.get('name')=='gloo:all_reduce']
        native=[e for e in cpu if e.get('name','').startswith('_C_ascend::npu_') and 'metadata' in e.get('name','')]
        assert len(gloo)==1 and native,('boundary qualification',path,len(gloo),len(native))
        cut1=span(gloo[0])[1];cut2=min(span(e)[0] for e in native)
        assert a<=cut1<=cut2<=b
        parts=[('input_through_dp_sync',a,cut1),('after_dp_before_first_metadata_submission',cut1,cut2),('metadata_submission_and_finish',cut2,b)]
        rows=[]
        for name,x,y in parts:
            where=[e for e in cpu if e.get('name')=='aten::where' and span(e)[0]>=x and span(e)[1]<=y]
            rows.append({'key':name,'wall_ms':(y-x)/1000,'free_ms':duration(intersection(free,[[x,y]])),'where_calls':len(where)})
        total=duration(intersection(free,[[a,b]]));assert abs(sum(r['free_ms'] for r in rows)-total)<1e-5
        records.append({'start_us':a,'end_us':b,'wall_ms':(b-a)/1000,'free_ms':total,'parts':rows})
    out.append({'ep':0 if prefix.startswith('dp0') else 8,'trace':str(path),'closed_full_prepare_calls':len(records),'records':records,'note':'Boundaries are DP all_reduce completion and first native metadata wrapper start. Middle part is not an exact Python function range.'})
(root/'prepare-boundary-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('PREPARE_BOUNDARIES_COMPLETE',[(r['ep'],r['closed_full_prepare_calls']) for r in out])
