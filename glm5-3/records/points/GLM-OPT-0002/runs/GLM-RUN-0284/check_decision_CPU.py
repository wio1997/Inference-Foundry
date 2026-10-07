from pathlib import Path
import ast,copy,hashlib,json,math
r=Path(__file__).resolve().parent;s=r/'pd_compare.py'
nodes=[n for n in ast.parse(s.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('decide','effective_decision','validate_cache')]
P='172.16.10.166:9081';D='172.16.10.167:9900';ns=dict(math=math,PODS=[P,D]);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(s),'exec'),ns)
cache={P:dict(hbm_queries=58,hbm_hits=0,ext_queries=58,ext_hits=0),D:dict(hbm_queries=58,hbm_hits=0,ext_queries=58,ext_hits=58)}
rows=[dict(D_wall_s=a,PD_wall_s=b,P_wall_s=b-a,TPOT_ms=t,workload_signature=dict(drafts=11,draft_tokens=11,accepted=11,invalid=0),cache_by_endpoint=copy.deepcopy(cache)) for a,b,t in ((1,1.3,40),(.8,1.1,32),(1.01,1.32,40.4),(.81,1.12,32.4))]
positive=ns['decide'](rows);assert positive['H13_research_stack']
cases=[]
for name in ('different_accepted_work','only_one_D_pair_improves','only_one_PD_pair_improves','D_below_drift','PD_below_drift','no_second_TPOT_gain','unmatched_cache_queries'):
    x=copy.deepcopy(rows)
    if name=='different_accepted_work':x[3]['workload_signature']['drafts']=12
    elif name=='only_one_D_pair_improves':x[3]['D_wall_s']=1.02
    elif name=='only_one_PD_pair_improves':x[3]['PD_wall_s']=1.33
    elif name=='D_below_drift':x[1]['D_wall_s']=.995
    elif name=='PD_below_drift':x[1]['PD_wall_s']=1.295
    elif name=='no_second_TPOT_gain':x[3]['TPOT_ms']=40.4
    else:x[3]['cache_by_endpoint'][P]['hbm_queries']=59
    result=ns['decide'](x);assert not result['H13_research_stack'] and result['verdict']=='INCONCLUSIVE';cases.append(name)
for args in ((False,False,None),(True,True,None),(True,False,'failed terminal')):
    z=ns['effective_decision'](positive,*args);assert z['verdict']=='INVALIDATED' and not z['H13_research_stack']
for key in ('D_wall_s','PD_wall_s','P_wall_s','TPOT_ms'):
    x=copy.deepcopy(rows);x[1][key]=float('nan')
    try:ns['decide'](x)
    except AssertionError:pass
    else:raise AssertionError('nonfinite accepted')
ns['validate_cache'](cache,58)
for pod,key,value in ((P,'ext_queries',0),(P,'ext_hits',58),(P,'hbm_hits',58),(D,'hbm_queries',0),(D,'ext_hits',57)):
    x=copy.deepcopy(cache);x[pod][key]=value
    try:ns['validate_cache'](x,58)
    except AssertionError:pass
    else:raise AssertionError('wrong endpoint cache accepted')
proof=dict(passed=True,source_sha256=hashlib.sha256(s.read_bytes()).hexdigest(),rejected_counterexamples=cases,nonfinite_rejections=4,terminal_invalidation_cases=3,cache_endpoint_rejections=5,model_requests=0,NPU_initialization=False)
p=r/'decision_CPU_result.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
