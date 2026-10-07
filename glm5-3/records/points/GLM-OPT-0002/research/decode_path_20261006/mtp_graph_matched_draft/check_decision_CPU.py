"""Falsify unsafe performance promotion from unmatched work/drift/single pair."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'pd_compare.py';tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('decide','effective_decision')];env=dict(math=math,PODS=['P','D']);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
base=[dict(D_wall_s=d,PD_wall_s=pd,P_wall_s=pd-d,TPOT_ms=t,workload_signature=dict(drafts=11,accepted=11)) for d,pd,t in ((.90,1.0,40),(.80,.89,35),(.905,1.005,40),(.805,.895,35))]
for row in base:row['cache_by_endpoint']=dict(P=dict(hbm_queries=58,hbm_hits=0,ext_queries=0,ext_hits=0),D=dict(hbm_queries=58,hbm_hits=0,ext_queries=58,ext_hits=58))
assert env['decide'](base)['H11_research_stack']
cases={}
r=deepcopy(base);r[3]['workload_signature']['accepted']=10;cases['different_accepted_work']=r
r=deepcopy(base);r[3]['D_wall_s']=.906;cases['only_one_D_pair_improves']=r
r=deepcopy(base);r[3]['PD_wall_s']=1.006;cases['only_one_complete_PD_pair_improves']=r
r=deepcopy(base);r[2]['D_wall_s']=1.0;r[3]['D_wall_s']=.905;cases['saving_below_D_drift']=r
r=deepcopy(base);r[2]['PD_wall_s']=1.2;r[3]['PD_wall_s']=1.095;cases['saving_below_PD_drift']=r
r=deepcopy(base);r[3]['TPOT_ms']=40;cases['no_second_TPOT_improvement']=r
r=deepcopy(base);r[3]['cache_by_endpoint']['D']['hbm_queries']=0;cases['unmatched_local_query_work']=r
for name,rows in cases.items():
 assert not env['decide'](rows)['H11_research_stack'],name
for key in ('D_wall_s','PD_wall_s','P_wall_s','TPOT_ms'):
 r=deepcopy(base);r[1][key]=float('nan')
 try:env['decide'](r)
 except AssertionError:pass
 else:raise AssertionError('nonfinite timing admitted')
evidence=env['decide'](base)
assert env['effective_decision'](evidence,True,False,None)['H11_research_stack']
for args in ((evidence,False,False,None),(evidence,True,True,None),(evidence,True,False,'terminal failure'),(None,True,False,None)):
 z=env['effective_decision'](*args);assert z['verdict']=='INVALIDATED' and not z['H11_research_stack']
row=dict(passed=True,CPU_only=True,model_requests=0,positive_case=1,rejected_counterexamples=list(cases),nonfinite_rejections=4,terminal_invalidation_cases=4,source_sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest())
(HERE/'decision_CPU_result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
