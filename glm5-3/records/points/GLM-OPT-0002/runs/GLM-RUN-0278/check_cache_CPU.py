"""Regression against actual Run274 native P/D counter deltas; no requests."""
import ast,json
from copy import deepcopy
from pathlib import Path
HERE=Path(__file__).resolve().parent;SOURCE=HERE/'pd_compare.py'
node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_cache')
pods=['172.16.10.166:9081','172.16.10.167:9900'];env=dict(PODS=pods)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),env)
raw=HERE.parent/'GLM-RUN-0274/A1_complete_cache_delta.json'
endpoints=json.loads(raw.read_text())['per_endpoint'];rows={p:endpoints[p]['dp0'] for p in pods};env['validate_cache'](rows,58)
rejected=[]
for pod,key,value in [(pods[0],'ext_queries',0),(pods[0],'ext_hits',58),(pods[0],'hbm_hits',58),(pods[1],'hbm_queries',0),(pods[1],'ext_hits',57)]:
 bad=deepcopy(rows);bad[pod][key]=value
 try:env['validate_cache'](bad,58)
 except AssertionError:rejected.append([pod,key,value])
 else:raise AssertionError('incorrect work admitted')
row=dict(passed=True,CPU_only=True,HTTP_NPU_model_requests=0,actual_Run274_P_queries58_hits0_accepted=True,
    rejected_counterexamples=rejected,source_sha256=__import__('hashlib').sha256(SOURCE.read_bytes()).hexdigest())
(HERE/'cache_CPU_result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
