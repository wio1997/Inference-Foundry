#!/usr/bin/env python3
"""Stdlib-only mutation oracle; no torch import or device execution."""
import ast, copy, hashlib, itertools, json, struct
from pathlib import Path
ROOT = Path('/data/wio/Inference_Foundry')
SCRIPT = ROOT / 'scripts/loop079_moe_row_sentinel.py'
source = SCRIPT.read_text()
fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
start = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'starts' for t in n.targets))
loop = next(n for n in fn.body if isinstance(n, ast.For) and any(isinstance(b, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'expert' for t in b.targets) for b in n.body))
loop = copy.deepcopy(loop)
# Extract actual source expert assignment and segment assert; omit torch argmax.
loop.body = loop.body[:2]
assert isinstance(loop.body[1], ast.Assert)
segment = compile(ast.fix_missing_locations(ast.Module(body=[start, loop], type_ignores=[])), str(SCRIPT), 'exec')
route = json.loads((ROOT/'evidence/20260927_loop078_bound/run403/capture/rank0_cohort1.json').read_text())['records']['64']['target'][0]['ids']
cases = [('small', [[0,33],[63,1],[32,64],[2,31]], 0,32), ('rank0', route,0,32), ('nonzero',route,96,128)]
results=[]
for name, ids, lo,hi in cases:
    K=len(ids[0]); counts=[sum(e==v for row in ids for v in row) for e in range(lo,hi)]
    slots=sorted([(t,k) for t,row in enumerate(ids) for k,v in enumerate(row) if lo<=v<hi], key=lambda tk:ids[tk[0]][tk[1]])
    g={(t,k):pos for pos,(t,k) in enumerate(slots)}
    def check(mapping):
        ns=dict(count=counts, lo=lo, hi=hi, ids=ids, name=name, valid=[(t,k,pos) for (t,k),pos in mapping.items()])
        exec(segment,ns)
    check(g)
    rejected=0
    for t,row in enumerate(ids):
        for a,b in itertools.combinations([k for k in range(K) if (t,k) in g],2):
            assert row[a] != row[b], 'duplicate per-token expert needs separate equivalence definition'
            bad=g.copy();bad[t,a],bad[t,b]=bad[t,b],bad[t,a]
            try:check(bad)
            except AssertionError:rejected+=1
            else:raise AssertionError(('same-token different-expert swap survived',name,t,a,b))
    results.append(dict(name=name, valid_routes=len(g), same_token_expert_swaps_rejected=rejected))
def bf16(x):
    bits=struct.unpack('>I',struct.pack('>f',x))[0]
    bits=(bits+0x7fff+((bits>>16)&1)) & 0xffff0000
    return struct.unpack('>f',struct.pack('>I',bits))[0]
errors=[]
for t in range(96):
    for a,b in itertools.combinations(range(6),2):
        sa=1+4*a+t/128;sb=1+4*b+t/128
        wa=2**-(a+1);wb=2**-(b+1)
        expected=sa*wa+sb*wb
        mutated=bf16(bf16(sb)*wa+bf16(sa)*wb)
        error=abs(mutated-expected)
        assert error>.015,(t,a,b,error)
        errors.append(error)
result=dict(status='PASS',device_imports=False,source_sha256=hashlib.sha256(source.encode()).hexdigest(), segment_assert='extracted from current source AST',cases=results,k_swap_numeric_cases=len(errors),min_k_swap_error=min(errors), threshold=.015, limitation='CPU BF16 round-to-nearest model and pairwise mutations; not native arbitrary-permutation exhaustive proof or raw-index replay')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
