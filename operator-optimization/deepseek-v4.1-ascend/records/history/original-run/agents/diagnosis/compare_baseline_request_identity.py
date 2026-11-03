#!/usr/bin/env python3
"""Read frozen results; compare saved inputs and configuration without exposing content or IDs."""
import ast, hashlib, json, pathlib, re
ROOT = pathlib.Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results')
OUT = pathlib.Path(__file__).with_name('baseline-request-identity-anonymized.json')

def norm(s):
    return re.sub(r'202610\d{2}[_-]\d{6}', 'TIMESTAMP', s.replace('measure1','measureX').replace('measure2','measureX'))

class Normalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(norm(node.value)),node)
        return node

phases={}
for phase in ('measure1','measure2'):
    base=ROOT/phase
    ds=list(base.rglob('*details.jsonl')); configs=list(base.rglob('configs/*.py'))
    if len(ds)!=1 or len(configs)!=1: raise ValueError('ambiguous frozen sources')
    rows=[json.loads(x) for x in ds[0].open() if x.strip()]
    if len(rows)!=8 or len({r['data_id'] for r in rows})!=8: raise ValueError('bad request cardinality')
    rows.sort(key=lambda r:r['data_id'])
    phases[phase]=(rows,ds[0],configs[0])
a,b=phases['measure1'][0],phases['measure2'][0]
if any(x['data_id']!=y['data_id'] for x,y in zip(a,b)): raise ValueError('different position identity set')
positions=[]
for i,(x,y) in enumerate(zip(a,b),1):
    positions.append({'position':i,'saved_input_exact_equal':x['input']==y['input'],
      'input_token_count_equal':x['input_tokens']==y['input_tokens'],
      'output_token_count_equal':x['output_tokens']==y['output_tokens'],
      'cache_field_available':any('cache' in str(k).lower() for r in (x,y) for q in (r,r.get('extra_perf_data',{}),r.get('extra_details_data',{})) for k in q)})
c1=ast.parse(phases['measure1'][2].read_text());c2=ast.parse(phases['measure2'][2].read_text())
raw_equal=ast.dump(c1,include_attributes=False)==ast.dump(c2,include_attributes=False)
norm_equal=ast.dump(Normalizer().visit(c1),include_attributes=False)==ast.dump(Normalizer().visit(c2),include_attributes=False)
r={'sources':{s:{'details_sha256':hashlib.sha256(v[1].read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(v[2].read_bytes()).hexdigest()} for s,v in phases.items()},'positions':positions,'config_ast_exact_equal':raw_equal,'config_ast_equal_after_measure_and_timestamp_normalization':norm_equal,'limits':['saved input comparison does not establish exact transmitted tokens or common-prefix token count','matching token counts do not imply identical workload','normalized configuration text is not proof of runtime environment or server state identity','cache field presence scans only top level and extra dictionaries of frozen details, not other potential sources']}
OUT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print('saved anonymous comparison; positions',len(positions),'input exact equal',sum(x['saved_input_exact_equal'] for x in positions),'normalized config AST equal',norm_equal)
