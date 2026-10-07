"""AST equality only; does not import torch or execute model code."""
import ast,copy,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;Q=R.parents[1]
production=Q/'post_full_local_prepare/prepare_finalize_candidate.py'
diagnostic=R/'prepare_finalize.py'
def methods(path,classname):
 c=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==classname)
 return {n.name:n for n in c.body if isinstance(n,ast.FunctionDef)}
a=methods(production,'PrepareAndFinalizeWithMC2');b=methods(diagnostic,'_H13Candidate');checks={}
for name in ['_select_tp_slice','_pad_local_slice','prepare']:
 other='_h13_candidate_prepare' if name=='prepare' else name
 left=copy.deepcopy(a[name]);right=copy.deepcopy(b[other]);right.name=left.name
 checks[name]=ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
 assert checks[name],name
assert hashlib.sha256(production.read_bytes()).hexdigest().startswith('7bb35edf')
result=dict(passed=True,exact_function_AST=checks,only_normalized_function_name=True,production_sha256=hashlib.sha256(production.read_bytes()).hexdigest(),diagnostic_sha256=hashlib.sha256(diagnostic.read_bytes()).hexdigest(),NPU_requests=0)
(R/'production_equivalence_CPU_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
