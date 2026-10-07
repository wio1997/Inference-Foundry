"""Reject each missing immutable action dependency using actual admission AST."""
from pathlib import Path
import ast,hashlib,json,tempfile,shutil
r=Path(__file__).resolve().parent;s=r/'pd_local_prepare_correctness.py';deps=json.loads((r/'startup_dependencies.json').read_text());boot_sha=deps['epochs/candidate/runtime_bundle/native_acl_lifecycle.py']
node=next(n for n in ast.parse(s.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='verify_boot')
plan=dict(startup_dependencies=deps,native_boot_sha256=boot_sha)
ns=dict(ROOT=r,PLAN=plan,hashlib=hashlib);exec(compile(ast.Module(body=[node],type_ignores=[]),str(s),'exec'),ns)
assert len(ns['verify_boot']()['native_boot'])==len(deps)+2
failed=[]
with tempfile.TemporaryDirectory(prefix='glm283-closure-') as temp:
    root=Path(temp)
    for rel in deps:
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((r/rel).read_bytes())
    ns['ROOT']=root
    for rel in deps:
        p=root/rel;raw=p.read_bytes();p.unlink()
        try:ns['verify_boot']()
        except FileNotFoundError:failed.append(rel)
        else:raise AssertionError('missing input admitted: '+rel)
        p.write_bytes(raw)
        p.write_bytes(raw+b'\n')
        try:ns['verify_boot']()
        except AssertionError:pass
        else:raise AssertionError('changed input admitted: '+rel)
        p.write_bytes(raw)
assert len(failed)==11
proof=dict(passed=True,driver_sha256=hashlib.sha256(s.read_bytes()).hexdigest(),dependencies=deps,all_missing_and_changed_rejected=failed,ACL_NPU_model_imports=0,model_requests=0,limitations='CPU filesystem admission only; no native graph or PD output claim')
p=r/'startup_closure_CPU_result.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k not in ('dependencies','all_missing_and_changed_rejected')}))
