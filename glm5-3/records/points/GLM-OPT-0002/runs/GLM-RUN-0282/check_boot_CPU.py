"""Execute actual epoch boot main and admission AST using CPU-only import doubles."""
from pathlib import Path
from types import SimpleNamespace
import ast,builtins,hashlib,json,tempfile
r=Path(__file__).resolve().parent;plan=json.loads((r/'functional_plan.json').read_text())
boot=r/'epochs/candidate/runtime_bundle/native_acl_lifecycle.py';raw=boot.read_bytes()
other=r/'epochs/baseline_recovery/runtime_bundle/native_acl_lifecycle.py'
assert raw==other.read_bytes() and hashlib.sha256(raw).hexdigest()==plan['native_boot_sha256']
main=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='main')
rows=[]
for scenario in ('cli_success','cli_failure','acl_init_failure','wrong_mode'):
    calls=[]
    def init():calls.append('acl.init');return 1 if scenario=='acl_init_failure' else 0
    def finalize():calls.append('acl.finalize');return 0
    def native_main():
        calls.append('native_cli')
        if scenario=='cli_failure':raise ValueError('fixture native failure')
    acl=SimpleNamespace(init=init,finalize=finalize)
    def imports(name,*args,**kwargs):
        if name=='acl':calls.append('acl_import');return acl
        if name=='vllm.entrypoints.cli.main':return SimpleNamespace(main=native_main)
        raise AssertionError('unexpected native import: '+name)
    custom=dict(vars(builtins));custom['__import__']=imports
    sys=SimpleNamespace(argv=[str(boot),'invalid' if scenario=='wrong_mode' else 'cli','serve','/fixture-model','--fixture'])
    ns={'sys':sys,'os':SimpleNamespace(getpid=lambda:1),'json':json,'__builtins__':custom,'print':lambda *a,**k:None}
    exec(compile(ast.Module(body=[main],type_ignores=[]),str(boot),'exec'),ns)
    error=None
    try:ns['main']()
    except (ValueError,RuntimeError) as e:error=repr(e)
    if scenario=='cli_success':assert error is None and calls==['acl_import','acl.init','native_cli','acl.finalize'] and sys.argv==['vllm','serve','/fixture-model','--fixture']
    elif scenario=='cli_failure':assert 'fixture native failure' in error and calls==['acl_import','acl.init','native_cli','acl.finalize']
    elif scenario=='acl_init_failure':assert 'acl.init failed' in error and calls==['acl_import','acl.init']
    else:assert error and not calls
    rows.append(dict(scenario=scenario,calls=calls,error=error))
source=r/'pd_local_prepare_correctness.py';tree=ast.parse(source.read_text())
verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_boot')
with tempfile.TemporaryDirectory(prefix='glm282-boot-') as folder:
    tmp=Path(folder)
    for epoch in ('candidate','baseline_recovery'):
        p=tmp/'epochs'/epoch/'runtime_bundle/native_acl_lifecycle.py';p.parent.mkdir(parents=True);p.write_bytes(raw)
    ns=dict(ROOT=tmp,PLAN=plan,hashlib=hashlib)
    exec(compile(ast.Module(body=[verify],type_ignores=[]),str(source),'exec'),ns)
    assert len(ns['verify_boot']()['native_boot'])==2
    bad=tmp/'epochs/baseline_recovery/runtime_bundle/native_acl_lifecycle.py';bad.unlink()
    try:ns['verify_boot']()
    except FileNotFoundError:pass
    else:raise AssertionError('missing recovery boot admitted')
    bad.write_bytes(raw+b'\n')
    try:ns['verify_boot']()
    except AssertionError:pass
    else:raise AssertionError('changed recovery boot admitted')
# Actual inherited native entry resolves exactly this epoch-local file.
inherited=r.parent/'GLM-RUN-0246/pd_functional_run.py'
native=next(n for n in ast.parse(inherited.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='native')
assert "runtime_bundle/native_acl_lifecycle.py" in ast.unparse(native)
row=dict(passed=True,boot_sha256=hashlib.sha256(raw).hexdigest(),driver_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),same_boot_both_epochs=True,missing_or_changed_boot_rejected=True,actual_main_cli_cases=rows,model_imports=0,NPU_initialization=False,model_requests=0)
p=r/'boot_CPU_result.json';assert not p.exists();p.write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps({k:v for k,v in row.items() if k!='actual_main_cli_cases'}))
