"""CPU-only state ordering mock. Does not invoke subprocesses, SSH, Docker, or NPU."""
from pathlib import Path
import json, os, runpy, subprocess, sys, time, uuid
from unittest import mock

here=Path(__file__).resolve().parent
script=here/'orchestrate_binding.py'
results=[]
for failure in ("none","start","recovery"):
    run_id=uuid.uuid4().hex; seen=[]
    class Proc:
        pid=24680
        def poll(self):return None
    def fake_popen(args,**kwargs):
        assert args[1].endswith('watchdog_binding.py')
        token=args[3];assert token==run_id
        (here/('watchdog-'+token+'.json')).write_text(json.dumps({'state':'READY','binding_run_id':token,'pid':24680,'time':time.time()}))
        seen.append('watchdog_ready');return Proc()
    def fake_run(args,**kwargs):
        name=Path(args[1]).name;seen.append(name)
        assert kwargs['env']['WIO_BINDING_RUN_ID']==run_id
        return subprocess.CompletedProcess(args,1 if (failure=='start' and name=='start_binding_recapture.py') or (failure=='recovery' and name=='recover_binding.py') else 0)
    old=sys.argv;sys.argv=[str(script),'--execute']
    try:
        with mock.patch('uuid.uuid4',return_value=uuid.UUID(run_id)),mock.patch('subprocess.Popen',side_effect=fake_popen),mock.patch('subprocess.run',side_effect=fake_run):
            try:runpy.run_path(str(script),run_name='__main__')
            except RuntimeError:
                assert failure!="none"
    finally:sys.argv=old
    expected=['watchdog_ready','start_binding_recapture.py']+([] if failure=='start' else ['supervise_binding_recapture.py'])+['recover_binding.py']
    assert seen==expected,(seen,expected)
    assert (here/('watchdog-'+run_id+'.done')).exists() == (failure!='recovery')
    results.append({'scenario':failure,'order':seen,'recovery':'called','watchdog_completes':failure!='recovery','status':'PASS'})
# Check generated start remote program has pre-stop lease and token, and honestly records unknown semantics.
import ast
start=(here/'start_binding_recapture.py').read_text()
tree=ast.parse(start)
remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets))
compile(remote,'remote','exec')
assert remote.index("watchdog-lease.json") < remote.index("cmd(['docker','stop','--time','45',baseline['Id']]")
assert "'inference_semantics_unchanged':None" in remote and "'binding_run_id':RUN_ID" in remote
print(json.dumps({'status':'PASS','scope':'CPU mock/static only; no SSH or Docker','cases':results,'lease_guard_precedes_stop':'PASS','semantic_equivalence':'unknown'},indent=2))
