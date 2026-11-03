"""Pure static checks of isolation/locking/recorder bundle, with no remote operations."""
from pathlib import Path
import ast,hashlib,json
p=Path(__file__).resolve().parent
start=(p/'start_binding_recapture.py').read_text()
supervisor=(p/'supervise_binding_recapture.py').read_text()
recover=(p/'recover_binding.py').read_text()
watchdog=(p/'watchdog_binding.py').read_text()
old='wio-vllm-v41-binding-diag169-20261029'
new='wio-vllm-v41-binding-diag2-169-20261029'
for text in (start,supervisor,recover):
    assert new in text and old not in text
    assert 'binding-diagnostic-selective2' in text
assert 'binding-diagnostic-selective2' in watchdog
assert start.count("acquire_owner_fence(ROOT+'/analysis/")==1
assert supervisor.count("acquire_owner_fence(ROOT+'/analysis/")==2
assert 'acquire_owner_fence(ROOT.removesuffix("-selective2"))' in recover
assert "assert DIAG not in existing" in start
assert "if cols[1]==DIAG:found.append(cols[0])" in recover
assert "get('wio.binding_run_id')==RUN_ID" in supervisor
assert "get('wio.binding_run_id')!=RUN_ID" in recover
assert "bundle['instrumentation/recorder.c']=(HERE/'selective_recorder_next_graphgate.c').read_text()" in start
assert "source=dst/'instrumentation/recorder.c'" in start
assert "assert not dst.exists()" in start
assert "'export WIO_DIAG_DIR='" in start
assert "save('OFFLINE_EXPORT_REQUIRED'" in supervisor and 'export_deadline=' not in supervisor
assert supervisor.index("save('CAPTURE_COMPLETE')") < supervisor.index("save('OFFLINE_EXPORT_REQUIRED'") < supervisor.index('summary=json.loads(ssh(SUMMARY')
t=ast.parse(supervisor)
assert any(isinstance(n,ast.Try) and n.finalbody and any(isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='summary' for y in x.targets) for x in n.body) for n in t.body)

for s in (start,supervisor,recover,watchdog):ast.parse(s)
print(json.dumps({'status':'PASS','scope':'static only','shared_existing_lock':'PASS','separate_runtime_diagnostic_and_watchdog':'PASS','exact_container_name_and_run_id':'PASS','target_recorder_bundled':'PASS','recorder_sha256':hashlib.sha256((p/'selective_recorder_next_graphgate.c').read_bytes()).hexdigest()}))
