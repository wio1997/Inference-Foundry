"""Exercise the real post-capture AST with fake SSH: no service, Docker or NPU."""
from pathlib import Path
import ast,json,tempfile
p=Path(__file__).resolve().parent
module=ast.parse((p/'supervise_binding_recapture.py').read_text())
outer=next(n for n in module.body if isinstance(n,ast.Try) and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='summary' for t in x.targets) for x in n.body))
start=next(i for i,n in enumerate(outer.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for t in n.targets) and 'export_incomplete_requires_offline_parse' in ast.dump(n))
part=outer.body[start:]
for scenario in ('incomplete','error','stable'):
 with tempfile.TemporaryDirectory() as d:
  state={}
  def save(label,**kw):state.update(state=label,**kw)
  def ssh(code,timeout):
   if code=='raw':
    if scenario=='error':raise RuntimeError('probe failed')
    return json.dumps({'status':'RAW_VISIBILITY_STABLE_CLUE' if scenario=='stable' else 'RAW_PROFILE_INCOMPLETE_OR_UNSTABLE'})
   assert code=='summary';return json.dumps({'summary':'ok'})
  env={'state':state,'save':save,'ssh':ssh,'RAW_PROBE':'raw','SUMMARY':'summary','HERE':Path(d),'WORK':'/work','REL':'/task','json':json}
  exec(compile(ast.Module(body=part,type_ignores=[]),'mock-postcapture','exec'),env)
  assert state['state']=='EXPORT_SUMMARY_SAVED'
  assert state['raw_profile_visibility_complete']==(scenario=='stable'),state
  if scenario=='incomplete':assert state['raw_profile_status']=='RAW_PROFILE_INCOMPLETE_OR_UNSTABLE'
  if scenario=='error':assert state['raw_profile_status']=='RAW_CHECK_ERROR'
print(json.dumps({'status':'PASS','summary_after_probe':'PASS','incomplete_persists':'PASS','probe_error_persists':'PASS','scope':'CPU mock'}))
