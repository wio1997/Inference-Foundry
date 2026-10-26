"""New selector semantics; GPU arithmetic reused unchanged from admitted Run666."""
import json,sys,tempfile,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serving.cohort_mode import read_cohort_mode
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'mode.json'
 for phase,on,v in [('off_a',False,1),('on',True,2),('off_b',False,3)]:
  expected=dict(phase=phase,enabled=on,version=v)
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(expected));tmp.replace(p)
  old=read_cohort_mode(p);assert old==expected
  p.write_text('{}');assert old==expected
 for bad in ({},{'phase':'on','enabled':1,'version':1},{'phase':'on','enabled':True,'version':True},{'phase':'on','enabled':True,'version':0},{'phase':'','enabled':True,'version':1},{'phase':'on','enabled':True,'version':1,'extra':1}):
  p.write_text(json.dumps(bad))
  try:read_cohort_mode(p)
  except (RuntimeError,KeyError,TypeError):pass
  else:raise AssertionError(bad)
 p.unlink()
 try:read_cohort_mode(p)
 except FileNotFoundError:pass
 else:raise AssertionError('missing mode accepted')
# Exact function AST equality proves unchanged Graph arithmetic/capture implementation.
import loop081_metadata_graph_patch_run666 as old
root=Path('/data/wio/Inference_Foundry')
original=old.PATCH['metadata'](old.SOURCES['metadata'][0].read_text()) if 'capture_graph' not in old.SOURCES['metadata'][0].read_text() else None
# During guarded preflight source is installed; regenerate prior candidate from saved original.
if original is None:
 raw=(root/'evidence/20260929_loop081_bound/run675/preflight/meta_patch_state/metadata.orig').read_text()
 original=old.PATCH['metadata'](raw)
new=(root/'evidence/20260929_loop081_bound/run675/candidate/metadata.py').read_text()
def functions(s):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef)}
a,b=functions(original),functions(new)
for name in ('capture_graph','update','_update_eager'):assert a[name]==b[name],name
print(json.dumps({'pass':True,'mode_cases':10,'unchanged_graph_functions':['capture_graph','update','_update_eager']}))
