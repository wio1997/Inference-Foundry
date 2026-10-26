"""Require both roles' resolved native preflights before any model starts."""
import pathlib,json,subprocess,sys,time,os
R=pathlib.Path('/run-evidence');q=subprocess.run([sys.executable,str(R/'preflight.py')]);assert q.returncode==0;deadline=time.monotonic()+180
while not (R/'start-model.json').exists():
 assert not (R/'stop.json').exists(),'Owned startup stopped before model permit'
 assert time.monotonic()<deadline,'Bothrolepreflightpermitdeadline';time.sleep(.5)
x=json.loads((R/'start-model.json').read_text());assert x['project_id']=='glm53-flash' and x['run_id']=='r247' and x['both_native_preflights_passed'] is True
os.execv(sys.executable,[sys.executable,'-m','sglang.launch_server',*sys.argv[1:]])
