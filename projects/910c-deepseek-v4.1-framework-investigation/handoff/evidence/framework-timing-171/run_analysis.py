from pathlib import Path
import json,subprocess,sys
P=Path(__file__).resolve().parent;root=json.loads((P/'launch.json').read_text())['remote']
runner='''from pathlib import Path
import sys,json,subprocess,hashlib
p=Path(sys.argv[1]);state={'status':'ANALYZING_EXISTING_EXPORTS','new_capture':False,'original_service_actions':False,'scripts':{}}
def write(): (p/'analysis-state.json').write_text(json.dumps(state,indent=2))
try:
 for name in ['analyze_functions.py','analyze_detailed.py']:
  state['script']=name;state['scripts'][name]=hashlib.sha256((p/name).read_bytes()).hexdigest();write()
  with (p/(name+'.log')).open('wb') as f:
   r=subprocess.run(['python3','-B','-I','-S',str(p/name),str(p)],stdout=f,stderr=subprocess.STDOUT,timeout=1800);r.check_returncode()
 state['status']='ANALYZED';write()
except BaseException as e:
 state['status']='ERROR';state['error']=str(e);write();raise
'''
if '--functions-only' in sys.argv:
 runner=runner.replace("['analyze_functions.py','analyze_detailed.py']","['analyze_functions.py']").replace('analysis-state.json','function-body-state.json').replace("name+'.log'","name+'.body.log'")
files={n:(P/n).read_text() for n in ['analyze_functions.py','analyze_detailed.py']};files['run_analysis_remote.py']=runner
source='''from pathlib import Path
import json,sys,subprocess
p=Path(sys.argv[1]);files=json.load(sys.stdin)
for name,s in files.items():(p/name).write_text(s)
f=open(p/'analysis-runner.log','wb');c=subprocess.Popen(['python3','-u',str(p/'run_analysis_remote.py'),str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'analysis_pid':c.pid,'new_capture':False,'source':'already exported 16 traces and JSONL'}))
'''
if '--functions-only' in sys.argv:
 files['run_body_analysis_remote.py']=files.pop('run_analysis_remote.py')
 source=source.replace('run_analysis_remote.py','run_body_analysis_remote.py').replace('analysis-runner.log','function-body-runner.log')
r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',source,root],input=json.dumps(files).encode(),capture_output=True,timeout=90,check=True)
(P/('function-body-launch.json' if '--functions-only' in sys.argv else 'analysis-launch.json')).write_bytes(r.stdout);print(r.stdout.decode())
