from pathlib import Path
import subprocess,json,time
p=Path('/data/tiankuan/wio/910c-deepinfer-172');status=p/'runs/prefill-cached-cp-v16-benchmark-status.json';assert not status.exists()
source=p/'tools/aisbench-7bcb6440da11';py=p/'tools/aisbench-venv-7bcb6440da11/bin/python'
env='PYTHONPATH='+':'.join(map(str,[p/'tools/aisbench-adapter',source,source/'plugins/prefix_cache']))
command=['docker','exec','-e',env,'-e','PYTHONDONTWRITEBYTECODE=1','-e','NO_PROXY=*','-e','no_proxy=*','-e','HTTP_PROXY=','-e','HTTPS_PROXY=','-e','ALL_PROXY=','-e','MPLCONFIGDIR='+str(p/'cache/matplotlib'),'wio-deepinfer-s172','timeout','--kill-after=20','7200',str(py),str(p/'tools/native_prefill_cached_cp_v16_benchmark.py')]
status.write_text(json.dumps({'state':'RUNNING','started':time.time(),'command':command},indent=2))
with (p/'logs/prefill-cached-cp-v16-benchmark.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
status.write_text(json.dumps({'state':'PASS' if r.returncode==0 else 'FAIL','exit_code':r.returncode,'command':command},indent=2))
