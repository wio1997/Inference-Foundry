
from pathlib import Path
import json,subprocess,time
p=Path('/data/tiankuan/wio/910c-deepinfer-172');root=p/'benchmarks/allopt-mc2-cap4096-v1-20261028';status=p/'runs/allopt-mc2-cap4096-v1-20261028-benchmark-status.json'
assert not status.exists()
source=p/'tools/aisbench-7bcb6440da11';py=p/'tools/aisbench-venv-7bcb6440da11/bin/python'
env='PYTHONPATH='+':'.join(map(str,[p/'tools/aisbench-adapter',source,source/'plugins/prefix_cache']))
command=['docker','exec','-e',env,'-e','PYTHONDONTWRITEBYTECODE=1','-e','NO_PROXY=*','-e','no_proxy=*','-e','HTTP_PROXY=','-e','HTTPS_PROXY=','-e','ALL_PROXY=','-e','MPLCONFIGDIR='+str(p/'cache/matplotlib'),'wio-deepinfer-s172','timeout','--kill-after=20','3600',str(py),str(p/'tools/allopt-mc2-cap4096-v1-20261028-benchmark.py')]
status.write_text(json.dumps({'state':'RUNNING','started':time.time(),'command':command},indent=2))
with (root/'driver.log').open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
status.write_text(json.dumps({'state':'PASS' if r.returncode==0 else 'FAIL','exit_code':r.returncode,'command':command},indent=2))

(root/'npu-smi-after-benchmark.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
