"""Submit procfs-only EngineCore sampling; no NPU or request is launched."""
import base64
from pathlib import Path
import shlex
import subprocess
import sys

SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE='wio-vllm-v41-repro169-20261028'
BASE_ID='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'

def main():
    if sys.argv[1:]!=['--execute']:
        raise SystemExit('Review the plan first; pass --execute to sample existing workload for 20 seconds.')
    source=Path(__file__).with_name('sample_engine_core_proc.py').read_bytes()
    compile(source,'sample_engine_core_proc.py','exec')
    payload=base64.b64encode(source).decode('ascii')
    inner='import base64;exec(compile(base64.b64decode('+repr(payload)+'),"sample_engine_core_proc.py","exec"))'
    remote='import subprocess\n'
    remote+='v=subprocess.check_output(["docker","inspect","--format","{{.Id}}",'+repr(BASE)+'],text=True).strip()\n'
    remote+='assert v=='+repr(BASE_ID)+',"baseline container id changed"\n'
    remote+='subprocess.run(["docker","exec",'+repr(BASE)+',"python3","-c",'+repr(inner)+'],check=True)\n'
    result=subprocess.run([SSH,shlex.join(['python3','-c',remote])],text=True,timeout=40)
    if result.returncode: raise SystemExit(result.returncode)

if __name__=='__main__': main()
