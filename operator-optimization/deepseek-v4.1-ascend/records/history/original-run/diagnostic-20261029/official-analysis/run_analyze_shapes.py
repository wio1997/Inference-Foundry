#!/usr/bin/env python3
import base64,pathlib,subprocess,sys
p=pathlib.Path(__file__).resolve().parent
data=base64.b64encode((p/"analyze_shapes_remote.py").read_bytes()).decode("ascii")
code="import base64;exec(compile(base64.b64decode('"+data+"'),'<offline_shapes>','exec'))"
remote='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c "'+code+'"'
with (p/"operator-shapes.json").open("w") as out, (p/"shapes-analysis.stderr").open("w") as err:
    proc=subprocess.run(["/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh",remote],stdout=out,stderr=err,timeout=300)
print("exit",proc.returncode)
sys.exit(proc.returncode)
