#!/usr/bin/env python3
"""Run bounded analysis on copied exported CSVs through the established SSH."""
import base64, pathlib, subprocess, sys

p=pathlib.Path(__file__).resolve().parent
payload=base64.b64encode((p/"analyze_remote.py").read_bytes()).decode("ascii")
code="import base64;exec(compile(base64.b64decode('"+payload+"'),'<offline_analysis>','exec'))"
remote='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c "'+code+'"'
with (p/"operators.json").open("w") as out, (p/"operators-analysis.stderr").open("w") as err:
    proc=subprocess.run(["/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh",remote],
                        stdout=out,stderr=err,timeout=300,check=False)
print("exit",proc.returncode)
sys.exit(proc.returncode)
