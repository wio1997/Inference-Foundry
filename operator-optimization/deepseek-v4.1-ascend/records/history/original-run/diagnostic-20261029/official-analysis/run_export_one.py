#!/usr/bin/env python3
"""Transport the fixed offline exporter to stage without copying credentials."""
import base64, pathlib, re, subprocess, sys

name = sys.argv[1]
assert re.fullmatch(r"[A-Za-z0-9_]+_ascend_pt", name)
p = pathlib.Path(__file__).resolve().parent
payload = base64.b64encode((p / "export_one.py").read_bytes()).decode("ascii")
code = "import base64;exec(compile(base64.b64decode('" + payload + "'),'<offline_export>','exec'))"
remote = "docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c \"" + code + "\" " + name
with (p / (name + ".stdout")).open("w") as out, (p / (name + ".stderr")).open("w") as err:
    result = subprocess.run(["/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh", remote],
                            stdout=out, stderr=err, timeout=600, check=False)
print(name, result.returncode)
sys.exit(result.returncode)
