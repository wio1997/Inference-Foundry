#!/usr/bin/env python3
"""Inspect stock prefix-reset API signatures and implementation before any reset."""
import json
import socket
import subprocess


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    code = '''import ast,hashlib,importlib.util,json,pathlib
root=pathlib.Path(importlib.util.find_spec('vllm').origin).parent
rows=[]
for p in root.rglob('*.py'):
 raw=p.read_bytes()
 if b'def reset_prefix_cache' not in raw:continue
 text=raw.decode();tree=ast.parse(text)
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('reset_prefix_cache','reset_prefix_cache_async'):
   rows.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'source':ast.get_source_segment(text,n)})
print(json.dumps(rows))
'''
    raw = subprocess.check_output(['docker','run','--rm','--pull','never','--network','none','--read-only','--cap-drop','ALL','--entrypoint','python3','sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83','-c',code],timeout=45)
    print(json.dumps({'functions':json.loads(raw),'scope':'Pinned installed source only, no model/cache changes or NPU mounts.'},indent=2))


if __name__ == '__main__':
    main()
