#!/usr/bin/env python3
"""Read pinned installed synchronization API source without attaching an NPU."""
import ast
import hashlib
import json
import socket
import subprocess


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    code = '''import ast,hashlib,importlib.util,json,pathlib
s=importlib.util.find_spec('torch_npu')
root=pathlib.Path(s.origin).parent
result=[]
for p in sorted((root/'npu').glob('*.py')):
 raw=p.read_bytes();text=raw.decode();tree=ast.parse(text)
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('synchronize','set_device','current_device'):
   result.append({'path':str(p),'file_sha256':hashlib.sha256(raw).hexdigest(),'function':node.name,'source':ast.get_source_segment(text,node)})
print(json.dumps(result))
'''
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    raw = subprocess.check_output(['docker','run','--rm','--pull','never','--network','none','--read-only','--cap-drop','ALL','--entrypoint','python3',image,'-c',code], timeout=40)
    functions = json.loads(raw)
    assert {r['function'] for r in functions} == {'synchronize','set_device','current_device'}
    print(json.dumps({'image':image,'functions':functions,'scope':'Installed pinned Python source only; no torch_npu import, NPU mounts or device initialization. Backend.set_device calls torch.npu.set_device(device_id) before sender queue loop. Runtime log still must confirm each sender current device.'},indent=2))


if __name__ == '__main__':
    main()
