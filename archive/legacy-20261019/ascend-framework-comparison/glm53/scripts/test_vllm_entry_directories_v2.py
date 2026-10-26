"""Real filesystem/UNIX-socket regression for the observed missing IPC directory."""
import hashlib,importlib.util,json,socket,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/vllm_correctness_entry_v2.py';s=importlib.util.spec_from_file_location('entry',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main():
 with tempfile.TemporaryDirectory(prefix='g53-',dir='/tmp') as temp:
  root=Path(temp);env={'VLLM_RPC_BASE_PATH':str(root/'rpc'),'VLLM_CACHE_ROOT':str(root/'cache')};probe=root/'rpc'/'probe.sock'
  client=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
  try:
   try:client.bind(str(probe))
   except FileNotFoundError:pass
   else:raise AssertionError('Fixture did not reproduce missing IPC parent')
   prepared=m.prepare_runtime_directories(root,env);client.bind(str(probe));assert probe.exists()
   assert m.prepare_runtime_directories(root,env)==prepared
  finally:client.close();probe.unlink(missing_ok=True)
  try:m.prepare_runtime_directories(root,dict(env,VLLM_RPC_BASE_PATH='/tmp/outside-glm53'))
  except AssertionError:pass
  else:raise AssertionError('Foreign IPC path accepted')
 with tempfile.TemporaryDirectory(prefix='g53-',dir='/tmp') as temp,tempfile.TemporaryDirectory(prefix='g53-',dir='/tmp') as foreign:
  root=Path(temp);(root/'rpc').symlink_to(foreign,target_is_directory=True)
  try:m.prepare_runtime_directories(root,{})
  except AssertionError:pass
  else:raise AssertionError('Symlink escape accepted')
 out={'project_id':'glm53','pass':True,'entry_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'actual_missing_parent_reproduced':True,'actual_UNIX_bind_after_prepare':True,'idempotent':True,'foreign_path_and_symlink_rejected':True,'SDK_init_calls':0,'model_requests':0,'scope':'Local filesystem/UNIX socket only; no SSH/Docker/native model qualification'}
 (ROOT/'results/vllm-entry-directories-local-fixture-v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
