"""Recover only incomplete node31 JSON; reuse other complete original snapshots."""
import ast,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1]
def load(file,sha,name):
 path=P.with_name(file);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
SOURCE_SHA='552e1b7b9befa3c4f3586a305db4aa6da15d6bd261ce6f6dc1b07e0756ecf76e'
TRANSPORT_SHA='ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5'
source=load('inspect_vllm_formal_C32_native_v2.py',SOURCE_SHA,'source');codec=load('glm53_native_output_transport_v1.py',TRANSPORT_SHA,'codec')
OLD=ROOT/'runs/20261017T1804-vllm-formal-C32-native-v2';RUN=ROOT/'runs/20261017T1805-vllm-formal-C32-recovered-native-v3'
def main():
 assert json.loads((ROOT/'runs/20261017T1800-vllm-formal-C32-v1/controller-rpc-state.json').read_text())['done_received']
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'reused':{},'node31_rc':None,'scope':'Only read-only evidence transport recovery after completed C32; no subsequent model calls. Original incomplete output preserved.'}
 try:
  for node in [30,43,45]:
   path=OLD/f'node-{node}.json';raw=path.read_bytes();outer=json.loads(raw);assert outer['rc']==0;obj=json.loads(outer['stdout']);assert obj['node']==node
   (RUN/path.name).write_bytes(raw);receipt['reused'][str(node)]={'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()}
  real=source.base.invoke
  def invoke(node,code,timeout):
   assert node==31
   code=code.replace('20261017T1804-vllm-formal-C32-native-v2','20261017T1805-vllm-formal-C32-recovered-native-v3')
   before,found,after=code.rpartition('print(json.dumps(out))');assert found and not after.strip();code=before+codec.ENCODER;ast.parse(code)
   response=real(node,code,timeout);(RUN/'node-31-wire.json').write_text(json.dumps(response,indent=2)+'\n');return codec.decode(response)
  source.base.invoke=invoke;cap=json.loads((ROOT/'results/vllm-shared-pool-capacity-v4.json').read_text());response=source.capture(31,cap);(RUN/'node-31.json').write_text(json.dumps(response,indent=2)+'\n');receipt['node31_rc']=response['rc'];receipt['transport']=response['native_output_transport']
 except BaseException as error:receipt['error']=repr(error);raise
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
