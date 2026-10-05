"""Explicit task-private dispatch policy; native target FULL and draft NONE."""
import functools,hashlib,inspect,json,os
from pathlib import Path
def install():
 from vllm.config.compilation import CUDAGraphMode
 from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator as C
 original=C.init_cudagraph_manager
 assert hashlib.sha256(Path(inspect.getfile(C)).read_bytes()).hexdigest()=="4bde6427b107e689f7af58d784d985ee73e8977f14727ffb29c9ebfc2fc57edc"
 assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()=="f4cc23a2a0dc3e90406890e3e92deed7ade3e89b1c1571dcc0216b49fe542f0a"
 @functools.wraps(original)
 def honor_eager(self,cudagraph_mode):
  assert self.speculative_config.enforce_eager is True
  result=original(self,CUDAGraphMode.NONE)
  assert not self.prefill_cudagraph_manager.needs_capture() and not self.decode_cudagraph_manager.needs_capture()
  print("GLM_DRAFT_EAGER_DISPATCH "+json.dumps(dict(pid=os.getpid(),requested_mode=str(cudagraph_mode),draft_mode="NONE",native_spec_enforce_eager=True,prefill_needs_capture=False,decode_needs_capture=False,native_math_changes=0)),flush=True)
  return result
 C.init_cudagraph_manager=honor_eager
 print("GLM_DRAFT_EAGER_DISPATCH_INSTALLED "+json.dumps(dict(pid=os.getpid(),native_math_changes=0)),flush=True)
