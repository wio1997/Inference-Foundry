from vllm_ascend.worker.worker import NPUWorker
from atomic_mq_bind import install
from pp_empty_token_guard import install as install_pp_guard
from sfa_dcp_diagnostic import install as install_diagnostic
from draft_eager_dispatch import install as install_draft_eager
class AtomicMQWorker(NPUWorker):
 def __init__(self,*args,**kwargs):
  install()
  install_pp_guard()
  install_diagnostic()
  super().__init__(*args,**kwargs)
  install_draft_eager()
