from vllm_ascend.worker.worker import NPUWorker
from atomic_mq_bind import install
class AtomicMQWorker(NPUWorker):
 def __init__(self,*args,**kwargs):
  install()
  super().__init__(*args,**kwargs)
