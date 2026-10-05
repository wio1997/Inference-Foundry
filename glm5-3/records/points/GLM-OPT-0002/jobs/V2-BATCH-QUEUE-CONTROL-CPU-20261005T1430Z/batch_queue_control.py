"""Task-private EngineCore admission cap; native scheduler/output/KV/math unchanged."""
import functools,hashlib,inspect,json,os,time
from collections import deque
from pathlib import Path
EXPECTED_MODULE="6fdd067f54e5d42c57ff413e292685f5bdf6f343498d85c9262567f1eb746916"
EXPECTED_METHOD="84d0cd0ecaa08eb2f0e1b25d3d28e66415155370c4808d23f89faff9c99116c1"
def checked_policy(path):
 p=json.loads(Path(path).read_text())
 assert set(p)=={"schema_version","cohort_id","cap","serial","diagnostic","max_records"}
 assert p["schema_version"]==1 and p["cohort_id"]=="GLM-COHORT-0228"
 assert type(p["cap"])is int and p["cap"]in[2,3]
 assert type(p["serial"])is int and p["serial"]>=1
 assert type(p["diagnostic"])is bool and type(p["max_records"])is int and 0<=p["max_records"]<=256
 return p
def apply_empty_queue_cap(engine,cap):
 # Resource configuration and all buffer pools stay provisioned for native3.
 config=engine.scheduler.vllm_config;p=config.parallel_config
 assert config.use_v2_model_runner and config.scheduler_config.async_scheduling
 assert(p.data_parallel_size,p.tensor_parallel_size,p.pipeline_parallel_size,p.decode_context_parallel_size)==(1,8,2,8)
 assert not p.enable_expert_parallel and config.kv_transfer_config is None
 assert config.max_concurrent_batches==3 and cap in[2,3]
 assert isinstance(engine.batch_queue,deque)and engine.batch_queue.maxlen==engine.batch_queue_size
 assert engine.batch_queue_size in[2,3]and len(engine.batch_queue)<engine.batch_queue_size
 if cap==engine.batch_queue_size:return True
 if engine.batch_queue:return False
 engine.batch_queue=deque(maxlen=cap);engine.batch_queue_size=cap
 return True
def install():
 from vllm.v1.engine.core import EngineCore as C
 original=C.step_with_batch_queue
 if getattr(original,"_glm_queue_cap_installed",False):return
 assert hashlib.sha256(Path(inspect.getfile(C)).read_bytes()).hexdigest()==EXPECTED_MODULE
 assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==EXPECTED_METHOD
 path=Path(__file__).with_name("batch_queue_policy.json")
 @functools.wraps(original)
 def step(self):
  policy=checked_policy(path);applied=apply_empty_queue_cap(self,policy["cap"])
  if getattr(self,"_glm_queue_serial",None)!=policy["serial"]and applied:
   self._glm_queue_serial=policy["serial"];self._glm_queue_records=0
   print("GLM_BATCH_QUEUE_CAP_SELECTED "+json.dumps(dict(pid=os.getpid(),cohort=policy["cohort_id"],serial=policy["serial"],cap=self.batch_queue_size,native_resource_capacity=self.scheduler.vllm_config.max_concurrent_batches,native_math_changes=0,transition_only_empty=True)),flush=True)
  size=len(self.batch_queue);start=time.perf_counter()
  out=original(self)
  if policy["diagnostic"]and getattr(self,"_glm_queue_records",0)<policy["max_records"]:
   self._glm_queue_records=getattr(self,"_glm_queue_records",0)+1
   print("GLM_BATCH_QUEUE_CPU_STEP "+json.dumps(dict(pid=os.getpid(),serial=policy["serial"],cap=self.batch_queue_size,before=size,after=len(self.batch_queue),CPU_wall_s=time.perf_counter()-start,has_output=out[0]is not None,model_executed=out[1],device_completion_time=False)),flush=True)
  return out
 step._glm_queue_cap_installed=True;C.step_with_batch_queue=step
 print("GLM_BATCH_QUEUE_CAP_INSTALLED "+json.dumps(dict(pid=os.getpid(),module_sha256=EXPECTED_MODULE,method_sha256=EXPECTED_METHOD,native_math_changes=0)),flush=True)
