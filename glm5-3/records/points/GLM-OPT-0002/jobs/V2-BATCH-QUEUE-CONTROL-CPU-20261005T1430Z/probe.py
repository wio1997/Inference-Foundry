from pathlib import Path
import json,hashlib,types
from collections import deque
from batch_queue_control import checked_policy,apply_empty_queue_cap
j=Path(__file__).parent
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
cfg=types.SimpleNamespace(use_v2_model_runner=True,scheduler_config=types.SimpleNamespace(async_scheduling=True),parallel_config=types.SimpleNamespace(data_parallel_size=1,tensor_parallel_size=8,pipeline_parallel_size=2,decode_context_parallel_size=8,enable_expert_parallel=False),kv_transfer_config=None,max_concurrent_batches=3)
engine=types.SimpleNamespace(scheduler=types.SimpleNamespace(vllm_config=cfg),batch_queue_size=3,batch_queue=deque(maxlen=3))
resource_config_before=repr(vars(cfg));a,b=object(),object();engine.batch_queue.extend([b,a]);q=engine.batch_queue
assert not apply_empty_queue_cap(engine,2)and engine.batch_queue is q and engine.batch_queue_size==3
assert engine.batch_queue.pop()is a and engine.batch_queue.pop()is b
assert apply_empty_queue_cap(engine,2)and engine.batch_queue.maxlen==engine.batch_queue_size==2 and not engine.batch_queue
engine.batch_queue.append(a);q=engine.batch_queue;assert not apply_empty_queue_cap(engine,3)and engine.batch_queue is q and engine.batch_queue.pop()is a
assert apply_empty_queue_cap(engine,3)and engine.batch_queue.maxlen==3 and repr(vars(cfg))==resource_config_before
cfg.use_v2_model_runner=False
try:apply_empty_queue_cap(engine,2)
except AssertionError:pass
else:raise RuntimeError("V1 guard was lost")
cfg.use_v2_model_runner=True
f=j/"CPU_policy.json";base=dict(schema_version=1,cohort_id="GLM-COHORT-0228",cap=3,serial=1,diagnostic=True,max_records=64);f.write_text(json.dumps(base));assert checked_policy(f)==base
for key,val in[("cap",1),("cap",4),("cap",True),("serial",0),("diagnostic",1),("max_records",257),("cohort_id","other")]:
 v=dict(base);v[key]=val;f.write_text(json.dumps(v))
 try:checked_policy(f)
 except AssertionError:pass
 else:raise RuntimeError("invalid policy accepted")
f.unlink()
out=dict(CPU_contract_VALID=True,checks=["nonempty transition deferred without queue mutation","empty3to2to3 reallocates deque only","pending FIFO Future identity preserved","native resource config remains3","V1 rejected","invalid cap/cohort/serial/diagnostic/boundedrecords rejected"],CPU_fixture=True,native_step_not_executed=True,native_Graph_or_NPU_or_model_calls=0,SDK_or_communicators_created=0,math_operator_changes=0,candidate=ref(j/"batch_queue_control.py"),limits=["Synthetic admission controller invariants only; no native GPU in-flight/PP/reuse/speed/fullAPI proof"])
f=j/"reduction.json";f.write_text(json.dumps(out,indent=2)+"\n")
v=dict(schema_version=1,job_id=j.name,status="completed",summary="CPU queue-control invariants passed; onlyempty cap3to2to3 native provision remains3/FIFO/noV1/noNPU; actualGPU pending",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="CPU synthetic queue transition invariants")],unknowns=out["limits"],decision_request=None,next_check_at=None);(j/"result.json").write_text(json.dumps(v,indent=2)+"\n");print(json.dumps(dict(CPU_VALID=True,**ref(f))))
