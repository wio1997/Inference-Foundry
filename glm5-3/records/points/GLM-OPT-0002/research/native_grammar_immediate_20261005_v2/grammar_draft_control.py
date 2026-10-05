"""Private native CPU draft validation for immediate grammar sampling.
No native kernels, scheduler allocation, sampling or wire implementation changes.
Activation requires native source guards and an existing queue-cap wrapper.
"""
import functools,hashlib,inspect,json,os
from pathlib import Path

CORE_SHA="6fdd067f54e5d42c57ff413e292685f5bdf6f343498d85c9262567f1eb746916"
STEP_SHA="84d0cd0ecaa08eb2f0e1b25d3d28e66415155370c4808d23f89faff9c99116c1"
SCHEDULER_SHA="c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a"

def validate_immediate_drafts(engine,output):
    if output.pending_structured_output_tokens or not engine.check_for_draft_tokens:
        return None
    wanted={}
    for req_id,tokens in output.scheduled_spec_decode_tokens.items():
        request=engine.scheduler.requests.get(req_id)
        if request is not None and request.use_structured_output and not request.is_prefill_chunk and -1 in tokens:
            wanted[req_id]=list(tokens)
    if not wanted:
        return None
    drafts=engine.model_executor.take_draft_token_ids()
    if drafts is None:
        raise RuntimeError("immediate grammar requires native draft IDs; none returned")
    if len(drafts.req_ids)!=len(drafts.draft_token_ids) or len(set(drafts.req_ids))!=len(drafts.req_ids):
        raise RuntimeError("native draft request identity mapping is ambiguous")
    by_request=dict(zip(drafts.req_ids,drafts.draft_token_ids))
    for req_id,tokens in wanted.items():
        actual=by_request.get(req_id)
        if actual is None or len(actual)<len(tokens) or any(type(x)is not int or x<0 for x in actual[:len(tokens)]):
            raise RuntimeError("native draft coverage missing for immediate structured request "+req_id)
    # Delegate validation and invalid-draft padding to the unchanged native method.
    engine.scheduler.update_draft_token_ids_in_output(drafts,output)
    return dict(request_ids=sorted(wanted),before=wanted,
                after={k:list(output.scheduled_spec_decode_tokens[k])for k in wanted},
                draft_request_ids=list(drafts.req_ids),native_filter_called=True,
                pending_structured_output_tokens=False,math_changes=0,
                draft_generation_identity_not_independently_proven=True)

def install():
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.core.sched.scheduler import Scheduler
    original=EngineCore.step_with_batch_queue
    if getattr(original,"_glm_immediate_grammar_drafts_installed",False):
        return
    assert getattr(original,"_glm_queue_cap_installed",False)
    assert hashlib.sha256(Path(inspect.getfile(EngineCore)).read_bytes()).hexdigest()==CORE_SHA
    assert hashlib.sha256(inspect.getsource(inspect.unwrap(original)).encode()).hexdigest()==STEP_SHA
    assert hashlib.sha256(Path(inspect.getfile(Scheduler)).read_bytes()).hexdigest()==SCHEDULER_SHA
    @functools.wraps(original)
    def step(self):
        if not getattr(self,"_glm_immediate_grammar_hook",False):
            config=self.scheduler.vllm_config;p=config.parallel_config
            assert config.use_v2_model_runner and config.scheduler_config.async_scheduling
            assert(p.data_parallel_size,p.tensor_parallel_size,p.pipeline_parallel_size,p.decode_context_parallel_size)==(1,8,2,8)
            assert not p.enable_expert_parallel and config.kv_transfer_config is None
            assert config.max_concurrent_batches==3
            native_get=self.scheduler.get_grammar_bitmask
            assert native_get.__func__ is Scheduler.get_grammar_bitmask
            self._glm_immediate_grammar_records=0
            def get(output):
                evidence=validate_immediate_drafts(self,output)
                if evidence is not None and self._glm_immediate_grammar_records<8:
                    self._glm_immediate_grammar_records+=1
                    print("GLM_IMMEDIATE_GRAMMAR_DRAFT_VALIDATED "+json.dumps(dict(
                        pid=os.getpid(),cohort=os.environ.get("GLM_ISSUE_BUDGET_COHORT"),**evidence)),flush=True)
                return native_get(output)
            self.scheduler.get_grammar_bitmask=get
            self._glm_immediate_grammar_hook=True
        return original(self)
    step._glm_immediate_grammar_drafts_installed=True
    EngineCore.step_with_batch_queue=step
    print("GLM_IMMEDIATE_GRAMMAR_DRAFT_CONTROL_INSTALLED "+json.dumps(dict(
        pid=os.getpid(),core_sha256=CORE_SHA,step_sha256=STEP_SHA,scheduler_sha256=SCHEDULER_SHA,
        native_filter_reused=True,kernels_changed=0,queue_FIFOs_untouched=True)),flush=True)
