import importlib.util,json,sys,types,pickle,inspect
from pathlib import Path

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,'/audit/'+file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
out=load('vllm.v1.core.sched.output','output.py')
scheduler=load('_glm53_candidate_scheduler','scheduler.py')
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request,RequestStatus
from vllm.distributed.device_communicators import shm_broadcast
from vllm.v1.worker.gpu_input_batch import CachedRequestState
assert "pickle.dumps(" in Path(shm_broadcast.__file__).read_text()
params={'_glm53_adopt_first':True,'last_token_id':20,'remote_num_prompt_tokens':256,'remote_block_ids':[[1,2]],'remote_request_id':'P-cpu-test','do_remote_decode':False}
sp=SamplingParams(max_tokens=64,min_tokens=1,temperature=0,ignore_eos=True,extra_args={'kv_transfer_params':params})
r=Request('cpu-native-test',[17]*256,sp,None);r.status=RequestStatus.WAITING_FOR_REMOTE_KVS;r.num_computed_tokens=256
owner=types.SimpleNamespace(_glm53_adopt_enabled=True,finished_recving_kv_req_ids={r.request_id},failed_recving_kv_req_ids=set(),enable_return_routed_experts=False,return_sampling_mask=False)
checks={}
assert scheduler.Scheduler._glm53_can_adopt_first(owner,r);checks['full_native_Request_constructor_and_positive_eligibility']=True
for key,value in [('_glm53_adopt_first',False)]:
 old=params[key];params[key]=value;assert not scheduler.Scheduler._glm53_can_adopt_first(owner,r);params[key]=old
checks['per_request_default_native']=True
r.max_tokens=1;assert not scheduler.Scheduler._glm53_can_adopt_first(owner,r);r.max_tokens=64;checks['max1_native_fallback']=True
r.append_output_token_ids(20);r._glm53_adopted=True
assert r.num_tokens==257 and r.num_output_tokens==1 and r.num_computed_tokens==256
first=out.NewRequestData.from_request(r,([1,2],));assert first.initial_output_token_ids==[20]
buffers=[]
encoded=pickle.dumps(first,protocol=pickle.HIGHEST_PROTOCOL,buffer_callback=buffers.append);decoded=pickle.loads(encoded,buffers=buffers)
assert decoded.initial_output_token_ids==[20] and decoded.num_computed_tokens==256
checks['actual_native_RPC_pickle_protocol_history_roundtrip']=True
cached=CachedRequestState(req_id=r.request_id,prompt_token_ids=decoded.prompt_token_ids,prompt_embeds=None,prompt_is_token_ids=None,mm_features=[],sampling_params=sp,pooling_params=None,generator=None,block_ids=decoded.block_ids,num_computed_tokens=decoded.num_computed_tokens,output_token_ids=list(decoded.initial_output_token_ids or ()),lora_request=None)
assert cached.num_tokens==257 and cached.output_token_ids==[20]
checks['full_native_CachedRequestState_initial_history']=True
first.initial_output_token_ids.append(19);assert list(r.output_token_ids)==[20];checks['DTO_history_independent_copy']=True
print(json.dumps({'checks':checks,'full_native_Request_and_SamplingParams_import':True,'full_scheduler_initialized':False,'real_KV_manager_or_InputBatch_run':False,'device_run':False},indent=2))
