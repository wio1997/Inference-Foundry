import importlib.util,json,sys,types,pickle,inspect,ast
from pathlib import Path

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,'/audit/'+file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
out=load('vllm.v1.core.sched.output','output.py')
scheduler=load('_glm53_candidate_scheduler','scheduler.py')
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request,RequestStatus
from vllm.distributed.device_communicators import shm_broadcast
from vllm.v1.worker.gpu_input_batch import CachedRequestState
import hashlib
assert hashlib.sha256(Path(shm_broadcast.__file__).read_bytes()).hexdigest()=="f3e7a3f3ff713fccbd535e0d41650ee05bdf8677e83d4b90519910645022c30b"
params={'_glm53_export_p_draft':True,'_glm53_adopt_first':True,'last_token_id':20,'remote_num_prompt_tokens':256,'remote_block_ids':[[1,2]],'remote_request_id':'P-cpu-test','do_remote_decode':False}
sp=SamplingParams(max_tokens=64,min_tokens=1,temperature=0,ignore_eos=True,extra_args={'kv_transfer_params':params})
r=Request('cpu-native-test',[17]*256,sp,None);r.status=RequestStatus.WAITING_FOR_REMOTE_KVS;r.num_computed_tokens=256
owner=types.SimpleNamespace(_glm53_adopt_enabled=True,finished_recving_kv_req_ids={r.request_id},failed_recving_kv_req_ids=set(),enable_return_routed_experts=False,return_sampling_mask=False)
checks={}
init_ast=next(n for n in ast.parse(Path('/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
init_fn=next(n for n in init_ast.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
attrs={n.attr for n in ast.walk(init_fn) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and isinstance(n.ctx,ast.Store)}
assert 'vllm_config' in attrs and 'model_config' not in attrs
from vllm.config import ModelConfig
assert callable(ModelConfig.get_vocab_size)
checks['native_scheduler_configuration_attribute_contract']=True
owner.vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(get_vocab_size=lambda:1000))
owner._glm53_valid_p_draft=lambda request:scheduler.Scheduler._glm53_valid_p_draft(owner,request)
assert scheduler.Scheduler._glm53_can_adopt_first(owner,r);checks['full_native_Request_constructor_and_positive_eligibility']=True
for key,value in [('_glm53_adopt_first',False)]:
 old=params[key];params[key]=value;assert not scheduler.Scheduler._glm53_can_adopt_first(owner,r);params[key]=old
checks['per_request_default_native']=True
r.max_tokens=1;assert not scheduler.Scheduler._glm53_can_adopt_first(owner,r);r.max_tokens=64;checks['max1_native_fallback']=True
params['_glm53_use_p_draft']=True
assert not scheduler.Scheduler._glm53_can_adopt_first(owner,r)
params['p_first_draft']={'request_id':'P-cpu-test','num_prompt_tokens':256,'first_token_id':20,'rank_agreement':16,'token_ids':[21]}
assert scheduler.Scheduler._glm53_can_adopt_first(owner,r)
params['p_first_draft']['rank_agreement']=15;assert not scheduler.Scheduler._glm53_can_adopt_first(owner,r)
params['p_first_draft']['rank_agreement']=16
checks['seed_metadata_guard']=True
r.append_output_token_ids(20);r._glm53_adopted=True
assert r.num_tokens==257 and r.num_output_tokens==1 and r.num_computed_tokens==256
first=out.NewRequestData.from_request(r,([1,2],));assert first.initial_output_token_ids==[20]
buffers=[]
writer=types.SimpleNamespace(_is_writer=True,n_local_reader=0,n_remote_reader=1,remote_socket=types.SimpleNamespace(send_multipart=lambda parts,copy:buffers.extend(parts)))
shm_broadcast.MessageQueue.enqueue(writer,first)
reader=types.SimpleNamespace(poll=lambda timeout:True,recv_multipart=lambda copy:buffers)
decoded=shm_broadcast.MessageQueue.recv(reader,None)
assert decoded.initial_output_token_ids==[20] and decoded.num_computed_tokens==256 and decoded.export_p_first_draft is True
checks['actual_native_MessageQueue_enqueue_recv_with_socket_recorder']=True
cached=CachedRequestState(req_id=r.request_id,prompt_token_ids=decoded.prompt_token_ids,prompt_embeds=None,prompt_is_token_ids=None,mm_features=[],sampling_params=sp,pooling_params=None,generator=None,block_ids=decoded.block_ids,num_computed_tokens=decoded.num_computed_tokens,output_token_ids=list(decoded.initial_output_token_ids or ()),lora_request=None)
assert cached.num_tokens==257 and cached.output_token_ids==[20]
checks['full_native_CachedRequestState_initial_history']=True
first.initial_output_token_ids.append(19);assert list(r.output_token_ids)==[20];checks['DTO_history_independent_copy']=True

# Invoke the actual installed Ascend compatibility function, extracted intact
# without running unrelated PP scheduler monkeypatches on this CPU harness.
import ast,os
compat=Path('/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_pp_mtp.py').read_text()
assert hashlib.sha256(compat.encode()).hexdigest()=='367056c95c24f58f0fdcf8f126900aa1a6558dc3aa53433a5721f30d60098359'
f=next(n for n in ast.parse(compat).body if isinstance(n,ast.FunctionDef) and n.name=='_patch_model_runner_output')
from functools import wraps
exec(compile(ast.Module(body=[f],type_ignores=[]),'<native-Ascend-compat-function>','exec'));_patch_model_runner_output()
from vllm.v1.outputs import ModelRunnerOutput,KVConnectorOutput
utils=load('_glm53_seed_utils','utils.py');os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='1'
def items():return [ModelRunnerOutput(req_ids=['P-test'],req_id_to_index={'P-test':0},sampled_token_ids=[[20]],spec_token_ids=[[21]],kv_connector_output=KVConnectorOutput(finished_sending={'P-test'},failed_recving=set())) for i in range(16)]
rows=items();result=utils.KVOutputAggregator(16).aggregate(rows);assert result.spec_token_ids==[[21]] and result.kv_connector_output.finished_sending=={'P-test'}
checks['native_16_rank_aggregator_agreement']=True
rows=items();rows[-1].spec_token_ids=[[22]];assert utils.KVOutputAggregator(16).aggregate(rows).spec_token_ids is None
rows=items()[:15];assert utils.KVOutputAggregator(16).aggregate(rows).spec_token_ids is None
checks['native_aggregator_mismatch_and_missing_rank_reject']=True
print(json.dumps({'checks':checks,'full_native_Request_and_SamplingParams_import':True,'full_scheduler_initialized':False,'real_KV_manager_or_InputBatch_run':False,'actual_IPC_transport_run':False,'device_run':False},indent=2))
