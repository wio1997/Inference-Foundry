from pathlib import Path
import sys,json,hashlib
j=Path(__file__).parent;p=j.parents[1];runtime=p.parents[2]/"runtime";sys.path.insert(0,str(runtime))
import pd_semantic_probe as task
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys,hash_block_tokens
ids=[11,22,33,44,55,66,77,88];base=dict(model="glm-52",messages=[dict(role="user",content="Native cache-salt CPU control")],temperature=0,max_tokens=64,return_token_ids=True,include_reasoning=True)
api=[ChatCompletionRequest(**base,cache_salt=s)for s in["CPU-A","CPU-A","CPU-B",None]]
try:ChatCompletionRequest(**base,cache_salt="")
except VLLMValidationError:empty_rejected=True
else:raise AssertionError("native API accepted empty salt")
assert all(x.messages==api[0].messages and x.return_token_ids and x.include_reasoning for x in api)
def chain(s):
 r=Request(request_id="fixture",prompt_token_ids=ids.copy(),sampling_params=SamplingParams(temperature=0,max_tokens=64),pooling_params=None,cache_salt=s)
 parent=b"0"*32;hashes=[];extras=[]
 for start in[0,4]:
  extra,mm=generate_block_hash_extra_keys(r,start,start+4,0);assert mm==0
  parent=hash_block_tokens(lambda x:hashlib.sha256(repr(x).encode()).digest(),parent,ids[start:start+4],extra);hashes.append(bytes(parent).hex());extras.append(extra)
 assert r.prompt_token_ids==ids and r.num_prompt_tokens==len(ids)
 return dict(hashes=hashes,extra_keys=extras)
chains=[chain(x.cache_salt)for x in api]
assert chains[0]["hashes"]==chains[1]["hashes"]and all(a!=b for a,b in zip(chains[0]["hashes"],chains[2]["hashes"]))and chains[0]["hashes"]!=chains[3]["hashes"]
assert chains[0]["extra_keys"]==[("CPU-A",),None]and chains[2]["extra_keys"]==[("CPU-B",),None]
f=p/"runs/GLM-RUN-0054/P_166.resident.metrics";raw=f.read_text()
names=["num_requests_running","num_requests_waiting"]+list(task.COUNTERS);metrics={n:task.counter(raw,n)for n in names};assert all(metrics[n]is not None for n in names)
out=dict(native_request_class=True,native_hash_helpers=True,prompt_ids_preserved=True,native_API_salt_fields_preserved=True,empty_salt_rejected=empty_rejected,two_block_chains=chains,metrics_parser=dict(source=task.ref(f),actual=metrics),probe_source=task.ref(runtime/"pd_semantic_probe.py"),source_refs=[task.ref(Path(x))for x in["/vllm-workspace/vllm/vllm/entrypoints/openai/chat_completion/protocol.py","/vllm-workspace/vllm/vllm/v1/request.py","/vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py"]],new_NPU_workers=0,new_models=0,inference=0,signals=0,limits=["NativeCPU API/hash key fixture verifies separate cache namespace, not GPU KV transfer/committed token equality","Toy hash_function serializes fixture object; native hash_block_tokens and extra-key helpers are real; does not certify configured production hash backend","Actual P resident Prometheus snapshot parser checked, no live new requests/cache performance"])
(j/"reduction.json").write_text(json.dumps(out,indent=2)+"\n")
(j/"result.json").write_text(json.dumps(dict(schema_version=1,job_id=j.name,status="completed",summary="NativeCPU request/salt/hash controls: different salts separate both block hashes, same salt positive; prompt IDs preserved, actualnative Prometheus parserchecked;0NPU/inference",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="nativeRequest/API/hash chain positive-negative/actualmetrics/probeSHA",**task.ref(j/"reduction.json"))],unknowns=out["limits"],decision_request=None,next_check_at=None),indent=2)+"\n");print(json.dumps(out))

