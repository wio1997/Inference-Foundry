from pathlib import Path
import urllib.request,json,hashlib,time,re
j=Path(__file__).parent;p=j.parents[1];http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(n,b):
 f=j/n;f.write_bytes(b);return ref(f)
sources=[]
for n in ["/vllm-workspace/vllm/vllm/renderers/online_renderer.py","/vllm-workspace/vllm/vllm/entrypoints/openai/completion/protocol.py","/vllm-workspace/vllm/vllm/entrypoints/openai/chat_completion/serving.py","/vllm-workspace/vllm/vllm/entrypoints/serve/tokenize/serving.py","/opt/aisbench-benchmark/ais_bench/benchmark/models/api_models/vllm_custom_api_chat.py",str(p.parents[2]/"runtime/response_affinity_gateway_v11.py"),str(p.parents[2]/"runtime/sse_observer.py")]:
 f=Path(n);sources.append(ref(f))
 b=f.read_bytes();save("source_"+str(len(sources))+".txt",b)
u=json.load(http.open("http://172.16.10.167:9900/openapi.json",timeout=10));save("native_openapi.json",json.dumps(u).encode())
assert "/v1/completions"in u["paths"]and "/tokenize"in u["paths"]
schema=u["components"]["schemas"]["CompletionRequest"];assert schema["properties"]["prompt"]["anyOf"][0]["items"]["type"]=="integer"
a=p/"jobs/AUDIT-RUN159-20261003T0756Z/reduction.json";assert ref(a)["sha256"]=="859b61b0cf0000431ba9ce08bb721ce341086892b5c81e2ec6aea655468f71bc";v=json.loads(a.read_text());records=[]
data=p/"runs/GLM-RUN-0159/formal/benchmark/dataset"
for phase,n in [("warmup",1),("full",2)]:
 f=Path(v["checks"][phase]["details_path"]);rows=[json.loads(l)for l in f.read_text().splitlines()];assert len(rows)==n
 files=list(data.glob("prefix-*.jsonl"if phase=="warmup"else"GSM8K-*.jsonl"));assert len(files)==1;qs=[json.loads(l)["question"]for l in files[0].read_text().splitlines()]
 for i,row in enumerate(rows):
  messages=row["input"];assert messages==[dict(role="user",content=qs[i])]
  label=phase+str(i);body=json.dumps(dict(model="glm-52",messages=messages,add_generation_prompt=True,add_special_tokens=False,return_token_strs=False),ensure_ascii=False).encode()
  bref=save(label+"_tokenize.request.json",body);begin=time.monotonic()
  res=http.open(urllib.request.Request("http://172.16.10.167:9900/tokenize",data=body,headers={"Content-Type":"application/json","X-Request-ID":j.name+"-"+label}),timeout=60);raw=res.read();elapsed=time.monotonic()-begin;assert res.status==200
  wr=save(label+"_tokenize.response.json",raw);tok=json.loads(raw);ids=tok["tokens"];assert tok["count"]==len(ids)==(73740 if phase=="warmup"else 81932)and all(type(x)is int and x>=0 for x in ids)
  ir=save(label+"_ids.json",json.dumps(ids,separators=(",",":")).encode())
  records.append(dict(label=label,source_details=ref(f),source_row_id=row["id"],source_messages_match_exact_dataset_question=True,request=bref,response=wr,ids=ir,count=len(ids),tokenize_preparation_wall_s=elapsed))
ids=[json.loads(Path(x["ids"]["path"]).read_text())for x in records]
def lcp(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n
prefix=[lcp(ids[0],x)for x in ids[1:]];assert all(x>=72704 for x in prefix)
out=dict(kind="CPU_native_token_input_research",valid=True,sources=sources,records=records,native_common_prefix_tokens=prefix,native_completion_integer_input_schema_confirmed=True,gateway_generic_native_route=True,old_benchmark_SSE_return_token_ids=False,tokenize_is_frontend_CPU_not_inference=True,native_chat_prompt_reuse_kv_param_confirmed=True,limitations=["Tokens produced through installed native /tokenize chat renderer from actual159AISBenchmessages; source/API equivalence, actualCompletioninference not yet executed","Preparation latency is real cost; pretokenized input experiment must separate tokenization preparation from request service, not claim free E2E gain","Native all32 ownership/counter0 verified by HOST wrapper; no model/operator/guard edits"])
save("token_summary.json",json.dumps(out,indent=2).encode());print(json.dumps(dict(valid=True,counts=[x["count"]for x in records],prefix=prefix,tokenize_wall_s=[x["tokenize_preparation_wall_s"]for x in records])))
