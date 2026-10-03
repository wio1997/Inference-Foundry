from pathlib import Path
import asyncio,sys,json,hashlib,time,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from native_chat_token_memo import NativeChatTokenMemoTransport,plain_tokenize_payload
from contract_tests import run as mock_tests
import httpx
j=Path(__file__).parent;p=j.parents[1];http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
async def main():
 mock=await mock_tests();(j/"mock_contract.json").write_text(json.dumps(mock,indent=2)+"\n")
 config=json.loads((p/"runs/GLM-RUN-0175/restored/service_config.json").read_text())
 replicas=json.loads(config["environment"]["GLM_REPLICAS"]);groups=json.loads(config["environment"]["GLM_EXECUTION_GROUPS"])
 epochs={x["url"]:next(g["epoch"]for g in groups if x["id"]in g["members"])for x in replicas}
 renderer=Path("/vllm-workspace/vllm/vllm/renderers/online_renderer.py")
 assert ref(renderer)["sha256"]=="5a8500fc56ca5025005ec3931f7949859ff800ad7e18c8766f7d2d36de8b207b"
 data=p/"runs/GLM-RUN-0152/formal/benchmark/dataset"
 fullfiles=list(data.glob("GSM8K-*.jsonl"));warmfiles=list(data.glob("prefix-*.jsonl"))
 assert len(fullfiles)==len(warmfiles)==1
 full=[json.loads(l)["question"]for l in fullfiles[0].read_text().splitlines()]
 warm=[json.loads(l)["question"]for l in warmfiles[0].read_text().splitlines()]
 assert len(full)==4 and len(warm)==2
 memo=NativeChatTokenMemoTransport(httpx.AsyncHTTPTransport(retries=0),epochs,audit_dir=j/"native_tokenize",
       trace_path=str(j/"memo_trace.jsonl"),stats_path=str(j/"memo_stats.json"))
 records=[]
 for origin,epoch in epochs.items():
  schema=json.load(http.open(origin+"/openapi.json",timeout=15))
  props=schema["components"]["schemas"]["TokenizeChatRequest"]["properties"]
  assert all(k in props for k in ["model","messages","add_generation_prompt","add_special_tokens","return_token_strs","chat_template_kwargs"])
  for phase,questions in [("warmup",warm),("full",full)]:
   for i,text in enumerate(questions):
    body=json.dumps(dict(model="glm-52",messages=[dict(role="user",content=text)]),ensure_ascii=False).encode()
    payload=plain_tokenize_payload(body);assert payload is not None
    canonical=json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
    key=hashlib.sha256(origin.encode()+b"\0"+epoch.encode()+b"\0"+canonical).hexdigest()
    start=time.monotonic();entry,result=await memo.lookup(key,origin,epoch,payload,j.name+"-"+phase+str(i))
    elapsed=time.monotonic()-start;assert entry is not None and result in ["miss","hit"]
    ids_json,count,source=entry;assert count==(73740 if phase=="warmup"else 81932)
    filename=("D0"if "166" in origin else"D1")+"_"+phase+str(i)+"_ids.json"
    (j/filename).write_bytes(ids_json);idsref=ref(j/filename)
    if phase=="full"and i<2 or phase=="warmup":
     old=p/"jobs/TOKEN-INPUT-CPU3-20261003T0837Z"/(phase+str(i if phase=="full"else 0)+"_ids.json")
     assert ids_json==old.read_bytes()
    hit,hitresult=await memo.lookup(key,origin,epoch,payload,"hit")
    assert hitresult=="hit"and hit==entry
    records.append(dict(origin=origin,epoch=epoch,phase=phase,row=i,count=count,ids=idsref,lookup=result,
                        elapsed_s=elapsed,tokenize_source=source))
 beforeclose=memo.snapshot();await memo.aclose();assert memo.snapshot()["pending"]==0
 out=dict(kind="CPU_native_token_memo_contract",valid=True,mock=mock,records=records,
     sources=[ref(renderer),ref(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/native_chat_token_memo.py"))],
     dataset_sources=[ref(fullfiles[0]),ref(warmfiles[0])],snapshot=beforeclose,final_snapshot=memo.snapshot(),
     limitations=["Mock contract checks are simulation only, not native inference or performance",
     "Real selected-native /tokenize inside CPU lookup, not a full HTTP Chat E2E",
     "Existing token arrays used only for equality checking after actual calls, never preloaded",
     "No model/operator/device tensors; actual public gateway not changed; performance KEEP remains unknown"])
 (j/"token_summary.json").write_text(json.dumps(out,indent=2)+"\n")
 print(json.dumps(dict(valid=True,mock_checks=len(mock["checks"]),real_CPU_calls=memo.stats["tokenizer_CPU_calls"],
                      counts=[v["count"]for v in records],snapshot=memo.snapshot())))
asyncio.run(main())
