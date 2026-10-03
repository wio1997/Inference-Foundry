"""Short semantic proof using exact prompt IDs from audited156 native responses."""
from pathlib import Path
import json,hashlib,time
from phase_runner import atomic_json,utc
async def semantics(client,root,run_id):
 p=Path(__file__).parents[2];s=p/"runs/GLM-RUN-0156/client_final_summary.json";assert hashlib.sha256(s.read_bytes()).hexdigest()=="b3b206687313e8ecdf465ddb21b9e00bcea09074f96efad2f97ac1d501cad0f7"
 source=json.loads(s.read_text());rows=[]
 for label,answer in [("add2","4"),("add17","42"),("literal","GLM_OK_731")]:
  old=next(x for x in source["requests"]if x["name"]==label)
  f=Path(old["wire"]["path"]);raw=f.read_bytes();assert len(raw)==old["wire"]["bytes"]and hashlib.sha256(raw).hexdigest()==old["wire"]["sha256"];original=json.loads(raw);ids=original["prompt_token_ids"];assert len(ids)==old["usage"]["prompt_tokens"]
  f=Path(old["body"]["path"]);b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==old["body"]["sha256"];body=json.loads(b);body.update(cache_salt=run_id+"-semantic-"+label,kv_transfer_params={"prompt_token_ids":ids})
  b=json.dumps(body).encode();bp=root/(label+".body");bp.write_bytes(b);started=time.monotonic();res=await client.post("http://127.0.0.1:8000/v1/chat/completions",content=b,headers={"Content-Type":"application/json","X-Request-ID":run_id+"-"+label});f=root/(label+".wire");f.write_bytes(res.content);res.raise_for_status();v=res.json();c=v["choices"][0];u=v["usage"]
  assert v["prompt_token_ids"]==ids and c["message"]["content"].strip()==answer and c["finish_reason"]=="stop"and len(c["token_ids"])==u["completion_tokens"]and u["completion_tokens"]==old["usage"]["completion_tokens"]and u["total_tokens"]==u["prompt_tokens"]+u["completion_tokens"]
  rows.append(dict(name=label,answer=answer,native_id=v["id"],usage=u,semantic_pass=True,prompt_ids_exact_native156=True,wall_s=time.monotonic()-started,source=old,body=dict(path=str(bp),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),wire=dict(path=str(f),bytes=len(res.content),sha256=hashlib.sha256(res.content).hexdigest())))
 atomic_json(root/"semantic_summary.json",dict(valid=True,requests=rows,effective_output_tokens=sum(x["usage"]["completion_tokens"]for x in rows)));return rows
