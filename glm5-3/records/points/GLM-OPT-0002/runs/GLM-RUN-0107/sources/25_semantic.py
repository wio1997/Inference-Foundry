from pathlib import Path
import json,sys,time,hashlib,urllib.request,urllib.error,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;owners=json.loads((r/"adopted_model_identities.json").read_text());rows=[];opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
for key,o in owners.items():
 if key!="D1":continue
 url="http://172.16.10."+o["host"]+":"+str(o["port"])
 for name,prompt,answer in[("add2","Compute 2 + 2. Reply with only the number.","4"),("add17","Compute 17 + 25. Reply with only the number.","42"),("literal","Reply with exactly this text and nothing else: GLM_OK_731","GLM_OK_731")]:
  guard();label=key+"_"+name;body=dict(model="glm-52",messages=[dict(role="user",content=prompt)],temperature=0,seed=20260930,max_tokens=32,ignore_eos=False,stream=False,return_token_ids=True,cache_salt=r.name+"-"+label,chat_template_kwargs=dict(enable_thinking=False))
  f=r/(label+".body.json");atomic_json(f,body);row=dict(name=label,owner=key,expected=answer,body=ref(f),at=utc(),status="started");rows.append(row);atomic_json(r/"attempts.json",rows)
  started=time.monotonic();req=urllib.request.Request(url+"/v1/chat/completions",data=f.read_bytes(),headers={"Content-Type":"application/json","X-Request-ID":r.name+"-"+label})
  try:
   with opener.open(req,timeout=120)as res:status=res.status;raw=res.read()
  except urllib.error.HTTPError as err:status=err.code;raw=err.read()
  wire=r/(label+".wire");wire.write_bytes(raw);row.update(http_status=status,wire=ref(wire),wall_s=time.monotonic()-started)
  v=json.loads(raw);assert status==200 and not v.get("error")and len(v["choices"])==1
  usage=v["usage"];assert usage["prompt_tokens"]>0 and usage["completion_tokens"]>0 and usage["total_tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
  choice=v["choices"][0];content=choice["message"].get("content")or"";reason=choice["message"].get("reasoning")or""
  row.update(status="completed",usage=usage,content=content,reasoning=reason,finish_reason=choice["finish_reason"],semantic_pass=content.strip()==answer,effective_public_output_credit=usage["completion_tokens"])
  atomic_json(r/"attempts.json",rows);print(json.dumps(row),flush=True)
  for n in range(60):
   with opener.open(url+"/metrics",timeout=10)as res:b=res.read()
   (r/(label+"_after.metrics")).write_bytes(b);counts=[float(x)for x in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M)]
   if counts and all(x==0 for x in counts):break
   time.sleep(.25)
  else:raise RuntimeError("semanticprobe nativeidle deadline")
out=dict(at=utc(),transport_valid=True,semantic_acceptance=all(x["semantic_pass"]for x in rows),semantic_pass_cases=sum(x["semantic_pass"]for x in rows),semantic_cases=len(rows),requests=rows,effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows),verdict="INCONCLUSIVE"if all(x["semantic_pass"]for x in rows)else"REJECT",limits=["Finite3 newD exactanswer; P89 sixpriorpair exactanswers reused; simpleprompt semantic diagnostic withthinkingoff, not comprehensiveGLMqualitytest/SLO/capacity","P89 DP1TP16DCP16EP16 retained, freshD107 DP1TP8PP2DCP8 EP8perPP nativepartition42,36; noDSACP/PnoSP/DnoSP/P_K5_D_K3/nativeweights/math/retainedMLAworkspace; no rootcause/speedup/PD proof"])
atomic_json(r/"semantic_summary.json",out)
# Exactsameepochs/source/nativeidle after allrequests.
z=__import__("subprocess").run([sys.executable,str(r/"epoch_check.py"),"semantic_final"],capture_output=True,timeout=240);(r/"final_epoch.stdout").write_bytes(z.stdout);(r/"final_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
print(json.dumps(dict(semantic_pass_cases=out["semantic_pass_cases"],semantic_cases=len(rows),epoch_same=True)),flush=True)

if not out["semantic_acceptance"]:raise RuntimeError("native semantic candidate rejected; no cold performance workload")
