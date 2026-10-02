from pathlib import Path
import asyncio,json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/tests")
import httpx
from test_response_affinity import CPUStores,CONFIG,GROUPS
from response_affinity_gateway import create_app
j=Path(__file__).parent
class ConcurrentStore(CPUStores):
 def __init__(self):
  super().__init__();self.first_arrived=asyncio.Event();self.release_first=asyncio.Event();self.posts=[]
 async def __call__(self,req):
  if req.method=="POST"and req.url.path=="/v1/responses":
   body=await req.aread();self.posts.append(dict(node=req.url.host,body_sha256=hashlib.sha256(body).hexdigest()))
   if len(self.posts)==1:self.first_arrived.set();await self.release_first.wait()
  return await super().__call__(req)
async def run():
 store=ConcurrentStore();journal=j/"owners_cpu.json";app=create_app(CONFIG,transport=httpx.MockTransport(store),groups=GROUPS,response_owner_state_path=str(journal));ident="resp_cpu_concurrent_custom"
 body=json.dumps(dict(model="glm-52",input="CPU fixture only",max_output_tokens=2,request_id=ident,store=True),separators=(",",":")).encode()
 async with app.router.lifespan_context(app):
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://gateway",timeout=10)as c:
   first=asyncio.create_task(c.post("/v1/responses",content=body,headers={"Content-Type":"application/json"}))
   await asyncio.wait_for(store.first_arrived.wait(),5)
   try:
    second=await c.post("/v1/responses",content=body,headers={"Content-Type":"application/json"});assert second.status_code==200 and second.json()["id"]==ident
    after_second=app.state.response_owner_index.resolve(ident);got_second=await c.get("/v1/responses/"+ident);assert got_second.status_code==200
   finally:store.release_first.set()
   one=await first;assert one.status_code==200 and one.json()["id"]==ident
   after_first=app.state.response_owner_index.resolve(ident);got_first=await c.get("/v1/responses/"+ident);assert got_first.status_code==200
   assert not app.state.placement.leases
 out=dict(posts=store.posts,owner_after_second=after_second,owner_after_first=after_first,cross_owner_race=len({v["node"]for v in store.posts})==2 and after_second!=after_first,body_bytes_preserved=all(v["body_sha256"]==hashlib.sha256(body).hexdigest()for v in store.posts),scope="CPUASGI/mocktransport/native39fixture only; zero actualbackend/model/GPU requests; no realnative duplicateID acceptance or performance proof")
 (j/"race_reduction.json").write_text(json.dumps(out,indent=2)+"\n");assert out["cross_owner_race"]and out["body_bytes_preserved"];print(json.dumps(out))
asyncio.run(run())
