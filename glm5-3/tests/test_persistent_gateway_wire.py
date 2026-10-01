
import json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/"runtime"))
from persistent_coupled_gateway import create_app
from test_coupled_fault_domain import CONFIG,GROUPS
class PersistentWireContract(unittest.IsolatedAsyncioTestCase):
 async def test_native_historical_error_payload_is_raw_and_blocks_same_epoch_new_gateway(self):
  import httpx
  path=pathlib.Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0032/control_DP0/cold_DP0.response.jsonl")
  event=json.loads(path.read_text().splitlines()[0]);payload=event.get("data")or event.get("raw")
  self.assertIsInstance(payload,str)
  wire=("data: "+payload+"\n\ndata: [DONE]\n\n").encode()
  class Raw(httpx.AsyncByteStream):
   async def __aiter__(self):
    for i in range(0,len(wire),7):yield wire[i:i+7]
  calls=[]
  async def backend(request):
   calls.append((request.method,request.url.path))
   if request.url.path=="/health":return httpx.Response(200,content=b"healthy")
   return httpx.Response(200,headers={"content-type":"text/event-stream"},stream=Raw())
  with tempfile.TemporaryDirectory()as d:
   state=str(pathlib.Path(d)/"faults.json")
   app=create_app(CONFIG,groups=GROUPS,fault_state_path=state,transport=httpx.MockTransport(backend))
   async with app.router.lifespan_context(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://router")as client:
     response=await client.post("/v1/chat/completions",content=b"{}")
     self.assertEqual(response.status_code,200);self.assertEqual(response.content,wire)
     self.assertFalse(app.state.placement.leases)
   stored=json.loads(pathlib.Path(state).read_text());self.assertFalse(stored["open"]);self.assertTrue(stored["groups"]["EP32"]["faulted"])
   fresh=create_app(CONFIG,groups=GROUPS,fault_state_path=state,transport=httpx.MockTransport(backend))
   async with fresh.router.lifespan_context(fresh):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh),base_url="http://router")as client:
     health=await client.get("/healthcheck");self.assertEqual(health.status_code,503)
     for id in ["DP0","DP1"]:
      self.assertEqual((await client.delete("/control/replicas/"+id)).status_code,200)
      self.assertEqual((await client.post("/control/replicas",json=next(v for v in CONFIG if v["id"]==id))).status_code,200)
     blocked=await client.post("/v1/chat/completions",content=b"{}");self.assertEqual(blocked.status_code,503)
     self.assertTrue(all(v["group_faulted"]for v in (await client.get("/control/replicas")).json()["replicas"]))
   self.assertEqual(sum(method=="POST"and path=="/v1/chat/completions"for method,path in calls),1)
if __name__=="__main__":unittest.main()
