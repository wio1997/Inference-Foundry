import asyncio,unittest,json,tempfile,hashlib
from pathlib import Path
from shape_split_placement import ShapeSplitPlacement
from native_engines_service_config import compile_config,render,checked_config
C=[dict(id="D0",url="http://d0"),dict(id="D1",url="http://d1")]
S=dict(input_threshold_bytes=32768,prefill_members=["D1"],decode_members=["D0"])
G=[dict(id=k,epoch="epoch-"+k,members=[k])for k in["D0","D1"]]
class Candidate(unittest.IsolatedAsyncioTestCase):
 def pool(self):return ShapeSplitPlacement(C,"shape_split_idle_spill",G,shape_config=S)
 async def test_busy_prefill_not_borrowed_after_first_output(self):
  p=self.pool();large=await p.acquire(512,81932);small=await p.acquire(4096,21)
  await p.output_started(large)
  more=await p.acquire(1024,21);self.assertEqual(more.replica.key,"D0")
  self.assertEqual(large.replica.key,"D1")
  for lease in[large,small,more]:await p.release(lease)
 async def test_lease_empty_borrow_once_under_concurrent_arrival_then_return_after_release(self):
  p=self.pool();held=await p.acquire(4096,21)
  arrivals=await asyncio.gather(*(p.acquire(1024,21)for _ in range(20)))
  peer=[l for l in arrivals if l.replica.key=="D1"];self.assertEqual(len(peer),1)
  self.assertEqual(len(p.leases),21)
  self.assertEqual(peer[0].replica.active[peer[0].lease_id],(1024,21))
  await p.release(peer[0]);again=await p.acquire(64,21);self.assertEqual(again.replica.key,"D1")
  self.assertTrue(await p.release(again));self.assertFalse(await p.release(again))
  for l in arrivals:
   if l is not peer[0]:await p.release(l)
  await p.release(held);self.assertFalse(p.leases)
 async def test_preferred_lease_empty_does_not_spill_and_large_does_not_spill(self):
  p=self.pool();short=await p.acquire(32,21);self.assertEqual(short.replica.key,"D0")
  large=await p.acquire(64,32768);self.assertEqual(large.replica.key,"D1")
  morelarge=await p.acquire(64,32768);self.assertEqual(morelarge.replica.key,"D1")
  for l in[short,large,morelarge]:await p.release(l)
 async def test_affinity_and_fault_epoch_override_shape_spill(self):
  p=self.pool();held=await p.acquire(32,21);borrowed=await p.acquire(32,21)
  owner=dict(replica="D0",url="http://d0",group="D0",epoch="epoch-D0")
  bound=await p.acquire_for_owner(64,21,owner);self.assertEqual(bound.replica.key,"D0")
  with self.assertRaises(RuntimeError):await p.acquire_for_owner(64,21,{**owner,"epoch":"old"})
  await p.release(borrowed,backend_failure=True)
  more=await p.acquire(64,21);self.assertEqual(more.replica.key,"D0")
  rows=await p.snapshot();self.assertTrue(next(x for x in rows if x["id"]=="D1")["group_faulted"])
  for l in[held,bound,more]:await p.release(l)
 async def test_drain_peer_and_last_preferred_lease_keeps_owners_until_release(self):
  p=self.pool();held=await p.acquire(32,21);await p.remove("D1")
  more=await p.acquire(32,21);self.assertEqual(more.replica.key,"D0")
  await p.remove("D0")
  with self.assertRaises(RuntimeError):await p.acquire(1,0)
  self.assertTrue(await p.release(more));self.assertTrue(await p.release(held));self.assertFalse(p.replicas)
 async def test_gateway_full_body_duplicate_headers_and_bound_STORE_remain_same_owner(self):
  import httpx
  from response_affinity_gateway_v12 import create_app
  from test_response_affinity import JSON_RAW
  calls=[]
  async def backend(request):
   calls.append((request.url.host,request.url.path,await request.aread()))
   if request.method=="GET":return httpx.Response(200,content=JSON_RAW)
   return httpx.Response(200,content=JSON_RAW,headers=[("content-type","application/json"),("x-duplicate","one"),("x-duplicate","two")])
  with tempfile.TemporaryDirectory()as tmp:
   app=create_app(C,transport=httpx.MockTransport(backend),policy="shape_split_idle_spill",groups=G,response_owner_state_path=str(Path(tmp)/"owner.json"),shape_config=S,pd_config={})
   async with app.router.lifespan_context(app):
    p=app.state.placement;held=await p.acquire(4096,21)
    raw=b'{"model":"glm-52","input":"preserve bytes","max_output_tokens":32}'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://router")as client:
     reply=await client.post("/v1/responses",content=raw)
     self.assertEqual(reply.content,JSON_RAW);self.assertEqual(reply.headers.get_list("x-duplicate"),["one","two"])
     self.assertEqual(calls[-1],("d1","/v1/responses",raw))
     obj=json.loads(JSON_RAW);rid=obj["id"]
     bound=await client.get("/v1/responses/"+rid);self.assertEqual(bound.content,JSON_RAW)
     self.assertEqual(calls[-1][0],"d1")
    await p.release(held);self.assertFalse(p.leases)
class Compiler(unittest.TestCase):
 def test_exact_material_compilation_and_roundtrip_preserve_all_native_domains(self):
  base=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0204/restored/native_engines_resident.json")
  material=json.loads(base.read_text());old=compile_config(material,"/tmp/test-native-shape")
  material["placement"]["kind"]="shape_split_idle_spill";new=compile_config(material,"/tmp/test-native-shape")
  self.assertEqual(new["native_domains"],old["native_domains"]);self.assertEqual(new["engine_geometry"],old["engine_geometry"])
  expected=dict(old["environment"],GLM_PLACEMENT_POLICY="shape_split_idle_spill");self.assertEqual(new["environment"],expected)
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/"native_engines_resident.json").write_text(json.dumps(material))
   conf=render(root,root/"state");f=root/"service_config.json";f.write_text(json.dumps(conf))
   actual,state=checked_config(f);self.assertEqual(actual,conf);self.assertEqual(state,root/"state")
  material["placement"]["unexpected"]=True
  with self.assertRaises(ValueError):compile_config(material,"/tmp/test-native-shape")
if __name__=="__main__":
 # Candidate imported before historic tests can prepend their runtime path.
 import test_shape_split,test_response_affinity,test_response_affinity_v2,test_response_affinity_v3,test_durable_coupled_group
 suite=unittest.TestSuite()
 loader=unittest.defaultTestLoader
 for mod in[__import__(__name__),test_shape_split,test_response_affinity,test_response_affinity_v2,test_response_affinity_v3,test_durable_coupled_group]:suite.addTests(loader.loadTestsFromModule(mod))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 print(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful())))
 raise SystemExit(0 if result.wasSuccessful()else 1)
