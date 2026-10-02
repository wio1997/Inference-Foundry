"""CPU protocol contracts; native GLM plans are inputs, no inference."""
import copy,json,unittest
from pathlib import Path
import httpx
from native_pd_geometry import assess_pd_geometry,MOONCAKE_SOURCE_SHA256
from native_pd_transport_v3 import NativePDTransport
ROOT=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs")
D="http://172.16.10.167:9900";P="http://172.16.10.166:9081"
RAW=b'data: {"id":"cpu_fixture","choices":[]}\n\ndata: [DONE]\n\n'
def pair(n=104):
    plans=json.loads((ROOT/("GLM-RUN-%04d"%n)/"planned_launch.json").read_text())
    return dict(producer=next(x for x in plans if x["node"]=="166"),decoder=next(x for x in plans if x["node"]=="167"),connector_source_sha256=MOONCAKE_SOURCE_SHA256)
MAPPING={D:dict(url=P,remote_host="172.16.10.166",remote_port=28000,dcp_size=16)}
KV=dict(do_remote_prefill=True,remote_host="172.16.10.166",remote_port=28000,remote_engine_id="cpu_fixture_only",remote_dcp_size=16,remote_pcp_size=1,remote_block_ids=[[1,2]])
class GeometryProtocols(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,entry,body,path="/v1/chat/completions",mapping=None,status=200):
        calls=[];traces=[]
        async def native(req):
            wire=await req.aread();calls.append((str(req.url),wire,req.headers.get("x-request-id")))
            if str(req.url).startswith(P):
                usage={"output_tokens":1}if path=="/v1/responses"else{"completion_tokens":1}
                return httpx.Response(200,json=dict(kv_transfer_params=KV,usage=usage))
            return httpx.Response(status,headers=[("content-type","text/event-stream"),("x-native","fixture"),("x-duplicate","one"),("x-duplicate","two")],content=RAW)
        t=NativePDTransport(mapping or MAPPING,base=httpx.MockTransport(native),native_plans={}if entry is None else{D:entry},minimum_bytes=0,trace=lambda event,**fields:traces.append((event,fields)))
        async with httpx.AsyncClient(transport=t)as c:
            res=await c.post(D+path,content=body,headers={"content-type":"application/json","x-request-id":"original_native_id"})
            self.assertEqual(res.status_code,status);self.assertEqual(res.content,RAW)
            self.assertEqual(res.headers.get_list("x-duplicate"),["one","two"])
        return calls,traces
    def test_known_native_pair_necessary_geometry(self):
        a=pair();result=assess_pd_geometry(**a);self.assertTrue(result["eligible"]);self.assertEqual(result["reasons"],[])
    def test_real_pp2_candidates_ineligible(self):
        for n in [107,108,109]:
            with self.subTest(run=n):
                r=assess_pd_geometry(**pair(n));self.assertFalse(r["eligible"]);self.assertIn("native_decode_pp_must_be_1",r["reasons"])
    async def test_unknown_incompatible_pair_never_creates_helper_and_keeps_error_wire(self):
        original=b'{ "model":"glm-52", "messages":[{"role":"user","content":"fixture"}], "max_tokens":3, "stream":true }'
        stale=pair();i=stale["decoder"]["argv"].index("--tensor-parallel-size");stale["decoder"]["argv"][i+1]="8"
        unknown=pair();unknown["connector_source_sha256"]="unknown"
        malformed=pair();malformed["decoder"]=None
        boolean=pair();i=boolean["decoder"]["argv"].index("--kv-transfer-config");v=json.loads(boolean["decoder"]["argv"][i+1]);v["kv_connector_extra_config"]["decode"]["pp_size"]=True;boolean["decoder"]["argv"][i+1]=json.dumps(v)
        for entry in [None,pair(108),pair(109),stale,unknown,malformed,boolean]:
            with self.subTest(entry=entry is None):
                calls,traces=await self.exercise(entry,original,status=400)
                self.assertEqual(calls,[(D+"/v1/chat/completions",original,"original_native_id")]);self.assertEqual(traces[0][0],"pd_geometry_native_fallback")
    async def test_mapping_disagreement_falls_back_before_helper(self):
        original=b'{"model":"glm-52","messages":[{"role":"user","content":"fixture"}],"max_tokens":3}'
        mapping=copy.deepcopy(MAPPING);mapping[D]["remote_port"]=28001
        calls,_=await self.exercise(pair(),original,mapping=mapping);self.assertEqual(len(calls),1);self.assertEqual(calls[0][1],original)
    async def test_eligible_chat_keeps_v2_one_helper_one_original_decoder_plus_kv(self):
        payload=dict(model="glm-52",messages=[dict(role="user",content="fixture")],max_tokens=3,stream=True,seed=1024,request_id="native_caller")
        calls,traces=await self.exercise(pair(),json.dumps(payload).encode());self.assertEqual(len(calls),2)
        helper=json.loads(calls[0][1]);decoder=json.loads(calls[1][1])
        self.assertEqual(helper["max_tokens"],1);self.assertEqual(helper["min_tokens"],1);self.assertFalse(helper["stream"]);self.assertNotEqual(helper["request_id"],"native_caller")
        self.assertEqual(decoder,{**payload,"kv_transfer_params":KV});self.assertEqual(calls[1][2],"original_native_id")
    async def test_eligible_responses_helper_is_private_unstored_and_original_store_retained(self):
        payload=dict(model="glm-52",input="fixture",max_output_tokens=3,store=True,request_id="resp_native_caller")
        calls,_=await self.exercise(pair(),json.dumps(payload).encode(),path="/v1/responses");self.assertEqual(len(calls),2)
        helper=json.loads(calls[0][1]);self.assertEqual(helper["max_output_tokens"],1);self.assertFalse(helper["store"]);self.assertTrue(helper["request_id"].startswith("resp_glm_pdhelper_"))
        self.assertEqual(json.loads(calls[1][1]),{**payload,"kv_transfer_params":KV})
    async def test_stateful_request_preserves_native_decoder_without_helper(self):
        raw=b'{"model":"glm-52","input":"fixture","max_output_tokens":3,"previous_response_id":"resp_cpu_fixture"}'
        calls,_=await self.exercise(pair(),raw,path="/v1/responses");self.assertEqual(calls,[(D+"/v1/responses",raw,"original_native_id")])
