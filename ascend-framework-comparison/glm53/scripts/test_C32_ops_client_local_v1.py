"""Pure async HTTP stubs exercise success/failure draining of the actual C32 functional client."""
import ast,asyncio,contextlib,hashlib,io,json,tempfile,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/probe_sglang_C32_ops_pd_payload_v1.py';source=p.read_text();tree=ast.parse(source)
 function=next(x for x in tree.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='main')
 code=compile(ast.Module(body=[function],type_ignores=[]),str(p),'exec');cases=[]
 for fail_one in [False,True]:
  active=0;peak=0;finished=0
  class Response:
   def __init__(self,body):
    self.body=body;self.status=500 if fail_one and body['rid'].endswith('p03-functional.D') else 200
   async def __aenter__(self):
    nonlocal active,peak
    active+=1;peak=max(peak,active);await asyncio.sleep(.002);return self
   async def __aexit__(self,*args):
    nonlocal active,finished
    active-=1;finished+=1
   async def text(self):
    return json.dumps({'usage':{'prompt_tokens':512,'completion_tokens':self.body['max_tokens']},'choices':[{'finish_reason':'length','text':'fixture-output'}]})
  class Session:
   def __init__(self,*args,**kwargs):pass
   async def __aenter__(self):return self
   async def __aexit__(self,*args):pass
   def post(self,url,*,json,headers):return Response(json)
  def route(pid):return {'P':{'global_dp':pid%4},'D':{'global_dp':pid%16}}
  fake=types.SimpleNamespace(ClientSession=Session,ClientTimeout=lambda **k:None,TCPConnector=lambda **k:None)
  with tempfile.TemporaryDirectory(prefix='glm53-C32-client-stubs-') as tmp:
   env={'asyncio':asyncio,'hashlib':hashlib,'json':json,'struct':__import__('struct'),'time':time,'Path':Path,'RUN':Path(tmp),'BASE':'fixture','sha':'fixture-input','ids':[0]*512,'expected':'fixture-output','tok':types.SimpleNamespace(decode=lambda *a,**k:'fixture-output'),'aiohttp':fake,'route':route,'bootstrap_room':lambda rid,rank:rank}
   exec(code,env);raised=False
   with contextlib.redirect_stdout(io.StringIO()):
    try:asyncio.run(env['main']())
    except AssertionError:raised=True
   report=json.loads((Path(tmp)/'result.json').read_text());assert len(report['cases'])==32 and len(report['backend_attempts'])==64 and finished==64 and active==0 and peak==64
   assert report['model_requests_attempted']==64 and report['successful_inferences']==(63 if fail_one else 64) and report['actual_successful_output_tokens']==(528 if fail_one else 544)
   assert raised==fail_one and report.get('pass',False)==(not fail_one)
   assert {r['D_DP'] for r in report['cases']}==set(range(16))
   cases.append({'injected_failure':fail_one,'all32_logical_cases_and64_backend_terminals_drained':True,'peak_backend_inflight':peak,'remaining_inflight':active,'raw_failure_preserved':True})
 proof={'project_id':'glm53','pass':True,'pure_stubs':True,'actual_image_or_model_test':False,'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'cases':cases,'source_evidence_sha256':{'scripts/probe_sglang_C32_ops_pd_payload_v1.py':hashlib.sha256(p.read_bytes()).hexdigest()}}
 (ROOT/'results/C32-ops-functional-client-pure-stub-v1.json').write_text(json.dumps(proof,indent=2)+'\n');print('PASS: C32success/failure bothdrain64terminals; zero network/modelcalls')
if __name__=='__main__':main()
