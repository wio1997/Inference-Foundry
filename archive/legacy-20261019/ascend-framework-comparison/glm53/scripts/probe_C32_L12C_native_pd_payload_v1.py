"""Native concurrent bootstrap functional check; no benchmark clock acceptance."""
import asyncio,hashlib,json,struct,time
from pathlib import Path
import aiohttp
from tokenizers import Tokenizer
from glm53_sglang_pd_gateway_v2 import route,bootstrap_room
RUN=Path('/run/glm53');BASE='glm53-sg-C32-L12C-native-functional-v1-20261019'
p=Path('/tokenizer.json');assert hashlib.sha256(p.read_bytes()).hexdigest()=='19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
tok=Tokenizer.from_file(str(p));text=('Inventory verification. The code maple means 17, cedar means 23, and river means 41. Read the complete record and explain how to check that the numbers add up. A reliable answer uses each stated number exactly once and shows the result.\n')*40
ids=tok.encode(text,add_special_tokens=False).ids[:512];sha=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest();assert sha=='d6e4cf16d46a7e29461cd4a7e19fa45725cb1b42ac3f76fda226260074db107e'
expected=tok.decode([220,99619,11,323,14785,3363,220,102340,13,4457,279,4583,3255,323,10334,1246],skip_special_tokens=False)
async def main():
 rows=[];attempts=[];report={'project_id':'glm53','namespace':BASE,'functional_only':True,'offered_concurrency':32,'D_DP_count':16,'cases_per_D_DP':2,'input_sha256_le_u32':sha,'input_tokens':512,'cases':rows,'backend_attempts':attempts,'baseline_accepted':False,'formal_requests':0}
 async def leg(session,body,url,role,rank):
  row={'rid':body['rid'],'role':role,'global_dp':rank,'dispatch_monotonic_ns':time.monotonic_ns()};attempts.append(row)
  try:
   async with session.post(url,json=body,headers={'X-Request-Id':body['rid']}) as response:
    row['status']=response.status;raw=await response.text();row['raw_response']=raw;assert response.status==200,(response.status,raw[-2000:]);d=json.loads(raw);row['response']=d;maximum=1 if role=='P' else 16
    assert d['usage']['prompt_tokens']==512 and d['usage']['completion_tokens']==maximum
    choice=d['choices'][0];assert choice['finish_reason']=='length'
    if role=='D':assert choice['text']
    row['pass']=True;row['text_sha256']=hashlib.sha256(choice['text'].encode()).hexdigest();row['vllm_short_reference_text_equal']=choice['text']==(tok.decode([220],skip_special_tokens=False) if role=='P' else expected)
  except Exception as e:row['pass']=False;row['error']=repr(e)
  finally:row['terminal_monotonic_ns']=time.monotonic_ns();print('GLM53_SG_MATRIX_STEP='+json.dumps(row),flush=True)
  return row
 try:
  async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=180),connector=aiohttp.TCPConnector(limit=0,force_close=True)) as session:
   async def one(pid):
    r=route(pid);rid=f'{BASE}-p{pid:02d}-functional';room=bootstrap_room(rid,r['P']['global_dp']);assert room%4==r['P']['global_dp']
    common={'model':'glm53-w8a8','prompt':ids,'min_tokens':1,'ignore_eos':True,'temperature':0,'seed':20261017,'cache_salt':f'{BASE}-p{pid:02d}','stream':False,'skip_special_tokens':False,'return_token_ids':False,'bootstrap_host':'172.16.12.34','bootstrap_port':29898,'bootstrap_room':room,'return_cached_tokens_details':True,'return_spec_tokens_details':True}
    p={**common,'rid':rid+'.P','routed_dp_rank':r['P']['global_dp'],'max_tokens':1};d={**common,'rid':rid+'.D','routed_dp_rank':r['D']['global_dp'],'disagg_prefill_dp_rank':r['P']['global_dp'],'max_tokens':16,'min_tokens':16}
    # Same native bootstrap room; send concurrently, retaining each terminal result.
    pres,dres=await asyncio.gather(leg(session,p,'http://172.16.12.34:30500/v1/completions','P',r['P']['global_dp']),leg(session,d,'http://172.16.12.36:30600/v1/completions','D',r['D']['global_dp']))
    row={'P_DP':pid%4,'D_DP':pid%16,'prefill':pres,'decode':dres,'pass':pres['pass'] and dres['pass']};rows.append(row);(RUN/'progress.json').write_text(json.dumps(report,indent=2)+'\n');assert row['pass'],'Stop after first failed logical pair; retain both native results'
   cohort=await asyncio.gather(*(one(pid) for pid in range(32)),return_exceptions=True)
   assert not any(isinstance(x,BaseException) for x in cohort),[repr(x) for x in cohort if isinstance(x,BaseException)]
  report['pass']=len(rows)==32 and all(x['pass'] for x in rows);assert report['pass'] and len(attempts)==64
  report['successful_output_tokens']=sum(x['response']['usage']['completion_tokens'] for x in attempts if x['pass']);report['D_text_hashes']=sorted({x['decode']['text_sha256'] for x in rows})
 finally:
  report['model_requests_attempted']=len(attempts);report['successful_inferences']=sum(bool(x.get('pass')) for x in attempts);report['actual_successful_output_tokens']=sum(x.get('response',{}).get('usage',{}).get('completion_tokens',0) for x in attempts if x.get('pass'))
  (RUN/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_SG_MATRIX_RESULT='+json.dumps(report),flush=True)
asyncio.run(main())
