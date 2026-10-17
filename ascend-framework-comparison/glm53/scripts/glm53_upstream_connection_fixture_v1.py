"""CPU loopback regression for reused POST disconnects; zero model/SDK calls."""
import asyncio,collections,hashlib,importlib.metadata as md,json
from pathlib import Path
import aiohttp

async def main():
 peers=[];calls=[];closed_on_reuse=[]
 async def serve(reader,writer):
  peer=writer.get_extra_info('peername');peers.append(peer);requests=0
  try:
   while True:
    try:raw=await reader.readuntil(b'\r\n\r\n')
    except asyncio.IncompleteReadError:break
    headers=raw.decode().split('\r\n');size=int(next(x.split(':',1)[1] for x in headers if x.lower().startswith('content-length:')));body=await reader.readexactly(size);requests+=1
    request=json.loads(body);calls.append({'id':request['request_id'],'peer':peer,'request_number_on_connection':requests})
    if requests>1:
     # Deterministic peer close during a reused POST; no response, no retry.
     closed_on_reuse.append(request['request_id']);writer.close();break
    response=json.dumps({'request_id':request['request_id'],'input_tokens':80000,'output_tokens':600}).encode()
    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '+str(len(response)).encode()+b'\r\nConnection: keep-alive\r\n\r\n'+response);await writer.drain()
  finally:
   writer.close();await writer.wait_closed()
 server=await asyncio.start_server(serve,'127.0.0.1',0);url='http://127.0.0.1:'+str(server.sockets[0].getsockname()[1])+'/v1/completions';results={}
 try:
  for fresh in [False,True]:
   outcomes=[]
   async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=10),connector=aiohttp.TCPConnector(limit=0,force_close=fresh)) as session:
    for i in range(4):
     rid=f'fresh{fresh}-{i}'
     try:
      async with session.post(url,json={'request_id':rid}) as response:
       data=await response.json();assert response.status==200 and data['request_id']==rid and data['input_tokens']==80000 and data['output_tokens']==600;outcomes.append({'id':rid,'success':True})
     except aiohttp.ServerDisconnectedError as error:outcomes.append({'id':rid,'success':False,'error':type(error).__name__})
   results[str(fresh)]=outcomes
  assert sum(x['success'] for x in results['False'])==2
  assert all(x['success'] for x in results['True'])
  assert collections.Counter(x['id'] for x in calls)==collections.Counter(x['id'] for group in results.values() for x in group),'No automatic POST retry'
  assert len({tuple(x['peer']) for x in calls if x['id'].startswith('freshTrue-')})==4
  source=Path('/run/glm53/glm53_formal_ais_client_v2.py').read_text()
  assert 'connector=aiohttp.TCPConnector(limit=0,force_close=True)) as upstream:' in source
  result={'project_id':'glm53','pass':True,'qualification':'Synthetic reused-POST-disconnect mechanism regression only; actual C64 root cause remains inference until successful new run','aiohttp':md.version('aiohttp'),'real_model_requests':0,'SDK_init_calls':0,'NPU_devices_mounted':False,'attempts':len(calls),'automatic_retries':0,'old_reuse':results['False'],'fresh_connection':results['True'],'fresh_distinct_TCP_connections':4,'client_v2_sha256':hashlib.sha256(source.encode()).hexdigest()}
  Path('/run/glm53/fixture-result.json').write_text(json.dumps(result,indent=2)+'\n');print('GLM53_UPSTREAM_FIXTURE='+json.dumps(result),flush=True)
 finally:server.close();await server.wait_closed()
if __name__=='__main__':asyncio.run(main())
