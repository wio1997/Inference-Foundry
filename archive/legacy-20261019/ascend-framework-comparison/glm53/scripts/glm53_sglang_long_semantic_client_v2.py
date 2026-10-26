"""Unchanged AISBench/native-ID long retrieval guards plus cross-DP MC+PD route."""
import asyncio,hashlib,importlib.metadata as md,json,re,struct,time,urllib.request
from pathlib import Path
import aiohttp,numpy as np,torch
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_sglang_pd_gateway_v2 import SGLangPDGateway,route
RUN=Path('/run/glm53');DATA=Path('/semantic');BASE='glm53-sem-sglang-v5-20261017'
assert torch.__version__=='2.10.0+cpu' and md.version('ais_bench_benchmark')=='3.1.0'
assert hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()=='ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece'
def default(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def metrics(label):
 out={}
 for node in [34,35,36,41]:
  port=30500 if node in [34,35] else 30600
  try:
   with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/metrics',timeout=5) as r:data=r.read()
   name=f'metrics-{label}-node{node}.txt';(RUN/name).write_bytes(data);out[node]={'file':name,'sha256':hashlib.sha256(data).hexdigest()}
  except Exception as error:out[node]={'error':repr(error)}
 try:
  with urllib.request.urlopen('http://172.16.12.34:29782/metrics',timeout=5) as r:data=r.read()
  name=f'memcache-metrics-{label}.txt';(RUN/name).write_bytes(data)
  prefixes=('memcache_segment_','memcache_total_capacity_','memcache_get_','memcache_evict_','memcache_bandwidth_','memcache_object_','memcache_allocated_')
  selected=[line for line in data.decode().splitlines() if line.startswith(prefixes)]
  out['MemCacheMeta']={'file':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'selected_native_lines':selected}
 except Exception as error:out['MemCacheMeta']={'error':repr(error)}
 return out
class Observed(GLM53TokenIDAPI):
 async def parse_stream_response(self,data,output):
  if data.get('id'):
   seen=output.extra_details_data.setdefault('glm53_server_response_ids',[])
   if data['id'] not in seen:seen.append(data['id'])
  if data.get('sglext') is not None:output.extra_details_data['glm53_native_sglext']=data['sglext']
  await super().parse_stream_response(data,output)
async def key_gate(index,record,namespace):
 req={'index':index,'namespace':namespace,'prefix_ids':record['prefix_ids'],'prefix_sha256_le_u32':record['prefix_sha256_le_u32']}
 pending=RUN/f'need-metadata-{index}.tmp';pending.write_text(json.dumps(req)+'\n');pending.replace(RUN/f'need-metadata-{index}.json');deadline=time.monotonic()+180
 while not (RUN/f'metadata-gate-{index}.json').is_file():
  if time.monotonic()>deadline:raise TimeoutError('Known-key metadata IPC gate timedout')
  await asyncio.sleep(.2)
 result=json.loads((RUN/f'metadata-gate-{index}.json').read_text());assert result['rc']==0 and result['pass'],result
 return result
async def main():
 rows=[];gwrows=[];runner=None;report={'project_id':'glm53','functional_only':True,'qualification':'SEMANTIC_GUARDS_NOT_PERFORMANCE_OR_WHOLE_OUTPUT_EQUALITY','cases':rows,'gateway':gwrows,'metadata_gates':{},'metrics':{'before':metrics('before')},'native_versions':{n:md.version(n) for n in ['ais_bench_benchmark','torch','aiohttp']}}
 try:
  namespaces=[BASE+f'-case{i}-'+suffix for i in range(2) for suffix in ['cold','warm']]
  async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=900),connector=aiohttp.TCPConnector(limit=0)) as upstream:
   gateway=SGLangPDGateway(upstream,namespaces,gwrows);runner=web.AppRunner(gateway.application());await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();gport=site._server.sockets[0].getsockname()[1]
   async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=900),connector=aiohttp.TCPConnector(limit=0)) as client:
    for index in range(2):
     record=json.loads((DATA/f'semantic-{index}.json').read_text());ids=record['input_ids'];prefix=record['prefix_ids'];assert len(ids)==80000 and len(prefix)==74400 and ids[:74400]==prefix
     for a,k in [(ids,'ids_sha256_le_u32'),(prefix,'prefix_sha256_le_u32')]:assert hashlib.sha256(struct.pack('<'+'I'*len(a),*a)).hexdigest()==record[k]
     base=BASE+f'-case{index}';normal=index*2;alternate=normal+1
     specs=[('cold-PD',normal,80000,600,base+'-cold','127.0.0.1',gport),('prefix-warmup-PD',normal,74400,1,base+'-warm','127.0.0.1',gport),('cross-DP-storage-PD',alternate,80000,600,base+'-warm','127.0.0.1',gport)]
     for label,pid,inputs,outputs,namespace,host,port in specs:
      if label=='cross-DP-storage-PD':report['metadata_gates'][index]=await key_gate(index,record,namespace)
      model=Observed(path='',model='glm53-w8a8',stream=True,host_ip=host,host_port=port,run_namespace=namespace,expected_input_tokens=inputs,expected_output_tokens=outputs)
      out=RequestOutput(perf_mode=True);out.uuid=f'{base}-p{pid:02d}-{label}';begin=time.monotonic_ns();await model.generate(prefix if inputs==74400 else ids,outputs,out,session=client);end=time.monotonic_ns();await gateway.wait_idle(10)
      usage=out.extra_details_data.get('glm53_service_usage');valid=out.extra_perf_data.get('glm53_valid_output_chunk_times',[])
      count_pass=bool(out.success and usage and usage.get('prompt_tokens')==inputs and usage.get('completion_tokens')==outputs and out.input_tokens==inputs and out.output_tokens==outputs and out.extra_details_data.get('glm53_finish_reason')=='length' and valid)
      answers=re.findall(r'FINAL_ANSWER:\s*([0-9]{12})(?![0-9])',out.content)
      semantic_pass=None if outputs==1 else bool(answers and answers[0]==record['expected_answer'])
      out.input=None;row={'semantic_case':record['case'],'case':label,'request_id':out.uuid,'route_prefix_id':pid,'namespace':namespace,'required_input':inputs,'required_output':outputs,'outer_dispatch_monotonic_ns':begin,'outer_terminal_monotonic_ns':end,'count_pass':count_pass,'expected_answer':record['expected_answer'],'observed_answer_fields':answers,'semantic_pass':semantic_pass,'output_text_sha256':hashlib.sha256(out.content.encode()).hexdigest(),'native':out.to_dict()};rows.append(row)
      report['metrics'][f'case{index}-{label}']=metrics(f'case{index}-{label}');(RUN/'semantic-progress.json').write_text(json.dumps(report,default=default,indent=2)+'\n');print('GLM53_SEMANTIC_CASE='+json.dumps(row,default=default),flush=True)
      assert count_pass and semantic_pass is not False,'Long retrieval/count gate failed; preserve raw, no baseline'
   await gateway.wait_idle(10);assert len(rows)==6 and len(gwrows)==6 and all(g['status']=='D_STREAM_COMPLETE' for g in gwrows)
   report['semantic_pass']=all(r['semantic_pass'] is not False for r in rows);report['whole_output_hash_equality']={}
   for index in range(2):
    selected=rows[index*3:(index+1)*3];reference=selected[0]['output_text_sha256'];report['whole_output_hash_equality'][index]={r['case']:r['output_text_sha256']==reference for r in selected if r['required_output']==600}
 finally:
  if runner:await runner.cleanup()
  report['external_backend_HTTP_attempts']=sum(int('prefill_dispatch_monotonic_ns' in row)+int('decode_dispatch_monotonic_ns' in row) for row in gwrows)
  p_success=[row for row in gwrows if row.get('prefill_http_status')==200 and isinstance(row.get('prefill_usage'),dict) and row['prefill_usage'].get('completion_tokens')==1]
  d_success=[row for row in rows if row['count_pass']]
  report['successful_backend_inferences']=len(p_success)+len(d_success)
  report['actual_successful_output_tokens']=len(p_success)+sum(row['required_output'] for row in d_success)
  (RUN/'semantic-result.json').write_text(json.dumps(report,default=default,indent=2)+'\n');print('GLM53_SEMANTIC_RESULT='+json.dumps(report,default=default),flush=True)
asyncio.run(main())
