"""Local raw-archive fixture for the independent native metrics post-auditor."""
import ast,gzip,hashlib,importlib.util,json,re,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/inspect_vllm_formal_C16_native_v2.py';spec=importlib.util.spec_from_file_location('capture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 node=next(x for x in ast.parse(m.HOST).body if isinstance(x,ast.If) and isinstance(x.test,ast.Compare) and isinstance(x.test.left,ast.Name) and x.test.left.id=='NODE' and isinstance(x.test.comparators[0],ast.Constant) and x.test.comparators[0].value==43)
 cases=[]
 with tempfile.TemporaryDirectory() as temporary:
  for mode in ['success','bad_native_body_sha','missing_native_DP']:
   root=Path(temporary)/mode;output=root/'output';output.mkdir(parents=True);rows=[]
   for sample in range(2):
    for host in [43,45]:
     ranks=range(8) if host==43 else range(8,16);body='';occupancy={}
     for rank in ranks:
      running=1 if sample==0 else 0
      for name,value in [('num_requests_running',running),('num_requests_waiting',0)]:
       if not (mode=='missing_native_DP' and rank==15):body+=f'vllm:{name}{{engine="{rank}",model_name="glm53-w8a8"}} {value}\n'
      occupancy[str(rank)]={'node':host,'native_dp':rank,'running':running,'waiting':0}
     raw=body.encode();rows.append({'node':host,'dispatch_perf_counter_ns':sample*1000,'terminal_perf_counter_ns':sample*1000+(1 if host==43 else 2),'http_status':200,'raw_utf8':body,'bytes':len(raw),'sha256':('0'*64 if mode=='bad_native_body_sha' else hashlib.sha256(raw).hexdigest()),'occupancy':occupancy})
   raw=b''.join((json.dumps(row)+'\n').encode() for row in rows);packed=gzip.compress(raw,mtime=0);(output/'formal-D-metrics.jsonl.gz').write_bytes(packed)
   expected={'raw_archive_sha256':hashlib.sha256(packed).hexdigest(),'sample_count':2,'complete_sample_count':2,'sampled_peak_total_D_running':16,'sampled_peak_D_running_by_DP':{str(i):1 for i in range(16)}}
   report={'project_id':'glm53','framework':'vllm','concurrency':16,'namespace':'glm53-formal-LOCAL-FIXTURE-ONLY-C16','phase':'COMPLETE','baseline_accepted':True,'D_metrics':expected};(output/'formal-result.json').write_text(json.dumps(report))
   fragment=ast.unparse(ast.Module(body=node.body,type_ignores=[]));fragment=fragment.replace("'/data/wio/inference-foundry/glm53/runs/20261017T1790-vllm-formal-C16-v1'",repr(str(root)))
   scope={'pathlib':__import__('pathlib'),'json':json,'hashlib':hashlib,'gzip':gzip,'re':re,'out':{}};error=None
   try:exec(compile(fragment,'<actual_native_metrics_CPU_fragment>','exec'),scope)
   except AssertionError as problem:error=repr(problem)
   if mode=='success':assert error is None and scope['out']['independent_formal_metric_archive_audit']['all_native_bodies_SHA_and_DP_gauges_recomputed']
   else:assert error is not None
   cases.append({'case':mode,'expected_success':mode=='success','actual_success':error is None})
 result={'project_id':'glm53','local_fixture_only':True,'pass':True,'actual_resource_calls':0,'actual_model_calls':0,'actual_SDK_calls':0,'cases':cases,'native_capture_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};(ROOT/'results/formal-native-metrics-local-fixture-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
