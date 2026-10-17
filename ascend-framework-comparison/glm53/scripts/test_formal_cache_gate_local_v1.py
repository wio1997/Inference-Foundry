"""CPU fixtures only: every prefix metadata/read/retention/close contract."""
import hashlib,importlib.metadata,importlib.util,json,os,struct,sys,tempfile,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 spec=importlib.util.spec_from_file_location('gate',ROOT/'scripts/glm53_formal_cache_gate_v1.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 old_version=importlib.metadata.version;old_sdk=sys.modules.get('memcache_hybrid');old_seed=os.environ.get('PYTHONHASHSEED');os.environ['PYTHONHASHSEED']='0';cases=[]
 try:
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);data=root/'data';data.mkdir();manifest={'concurrency':16,'prefixes':[]}
   for pid in range(16):
    ids=[pid]+[42]*74399;digest=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest();entry={'prefix_id':pid,'input_ids':ids,'ids_sha256_le_u32':digest};(data/f'prefix-{pid:02d}.json').write_text(json.dumps(entry));manifest['prefixes'].append({'prefix_id':pid,'ids_sha256_le_u32':digest})
   raw=json.dumps(manifest).encode();(data/'manifest.json').write_bytes(raw)
   payload=b'\x2a'*12369920
   for mode in ['success','short_payload','retention_loss']:
    output=root/mode;output.mkdir();namespace='glm53-formal-LOCAL-FIXTURE-ONLY-'+mode+'-C16';(output/'need-cache-gate.json').write_text(json.dumps({'project_id':'glm53','framework':'vllm','concurrency':16,'namespace':namespace,'manifest_sha256':hashlib.sha256(raw).hexdigest()}));events=[]
    class Info:
     def size(self):return 12369920
     def gva_list(self):return [1]
    class Store:
     def __init__(self):self.checks=0
     def setup(self,cfg):events.append(('setup',cfg.dram_size,cfg.max_dram_size));assert cfg.protocol=='device_sdma';return 0
     def init(self,device_id,init_bm):events.append(('init',init_bm));assert init_bm is True;return 0
     def batch_is_exist(self,keys):
      self.checks+=1;events.append(('check',len(keys)));return [0 if mode=='retention_loss' and self.checks>16 else 1]*len(keys)
     def batch_get_key_info(self,keys):events.append(('info',len(keys)));return [Info() for key in keys]
     def get(self,key):events.append(('read',));return payload if mode!='short_payload' else b'bad'
     def close(self):events.append(('close',));return 0
    sys.modules['memcache_hybrid']=types.SimpleNamespace(__file__='/opt/glm53/sdk13/LOCAL_FIXTURE_ONLY.py',LocalConfig=type('Config',(),{}),DistributedObjectStore=Store)
    importlib.metadata.version=lambda name:'1.3.0' if name=='memcache_hybrid' else old_version(name)
    settings={'expected_sdk_root':'/opt/glm53/sdk13/','output':str(output),'framework':'vllm','dataset':str(data),'meta_service_url':'tcp://FIXTURE-ONLY','config_store_url':'tcp://FIXTURE-ONLY','world_size':128,'protocol':'device_sdma','max_dram_size':'192GB','device_id':0}
    error=None
    try:gate.run(settings)
    except AssertionError as problem:error=repr(problem)
    report=json.loads((output/'cache-gate.json').read_text());assert events[-1]==('close',) and report['close_rc']==0
    if mode=='success':assert report['pass'] and len(report['prefixes'])==16 and sum(x[0]=='read' for x in events)==64 and sum(x[0]=='check' for x in events)==32
    else:assert error is not None and report['pass'] is False
    cases.append({'case':mode,'expected_pass':mode=='success','actual_pass':report['pass'],'metadata_prefixes':len(report['prefixes']),'payload_reads':sum(x[0]=='read' for x in events),'closed':True})
 finally:
  importlib.metadata.version=old_version
  if old_sdk is None:sys.modules.pop('memcache_hybrid',None)
  else:sys.modules['memcache_hybrid']=old_sdk
  if old_seed is None:os.environ.pop('PYTHONHASHSEED',None)
  else:os.environ['PYTHONHASHSEED']=old_seed
 result={'project_id':'glm53','local_fixture_only':True,'pass':True,'cases':cases,'actual_model_calls':0,'actual_SDK_calls':0,'numerical_KV_equivalence_claimed':False,'source_sha256':hashlib.sha256((ROOT/'scripts/glm53_formal_cache_gate_v1.py').read_bytes()).hexdigest()};(ROOT/'results/formal-cache-gate-local-fixture-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
