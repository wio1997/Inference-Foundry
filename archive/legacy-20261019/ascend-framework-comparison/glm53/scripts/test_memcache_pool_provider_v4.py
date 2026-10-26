"""Real provider validation/lifecycle with a fake SDK; no resources."""
import contextlib,copy,hashlib,importlib.util,io,json,signal,tempfile,threading,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/glm53_memcache_pool_provider_v4.py';s=importlib.util.spec_from_file_location('provider',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main():
 configs=[json.loads((ROOT/f'config/standalone-pool-providers-v4/vllm-node-{n}.json').read_text()) for n in [30,31,43,45]]
 configs += [json.loads((ROOT/f'config/standalone-pool-providers-v2/sglang-node-{n}.json').read_text()) for n in [34,35,36,41]]
 rows=[]
 for config in configs:
  expected=m.validate(config,'1.3.0');calls=[]
  class LocalConfig:
   meta_service_url='';config_store_url='';log_level='';world_size=0;protocol='';dram_size='';hbm_size='';max_dram_size='';max_hbm_size=''
  class Store:
   def __init__(self):calls.append('construct')
   def setup(self,cfg):assert cfg.dram_size==expected and cfg.max_dram_size==config['max_dram_size'];calls.append('setup');return 0
   def init(self,device_id,init_bm):assert device_id==0 and init_bm;calls.append('init');signal.signal(signal.SIGTERM,signal.SIG_IGN);return 0
   def get_local_service_id(self):calls.append('id');return 0
   def close(self):calls.append('close');return 0
  sdk=types.SimpleNamespace(__file__=config['expected_sdk_root']+'__init__.py',LocalConfig=LocalConfig,DistributedObjectStore=Store)
  original_event=threading.Event;original_signals={n:signal.getsignal(n) for n in [signal.SIGTERM,signal.SIGINT]}
  event=original_event();event.set()
  with tempfile.TemporaryDirectory(prefix='glm53-provider-lifecycle-') as tmp:
   # Keep exact public config validation, remap only the filesystem output.
   original_path=m.Path
   m.Path=lambda x:Path(tmp)/'provider-status.json' if x=='/run/glm53/provider-status.json' else original_path(x)
   try:
    threading.Event=lambda:event
    with contextlib.redirect_stdout(io.StringIO()):m.run(config,sdk,'1.3.0')
    assert callable(signal.getsignal(signal.SIGTERM))
    result=json.loads((Path(tmp)/'provider-status.json').read_text());assert result['phase']=='CLOSE_COMPLETE' and result['events'][-1]['rc']==0
    assert any(row['event']=='READY' for row in result['events'])
    assert calls==['construct','setup','init','id','close']
   finally:
    m.Path=original_path;threading.Event=original_event
    for n,v in original_signals.items():signal.signal(n,v)
  rows.append({'framework':config['framework'],'node':config['node'],'fake_SDK_lifecycle':'setup/init/READY/close0','post_SDK_signal_handler_restored':True})
 for invalid in [dict(configs[0],framework='sglang'),dict(configs[0],dram_size='136GB'),dict(configs[0],max_dram_size='160GB')]:
  try:m.validate(invalid,'1.3.0')
  except AssertionError:pass
  else:raise AssertionError('Invalid cross-framework/capacity accepted')
 result={'project_id':'glm53','pass':True,'provider_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'cases':rows,'invalid_cases_rejected':3,'network_calls':0,'SDK_init_calls':0,'model_requests':0,'scope':'Actual provider code with fake SDK; excludes native allocation/readback'}
 (ROOT/'results/memcache-provider-local-fixture-v4.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
