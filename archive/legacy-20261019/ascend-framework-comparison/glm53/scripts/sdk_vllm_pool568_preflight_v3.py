"""Installed SDK objects only; native CLI requires NPU; no stores/engines/NPU init."""
import argparse, importlib.metadata, json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--framework',choices=['vllm'],required=True);parser.add_argument('--root',required=True)
args=parser.parse_args();root=Path(args.root)
import memcache_hybrid as sdk
import memfabric_hybrid as fabric
assert importlib.metadata.version('memcache_hybrid')=='1.3.0'
assert sdk.__file__.startswith('/opt/glm53/sdk13/') and fabric.__file__.startswith('/opt/glm53/sdk13/')
sizes={30:'192GB',31:'192GB',43:'128GB',45:'56GB'};observed=[]
for n,dram in sizes.items():
    provider=json.loads((root/f'provider-config/node-{n}.json').read_text())
    assert provider['dram_size']==dram and provider['max_dram_size']=='192GB' and provider['init_bm'] is True
    assert provider['framework']=='vllm' and provider['node']==n and provider['expected_sdk_root']=='/opt/glm53/sdk13/'
    for contribution in ['0GB',dram]:
        config=sdk.LocalConfig()
        for key in ['meta_service_url','config_store_url','log_level','world_size','protocol','dram_size','max_dram_size','hbm_size','max_hbm_size']:
            assert hasattr(config,key);setattr(config,key,contribution if key=='dram_size' else provider[key])
        assert config.dram_size==contribution and config.max_dram_size=='192GB' and config.protocol=='device_sdma' and config.world_size==128
        observed.append({'node':n,'dram_size':config.dram_size,'max_dram_size':config.max_dram_size})
    supervisor=json.loads((root/f'supervisor-config/node-{n}.json').read_text())
    assert supervisor['provider_env']=={} and 'MMC_LOCAL_CONFIG_PATH' not in supervisor['provider_env']
    assert 'PYTHONPATH' not in supervisor['model_env'] and 'LD_LIBRARY_PATH' not in supervisor['model_env']
    assert supervisor['model_env']['MMC_LOCAL_CONFIG_PATH']=='/opt/glm53/config/local.conf'
meta=(root/'config/meta.conf').read_text();local=(root/'config/local.conf').read_text()
for key,value in [('ock.mmc.meta.lease_ttl_ms','60000'),('ock.mmc.evict_threshold_high','99'),('ock.mmc.evict_threshold_low','98')]:assert f'{key} = {value}' in meta
assert 'ock.mmc.local_service.dram.size = 0GB' in local and 'ock.mmc.local_service.max.dram.size = 192GB' in local
print('GLM53_SDK_CONFIG_PREFLIGHT='+json.dumps({'framework':'vllm','module_paths':[sdk.__file__,fabric.__file__],
      'SDK_version':'1.3.0','configurations':observed,'native_frontend_CLI_parsed_all4':False,'native_CLI_requires_NPU_visible_container':True,
      'store_objects_created':0,'SDK_init_calls':0,'engines_created':0,'NPU_allocations':0,'model_requests':0}))
