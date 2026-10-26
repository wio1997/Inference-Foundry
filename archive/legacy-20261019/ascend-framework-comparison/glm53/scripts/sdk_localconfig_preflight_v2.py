"""Native SDK1.3 config objects only; never create or initialize a store."""
import argparse,hashlib,importlib.metadata,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--framework',required=True,choices=['sglang','vllm']);parser.add_argument('--root',required=True);args=parser.parse_args()
root=Path(args.root);head=34 if args.framework=='sglang' else 30;size='160GB' if head==34 else '512GB'
import memcache_hybrid as mc
import memfabric_hybrid as mf
assert importlib.metadata.version('memcache_hybrid')=='1.3.0'
provider=json.loads((root/f'provider-config/node-{head}.json').read_text());assert provider['dram_size']=='136GB' and provider['max_dram_size']==size and provider['init_bm'] is True
assert mc.__file__.startswith(provider['expected_sdk_root'])
for name in ['init','setup','close','get_local_service_id','batch_get_into','batch_get_into_layers']:assert callable(getattr(mc.DistributedObjectStore,name))
observed=[]
for dram in ['0GB','112GB','136GB','160GB']:
    cfg=mc.LocalConfig()
    for key in ['meta_service_url','config_store_url','log_level','world_size','protocol','dram_size','hbm_size','max_dram_size','max_hbm_size']:
        assert hasattr(cfg,key),key
        setattr(cfg,key,dram if key=='dram_size' else provider[key])
    assert cfg.protocol=='device_sdma' and cfg.world_size==128 and cfg.dram_size==dram and cfg.max_dram_size==size
    observed.append({'dram_size':cfg.dram_size,'max_dram_size':cfg.max_dram_size,'protocol':cfg.protocol,'world_size':cfg.world_size})
meta=mc.MetaConfig();meta.evict_threshold_high=90;meta.evict_threshold_low=89;assert meta.evict_threshold_high==90 and meta.evict_threshold_low==89
meta_text=(root/'config/meta.conf').read_text();local_text=(root/'config/local.conf').read_text()
for key,value in [('ock.mmc.meta.lease_ttl_ms','60000'),('ock.mmc.evict_threshold_high','90'),('ock.mmc.evict_threshold_low','89')]:assert f'{key} = {value}' in meta_text
assert 'ock.mmc.local_service.dram.size = 0GB' in local_text and f'ock.mmc.local_service.max.dram.size = {size}' in local_text
print('GLM53_SDK_CONFIG_PREFLIGHT='+json.dumps({'framework':args.framework,'module_paths':[mc.__file__,mf.__file__],'SDK_version':'1.3.0','configurations':observed,'store_objects_created':0,'SDK_init_calls':0,'NPU_allocations':0,'model_requests':0}),flush=True)
