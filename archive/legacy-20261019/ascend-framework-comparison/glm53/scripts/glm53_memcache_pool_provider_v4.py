"""Own standalone MemCache pool via documented SDK1.3 API; no model import."""
import argparse,hashlib,importlib.metadata,json,os,signal,threading,time
from pathlib import Path


def validate(config, version):
    assert config['project_id']=='glm53' and version=='1.3.0'
    assert config['protocol']=='device_sdma' and config['init_bm'] is True
    assert config['hbm_size']=='0GB' and config['max_hbm_size']=='0GB'
    assert config['device_id']==0 and config['world_size']==128
    capacities={'sglang':({34:'136GB',35:'112GB',36:'160GB',41:'160GB'},'160GB'),
                'vllm':({30:'192GB',31:'192GB',43:'128GB',45:'56GB'},'192GB')}
    assert config['framework'] in capacities
    nodes,maximum=capacities[config['framework']]
    assert config['node'] in nodes
    expected=nodes[config['node']]
    assert config['dram_size']==expected and config['max_dram_size']==maximum
    assert config['status_path']=='/run/glm53/provider-status.json'
    return expected


def run(config, sdk, version):
    expected=validate(config,version)
    output=Path(config['status_path']);assert not output.exists()
    stopped=threading.Event()
    for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,lambda *_:stopped.set())
    state={'project_id':'glm53','framework':config['framework'],'pid':os.getpid(),
           'sdk_version':version,'sdk_module':sdk.__file__,'configuration':config,
           'phase':'PREPARING','events':[]}
    def emit(event,**fields):
        row={'event':event,'monotonic_ns':time.monotonic_ns(),**fields};state['events'].append(row);state['phase']=event
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(output)
        print('GLM53_POOL_PROVIDER='+json.dumps(row),flush=True)
    cfg=sdk.LocalConfig()
    for field in ['meta_service_url','config_store_url','log_level','world_size','protocol',
                  'dram_size','hbm_size','max_dram_size','max_hbm_size']:
        assert hasattr(cfg,field),field
        setattr(cfg,field,config[field])
    store=sdk.DistributedObjectStore();initialized=False
    try:
        emit('SETUP_START');rc=store.setup(cfg);emit('SETUP_COMPLETE',rc=rc);assert rc==0
        emit('INIT_START');rc=store.init(config['device_id'],init_bm=True);emit('INIT_COMPLETE',rc=rc);assert rc==0;initialized=True
        for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,lambda *_:stopped.set())
        local_id=int(store.get_local_service_id());emit('READY',local_service_id=local_id,configured_dram_bytes=int(expected[:-2])*1024**3,actual_capacity_qualification='Requires native allocation/segment audit; init0 alone is not measured capacity')
        stopped.wait()
    except BaseException as error:
        emit('FAILED',error=repr(error));raise
    finally:
        emit('CLOSE_START',init_completed=initialized);rc=store.close();emit('CLOSE_COMPLETE',rc=rc);assert rc==0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--sha256',required=True);args=parser.parse_args()
    raw=Path(args.config).read_bytes();assert hashlib.sha256(raw).hexdigest()==args.sha256
    config=json.loads(raw)
    validate(config,'1.3.0')
    import memcache_hybrid as sdk
    assert sdk.__file__.startswith(config['expected_sdk_root'])
    run(config,sdk,importlib.metadata.version('memcache_hybrid'))
if __name__=='__main__':main()
