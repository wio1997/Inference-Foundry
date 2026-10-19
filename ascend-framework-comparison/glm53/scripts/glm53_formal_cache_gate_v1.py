"""Documented SDK exact-key readiness plus bounded payload samples, no model calls.

Run only by a reviewed resource launcher. init_bm=True / dram0 is required for
payload reads; metadata-only initialization must never be represented as readback.
This verifies bytes can be retrieved, not full numerical equivalence of KV data.
"""
import argparse, hashlib, importlib.metadata as md, json, os, pickle, struct, time
from pathlib import Path

def id_sha(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()

def known_keys(framework,ids,namespace):
    assert len(ids)==74400 and namespace.startswith('glm53-formal-')
    if framework=='sglang':
        from glm53_sglang_prefix_keys_v1 import page_hashes,component_keys
        hashes=page_hashes(ids,namespace);keys=component_keys(hashes)
        sizes=[{'k':10354688,'v':1294336,'index_k':2555904}[k.rsplit('__',1)[1]] for k in keys]
        samples=list(range(3))+list(range(len(keys)-3,len(keys)))
    else:
        assert framework=='vllm' and os.environ.get('PYTHONHASHSEED')=='0'
        def h(x):return hashlib.sha256(pickle.dumps(x,protocol=pickle.HIGHEST_PROTOCOL)).digest()
        parent=h('0');hashes=[]
        for offset in range(0,len(ids)-127,128):
            parent=h((parent,tuple(ids[offset:offset+128]),(namespace,) if offset==0 else None));hashes.append(parent.hex())
        keys=['GLM-5.3-w8a8@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@'+digest for digest in hashes]
        sizes=[12369920]*len(keys);samples=[0,len(keys)-1]
    assert len(hashes)==581
    return keys,sizes,samples

def run(settings):
    import memcache_hybrid as sdk
    assert md.version('memcache_hybrid')=='1.3.0' and sdk.__file__.startswith(settings['expected_sdk_root'])
    root=Path(settings['output']);request=json.loads((root/'need-cache-gate.json').read_text());fw=request['framework'];C=request['concurrency'];namespace=request['namespace'];assert request['project_id']=='glm53' and fw==settings['framework'] and C in [16,32,64]
    data=Path(settings['dataset']);raw=(data/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==request['manifest_sha256'];manifest=json.loads(raw);assert manifest['concurrency']==C and len(manifest['prefixes'])==C
    report={'project_id':'glm53','framework':fw,'namespace':namespace,'prefix_count':C,'prefixes':[],'all_expected_keys_present':False,'actual_payload_samples_read':False,'pass':False,'model_requests':0,'SDK_init_bm':True,'SDK_dram_size':'0GB','scope':'Every complete-prefix key metadata verified; first/last complete-page payload sampled for each independent prefix. Byte hashes are repeatability/readability checks, not a full KV numerical oracle.'}
    cfg=sdk.LocalConfig()
    for key in ['meta_service_url','config_store_url','world_size','protocol','max_dram_size']:
        setattr(cfg,key,settings[key])
    cfg.dram_size='0GB';cfg.hbm_size='0GB';cfg.max_hbm_size='0GB';cfg.log_level='info';assert cfg.protocol=='device_sdma'
    store=sdk.DistributedObjectStore();report['setup_rc']=store.setup(cfg);assert report['setup_rc']==0
    report['init_dispatch_monotonic_ns']=time.monotonic_ns()
    try:
        report['init_rc']=store.init(settings['device_id'],init_bm=True);assert report['init_rc']==0
        for meta in manifest['prefixes']:
            pid=meta['prefix_id'];prefix=json.loads((data/f'prefix-{pid:02d}.json').read_text());ids=prefix['input_ids'];assert id_sha(ids)==meta['ids_sha256_le_u32']==prefix['ids_sha256_le_u32']
            keys,sizes,samples=known_keys(fw,ids,namespace);deadline=time.monotonic()+120;polls=0
            while True:
                polls+=1;states=store.batch_is_exist(keys);assert len(states)==len(keys)
                if all(x==1 for x in states):break
                if time.monotonic()>deadline:raise TimeoutError(f'Incomplete known prefix keys: p{pid}, found{sum(x==1 for x in states)}/{len(keys)}')
                time.sleep(.2)
            infos=store.batch_get_key_info(keys);assert len(infos)==len(keys)
            assert all(info is not None and int(info.size())==size and any(int(x)>0 for x in info.gva_list()) for info,size in zip(infos,sizes))
            row={'prefix_id':pid,'prefix_ids_sha256_le_u32':meta['ids_sha256_le_u32'],'keys':len(keys),'key_list_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),'unique_object_bytes':sum(sizes),'metadata_polls':polls,'samples':[]}
            report['prefixes'].append(row)
            for index in samples:
                key=keys[index];sample={'key_index':index,'key_sha256_utf8':hashlib.sha256(key.encode()).hexdigest(),'expected_bytes':sizes[index]};row['samples'].append(sample)
                for repeat in [1,2]:
                    begin=time.monotonic_ns();value=store.get(key);end=time.monotonic_ns();assert isinstance(value,bytes) and len(value)==sizes[index],'DocumentedSDKget payload read failed; do not substitute metadata presence'
                    digest=hashlib.sha256(value).hexdigest();sample[f'read{repeat}']={'bytes':len(value),'sha256':digest,'dispatch_monotonic_ns':begin,'terminal_monotonic_ns':end};del value
                assert sample['read1']['sha256']==sample['read2']['sha256'],'Immutable prefix sample changed during readback'
            (root/'cache-gate-progress.json').write_text(json.dumps(report,indent=2)+'\n')
        # Recheck all prefix key sets together after readback: an earlier prefix
        # may have been evicted while later prefixes were being inspected.
        for meta in manifest['prefixes']:
            prefix=json.loads((data/f'prefix-{meta["prefix_id"]:02d}.json').read_text());keys,_,_=known_keys(fw,prefix['input_ids'],namespace);states=store.batch_is_exist(keys);assert len(states)==len(keys) and all(x==1 for x in states),'All-prefix retention failed before formal dispatch'
        report['all_expected_keys_present']=True;report['actual_payload_samples_read']=True;report['pass']=True
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        report['close_rc']=store.close();report['terminal_monotonic_ns']=time.monotonic_ns();report['pass']=bool(report['pass'] and report['close_rc']==0)
        temp=root/'cache-gate.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(root/'cache-gate.json');print('GLM53_FORMAL_CACHE_GATE='+json.dumps({k:v for k,v in report.items() if k!='prefixes'}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--settings',required=True);p.add_argument('--sha256',required=True);a=p.parse_args();raw=Path(a.settings).read_bytes();assert hashlib.sha256(raw).hexdigest()==a.sha256;run(json.loads(raw))
if __name__=='__main__':main()
