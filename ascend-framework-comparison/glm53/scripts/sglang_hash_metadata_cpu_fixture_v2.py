"""Native B103 C++/RadixKey hash oracle and hardware-free SDK1.3 metadata client."""
import ast,hashlib,importlib.util,json,pathlib,subprocess,sys,sysconfig,typing
from array import array
ROOT=pathlib.Path('/run/glm53')
def extract(path,name):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]);ast.fix_missing_locations(mod)
    scope={'array':array,'hashlib':hashlib,'Optional':typing.Optional};exec(compile(mod,str(path),'exec'),scope);return scope[name]

report={'project_id':'glm53','CPU_only':True,'model_requests':0,'NPU_devices':False,'hash_cases':[]}
try:
    cpp=ROOT/'hash_binding.cpp';radix=ROOT/'radix_cache.py';native=ROOT/'native_hash.py';utils=ROOT/'utils.py'
    for name,sha in json.loads((ROOT/'native-sha.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    ref_spec=importlib.util.spec_from_file_location('reference',ROOT/'glm53_sglang_prefix_keys_v1.py');ref=importlib.util.module_from_spec(ref_spec);ref_spec.loader.exec_module(ref)
    RadixKey=extract(radix,'RadixKey');native_input=extract(native,'_native_hash_input');seed=extract(utils,'storage_namespace_seed')
    for n in [0,1,127,128,129,74400,80000]:
        ids=[(i*104729+31)%180000 for i in range(n)]
        for bigram in [False,True]:
            for namespace,extra in [(None,None),('',None),('glm53-hash-fixture-汉字','lora-fixture')]:
                key=RadixKey(array('I',ids),cache_salt=namespace,extra_key=extra);key,_=key.maybe_to_bigram_view(bigram)
                raw,logical,width,is_bigram=native_input(key);prior=seed(key.extra_key,key.cache_salt);prior=bytes.fromhex(prior) if prior else None
                assert prior==ref.namespace_seed(extra,namespace)
                assert logical==(max(n-1,0) if bigram else n) and is_bigram==bigram
                report['hash_cases'].append({'raw_tokens':n,'bigram':bigram,'namespaced':namespace or None,'extra_key':extra,'logical_units':logical,'pass':True})
    report['native_input_namespace_AST_pass']=len(report['hash_cases'])==42
    report['compiled_CXX_oracle']='NOT_RUN:1430rc1CPUimage lacksOpenSSLdevheaders; useobservedrealnativeknownkeysasendtoendoracle.'
    (ROOT/'hash-result.json').write_text(json.dumps(report,indent=2)+'\n')
    import memcache_hybrid as mc
    from importlib.metadata import version
    assert version('memcache_hybrid')=='1.3.0' and mc.__file__.startswith('/opt/glm53/sdk13/')
    s=mc.DistributedObjectStore();rc=s.init(0,init_bm=False);report['SDK_metadata_init_rc']=rc;assert rc==0
    try:
        absent=s.batch_is_exist(['glm53-cpu-hash-meta-fixture-v1-not-created']);assert absent==[0];report['absent_owned_key_status']=absent
        from tokenizers import Tokenizer
        p=pathlib.Path('/tokenizer.json');assert hashlib.sha256(p.read_bytes()).hexdigest()=='19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
        text=('Inventory verification. The code maple means 17, cedar means 23, and river means 41. Read the complete record and explain how to check that the numbers add up. A reliable answer uses each stated number exactly once and shows the result.\n')*40
        ids=Tokenizer.from_file(str(p)).encode(text,add_special_tokens=False).ids[:512]
        hashes=ref.page_hashes(ids,'glm53-sg-v4-short-v2-20261017-p00');assert len(hashes)==3
        keys=ref.component_keys(hashes);states=s.batch_is_exist(keys);assert len(states)==9
        infos=s.batch_get_key_info([k for k,v in zip(keys,states) if v==1]) if 1 in states else []
        report['observed_known_short_prefix']={'keys':9,'existing':sum(v==1 for v in states),'exist_status_values':sorted(set(states)),'key_list_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),'component_sizes':[int(info.size()) if info is not None else 0 for info in infos],'metadata_observation_only':'Smallpromptbackupthresholdmaypreventstorage; onlyfull74400warmwillbearequiredreadinessgate.'}
    finally:report['SDK_metadata_close_rc']=s.close()
    assert report['SDK_metadata_close_rc']==0;report['pass']=True
finally:
    (ROOT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_HASH_METADATA_FIXTURE='+json.dumps(report),flush=True)
