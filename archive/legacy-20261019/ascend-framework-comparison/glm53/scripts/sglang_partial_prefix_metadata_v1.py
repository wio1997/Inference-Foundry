"""Pure-client SDK metadata snapshot of the two failed-run namespaces."""
import collections,hashlib,json,pathlib,struct,time
from importlib.metadata import version
from glm53_sglang_prefix_keys_v1 import page_hashes,component_keys
import memcache_hybrid as mc
assert version('memcache_hybrid')=='1.3.0' and mc.__file__.startswith('/opt/glm53/sdk13/')
assert hashlib.sha256(pathlib.Path('/run/glm53/glm53_sglang_prefix_keys_v1.py').read_bytes()).hexdigest()=='a2188e0205d85615a35ca3dd80a5339a7e3a28707c778ecc882aeab6ac3a15e7'
data=pathlib.Path('/semantic');assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()=='ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece'
record=json.loads((data/'semantic-0.json').read_text());store=mc.DistributedObjectStore()
out={'project_id':'glm53','init_bm':False,'NPU_devices':False,'metadata_only_not_payload_readback':True,'namespaces':[]}
out['init_rc']=store.init(0,init_bm=False);assert out['init_rc']==0
try:
    for suffix,field in [('cold','input_ids'),('warm','prefix_ids')]:
        ids=record[field];assert len(ids)==(80000 if suffix=='cold' else 74400)
        expected='74351693c90718dc41c10b101536a9c85fd10de00e041e36aeaf5beb07f92d79' if suffix=='cold' else '61cb81d7a85d8642f5ab52dd700c8232e04ae141c8d03b46a9d968107c2ced75'
        assert hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==expected
        ns='glm53-sem-sglang-v5-20261017-case0-'+suffix
        hashes=page_hashes(ids,ns);keys=component_keys(hashes);states=store.batch_is_exist(keys);assert len(states)==len(keys)
        existing=[i for i,state in enumerate(states) if state==1]
        selected=[keys[i] for i in existing];infos=store.batch_get_key_info(selected) if selected else []
        assert len(infos)==len(selected)
        kind_names=['k','v','index_k'];by_kind={kind:[i//3 for i in existing if i%3==offset] for offset,kind in enumerate(kind_names)}
        complete=[page for page in range(len(hashes)) if all(states[page*3+i]==1 for i in range(3))]
        contiguous=0
        for page in complete:
            if page!=contiguous:break
            contiguous+=1
        rows=[]
        for i,info in zip(existing,infos):
            if info is None:rows.append({'key_index':i,'snapshot_race_missing':True});continue
            size=int(info.size());assert size==[10354688,1294336,2555904][i%3]
            gvas=[int(v) for v in info.gva_list()];locations=[int(v) for v in info.loc_list()];assert any(v>0 for v in gvas)
            rows.append({'key_index':i,'page':i//3,'component':kind_names[i%3],'bytes':size,'gvas':gvas,'locations':locations})
        out['namespaces'].append({'namespace':ns,'full_input_complete_pages':len(hashes),
            'known_component_keys':len(keys),'existing_component_keys':len(existing),
            'status_counts':dict(collections.Counter(states)),'complete_pages':complete,
            'contiguous_complete_prefix_pages':contiguous,'existing_pages_by_component':by_kind,
            'object_bytes_observed':sum(row.get('bytes',0) for row in rows),'metadata':rows,
            'key_list_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()})
finally:
    out['close_rc']=store.close();out['terminal_monotonic_ns']=time.monotonic_ns()
    print('GLM53_PARTIAL_PREFIX_METADATA='+json.dumps(out),flush=True)
assert out['close_rc']==0
