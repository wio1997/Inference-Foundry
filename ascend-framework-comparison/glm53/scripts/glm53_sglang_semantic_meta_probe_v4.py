"""Hardware-free SDK metadata readiness for exact native B103 prefix component keys."""
import hashlib,json,sys,time,struct,pathlib
from importlib.metadata import version
from glm53_sglang_prefix_keys_v1 import page_hashes,component_keys
req=json.load(sys.stdin);index=req['index'];assert type(index) is int and index in [0,1]
ns=f'glm53-sem-sglang-shared-v8-20261017-case{index}-warm';assert req['namespace']==ns
ids=req['prefix_ids'];expected=['61cb81d7a85d8642f5ab52dd700c8232e04ae141c8d03b46a9d968107c2ced75','0b9df0586836fac65d90ed0126d197183932708eeb3ed726522e0ff029ce0fa3'][index]
assert len(ids)==74400 and hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==req['prefix_sha256_le_u32']==expected
assert hashlib.sha256(pathlib.Path('/run/glm53/glm53_sglang_prefix_keys_v1.py').read_bytes()).hexdigest()=='a2188e0205d85615a35ca3dd80a5339a7e3a28707c778ecc882aeab6ac3a15e7'
hashes=page_hashes(ids,ns);assert len(hashes)==581;keys=component_keys(hashes);assert len(keys)==1743
import memcache_hybrid as mc
assert version('memcache_hybrid')=='1.3.0' and mc.__file__.startswith('/opt/glm53/sdk13/')
store=mc.DistributedObjectStore();out={'project_id':'glm53','namespace':ns,'init_bm':False,'NPU_devices':False,'dispatch_monotonic_ns':time.monotonic_ns(),'complete_pages':581,'known_component_keys':1743,'known_keys_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),'first_key':keys[0],'last_key':keys[-1],'component_order':['k','v','index_k'],'expected_component_bytes':[10354688,1294336,2555904]}
out['init_rc']=store.init(0,init_bm=False);assert out['init_rc']==0
try:
 deadline=time.monotonic()+120;polls=0
 while True:
  polls+=1;states=store.batch_is_exist(keys);assert len(states)==1743
  out['last_existing_component_keys']=sum(s==1 for s in states)
  if all(s==1 for s in states):break
  if time.monotonic()>deadline:raise TimeoutError('Exact1743nativeprefixcomponentkeys notallwrite-complete/readable in120s')
  time.sleep(.2)
 infos=store.batch_get_key_info(keys);assert len(infos)==1743
 sizes=[int(info.size()) if info is not None else 0 for info in infos];out['object_bytes']=sum(sizes);out['polls']=polls;out['exist_status_values']=sorted(set(states));out['component_size_values']={kind:sorted(set(sizes[i::3])) for i,kind in enumerate(out['component_order'])}
 assert sizes==out['expected_component_bytes']*581 and out['object_bytes']==8253063168
 assert all(info is not None and any(int(v)>0 for v in info.gva_list()) for info in infos)
 out['locations_by_component_values']={kind:sorted({int(v) for info in infos[i::3] for v in info.loc_list()}) for i,kind in enumerate(out['component_order'])};out['pass']=True
finally:
 out['close_rc']=store.close();out['terminal_monotonic_ns']=time.monotonic_ns();print('GLM53_PREFIX_METADATA='+json.dumps(out),flush=True)
assert out['close_rc']==0
