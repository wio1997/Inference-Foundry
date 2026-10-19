"""Wait for all581 exact known-model-prefix keys to be readable; metadata-only."""
import hashlib,json,os,pathlib,pickle,struct,sys,time
request=json.load(sys.stdin);index=request['index'];assert type(index) is int and index in (0,1)
NS='glm53-sem-vllm-shared-v1-20261017T1774-case'+str(index)+'-warm';assert request['namespace']==NS
expected=['61cb81d7a85d8642f5ab52dd700c8232e04ae141c8d03b46a9d968107c2ced75','0b9df0586836fac65d90ed0126d197183932708eeb3ed726522e0ff029ce0fa3'][index]
ids=request['prefix_ids'];assert len(ids)==74400 and hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==request['prefix_sha256_le_u32']==expected
def h(x):return hashlib.sha256(pickle.dumps(x,protocol=pickle.HIGHEST_PROTOCOL)).digest()
assert os.environ.get('PYTHONHASHSEED')=='0'
parent=h('0');keys=[]
for i in range(0,len(ids)-127,128):
 parent=h((parent,tuple(ids[i:i+128]),(NS,) if i==0 else None));keys.append('GLM-5.3-w8a8@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@'+parent.hex())
import memcache_hybrid as mc
assert __import__('importlib.metadata',fromlist=['version']).version('memcache_hybrid')=='1.3.0' and mc.__file__.startswith('/opt/glm53/sdk13/')
s=mc.DistributedObjectStore();out={'project_id':'glm53','namespace':NS,'init_bm':False,'dispatch_monotonic_ns':time.monotonic_ns(),'known_key_count':len(keys),'known_keys_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),'first_key':keys[0],'last_key':keys[-1]};out['init_rc']=s.init(0,init_bm=False);assert out['init_rc']==0
try:
 deadline=time.monotonic()+120;polls=0
 while True:
  polls+=1;states=s.batch_is_exist(keys);assert len(states)==581
  if all(x==1 for x in states):break
  if time.monotonic()>deadline:raise TimeoutError('Known581prefix keys notall write-complete/readable within120s')
  time.sleep(.2)
 infos=s.batch_get_key_info(keys);assert len(infos)==581
 out['polls']=polls;out['exist_status_values']=sorted(set(states));out['bytes_by_key']=[x.size() if x is not None else 0 for x in infos];out['locations_by_key']=[x.loc_list() if x else [] for x in infos];out['types_by_key']=[x.type_list() if x else [] for x in infos]
 assert all(x is not None and x.size()==12369920 and any(v>0 for v in x.gva_list()) for x in infos)
 out['object_bytes']=sum(out['bytes_by_key']);out['pass']=True
finally:out['close_rc']=s.close();out['terminal_monotonic_ns']=time.monotonic_ns();print('GLM53_PREFIX_METADATA='+json.dumps(out),flush=True)
assert out['close_rc']==0
