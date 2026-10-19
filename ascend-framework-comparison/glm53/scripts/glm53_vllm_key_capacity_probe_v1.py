"""Known-key metadata only: identify unique stored objects versus TP read descriptors."""
import hashlib,json,os,pickle,struct,sys,time
request=json.load(sys.stdin);assert request['project_id']=='glm53' and len(request['cases'])==2
import memcache_hybrid as mc
assert mc.__file__.startswith('/opt/glm53/sdk13/')
def h(x):return hashlib.sha256(pickle.dumps(x,protocol=pickle.HIGHEST_PROTOCOL)).digest()
store=mc.DistributedObjectStore();assert store.init(0,init_bm=False)==0
report={'project_id':'glm53','init_bm':False,'metadata_only':True,'model_requests':0,'cases':[]}
try:
 for i,case in enumerate(request['cases']):
  ids=case['prefix_ids'];ns=f'glm53-sem-vllm-v4-20261017-case{i}-warm';assert case['namespace']==ns and len(ids)==74400
  assert hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==case['prefix_sha256_le_u32']
  parent=h(os.environ.get('PYTHONHASHSEED','vllm-none-hash'));hashes=[]
  for j in range(0,len(ids)-127,128):parent=h((parent,tuple(ids[j:j+128]),(ns,) if j==0 else None));hashes.append(parent.hex())
  assert len(hashes)==581;row={'namespace':ns,'heads':[]};identities=set()
  for head in range(8):
   keys=[f'GLM-5.3-w8a8@dcp:0@head_or_tp_rank:{head}@pp_rank:0@group:0@cache_role:kv@cache_family:default@'+v for v in hashes]
   states=store.batch_is_exist(keys);assert len(states)==581;found=[k for k,s in zip(keys,states) if s==1];infos=store.batch_get_key_info(found) if found else [];assert len(infos)==len(found)
   sizes=[];gva_count=0
   for info in infos:
    if info is None:sizes.append(0);continue
    sizes.append(int(info.size()));gvas=tuple(int(v) for v in info.gva_list());locs=tuple(int(v) for v in info.loc_list());identities.add((locs,gvas,int(info.size())));gva_count+=int(any(gvas))
   row['heads'].append({'head_or_tp_rank':head,'keys':len(keys),'key_list_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),'first32_key_list_sha256':hashlib.sha256(json.dumps(keys[:32],separators=(',',':')).encode()).hexdigest(),'existing_keys':sum(x==1 for x in states),'exist_status_values':sorted(set(states)),'object_bytes':sum(sizes),'object_sizes':sorted(set(sizes)),'nonzero_gva_objects':gva_count})
  row['unique_location_gva_size_identities']=len(identities);row['object_bytes_all_found_heads']=sum(x['object_bytes'] for x in row['heads']);report['cases'].append(row)
finally:
 report['close_rc']=store.close();print('GLM53_KEY_CAPACITY_METADATA='+json.dumps(report),flush=True)
assert report['close_rc']==0
