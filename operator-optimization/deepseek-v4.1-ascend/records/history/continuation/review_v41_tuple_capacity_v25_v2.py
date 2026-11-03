from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import json,hashlib,ast
from finalize_origin_coverage_v9 import N,O,R

def main():
 out=O/'V41-tuple-capacity-and-F34-graph-review-v25-v2.json';assert not out.exists()
 s=R/'whole-scope-20261030/frozen-source-snapshot'
 remote_refs={}
 for label in ['C2-cache-spec166-v25','C2-allocation166-v25']:
  rec=json.loads((N/(label+'.json')).read_text());assert rec['returncode']==0
  d=json.loads(rec['stdout'])
  for f in d['result']['files']:remote_refs[f['path']]=f['sha256']
 specs=[('vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py',{'is_deepseek_v41_cache','get_layer_tuples','group_cache_specs','make_cache_groups','_specs_from_groups','get_deepseek_v41_pool_bytes_per_block','get_deepseek_v41_kv_cache_config'},[]),('vllm-ascend/vllm_ascend/worker/model_runner_v1.py',set(),[(4741,4779),(4946,4978),(5387,5461)])]
 files=[]
 for rel,names,ranges in specs:
  p=s/rel;b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();remote='/vllm-workspace/'+rel
  assert remote_refs[remote]==sha,(rel,sha)
  ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
  for f in ast.walk(tree):
   if isinstance(f,ast.FunctionDef)and f.name in names:
    if p.name=='kv_cache_interface.py':
     owners=[c.name for c in ast.walk(tree)if isinstance(c,ast.ClassDef)and c.lineno<=f.lineno<=c.end_lineno]
     if owners!=['AttentionSpec']:continue
    fs.append(dict(name=f.name,start=f.lineno,end=f.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(f.lineno-1,f.end_lineno)]))
  points={i for a,z in ranges for i in range(a-1,z)}
  files.append(dict(local_snapshot=str(p),installed_path=remote,sha256=sha,functions=fs,exact_ranges=ranges,range_lines=[str(i+1)+': '+ls[i]for i in sorted(points)]))
 # Max-operand invariance is the argument; finite examples only check arithmetic.
 tests=[]
 for swa in [131072,147712,262144]:
  for ring in [65536,131072]:
   for newring in [0,32768,65536]:
    for long_index_folded in [0,73856,147712,300000]:
     if not(0<=newring<=ring<=swa):continue
     before=max(swa,ring,long_index_folded);after=max(swa,newring,long_index_folded);assert before==after
     tests.append(dict(swa=swa,ring=ring,newring=newring,long_index_folded=long_index_folded,capacity_before=before,capacity_after=after))
 example=dict(conditional_SWA_BF16_page=[128,512],SWA_page_bytes=128*512*2,ring_before_FP32_page=[32,1024],ring_before_bytes=32*1024*4,ring_after_FP32_page=[16,1024],ring_after_bytes=16*1024*4)
 assert example['SWA_page_bytes']==example['ring_before_bytes']==131072 and example['ring_after_bytes']==65536
 old=json.loads((O/'F34-C2-ring-observability-candidate-v24.json').read_text());declared=set(old['candidate_graph']['nodes']);bad=[e for e in old['candidate_graph']['retained_edges']if e['a']not in declared or e['b']not in declared]
 assert len(bad)==2
 new=deepcopy(old)
 nodes=[dict(id=i,kind=k)for i,k in [('zero_pooled','VALUE_INIT'),('pool','STATEFUL_POOL'),('ring_update','STATE_MAINTENANCE'),('norm','VALUE_TRANSFORM'),('index_projection','OLD_LATENT_READER'),('index_publish','PERSISTENT_PUBLISH'),('long_publish','ROTARY_AND_PERSISTENT_PUBLISH'),('future_pool','EXTERNAL_STATE_READER'),('future_index','EXTERNAL_INDEX_READER'),('future_attention','EXTERNAL_CACHE_READER')]]
 edges=[dict(a=a,b=b,kind=k,value=v)for a,b,k,v in [('zero_pooled','pool','WAW','pooled_completed_rows'),('zero_pooled','norm','RAW','pooled_unproduced_zero_rows'),('pool','norm','RAW','pooled_BF16'),('pool','ring_update','WAR','old_ring_same_slot_read'),('ring_update','norm','CURRENT_SCHEDULE_CONTROL','retained_for_F34_alone;F18_fork_is_separate'),('ring_update','future_pool','PERSISTENT_STATE_GENERATION','even_ring_values'),('norm','index_projection','RAW','latent_before_long_rope'),('index_projection','index_publish','RAW','index_key'),('index_projection','long_publish','WAR','complete_old_latent_read_before_inplace_long_rope'),('norm','long_publish','RAW','latent'),('index_publish','long_publish','CURRENT_SCHEDULE_CONTROL','retained_for_F34_alone;F19_fork_is_separate'),('index_publish','future_index','PERSISTENT_STATE_GENERATION','INT8_key_and_FP16_scale'),('long_publish','future_attention','PERSISTENT_STATE_GENERATION','long_KV')]]
 ids={r['id']for r in nodes};assert all(e['a']in ids and e['b']in ids for e in edges)
 new['candidate_graph']=dict(nodes=nodes,retained_edges=edges,external_inputs=['FP32_kv','FP32_score','device_metadata','ring_old_even_values','RMS_parameters','slot_and_position_metadata'],scope='Explicit original source operation/refinement graph withexternalreaders;F34-Aonlychangesringwriteobservability,not scheduling. F18forkisaseparateconditionalcomposition.')
 new['candidate_graph']['alias_ownership_constraints']=['V41tupleviews can aliasSWA/long/index/ring backing;storageobjectdisjointnessisnot assumed.','Onlyring-ownedglobalblockIDs/pagesmaybewritten;retainexclusiveownership,generation andproperinitialization beforea pageisreusedbyanothercachetype.','F34-Akeepscurrentordering;F18/F19schedulechangesareseparateconditionalcompositions.']
 new['allocation_v25']=dict(current_tuple_contract='EachV41descriptorhasownbacking;itslayernamesaliasit. RingandSWAcanaliassamephysicalpagebytesatdifferentgloballyownedblockIDs.',page_capacity='max(long+index+optional_folded,eachalias_unpadded_page)',conditional_compaction_exclusion='IfSWAanchor>=oldringpage,shrinkingringalonecannotshrinktuplepagecapacity. Thenfixedgroups/fixedbudget/overridegiveunchangednum_blocks,descriptorsizesandtotalpoolallocation.',scope='Sourceconstructorcontract plusprofile-consistentpagegeometry,not actualpointer/groupconfiguration snapshot.',native_reader_and_ownership_closure=False)
 new['graph_review_v25']=dict(previous_shorthand_undefined_endpoint_count=2,correction='Declareexternalreaders anduseconsistentoperationIDs;addthevalue/publication/WAR edges andretainoriginalscheduleedgeforF34alone.',structural_validation_pass=True,semantic_native_proof=False)
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=files,universal_capacity_argument=['LetSbealowerboundgivenbyanunchangedSWAaliasandRtheoldringpage. Assume0<=R_new<=R<=S.','Foranylong/index/foldedpayloadL>=0,max(L,S,R)=max(L,S)=max(L,S,R_new).','Unchangedtuplecapacitiesimplyunchangedsum(page_sizes);withsameavailable_memory/override,num_blocksandper-descriptorN*page_sizeareunchanged.','A16slotviewmayalsobephysicallypadded;currenthelperstillrequires32rowsandcontiguity. Thisisnotaflag-onlynativechange.'],finite_capacity_cases=len(tests),profile_consistent_page_example=example,
  limits=['Not an actualruntimegroup/pointer snapshot.','No claim thatgenericmax-group allocatorappliestotheV41four-tuple path.','Fullringlogicalarraybytesarealiasviewaccounting,notadditiveextraallocations.','Filteringringwritesdoesnotfreeallocatedpagesbyitself.','Differenttuplemembership/SWArepresentation/globalpoolpolicyisaseparatewhole-cacheoptimizationdomain requiring allconsumers/ownership/lifetimes.'],
  old_graph_undefined_edges=bad,corrected_candidate=new,no_model_import_or_device_or_service_work=True,overall_complete=False)
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(output=str(out),finite_capacity_cases=len(tests),old_graph_undefined_edges=len(bad),corrected_graph_nodes=len(nodes),corrected_graph_edges=len(edges),overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
