"""Join saved Post/Pre value versions, MoE M/P operands and exact Pad siblings."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
 assert not(O/'REVIEW-CHECKPOINT-v147.json').exists()
 t=read(O/'hc-sp-eager-native-frontiers-v134.json');m=read(O/'hc-moe-native-frontiers-v141.json');d=read(O/'hc-draft-native-frontiers-v142.json');mr={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in m['records']};bindings=[];seen={};roles=Counter();apionly=[];intra=[];cross=[];dintra=[];dcross=[];pads=[];refs={r['database_index']:r for r in d['databases']}
 def add(di,s,role):
  roles[role]+=1
  if s['TASK']is None:
   apionly.append(dict(database_index=di,role=role,API=s['API'],source_frames=s.get('source_frames',s.get('selected_frames'))));return None
  k=di,s['TASK']['task_record_rowid']
  if k not in seen:
   seen[k]=len(bindings);bindings.append(dict(database_index=di,API=s['API'],TASK=s['TASK'],definition=s['definition'],selected_frames=s.get('selected_frames',s.get('source_frames')),roles=[role]))
  elif role not in bindings[seen[k]]['roles']:bindings[seen[k]]['roles'].append(role)
  return seen[k]
 for r in t['records']:
  di=r['database_index'];q=mr[di,r['forward_API_rowid'],r['source_block_ordinal']];st=q['stages'];intra.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],block=r['source_block_ordinal'],left=add(di,r['stages']['POST'],'TARGET_ATTN_POST'),right=add(di,r['stages']['FFN_PRE'],'TARGET_FFN_PRE'),earlier_premix_and_Post_coeff_provider=add(di,r['stages']['HC'],'TARGET_ATTN_PRE_PROVIDER'),norm=add(di,st['FFN_NORM'],'TARGET_FFN_NORM'),consumers={k:add(di,st[k],k)for k in ['ROUTER','MC2','SHARED_QUANT','SHARED_DOWN','MERGE']},future_Post=add(di,r['stages']['FFN_POST'],'TARGET_FFN_POST'),actual_tensor_alias_values_or_new_tile_publication_bound=False))
 for r in m['bridges']:
  di=r['database_index'];cross.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],from_block=r['from_block'],to_block=r['to_block'],left=add(di,r['previous_FFN_Post'],'CROSS_FFN_POST'),right=add(di,r['next_HC'],'CROSS_ATTN_PRE'),incoming_premix_provider=add(di,r['previous_FFN_Pre'],'CROSS_PREMIX_PROVIDER'),Engram_positive_APIs=r['next_Engram_positive_APIs'],aux_mean=add(di,r['next_aux_mean'],'AUX_MEAN')if r['next_aux_mean']else None,aux_allGather=r['next_aux_allGather'],all_raw_state_readers_or_actual_bank_versions_closed=False))
 for r in d['records']:
  di=r['database_index'];ls=r['layers']
  for j,l in enumerate(ls):
   s=l['stages'];dintra.append(dict(database_index=di,source_order_ordinal=r['source_order_ordinal'],block=j,ancestry=r['source_ancestry'],left=add(di,s['POST'],'DRAFT_ATTN_POST'),right=add(di,s['FFN_PRE'],'DRAFT_FFN_PRE'),earlier_premix_and_Post_coeff_provider=add(di,s['HC'],'DRAFT_ATTN_PRE_PROVIDER'),norm=add(di,s['FFN_NORM'],'DRAFT_FFN_NORM'),future_Post=add(di,s['FFN_POST'],'DRAFT_FFN_POST'),actual_new_handoff_or_alias_bound=False))
   if j:
    ps=ls[j-1]['stages'];dcross.append(dict(database_index=di,source_order_ordinal=r['source_order_ordinal'],from_block=j-1,to_block=j,left=add(di,ps['FFN_POST'],'DRAFT_CROSS_POST'),right=add(di,s['HC'],'DRAFT_CROSS_PRE'),incoming_premix_provider=add(di,ps['FFN_PRE'],'DRAFT_CROSS_PREMIX'),Engram_in_source_draft_constructor=False,actual_bank_reader_versions_closed=False))
 grouped=defaultdict(list)
 for r in intra:grouped[r['database_index']].append(r)
 for di,rr in sorted(grouped.items()):
  ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs=defaultdict(list)
  for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
   v=ss.get(x['stack'])
   if v and '/ops/fused_moe/prepare_finalize.py('in v:fs[x['id']].append(dict(x,value=v))
  apis=[dict(x,name_text=ss[x['name']])for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')if x['callchainId']in fs];cached=read(Path(ref['cached_native']['path']));defs={x['globalTaskId']:x for x in cached['COMPUTE_definitions']};byapi=defaultdict(list)
  for x in cached['TASK_records']:
   for aid in x.get('native_linked_API_rowids',[]):byapi[aid].append(x)
  for r in rr:
   m0=int(bindings[r['norm']]['definition']['inputShapes_text'].strip('"').split(',')[0]);p0=int(bindings[r['consumers']['MC2']]['definition']['inputShapes_text'].strip('"').split(',')[0]);a0,a1=bindings[r['norm']]['API'],bindings[r['consumers']['MC2']]['API'];scope=[a for a in apis if a['globalTid']==a0['globalTid']and int(a0['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(a1['startNs'])]
   for role,line,rid in [('HIDDEN_PAD',282,'OPR-23e591f55ada70b8'),('LOGITS_PAD',283,'OPR-c02a1d7e2f0e1c91'),('MASK_PAD',284,'OPR-5e17b8f72308562b')]:
    aa=[a for a in scope if a['name_text']=='aclnnConstantPadNd'and any(f'/prepare_finalize.py({line}):'in f['value']for f in fs[a['callchainId']])]
    if m0==p0:assert not aa;continue
    assert m0==800 and p0==816 and len(aa)==1,(di,r['forward_API_rowid'],r['block'],role,len(aa));a=aa[0];ii=[]
    for task in byapi[a['api_rowid']]:
     de=defs.get(task['globalTaskId'])
     if de:
      s=dict(API=a,TASK=task,definition={k:de[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},selected_frames=fs[a['callchainId']]);ii.append(add(di,s,role+'_'+de['opType_text']))
    ops=Counter(bindings[i]['definition']['opType_text']for i in ii);assert ops['PadV3']==ops['MemSet']==1
    pads.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],block=r['block'],role=role,source_line=line,API=a,M=m0,P=p0,sibling_bindings=ii,PadV3=next(i for i in ii if bindings[i]['definition']['opType_text']=='PadV3'),MemSet=next(i for i in ii if bindings[i]['definition']['opType_text']=='MemSet'),pending_MemSet_regime=rid,MemSet_address_length_workspace_or_destination_role_inferred=False,exact_shared_API_not_equal_storage_or_zero_cost=True))
  db.close()
 out=dict(target_intra=intra,target_cross=cross,draft_intra=dintra,draft_cross=dcross,pad_groups=pads,bindings=bindings,API_only=apionly,databases=[dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path']))for di,ref in sorted(refs.items())],unique_TASK_bindings=len(bindings),role_binding_count=sum(roles.values()),roles=dict(roles),source_Post_Pre_edges=len(intra)+len(cross)+len(dintra)+len(dcross),new_pad_sibling_TASK_bindings=sum(len(r['sibling_bindings'])for r in pads),new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 assert len(intra)==960 and len(cross)==936 and len(dintra)==312 and len(dcross)==208 and len(pads)==960
 write(O/'post-pre-native-frontiers-v147.json',out);print(json.dumps({k:v for k,v in out.items()if k in ['unique_TASK_bindings','role_binding_count','source_Post_Pre_edges','new_pad_sibling_TASK_bindings','new_native_operand_or_phase_promotions','new_device_work','overall_complete']}))


if __name__=='__main__':main()
