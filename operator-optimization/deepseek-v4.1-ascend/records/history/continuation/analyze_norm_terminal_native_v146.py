"""Saved-profile source/CID bindings for normalization, terminal and state cuts."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
 assert not(O/'REVIEW-CHECKPOINT-v146.json').exists()
 target=read(O/'hc-sp-eager-native-frontiers-v134.json');draft=read(O/'hc-draft-native-frontiers-v142.json');moe=read(O/'hc-moe-native-frontiers-v141.json');attn=read(O/'hc-norm-attention-native-frontiers-v145.json');aa={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in attn['records']};tr=defaultdict(list);dr=defaultdict(list);br=defaultdict(list);refs={r['database_index']:r for r in draft['databases']};targets=[];drafts=[];aux=[];bridges=[];bindings=[];comms=[];pins=[];seen={};counts=Counter();roles=Counter()
 for r in target['records']:tr[r['database_index']].append(r)
 for r in draft['records']:dr[r['database_index']].append(r)
 for r in moe['bridges']:br[r['database_index']].append(r)
 roots={r['database_index']:[] for r in target['forward_roots']}
 for r in target['forward_roots']:roots[r['database_index']].append(r)
 for di,ref in sorted(refs.items()):
  raw=Path(ref['raw_database']['path']);cp=Path(ref['cached_native']['path']);db=sqlite3.connect(raw.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs=defaultdict(list);unresolved=set()
  for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
   v=ss.get(x['stack'])
   if v is None:unresolved.add(x['stack'])
   else:fs[x['id']].append(dict(x,value=v))
  apis={x['api_rowid']:dict(x,name_text=ss[x['name']])for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};rawtasks=[dict(x)for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')];data=read(cp);ts={x['task_record_rowid']:x for x in data['TASK_records']};defs={x['globalTaskId']:x for x in data['COMPUTE_definitions']};byapi=defaultdict(list);ci=defaultdict(set)
  for x in db.execute('SELECT *FROM CONNECTION_IDS'):ci[x['id']].add(x['connectionId'])
  for t in ts.values():
   for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
  def source(a,path,line):return any(path+f'({line}):'in f['value']for f in fs[a['callchainId']])
  def add(a,t,role,new=False):
   k=di,t['task_record_rowid'];roles[role]+=1;counts['role_bindings']+=1
   if k not in seen:
    assert a['api_rowid']in t['native_linked_API_rowids']and t['connectionId']in ci[a['connectionId']],(k,role)
    d=defs[t['globalTaskId']];seen[k]=len(bindings);bindings.append(dict(database_index=di,API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},selected_frames=fs[a['callchainId']],roles=[role],new_source_CID_binding=new))
   elif role not in bindings[seen[k]]['roles']:bindings[seen[k]]['roles'].append(role)
   return seen[k]
  def old(s,role):return add(apis[s['API']['api_rowid']],ts[s['TASK']['task_record_rowid']],role)
  def select(scope,path,line,op,role):
   hits=[(a,t)for a in scope if source(a,path,line)for t in byapi[a['api_rowid']]if t['globalTaskId']in defs and defs[t['globalTaskId']]['opType_text']==op];assert len(hits)==1,(di,role,len(hits));a,t=hits[0];return add(a,t,role,True)
  def collective(scope,line,role):
   hits=[a for a in scope if a['name_text']=='HcclAllGather'and source(a,'/models/deepseek_v41/model.py',line)];assert len(hits)==1,(di,line,len(hits));a=hits[0];co=[dict(x)for x in db.execute('SELECT rowid AS COMM_OP_rowid,*FROM COMMUNICATION_OP')if x['connectionId']in ci[a['connectionId']]];assert len(co)==1
   c=co[0];ct={x['id']:x['name']for x in db.execute('SELECT *FROM ENUM_HCCL_DATA_TYPE')};idx=len(comms);comms.append(dict(database_index=di,role=role,API=a,COMM=c,dtype_text=ct[c['dataType']],selected_frames=fs[a['callchainId']],linked_TASK_rowids=[t['task_record_rowid']for t in byapi[a['api_rowid']]],raw_API_connection_ids=sorted(ci[a['connectionId']]),raw_API_connection_TASK_rows=[t for t in rawtasks if t['connectionId']in ci[a['connectionId']]],raw_COMM_exact_connection_TASK_rowids=[t['task_record_rowid']for t in rawtasks if t['connectionId']==c['connectionId']],raw_connection_membership_not_unique_Compute_TASK_role_proof=True,formal_request_or_clock_binding=False));return idx
  rows={(r['forward_API_rowid'],r['source_block_ordinal']):r for r in tr[di]}
  for root in roots.get(di,[]):
   a=root['forward_API'];r=rows[a['api_rowid'],39];scope=[x for x in apis.values()if x['globalTid']==a['globalTid']and int(a['startNs'])<=int(x['startNs'])<=int(x['endNs'])<=int(a['endNs'])];st={}
   for role,line,op in [('MTP_SNAPSHOT',1600,'TensorMove'),('COLLAPSE_MUL',1602,'Mul'),('COLLAPSE_SUM',1602,'ReduceSum'),('TERMINAL_NORM',1605,'RmsNorm')]:st[role]=select(scope,'/models/deepseek_v41/model.py',line,op,role)
   ca=[(x,t)for x in scope if source(x,'/models/deepseek_v41/model.py',1602)for t in byapi[x['api_rowid']]if t['globalTaskId']in defs and defs[t['globalTaskId']]['opType_text']=='Cast'];assert len(ca)==2
   for x,t in ca:
    role='COLLAPSE_CAST_IN'if defs[t['globalTaskId']]['inputDataTypes_text']=='DT_BF16'else'COLLAPSE_CAST_OUT';assert role not in st;st[role]=add(x,t,role,True)
   mr=next(x for x in moe['records']if x['database_index']==di and x['forward_API_rowid']==a['api_rowid']and x['source_block_ordinal']==39)
   for role,s in [('LAST_HC',r['stages']['HC']),('LAST_INPUT_NORM',r['stages']['NORM']),('LAST_FFN_NORM',mr['stages']['FFN_NORM']),('LAST_FFN_POST',r['stages']['FFN_POST'])]:st[role]=old(s,role)
   cs={role:collective(scope,line,role)for role,line in [('MTP_HIDDEN_AG',1597),('MTP_PREMIX_AG',1598)]};ar=aa[di,a['api_rowid'],39];targets.append(dict(database_index=di,forward_API_rowid=a['api_rowid'],forward_API=a,stages=st,collectives=cs,cache_rows=ar['cache_rows'],query_rows=ar['query_rows'],owner_rows=ar['carrier_sender_rows'],actual_snapshot_readers_or_all_final_output_rows_bound=False))
  for r in tr[di]:
   if r['aux_mean']:
    aux.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],entering_layer=r['source_block_ordinal'],stages={role:old(s,role)for role,s in [('AUX_MEAN',r['aux_mean']),('AUX_HC',r['stages']['HC']),('AUX_INPUT_NORM',r['stages']['NORM'])]},later_draft_bank_identity_bound=False))
  for r in dr[di]:
   scope=[a for a in apis.values()if a['globalTid']==r['seed']['API']['globalTid']and int(r['collapse']['CAST_OUT']['API']['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(r['seed']['API']['startNs'])];hd=select(scope,'/models/deepseek_v41/dspark.py',332,'RmsNorm','DRAFT_HEAD_NORM');layers=[]
   for l in r['layers']:layers.append(dict(ordinal=l['source_layer_ordinal'],INPUT_NORM=old(l['stages']['NORM'],'DRAFT_INPUT_NORM'),FFN_NORM=old(l['stages']['FFN_NORM'],'DRAFT_FFN_NORM')))
   collapse={k:old(s,'DRAFT_'+k)for k,s in r['collapse'].items()};main={};pc=r['context_parent']['preceding_main_projection']
   if pc:
    for role,field in [('MAIN_CONTEXT_PROJ','GEMM'),('MAIN_CONTEXT_NORM','full5120_norm')]:
     t=ts[pc[field]['TASK_rowid']];cand=[apis[x]for x in t['native_linked_API_rowids']if source(apis[x],'/models/deepseek_v41/dspark.py',282)];assert len(cand)==1,(di,role,len(cand));main[role]=add(cand[0],t,role)
   drafts.append(dict(database_index=di,source_order_ordinal=r['source_order_ordinal'],ancestry=r['source_ancestry'],context_rows=r['context_parent']['context_rows'],query_rows=r['context_parent']['query_KV_rows'],layers=layers,collapse=collapse,head_norm=hd,main=main,actual_request_context_query_weight_or_epoch_identity_bound=False))
  for b in br[di]:
   nr=rows[b['forward_API_rowid'],b['to_block']];bridges.append(dict(database_index=di,forward_API_rowid=b['forward_API_rowid'],from_block=b['from_block'],to_block=b['to_block'],previous_FFN_Post=old(b['previous_FFN_Post'],'PREVIOUS_FFN_POST'),next_input_norm=old(nr['stages']['NORM'],'NEXT_INPUT_NORM'),Engram_positive=bool(b['next_Engram_positive_APIs']),aux_positive=bool(b['next_aux_mean']),all_previous_Post_readers_bound=False))
  pins.append(dict(database_index=di,raw_database=pin(raw),cached_native=pin(cp),unresolved_unselected_stack_ids=sorted(unresolved)));db.close()
 assert len(targets)==24 and len(drafts)==104 and len(aux)==72 and len(bridges)==936 and len(comms)==48
 out=dict(target_tails=targets,draft_frontiers=drafts,aux_frontiers=aux,interlayer_bridges=bridges,bindings=bindings,collectives=comms,databases=pins,unique_TASK_bindings=len(bindings),new_source_CID_TASK_bindings=sum(x['new_source_CID_binding']for x in bindings),role_binding_count=counts['role_bindings'],roles=dict(roles),new_native_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 write(O/'norm-terminal-native-frontiers-v146.json',out);print(json.dumps({k:v for k,v in out.items()if k in ['unique_TASK_bindings','new_source_CID_TASK_bindings','role_binding_count','roles','new_device_work','overall_complete']}))


if __name__=='__main__':main()
