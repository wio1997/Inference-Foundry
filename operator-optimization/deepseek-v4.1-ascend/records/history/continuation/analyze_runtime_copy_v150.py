"""Reconcile saved DMA control origins; no native execution or source-code read."""
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_right
import sqlite3,json,hashlib
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def index(rows):
 groups=defaultdict(list)
 for r in rows:groups[r['globalTid']].append(r)
 out={}
 for k,v in groups.items():
  v.sort(key=lambda r:(r['startNs'],r['endNs'],r['api_rowid']));mx=[];m=-1
  for r in v:m=max(m,r['endNs']);mx.append(m)
  out[k]=(v,[r['startNs']for r in v],mx)
 return out


def containers(ix,child):
 v,starts,mx=ix.get(child['globalTid'],([],[],[]));j=bisect_right(starts,child['startNs'])-1;out=[]
 while j>=0 and mx[j]>=child['endNs']:
  r=v[j]
  if r['startNs']<=child['startNs'] and child['endNs']<=r['endNs']:out.append(r)
  j-=1
 return sorted(out,key=lambda r:r['api_rowid'])


def strict_innermost(rows):
 return [r for r in rows if not any(q['api_rowid']!=r['api_rowid']and r['startNs']<=q['startNs']and q['endNs']<=r['endNs']and(r['startNs'],r['endNs'])!=(q['startNs'],q['endNs'])for q in rows)]


def logical_extents(shapes,dtypes):
 if not isinstance(shapes,str)or not isinstance(dtypes,str):return None
 ss=[s.strip()for s in shapes.split(';')];ds=[s.strip()for s in dtypes.split(';')];out=[]
 widths={'bool':1,'char':1,'byte':1,'short':2,'int':4,'long':8,'half':2,'float':4,'double':8,'bfloat16':2,'complex64':8,'complex128':16}
 for i in range(min(len(ss),len(ds))):
  s,d=ss[i],ds[i]
  if not d or d=='Scalar':continue
  chunks=s.split(',')
  if not s or d not in widths or not all(z.strip().isdigit()for z in chunks):return None
  dims=[int(z.strip())for z in chunks];nel=1
  for z in dims:nel*=z
  out.append(dict(input_index=i,shape=dims,reported_dtype=d,assumed_element_bytes=widths[d],numel=nel,logical_extent_bytes=nel*widths[d]))
 return out or None


def main():
 assert not(O/'runtime-copy-control-origins-v150.json').exists()
 census=read(O/'task-namespace-census-v149.json');copies=read(O/'ta<REDACTED_TOKEN>.json')['runtime_copy_records'];bydb=defaultdict(list)
 for c in copies:bydb[c['database_index']].append(c)
 results=[];chains={};graphs=[];dbs=[];ct=Counter();sourcecounts=Counter();parentcounts=Counter();paths_hist=Counter();masks=Counter();sourceapis=set();queueflows=set();graphhits=Counter()
 for spec in census['databases']:
  di=spec['database_index'];p=spec['raw_database'];assert pin(p['path'])==p
  d=sqlite3.connect('file:'+p['path']+'?mode=ro',uri=True);d.row_factory=sqlite3.Row;ss=dict(d.execute('select id,value from STRING_IDS'));ci=defaultdict(set)
  for z in d.execute('select id,connectionId from CONNECTION_IDS'):ci[z[0]].add(z[1])
  pa=[]
  for z in d.execute('select rowid AS api_rowid,* from PYTORCH_API'):
   z=dict(z);z['startNs']=int(z['startNs']);z['endNs']=int(z['endNs']);z['name_text']=ss.get(z['name']);z['input_shapes_text']=ss.get(z['inputShapes']);z['input_dtypes_text']=ss.get(z['inputDtypes']);pa.append(z)
  pix=index(pa);ebyflow=defaultdict(list)
  for a in pa:
   if a['type']==50002 and a['name_text'].startswith('Enqueue@'):
    for fl in ci[a['connectionId']]:ebyflow[fl].append(a)
  chbyid=defaultdict(list)
  for z in d.execute('select id,stack,stackDepth from PYTORCH_CALLCHAINS order by id,stackDepth'):chbyid[z[0]].append(dict(stackDepth=z[2],frame=ss.get(z[1]),raw_stack_STRING_ID=z[1]))
  def chain(cid):
   if cid is None:return None
   frames=chbyid[cid];assert frames
   key=hashlib.sha256(json.dumps([r['frame']if r['frame']is not None else dict(unresolved_local_database=di,STRING_ID=r['raw_stack_STRING_ID'])for r in frames],ensure_ascii=False,separators=(',',':')).encode()).hexdigest();chains.setdefault(key,dict(frames=frames,all_frame_text_resolved=all(z['frame']is not None for z in frames),unresolved_frame_STRING_IDs=sorted({z['raw_stack_STRING_ID']for z in frames if z['frame']is None}),project_frames=[z['frame']for z in frames if z['frame']is not None and('/vllm_ascend/'in z['frame']or'/vllm/vllm/'in z['frame'])]))
   return key
  ni=N/'original-profile-cache-v53/analysis/native-execution-identity-v14'/('db-%02d.json'%di);native=read(ni);assert native['database_sha256']==p['sha256'];tasks={z['task_record_rowid']:z for z in native['TASK_records']};controls=[]
  for gb in native['graph_API_control_bindings']:
   if not gb['control_pair_valid']:continue
   cr=[tasks[tid]for tid in gb['control_task_rowids']];execute=[z for z in cr if z['task_type_name']=='MODEL_EXECUTE'];wait=[z for z in cr if z['task_type_name']=='MODEL_WAIT_COMPLETE'];assert len(execute)==len(wait)==1
   a,b=execute[0],wait[0];assert a['deviceId']==b['deviceId']and a['globalPid']==b['globalPid']and a['modelId']==b['modelId']==gb['model_id']and a['connectionId']==b['connectionId']and a['startNs']==gb['control_pair_start_ns']and b['endNs']==gb['control_pair_end_ns']
   controls.append(dict(database_index=di,model_id=gb['model_id'],CANN_API=gb['CANN_API'],MODEL_EXECUTE=a,MODEL_WAIT_COMPLETE=b,prepare_root_keys=gb['prepare_root_keys'],phase_records=gb['phase_records'],same_thread_markers=gb['same_thread_markers'],native_model_control_not_operand_identity=True))
  graphs+=controls
  for old in bydb[di]:
   t=old['TASK'];r=dict(database_index=di,TASK=t,exact_CANN_calls=old['CANN_API_same_connection'],direct_native_PYTORCH_family_count=0,host_control_paths=[],graph_control_candidates=[],native_byte_direction_pointer_epoch_not_proved=True)
   for c in old['CANN_API_same_connection']:
    assert c['globalTid']>>32==t['globalPid'];parents=containers(pix,c);dqs=[a for a in parents if a['type']==50002 and a['name_text'].startswith('Dequeue@')];pp=[]
    for dq in dqs:
     nm=dq['name_text'][len('Dequeue@'):];flow=ci[dq['connectionId']];ens={a['api_rowid']:a for fl in flow for a in ebyflow[fl]if a['name_text']=='Enqueue@'+nm}
     for en in ens.values():
      sources=[a for a in containers(pix,en)if a['type']==50001 and a['name_text']==nm];inner=strict_innermost(sources)
      for a in inner:
       assert a['globalTid']>>32==dq['globalTid']>>32==t['globalPid'];ck=chain(a['callchainId']);cp=[z for z in containers(pix,a)if z['type']==50001 and z['name_text']=='aten::copy_'and chain(z['callchainId'])==ck];inncp=strict_innermost(cp)
       path=dict(CANN_API=c,Dequeue=dq,queue_flow_ids=sorted(flow&ci[en['connectionId']]),Enqueue=en,source_API=a,source_chain_key=ck,chain_text_complete=(chains[ck]['all_frame_text_resolved']if ck is not None else False),chain_match_rule='ResolvedframecontentoridenticalunresolvedlocalSTRING_IDreferences;notinventedmissingtext.',all_same_named_source_container_rowids=[z['api_rowid']for z in sources],same_chain_copy_parent_candidates=cp,innermost_same_chain_copy_parents=inncp,CPU_enqueue_not_after_dequeue=en['endNs']<=dq['startNs'],direction_source_label={'acl_memcpy_host_to_device':'SOURCE_LABEL_HOST_TO_DEVICE','acl_memcpy_device_to_host':'SOURCE_LABEL_DEVICE_TO_HOST'}.get(nm,'SOURCE_PRIMITIVE_DIRECTION_UNRESOLVED'),logical_copy_parent_extents=logical_extents(inncp[0]['input_shapes_text'],inncp[0]['input_dtypes_text'])if len(inncp)==1 else None,logical_extents_not_DMA_length_or_physical_traffic=True)
       pp.append(path)
    r['host_control_paths']+=pp;r.setdefault('native_CANN_host_container_rowids',[]).append(dict(CANN_connection=c['connectionId'],all_host_container_rowids=[a['api_rowid']for a in parents],Dequeue_candidates=len(dqs)))
   for g in controls:
    a,b=g['MODEL_EXECUTE'],g['MODEL_WAIT_COMPLETE']
    if t['modelId']==g['model_id'] and t['globalPid']==a['globalPid']and t['deviceId']==a['deviceId']and a['startNs']<=t['startNs']<=t['endNs']<=b['endNs']:
     r['graph_control_candidates'].append(dict(CANN_connection=g['CANN_API']['connectionId'],MODEL_EXECUTE_rowid=a['task_record_rowid'],MODEL_WAIT_COMPLETE_rowid=b['task_record_rowid'],model_id=g['model_id'],phase_already_bound=bool(g['phase_records']),operand_template_source_ordinal_not_inferred=True))
   pp=r['host_control_paths'];r['unique_host_queue_source_control']=len(pp)==1;r['source_graph_or_phase_promoted']=False;paths_hist[str(len(pp))]+=1;graphhits[str(len(r['graph_control_candidates']))]+=1
   if len(pp)==1:
    q=pp[0];sourcecounts[q['source_API']['name_text']]+=1;parentcounts[str(len(q['innermost_same_chain_copy_parents']))]+=1;sourceapis.add((di,q['source_API']['api_rowid']));queueflows.add((di,tuple(q['queue_flow_ids'])));masks[bool(q['logical_copy_parent_extents']),q['direction_source_label']]+=1
   ct['native_CANN_link'if r['exact_CANN_calls']else'no_native_CANN_link']+=1;results.append(r)
  dbs.append(dict(database_index=di,raw_database=p,native_v14_input=pin(ni),copy_records=len(bydb[di])));d.close();print(json.dumps(dict(database_index=di,copies=len(bydb[di]),processed=len(results))),flush=True)
 assert len(results)==16342
 inputs=[pin(O/'task-namespace-census-v149.json'),pin(O/'ta<REDACTED_TOKEN>.json'),pin(O/'ta<REDACTED_TOKEN>.json'),pin(O/'raw-profile-capability-audit-v98.json')]+[z['raw_database']for z in dbs]+[z['native_v14_input']for z in dbs]
 write(O/'runtime-copy-inputs-v150.json',dict(inputs=inputs,new_remote_or_device_activity=False))
 write(O/'runtime-copy-control-origins-v150.json',dict(count=len(results),records=results,databases=dbs,canonical_chains=chains,source_API_call_count=len(sourceapis),queue_flow_group_count=len(queueflows),host_control_path_count_hist=dict(paths_hist),source_primitive_counts=dict(sourcecounts),copy_parent_count_hist=dict(parentcounts),logical_extent_and_direction_label_counts=[dict(logical_extents_present=k[0],source_direction_label=k[1],records=v)for k,v in masks.items()],native_CANN_link_counts=dict(ct),graph_control_candidate_count_hist=dict(graphhits),new_source_read=False,new_device_work=False,overall_complete=False))
 write(O/'runtime-copy-existing-graph-controls-v150.json',dict(count=len(graphs),controls=graphs,from_existing_v14_control_proofs=True,new_phase_binding=False,member_identity_rule='SamePID/device/modelANDrecordedMODEL_EXECUTE-throughMODEL_WAIT_COMPLETEcontainment;notmodelIdalone/count/ordinal. This is recorded graph-execution membership,notcopytemplate/phase/address/dataflow.',overall_complete=False))
 print(json.dumps(dict(copies=len(results),host_paths=dict(paths_hist),source_counts=dict(sourcecounts),parentcounts=dict(parentcounts),source_API_calls=len(sourceapis),queue_flow_groups=len(queueflows),graph_candidates=dict(graphhits),new_device_work=False)))


if __name__=='__main__':main()
