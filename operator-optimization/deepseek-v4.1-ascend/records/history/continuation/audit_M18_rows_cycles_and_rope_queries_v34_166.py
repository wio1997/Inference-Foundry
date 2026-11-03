from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');M=P/'native-PMU-complete-v18';OUT=M/'M18-aux-row-root-and-source-ordered-KV-cycle-audit-v34.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
d=pin(M/'M18-allrank-aux-projection-context-KV-and-graph-checkpoint-scope-v34.json','f3ffdf3f613cc129f213cbf9ea4dd303f2bc26f061f1e2b6424453edb6f0e2d7');prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
native=d['native_TASKs'];root_rows=[]
for root in prep['prepare_roots']:
 key=root['root_key'];nn=[r for r in native if r['roles']==['TARGET_AUX_MEAN_GATHER']and r['prepare_root_candidates']==[key]];co=[r for r in d['collectives']if r['roles']==['TARGET_AUX_MEAN_GATHER']and r['prepare_root_candidates']==[key]];ss=[r for r in d['graph_aux_segments']if r['prepare_root_keys']==[key]]
 means=[r for r in nn if r['definition']['opType_text']=='ReduceMean'];graph_means=[r for s in ss for r in s['compute']if r['definition']['opType_text']=='ReduceMean']
 localrows=sorted({int(r['definition']['inputShapes_text'].strip('"').split(',')[0])for r in means+graph_means})
 root_rows.append(dict(root_key=key,native_aux_means=len(means),native_aux_COMM=len(co),graph_segments=len(ss),local_mean_rows=localrows,physical_global_rows=[x*8 for x in localrows],native_COMM_counts=[r['operation']['count']for r in co],graph_before_layers=[s['before_layer']for s in ss]))
f29=[r for r in native if all(x.startswith('F29_')for x in r['roles'])];api_rows={(r['database_index'],r['API']['api_rowid']):r for r in d['native_APIs']if r['role'].startswith('F29_')};by_thread=defaultdict(list);tasks_by_api=defaultdict(list)
for (di,key),row in api_rows.items():by_thread[(di,row['API']['globalTid'],row['phase'])].append(row)
for r in f29:
 for a in r['source_APIs']:tasks_by_api[(r['database_index'],a['api_rowid'])].append(r)
cycles=[];unclosed=[];covered=set()
for (di,tid,phase),aa in by_thread.items():
 aa.sort(key=lambda r:(r['API']['startNs'],r['API']['api_rowid']))
 anchors=[j for j,r in enumerate(aa)if any(t['definition']and t['definition']['opType_text']=='MatMulV2'for t in tasks_by_api[(di,r['API']['api_rowid'])])]
 for n,start in enumerate(anchors):
  end=anchors[n+1]if n+1<len(anchors)else len(aa);part=aa[start:end];store=[j for j,r in enumerate(part)if any(t['definition']and t['definition']['opType_text']=='ScatterNdUpdateSk'for t in tasks_by_api[(di,r['API']['api_rowid'])])]
  if len(store)!=1:unclosed.append(dict(database_index=di,globalTid=tid,phase=phase,anchor_API=part[0]['API']['api_rowid'],store_anchors=len(store)));continue
  part=part[:store[0]+1];tt={t['TASK']['task_record_rowid']:t for r in part for t in tasks_by_api[(di,r['API']['api_rowid'])]};types=Counter(t['definition']['opType_text']if t['definition']else'NO_DEFINITION'for t in tt.values())
  if types['MatMulV2']!=1 or types['RmsNorm']!=1 or types['ScatterNdUpdateSk']!=1:
   unclosed.append(dict(database_index=di,globalTid=tid,phase=phase,anchor_API=part[0]['API']['api_rowid'],types=dict(types)));continue
  gemm=next(t['definition']for t in tt.values()if t['definition']['opType_text']=='MatMulV2');T=int(gemm['inputShapes_text'].strip('"').split(',')[0]);covered.update((di,key)for key in tt)
  cycles.append(dict(database_index=di,globalTid=tid,phase=phase,rows=T,anchor_API=part[0]['API']['api_rowid'],store_API=part[-1]['API']['api_rowid'],API_rowids=[r['API']['api_rowid']for r in part],TASK_rowids=sorted(tt),types=dict(types),source_grade='SOURCE_ORDER_PATTERN_NOT_PYTHON_INVOCATION_POINTER_GENERATION_PROOF'))
f27=[r for r in native if r['roles']==['F27_COMBINE']];f27_gemm=[r for r in f27 if r['definition']['opType_text']=='MatMulV2'];idx=[r for r in native if r['roles']==['AUX_ROWSELECT_CONCAT']and r['definition']['opType_text']=='Index'];cat=[r for r in native if r['roles']==['AUX_ROWSELECT_CONCAT']and r['definition']['opType_text']=='ConcatD']
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),target_aux_roots_with_work=sum(bool(r['native_aux_means']or r['graph_segments'])for r in root_rows),target_aux_root_regimes=[dict(native_means=k[0],native_COMM=k[1],graph_segments=k[2],local_rows=json.loads(k[3]),roots=v)for k,v in Counter((r['native_aux_means'],r['native_aux_COMM'],r['graph_segments'],json.dumps(r['local_mean_rows']))for r in root_rows).items()],graph_aux_segments=len(d['graph_aux_segments']),graph_aux_segments_nonunique_root=sum(len(r['prepare_root_keys'])!=1 for r in d['graph_aux_segments']),F27_GEMMs=len(f27_gemm),F27_row_regimes=dict(Counter(int(r['definition']['inputShapes_text'].strip('"').split(',')[0])for r in f27_gemm)),aux_Index_TASKs=len(idx),aux_Concat_TASKs=len(cat),aux_Index_shape_regimes=[dict(input=k[0],output=k[1],TASKs=v)for k,v in Counter((r['definition']['inputShapes_text'],r['definition']['outputShapes_text'])for r in idx).items()],F29_scope_TASKs=len(f29),source_order_complete_KV_cycles=len(cycles),unclosed_cycle_anchors=len(unclosed),covered_F29_TASKs=len(covered),unassigned_F29_TASKs=sum((r['database_index'],r['TASK']['task_record_rowid'])not in covered for r in f29),cycle_type_hist=[dict(types=json.loads(key),cycles=v)for key,v in Counter(json.dumps(r['types'],sort_keys=True)for r in cycles).items()],cycle_phase_rows=[dict(phase=k[0],rows=k[1],cycles=v)for k,v in Counter((r['phase'],r['rows'])for r in cycles).items()],whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,source_sha256='f3ffdf3f613cc129f213cbf9ea4dd303f2bc26f061f1e2b6424453edb6f0e2d7',target_aux_root_rows=root_rows,source_order_KV_cycles=cycles,unclosed_cycle_anchors=unclosed,limits=['Same-threadGEMM->scatterpatternrequirescompleteexpectedsource-order,notnearest-time;itstilldoesnotattestactualPythoncall/request/positions/slot/storagegeneration.','NoF27proposalismatchedtotargetrootbycoincidentequalcounts.','312auxcheckpointmeansand312KVpipelinesaredifferentpopulations;donotpairthembynumber.','CycleC/Squeryrepetitionisnotproofpositions/tablesequalacrossdifferentcyclesoriterations;source-scopedCSErequiresstableinputepoch.','Counts/norm/shape/layoutarenotphysicaltrafficorrequestgain;v20functionalresourcesnotretagged.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
