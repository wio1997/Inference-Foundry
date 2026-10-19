from pathlib import Path
import hashlib,json,collections
import numpy as np
r=Path('/Users/wio/work/Inference-Foundry-execution-plan/glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2');base=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/D-unused-draft/artifacts/divergence-probe');records=[json.loads(x) for f in base.glob('worker-*.jsonl') for x in f.read_text().splitlines() if x.strip()];records=[x for x in records if x['kind']=='target_tensor_snapshot'];cache={};failures=[];counts=collections.Counter();omissions=collections.Counter();rows=[]
def tensor(info):
 name=info['path']
 if name not in cache:
  data=(base/name).read_bytes();assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'];dtype={'torch.int32':'<i4','torch.int64':'<i8'}[info['dtype']];cache[name]=np.frombuffer(data,dtype=dtype).reshape(info['shape'])
 return cache[name]
for s in records:
 starts=np.asarray(s['query_start_loc']);lengths=np.diff(starts);attention=s['attention'];groups=[]
 for omission in attention['refused']:omissions[omission]+=1
 for g in attention['groups']:
  fields={key:tensor(info) for key,info in g['tensors'].items()};seq=fields['seq_lens'].reshape(-1);ends=fields['cum_query_lens'].reshape(-1);slots=fields['slot_mapping'].reshape(-1);table=fields['block_table'];block=int(g['scalars']['block_size']);actual=int(g['scalars']['num_actual_tokens']);checks={'num_actual_equals_query_total':actual==int(starts[-1]),'GPU_seq_lens_equals_full_cold_query_lengths':seq[:len(lengths)].tolist()==lengths.tolist(),'GPU_cum_query_ends_match':ends[:len(lengths)].tolist()==starts[1:].tolist(),'all_real_slots_nonnegative':bool(np.all(slots[:actual]>=0)),'all_real_slots_unique':len(np.unique(slots[:actual]))==actual}
  if 'query_start_loc' in fields:checks['GPU_query_start_loc_matches']=fields['query_start_loc'][:len(starts)].tolist()==starts.tolist()
  if 'seq_lens_cpu' in g['cpu_values']:checks['CPU_seq_lens_matches_GPU_real_rows']=g['cpu_values']['seq_lens_cpu'][:len(lengths)]==seq[:len(lengths)].tolist()
  used=[];map_ok=True
  for index,n in enumerate(lengths):
   n=int(n);positions=np.arange(n,dtype=np.int64);expected=table[index,positions//block].astype(np.int64)*block+positions%block;observed=slots[int(starts[index]):int(starts[index+1])];map_ok=map_ok and np.array_equal(observed,expected);used.append(table[index,:int((int(seq[index])+block-1)//block)].tolist())
  checks['slot_to_block_mapping_with_metadata_block_size']=bool(map_ok)
  for key,ok in checks.items():counts[key+'_pass' if ok else key+'_fail']+=1
  if not all(checks.values()):failures.append({'request_id':s['request_id'],'rank':s['tp_rank'],'class':g['class'],'checks':checks,'seq':seq.tolist(),'ends':ends.tolist(),'slot_head':slots[:8].tolist(),'table_used':used})
  groups.append({'class':g['class'],'layers':g['layers'],'checks':checks,'used_blocks':used})
 rows.append({'request_id':s['request_id'],'mode':s['diagnostic_mode'],'rank':s['tp_rank'],'groups':groups})
report={'snapshot_records':len(records),'checked_mapping_groups':sum(len(x['groups']) for x in rows),'counts':dict(counts),'failures':failures,'partial_capture_omissions':dict(omissions),'rows':rows,'scope':'Only captured Indexer/SFA metadata objects, target-return snapshot. Real query prefixes only; full unused block-table columns are excluded. Internal seq/query/slot-to-block relation, not cache byte contents or all layer groups. A map mismatch would require checking exact native builder semantics before calling it a fault.'};(r/'unused-sfa-map-summary-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ['snapshot_records','checked_mapping_groups','counts','partial_capture_omissions']}));print(json.dumps({'failures':failures[:2],'failure_count':len(failures)}))
