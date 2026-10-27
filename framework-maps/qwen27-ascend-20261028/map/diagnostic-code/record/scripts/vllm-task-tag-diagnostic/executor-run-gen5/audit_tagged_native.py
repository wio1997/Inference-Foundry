"""Exact tagged-graph keys inside explicit flow-linked native replay endpoints.

CPU-only. Inputs are immutable; unresolved identities are data, not filled gaps.
This does not estimate exclusive kernels, all-stream critical paths or performance.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,gzip,hashlib,json,sys

ROOT=Path(__file__).resolve().parent.parent
HELPER=ROOT/'native-sg-recovery/gen66/audit_vllm_gen66_by_rank.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='8eb88fff7e4b4c33bf6876387deb8cdc02b1e4112cd405f32f51121856b789cb','reviewed helper changed'
sys.path.insert(0,str(HELPER.parent))
from audit_vllm_gen66_by_rank import load_indexes,_resolve_outer,_native_key,event_end,dec,ds

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify(root,receipt,name):
 row=next(x for x in receipt['files'] if x['relative_path']==name)
 p=root/name
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
 return p

def classify(markers,outer,events,indexes,hardware_pid,label):
 """No ordinal pairing: each invocation comes only from explicit callback identity."""
 metadata,scopes,cpu,flows,native_at,native=indexes
 resolved=[];failures=[];grouped=defaultdict(dict)
 for row in outer:
  item,errors=_resolve_outer(row,scopes,cpu,flows,native_at,hardware_pid,label)
  if errors:failures.append({'invocation':row['invocation'],'side':row['side'],'errors':errors});continue
  resolved.append(item['row']);sides=grouped[row['invocation']]
  assert row['side'] not in sides,'duplicate explicit invocation/side'
  sides[row['side']]=item
 fences=[]
 for inv,sides in grouped.items():
  if set(sides)!={'start','end'}:
   failures.append({'invocation':inv,'errors':['missing_resolved_side']});continue
  a,b=sides['start']['native_event'],sides['end']['native_event']
  lo,hi=event_end(a),dec(b.get('ts'))
  if str(a['pid'])!=str(b['pid']) or str(a['tid'])!=str(b['tid']) or lo is None or hi is None or hi<=lo:
   failures.append({'invocation':inv,'errors':['invalid_native_endpoint_interval_or_stream']});continue
  fences.append({'invocation':inv,'lo':lo,'hi':hi,'start':sides['start']['row']['native_event_index'],'end':sides['end']['row']['native_event_index']})
 fences.sort(key=lambda f:f['lo'])
 overlaps=[{'previous':a['invocation'],'current':b['invocation']}for a,b in zip(fences,fences[1:])if a['hi']>b['lo']]
 keys={tuple(m['raw_task_key']):m for m in markers}
 assert len(keys)==len(markers),'duplicate accepted raw marker key'
 by_key=defaultdict(list)
 for i,e in native:
  # Raw graph uses EVENT_RECORD; the installed SDK trace spells it EVENT RECORD.
  # Keep both explicit spellings, never accept arbitrary task types by task ID.
  if str(e.get('pid'))==str(hardware_pid) and e.get('args',{}).get('Task Type') in {'EVENT_RECORD','EVENT RECORD'}:
   k=_native_key(e)
   if k in keys:by_key[k].append((i,e))
 matches=[];unknown=[];intervals=[];selected=set()
 # With any overlapping outer intervals, do not infer unique replay ownership.
 for f in fences:
  slots=defaultdict(dict)
  for k,m in keys.items():
   candidates=[(i,e)for i,e in by_key[k]if dec(e.get('ts')) is not None and event_end(e)is not None and f['lo']<=dec(e['ts']) and event_end(e)<=f['hi']]
   if overlaps or len(candidates)!=1:
    unknown.append({'invocation':f['invocation'],'marker_id':m['marker_id'],'key':k,'reason':'overlapping_outer_intervals'if overlaps else 'marker_candidate_count','candidate_count':len(candidates),'candidate_event_indices':[i for i,e in candidates]});continue
   i,e=candidates[0];selected.add(i)
   row={'invocation':f['invocation'],'marker_id':m['marker_id'],'layer_idx':m['layer_idx'],'side':m['marker_side'],'key':k,'native_event_index':i,'ts_us':ds(dec(e['ts'])),'dur_us':ds(dec(e.get('dur',0)))}
   matches.append(row);slots[m['layer_idx']][m['marker_side']]=(i,e)
  for layer in sorted({m['layer_idx']for m in markers}):
   sides=slots[layer]
   if set(sides)!={'pre','post'}:continue
   i,a=sides['pre'];j,b=sides['post'];lo,hi=event_end(a),dec(b['ts'])
   if lo is None or hi is None or hi<lo:
    unknown.append({'invocation':f['invocation'],'layer_idx':layer,'reason':'invalid_layer_endpoint_order'});continue
   intervals.append({'invocation':f['invocation'],'layer_idx':layer,'pre_event_index':i,'post_event_index':j,'start_us':ds(lo),'end_us':ds(hi),'endpoint_elapsed_us':ds(hi-lo)})
 remainder=[]
 for i,e in native:
  if str(e.get('pid'))!=str(hardware_pid) or i in selected:continue
  start,end=dec(e.get('ts')),event_end(e)
  owners=[f['invocation']for f in fences if None not in(start,end)and f['lo']<=start and end<=f['hi']]
  crossing=[f['invocation']for f in fences if None not in(start,end)and start<f['hi']and f['lo']<end and not(f['lo']<=start and end<=f['hi'])]
  remainder.append({'event_index':i,'key':_native_key(e),'name':e.get('name'),'task_type':e.get('args',{}).get('Task Type'),'ts_us':ds(start),'dur_us':ds(dec(e.get('dur',0))),'candidate_outer_owners':owners,'crossing_outer_intervals':crossing,'classification':'crossing'if crossing else 'other_native_row_in_outer'if owners else 'outside_accepted_outer'})
 return {'counts':{'outer_callbacks':len(outer),'resolved_endpoints':len(resolved),'explicit_outer_pairs':len(fences),'exact_marker_occurrences':len(matches),'layer_endpoint_intervals':len(intervals),'unknown_marker_or_order_rows':len(unknown),'other_native_rows_retained':len(remainder)},'resolved_endpoints':resolved,'unresolved_outer_rows':failures,'outer_overlaps':overlaps,'outer_pairs':[dict(invocation=f['invocation'],start_us=ds(f['lo']),end_us=ds(f['hi']),start_event_index=f['start'],end_event_index=f['end'])for f in fences],'marker_occurrences':matches,'unknown_markers':unknown,'layer_endpoint_intervals':intervals},remainder

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True,choices=range(4));ap.add_argument('--sdk',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
 acceptance=ROOT/'executor-run-gen4/model-raw-identity-acceptance.json';raw=json.loads(acceptance.read_text())
 assert sha(acceptance)=='7c47007ef4cff71d53d81787073b20f8e1daf47f5afd63982c3690fdf3d12c48'
 assert raw['executor_audit'] and raw['status']=='executor_verified_512_exact_raw_EVENT_RECORD_identities'
 for row in raw['inputs']:
  p=Path(row['path']);assert sha(p)==row['sha256'],str(p)
 snapshot=ROOT/'executor-run-gen4/model-snapshot';download=json.loads((snapshot/'snapshot-download.json').read_text())
 gp=verify(snapshot,download,'vllm-task-tag-model-gen81-exec-g3/state.json');gstate=json.loads(gp.read_text())
 outerp=verify(snapshot,download,'vllm-task-tag-model-gen81-exec-g3/replay-outer-rows.json')
 markers=[m for m in raw['rows']if m['rank_id']==a.rank];assert len(markers)==128
 pid={m['producer_pid']for m in markers};graph={m['graph_python_id']for m in markers};devices={m['device']for m in markers}
 assert len(pid)==len(graph)==len(devices)==1
 outer=[r for r in json.loads(outerp.read_text())if r['rank_id']==a.rank]
 assert {r['pid']for r in outer}==pid and {r['graph_python_id']for r in outer}==graph
 pairs=defaultdict(set)
 for r in outer:pairs[r['invocation']].add(r['side'])
 assert all(s=={'start','end'}for s in pairs.values()) and len(outer)==2*len(pairs)
 receipts=json.loads((a.sdk/'receipts-download.json').read_text())
 sp=verify(a.sdk,receipts,'state.json');ip=verify(a.sdk,receipts,'slot-intent.json');mp=verify(a.sdk,receipts,f'rank{a.rank}-raw-copy.json')
 state=json.loads(sp.read_text());assert state['status'].startswith('completed_')and state.get('cleanup')and not state.get('cleanup_error')
 assert state['raw_identity_gate']['receipt_sha256']==sha(acceptance)
 assert json.loads(ip.read_text())['phase']=='resumed'
 inp=next(x for x in state['inputs']if x['kind']==f'rank{a.rank}')
 assert sha(mp)==inp['manifest_sha256'] and int(Path(inp['source']).name.split('_')[1])==next(iter(pid))
 assert Path(inp['source']).parent.name==gstate['serving_wave']
 manifest=json.loads(mp.read_text());assert any(Path(x['relative_path']).name==f'profiler_info_{a.rank}.json'for x in manifest['files'])
 result=inp['result'];assert result['status']=='valid_nonempty_native_export_original_unchanged'
 dl=json.loads((a.sdk/f'rank{a.rank}-download.json').read_text());tp=verify(a.sdk,dl,f'rank{a.rank}-trace.json.gz')
 events,metadata,scopes,cpu,flows,native_at,native,digest,nbytes=load_indexes(tp,a.output.parent)
 assert digest==result['trace_sha256'] and nbytes==result['trace_bytes'] and len(events)==result['event_count']
 label=f'NPU {next(iter(devices))}';hw=[p for p,m in metadata.items()if m.get('process_name')=='Ascend Hardware'and m.get('process_labels')==label];assert len(hw)==1
 out,remainder=classify(markers,outer,events,(metadata,scopes,cpu,flows,native_at,native),hw[0],label)
 rest=a.output.with_suffix('.other-native.jsonl.gz')
 with gzip.open(rest,'xt')as f:
  for row in remainder:f.write(json.dumps(row)+'\n')
 out.update(task_id='Q27-FULL-MAP-20261025',generation=5,rank=a.rank,producer_pid=next(iter(pid)),graph_python_id=next(iter(graph)),device_label=label,review_level='standalone executor; root acceptance pending',scope='Exact keys within explicit native endpoint intervals; no exclusive kernels or all-stream critical path',limitations=['Diagnostic instrumentation timings are separate from frozen performance matrix.','Profiler RECORD-stop warning retained; count gaps remain unknown.','No quality/greedy equivalence or optimization adoption claim.'],inputs=[{'path':str(p),'sha256':sha(p)}for p in [acceptance,outerp,sp,ip,mp,tp,HELPER,Path(__file__)]],trace_uncompressed_sha256=digest,trace_uncompressed_bytes=nbytes,other_native_rows={'path':str(rest),'sha256':sha(rest)})
 out['deployment']={k:gstate.get(k)for k in ['host','host_architecture','container_id','image_id','container_started_at','source_SHA256','scope','serving_wave']}
 out['inputs'].append({'path':str(gp),'sha256':sha(gp)})
 with a.output.open('x')as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'rank':a.rank,'counts':out['counts'],'output':str(a.output)}))

if __name__=='__main__':main()
