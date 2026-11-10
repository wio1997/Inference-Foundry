#!/usr/bin/env python3
"""Read saved CPU span parent IDs under the owned data root, no new collection."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
ROOT=Path('audit/execution-search');TIMING=Path('audit/framework-timing-171')
REMOTE=r'''
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import json,sys
spec=json.loads(sys.argv[1]);root=Path(spec['root']).resolve()
assert str(root).startswith('/data/tiankuan/wio/')
names={'cp':'vllm_ascend.attention.context_parallel.dsa_v41_cp:AscendDSAV41CPMetadataBuilder.build','local':'vllm_ascend.attention.context_parallel.dsa_cp:AscendDSACPMetadataBuilder._build_local_token_metadata','base':'vllm_ascend.attention.dsa_v41:AscendDSAV41MetadataBuilder.build'}
out=[]
for plan in spec['ranks']:
 rows=[];pins=[]
 for pin in plan['files']:
  path=root/pin['path'];assert path.resolve().is_relative_to(root) and path.parent.name=='timing'
  raw=path.read_bytes();assert sha256(raw).hexdigest()==pin['sha256'];pins.append(pin)
  rows.extend(json.loads(line) for line in raw.splitlines() if line)
 byid={r['id']:r for r in rows if 'id' in r};clock=plan['clock'];origin=clock['origin_us'];offset=clock['offset_us'];slope=clock['slope_ppm']
 def mapped(ns):return origin+(ns/1000-origin-offset)/(1+slope/1e6)
 def ancestry(row):
  a=[];parent=row.get('parent');seen=set()
  while parent is not None:
   assert parent not in seen;seen.add(parent);p=byid[parent];a.append(p);parent=p.get('parent')
  return a
 def classification(a):
  if any(r['name'].endswith('AscendDSAV41MetadataBuilder.build_for_drafting') for r in a):return 'draft_build_for_drafting'
  if any(r['name'].endswith('NPUModelRunner._build_attention_metadata') for r in a):return 'runner_build_attention'
  return 'other'
 counts={k:Counter() for k in names};selected=[];batches=defaultdict(list)
 for r in byid.values():
  if r.get('mode')!='detailed' or r['name'] not in names.values():continue
  start=mapped(r['wall_ns'])
  if not (plan['window_start_us']<=start<plan['window_end_us']):continue
  key=next(k for k,v in names.items() if v==r['name']);anc=ancestry(r);kind=classification(anc);counts[key][kind]+=1
  selected.append(dict(kind=key,caller=kind,id=r['id'],parent=r.get('parent'),wall_ns=r['wall_ns'],duration_ns=r['duration_ns'],fully_enclosed=mapped(r['wall_ns']+r['duration_ns'])<=plan['window_end_us']))
  if key=='local' and kind=='runner_build_attention':
   parent=next(p for p in anc if p['name'].endswith('NPUModelRunner._build_attention_metadata'))
   batches[parent['id']].append(r)
 batch=[]
 for parent,rs in sorted(batches.items()):
  rs.sort(key=lambda r:r['wall_ns']);batch.append(dict(parent_id=parent,helper_calls=len(rs),first_helper_id=rs[0]['id'],first_helper_body_ns=rs[0]['duration_ns'],total_helper_body_ns=sum(r['duration_ns'] for r in rs)))
 out.append(dict(ep=plan['ep'],counts={k:dict(c) for k,c in counts.items()},target_batches=batch,selected=selected,files=pins,clock=clock,FT_body_unmatched_in_original_analysis=plan['unmatched']))
print(json.dumps(dict(status='READ_ONLY_ARCHIVED_CPU_PARENT_CLASSIFICATION',ranks=out,new_profile=0,new_NPU_work=0,service_requests=0,remote_writes=[])))
'''
def main():
    fine=json.loads((TIMING/'function-analysis.json').read_text());inv={r['path']:r for r in json.loads((TIMING/'raw-inventory.json').read_text())};receipt=json.loads((TIMING/'receipt.json').read_text());assert receipt['status']=='CAPTURED_RESTORED'
    root=receipt['output_host'];plans=[]
    for r in fine['ranks']:
        pins=[inv['timing/spans-'+str(pid)+'.jsonl'] for pid in r['python_pids']]
        plans.append(dict(ep=r['ep'],clock=r['clock_alignment'],window_start_us=r['window_start_us'],window_end_us=r['window_end_us'],unmatched=r['body_marker_unmatched'],files=pins))
    spec=dict(root=root,ranks=plans)
    result=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=110)
    if result.returncode:raise RuntimeError(result.stderr)
    d=json.loads(result.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [TIMING/'function-analysis.json',TIMING/'raw-inventory.json',TIMING/'receipt.json']],scope='Same saved CPU process-local parentIDs, sourcecaller classification and clockwindow only. No tensorarguments or exact firstFree overlap. Original2FT/body misses retained.')
    (ROOT/'common-frame-caller-logs.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(ep=r['ep'],counts=r['counts'],target_batch_helper_calls=[b['helper_calls'] for b in r['target_batches']]) for r in d['ranks']],ensure_ascii=False))
if __name__=='__main__':main()
